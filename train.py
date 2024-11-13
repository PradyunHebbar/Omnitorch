import os
import argparse
from datetime import datetime
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

#from torch.optim.lr_scheduler import CosineAnnealingLR
from lion_pytorch import Lion
#from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR
from library.network.utils import CosineDecayWithWarmup

from library.network import PET
from library.network.utils import get_logsnr_alpha_sigma, CosineDecayWithWarmup

from library.dataset import TopDataset, JetClassDataset

from get_logging import log_training_progress, log_roc_data
from sklearn.metrics import roc_curve, auc

def parse_args():
    parser = argparse.ArgumentParser(description="Train the PET model on JetClass dataset.")
    parser.add_argument("--folder", type=str, default="../../Datasets/JetClass/sample/", help="Folder containing input files")
    parser.add_argument("--batch", type=int, default=250, help="Batch size")
    parser.add_argument("--epoch", type=int, default=200, help="Max epoch")
    parser.add_argument("--dataset", type=str, default="top", choices=["jetclass", "top"], 
                        help="Dataset type to use")
    # ---------------------------------------------------------------------------------------------------
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--warm_epoch", type=int, default=3, help="Warm up epochs")
    parser.add_argument("--stop_epoch", type=int, default=30, help="Epochs before reducing lr")
    parser.add_argument("--b1", type=float, default=0.95, help="beta1 for optimizer")
    parser.add_argument("--b2", type=float, default=0.99, help="beta2 for optimizer")
    parser.add_argument("--lr_factor", type=float, default=10., help="factor for slower learning rate")
    parser.add_argument("--fine_tune", action='store_true', default=False, help='Fine tune a model')
   # ---------------------------------------------------------------------------------------------------
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--projection_dim", type=int, default=128, help="Projection dimension")
    parser.add_argument("--drop_probability", type=float, default=0.0, help="Drop probability")
    parser.add_argument("--feature_drop", type=float, default=0.1, help="Feature drop probability")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--local", action="store_true", default=False, help="Use local embedding")
    parser.add_argument("--simple", action="store_true",default=False, help="Use simple head model")
    parser.add_argument("--talking_head", action="store_true",default=False, help="Use talking head attention")
    parser.add_argument("--layer_scale", action="store_true", default=False, help="Use layer scale")
    parser.add_argument("--layer_scale_init", type=float, default=1e-5, help="Layer scale init value")
    #local_rank: Automatically provided by the launcher - torchrun.
    #parser.add_argument("--local_rank", type=int, default=-1, help=" Local rank for distributed training")
    parser.add_argument("--mode", type=str, default="all", help="Loss type to train the model")
    parser.add_argument("--simple_lr", action="store_true", default=False, help="Use a simpler learning rate schedule and optimizer")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth", help="Checkpoint file to save/load")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save logs and checkpoints")
    return parser.parse_args() 

# def setup(seed=42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     dist.init_process_group(backend='nccl', init_method='env://')
#     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def setup(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Get environment variables
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # Set the device
    torch.cuda.set_device(local_rank)
    
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def create_log_directory(args):
    timestamp = datetime.now().strftime("%m_%d_%y")
    log_dir = os.path.join(args.log_dir, f"logs_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def configure_optimizers(params, args, train_loader, lr_factor=1.0):
    world_size = dist.get_world_size()

    if args.simple_lr:
        # Simple configuration
        base_lr = args.lr / world_size  # Simple linear scaling
        optimizer = optim.Adam(
            params,  # to only optimize the given params
            lr=base_lr,
            weight_decay=args.wd
        )
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    else:
        # Original complex configuration
        scale_lr = args.lr * (world_size ** 0.5)
        warmup_start_lr = args.lr / lr_factor
        warmup_target_lr = scale_lr / lr_factor

        warmup_steps = args.warm_epoch * len(train_loader) // world_size
        total_steps = args.epoch * len(train_loader) // world_size

        optimizer = Lion(
            params,  # to only optimize the given params
            lr=warmup_start_lr,
            weight_decay=args.wd * lr_factor,
            betas=(args.b1, args.b2)
        )

        scheduler = CosineDecayWithWarmup(
            optimizer,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            warmup_start_lr=warmup_start_lr,
            warmup_target_lr=warmup_target_lr
        )

    return optimizer, scheduler

def get_dataloader(dataset, batch_size, num_workers=16, distributed=False, shuffle=False, dist=None):
    """
    Common dataloader function that works for both distributed and non-distributed scenarios.
    
    Args:
        dataset: Dataset object (TopDataset or JetClassDataset)
        batch_size: Size of each batch
        num_workers: Number of worker processes for data loading
        distributed: Whether to use distributed training
        shuffle: Whether to shuffle the data
    """
    if distributed:
        if dist is not None:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        sampler = DistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=shuffle
        )
        shuffle_loader = False  # When using DistributedSampler, shuffle is handled by the sampler
    else:
        sampler = None
        shuffle_loader = shuffle  # When not using DistributedSampler, use shuffle parameter directly
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle_loader,
        num_workers=num_workers,
        pin_memory=True
    )

def train(args):
    setup()
    
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    device = torch.device(f'cuda:{local_rank}')
    
    # Create timestamped log directory
    log_dir = create_log_directory(args)
    
    # Dataset
    if args.dataset == "jetclass":
        train_dataset = JetClassDataset(os.path.join(args.folder, 'train'))
        val_dataset = JetClassDataset(os.path.join(args.folder, 'val'))
    elif args.dataset == "top":  # top dataset
        train_dataset = TopDataset(os.path.join(args.folder, 'corrected_train_ttbar.h5'))
        val_dataset = TopDataset(os.path.join(args.folder, 'corrected_val_ttbar.h5'))
    else:
        raise ValueError('Invalid dataset selected :(')
        

    # Create dataloaders using the modified function
    train_loader = get_dataloader(
        train_dataset,
        batch_size=args.batch,
        distributed=True,  # Since this is in the train function which uses DDP
        shuffle=True,
        num_workers=16,
        dist=dist
    )
    
    val_loader = get_dataloader(
        val_dataset,
        batch_size=args.batch,
        distributed=True,  # Since this is in the train function which uses DDP
        shuffle=False,
        num_workers=16,
        dist=dist
    )
    
    # Model initialization
    model = PET(num_feat=train_dataset.num_feat,
                num_jet=train_dataset.num_jet,
                num_classes=train_dataset.num_classes,
                num_keep=7,
                feature_drop=args.feature_drop,
                projection_dim=args.projection_dim,
                local=args.local,
                K=10,
                num_local=2,
                num_layers=args.num_layers,
                num_class_layers=2,
                num_gen_layers=2,
                num_heads=args.num_heads,
                drop_probability=args.drop_probability,
                simple=args.simple,
                layer_scale=args.layer_scale,
                layer_scale_init=args.layer_scale_init,
                talking_head=args.talking_head,
                mode=args.mode,
                num_diffusion=3,
                dropout=args.dropout,
               device=device).to(device)
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    #--------NAN in Inputs DEBUG START --------
    
    # Added a NaN-only gradient hook and nan-tensor check here for finding NaN gradient or tensor issues
    def check_nan(tensor, name):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")
            print(f"Shape: {tensor.shape}")
            #print(f"Values: {tensor}")
            #raise RuntimeError(f"NaN detected in {name}")
            
    def hook(name):
        def hook_fn(grad):
            if torch.isnan(grad).any():
                print(f"NaN gradient detected in {name}")
                print(f"Gradient shape: {grad.shape}")
                print(f"Gradient mean: {grad.abs().mean()}")
                print(f"Gradient max: {grad.abs().max()}")
                print(f"Gradient min: {grad.abs().min()}")
                raise RuntimeError(f"NaN gradient in {name}")
        return hook_fn
    
    
        
    def check_input_nan(inputs):
        for key, value in inputs.items():
            if torch.isnan(value).any():
                print(f"NaN detected in input: {key}")
                print(f"Shape: {value.shape}")
                print(f"Values: {value}")
                raise RuntimeError(f"NaN detected in input: {key}")
   

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(hook(name))
            
    #--------NAN  DEBUG END --------------------------------------------------------------
    
    # Optimizer for PET body
    optimizer_body, scheduler_body = configure_optimizers(model.module.body.parameters(), args, train_loader, lr_factor=args.lr_factor if args.fine_tune else 1.0)
    
    # Combine parameters of classifier_head and generator_head
    classifier_and_generator_params = chain(
        model.module.classifier_head.parameters(), 
        model.module.generator_head.parameters()
    )

    # Optimizer for Heads
    optimizer_head, scheduler_head = configure_optimizers(classifier_and_generator_params, args, train_loader)
    #scaler = GradScaler()  #use with autocast
    
    from torch.nn import CrossEntropyLoss
    criterion_cls = CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    # Update checkpoint path to use the new log directory
    checkpoint_path = os.path.join(log_dir, args.checkpoint)
    
    if args.resume:
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer_body.load_state_dict(checkpoint['optimizer_body_state_dict'])
            optimizer_head.load_state_dict(checkpoint['optimizer_head_state_dict'])
            scheduler_body.load_state_dict(checkpoint['scheduler_body_state_dict'])
            scheduler_head.load_state_dict(checkpoint['scheduler_head_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            print(f"Resumed training from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
            
    # Update log file paths to use the new log directory
    log_file = os.path.join(log_dir, 'training_log.jsonl')
    roc_file = os.path.join(log_dir, 'roc_data.jsonl')
    
    torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(args.epoch):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        # Initialize counters
        total_correct = 0
        total_samples = 0
        train_pred = []
        train_true = []
        
        # Attempting to call set_epoch without a sampler will cause an error
        if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        
        for batch in train_loader:
            inputs, labels = batch
            labels = labels.to(device)
            inputs['input_label'] = labels  # Add labels to the inputs dictionary
            inputs = {k: v.to(device) for k, v in inputs.items()}
            batch_size = inputs['input_jet'].shape[0]
            
            #print(f"Input dtypes after to(rank): {[(k, v.dtype) for k, v in inputs.items()]}")
            
            #labels = labels.float().to(rank)
            
            # ------ DEBUG START ----------
            #print(f"Input keys: {inputs.keys()}")
            #print(f"Labels shape: {labels.shape}")
        
            #for k,v in inputs.items():
             #   print(f"key:{k}"," ",f"shape:{v.shape}")
            # ------ DEBUG END ----------
            
            # optimizer.zero_grad()
            optimizer_body.zero_grad()
            optimizer_head.zero_grad()
            loss = 0.0
            
            # Use autocast() for mixed-precision training (computational boost)
            # with autocast():
                #-------- DEBUG START --------
#                 print("Model parameters:")
#                 for name, param in model.named_parameters():
#                     print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")

#                 print("\nInputs:")
#                 for key, value in inputs.items():
#                     print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                #-------- DEBUG END --------
            check_input_nan(inputs)
            if args.mode in ['classifier', 'all']:
                body_output = model.module.body(inputs)
                classifier_output, regressor_output = model.module.classifier_head(body_output, inputs['input_jet'], inputs['input_mask'])
                # Classification loss
                # labels: shape=torch.Size([64, 10]), dtype=torch.int64 (One-Hot Encoded)
                # Use argmax to convert labels: shape=torch.Size([64]), dtype=torch.int64 (Class values)
                # classifier_output: shape=torch.Size([64, 10]), dtype=torch.float16
                loss_cls = criterion_cls(classifier_output, labels.argmax(dim=1))
                loss_reg = criterion_reg(regressor_output, inputs['input_jet'])
                
                check_nan(regressor_output, "regressor_output")
                check_nan(classifier_output, "classifier_output")
                
                loss += loss_cls + loss_reg  
                
                # For AUC, ROC
                probs = torch.softmax(classifier_output, dim=1)
                train_pred.extend(probs[:, 1].detach().cpu().numpy())
                train_true.extend(labels.argmax(dim=1).cpu().numpy())

            
            # Generator loss
            
            if args.mode in ['generator', 'all']:
                
                # Generate random timesteps for each sample in the batch
                # t.shape = (B, 1)
                t = torch.rand(batch_size, 1, device=device)
                _, alpha, sigma = get_logsnr_alpha_sigma(t)
                
                # Generate random noise with the same shape as the input features: (B, P, num_feat)
                # mask_1 = at particle level
                eps = torch.randn_like(inputs['input_features']) * inputs['input_mask'].unsqueeze(-1)
                
                # mask_2 = (feature level) Only apply diffusion to [:,:,0:num_diffusion] features
                mask_diffusion = torch.cat([
                    torch.ones_like(eps[:, :, :model.module.num_diffusion], dtype=torch.bool),
                    torch.zeros_like(eps[:, :, model.module.num_diffusion:], dtype=torch.bool)
                ], dim=-1)
                
                # torch.where(condition, x, y), when condition==False, take y
                eps = torch.where(mask_diffusion, eps, torch.zeros_like(eps))
                
                # alpha inverse prop |t| , sigma directly prop |t|
                perturbed_x = alpha.unsqueeze(1) * inputs['input_features'] + eps * sigma.unsqueeze(1)
                perturbed_x = torch.where(mask_diffusion, perturbed_x, torch.zeros_like(perturbed_x))
                
                # Create a new input dictionary with perturbed features
                perturbed_inputs = inputs.copy()
                perturbed_inputs['input_features'] = perturbed_x
                perturbed_inputs['input_time'] = t
                
                # Applying body to perturbed_inputs
                perturbed_body_output = model.module.body(perturbed_inputs)
                
                # Applying generator head to perturbed inputs
                generator_output = model.module.generator_head(perturbed_body_output, inputs['input_jet'], 
                                                               inputs['input_mask'], t, labels)
                
                v_pred = generator_output[:, :, :model.module.num_diffusion].reshape(generator_output.shape[0], -1)
                
                v = alpha.unsqueeze(1) * eps - sigma.unsqueeze(1) * inputs['input_features']
                v = v[:, :, :model.module.num_diffusion].reshape(v.shape[0], -1)
                
                loss_gen = torch.sum(torch.square(v - v_pred)) / (model.module.num_diffusion * torch.sum(inputs['input_mask']))
                loss += loss_gen
            
            # Classifier predicting using only classifier head
            if args.mode == 'all':
                perturbed_classifier_output, perturbed_regressor_output = model.module.classifier_head(perturbed_body_output, inputs['input_jet'], inputs['input_mask'])
                
                loss_cls_smear = criterion_cls(perturbed_classifier_output, labels.argmax(dim=1))
                loss_reg_smear = criterion_reg(perturbed_regressor_output, inputs['input_jet'])
                
                loss += (alpha ** 2).mean() * (loss_cls_smear + loss_reg_smear)
                
            print(f"Total loss: {loss.item():.4f}")    
            # All gradients will be zero before the backward calculation. set at optimizer.zero_grad()
            #scaler.scale(loss).backward()
            loss.backward()
            
            #--------GRADIENTS DEBUG START --------
            #print("-----SECOND ONE------")
            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(f"Parameter {name} has no gradient")
            #     elif param.grad.abs().sum() == 0:
            #         print(f"Parameter {name} has zero gradient")
            #--------GRADIENTS DEBUG END ----------
            
            #scaler.step(optimizer)
            #scaler.update()
            #optimizer.step()
            optimizer_body.step()
            optimizer_head.step()
            
            train_loss += loss.item()
            if args.mode in ['classifier', 'all']:
                # train_acc by calculating mean of batches
                train_acc += (classifier_output.argmax(dim=1) == labels.argmax(dim=1)).float().mean().item()
                
                # train acc by calculating total correct predictions
                predictions = classifier_output.argmax(dim=1)
                labels_cls = labels.argmax(dim=1)
                correct = (predictions == labels_cls).sum().item()
                total_correct += correct
                total_samples += labels_cls.size(0)
            
        # Calculate train AUC
        train_fpr, train_tpr, train_thresholds = roc_curve(train_true, train_pred)
        train_auc = auc(train_fpr, train_tpr)
        
        scheduler_body.step()
        scheduler_head.step()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        full_train_acc = total_correct / total_samples
        
        # Validation step
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_gen_loss = 0.0
        val_cls_smear_loss = 0.0
        val_reg_smear_loss = 0.0
        val_pred = []
        val_true = []
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                labels = labels.to(device)
                inputs['input_label'] = labels
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
                batch_size = inputs['input_jet'].shape[0]
            
                loss = 0.0
            
                if args.mode in ['classifier', 'all']:
                    body_output = model.module.body(inputs)
                    classifier_output, regressor_output = model.module.classifier_head(body_output, inputs['input_jet'], inputs['input_mask'])

                    loss_cls = criterion_cls(classifier_output, labels.argmax(dim=1))
                    loss_reg = criterion_reg(regressor_output, inputs['input_jet'])

                    loss += loss_cls + loss_reg
                    
                    # val accuracy by taking mean of batches
                    val_acc += (classifier_output.argmax(dim=1) == labels.argmax(dim=1)).float().mean().item()
                    
                    # val accuracy by considering total correct predictions
                    predictions = classifier_output.argmax(dim=1)
                    labels_cls = labels.argmax(dim=1)
                    correct = (predictions == labels_cls).sum().item()
                    total_correct += correct
                    total_samples += labels.size(0)
                    
                    # For AUC, ROC
                    probs = torch.softmax(classifier_output, dim=1)
                    val_pred.extend(probs[:, 1].detach().cpu().numpy())
                    val_true.extend(labels.argmax(dim=1).cpu().numpy())
                   
            
                if args.mode in ['generator', 'all']:
                    t = torch.rand(batch_size, 1, device=device)
                    _, alpha, sigma = get_logsnr_alpha_sigma(t)

                    eps = torch.randn_like(inputs['input_features']) * inputs['input_mask'].unsqueeze(-1)

                    mask_diffusion = torch.cat([
                        torch.ones_like(eps[:, :, :model.module.num_diffusion], dtype=torch.bool),
                        torch.zeros_like(eps[:, :, model.module.num_diffusion:], dtype=torch.bool)
                    ], dim=-1)

                    eps = torch.where(mask_diffusion, eps, torch.zeros_like(eps))

                    perturbed_x = alpha.unsqueeze(1) * inputs['input_features'] + eps * sigma.unsqueeze(1)
                    perturbed_x = torch.where(mask_diffusion, perturbed_x, torch.zeros_like(perturbed_x))

                    perturbed_inputs = inputs.copy()
                    perturbed_inputs['input_features'] = perturbed_x
                    perturbed_inputs['input_time'] = t

                    perturbed_body_output = model.module.body(perturbed_inputs)
                    generator_output = model.module.generator_head(perturbed_body_output, inputs['input_jet'], 
                                                                   inputs['input_mask'], t, labels)

                    v_pred = generator_output[:, :, :model.module.num_diffusion].reshape(generator_output.shape[0], -1)

                    v = alpha.unsqueeze(1) * eps - sigma.unsqueeze(1) * inputs['input_features']
                    v = v[:, :, :model.module.num_diffusion].reshape(v.shape[0], -1)

                    loss_gen = torch.sum(torch.square(v - v_pred)) / (model.module.num_diffusion * torch.sum(inputs['input_mask']))
                    loss += loss_gen
                    val_gen_loss += loss_gen.item()

                if args.mode == 'all':
                    perturbed_classifier_output, perturbed_regressor_output = model.module.classifier_head(perturbed_body_output, inputs['input_jet'], inputs['input_mask'])

                    loss_cls_smear = criterion_cls(perturbed_classifier_output, labels.argmax(dim=1))
                    loss_reg_smear = criterion_reg(perturbed_regressor_output, inputs['input_jet'])

                    loss += (alpha ** 2).mean() * (loss_cls_smear + loss_reg_smear)
                    val_cls_smear_loss += loss_cls_smear.item()
                    val_reg_smear_loss += loss_reg_smear.item()
            
                val_loss += loss.item()
    
    
        # Calculate validation AUC
        val_fpr, val_tpr, val_thresholds = roc_curve(val_true, val_pred)
        val_auc = auc(val_fpr, val_tpr)
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        # Calculate overall validation accuracy
        full_val_acc = total_correct / total_samples
        
        val_gen_loss /= len(val_loader)
        val_cls_smear_loss /= len(val_loader)
        val_reg_smear_loss /= len(val_loader)
        
        # Log training progress
        if global_rank == 0:  # Only log from the main process in distributed training
            log_training_progress(log_file, epoch, train_loss, val_loss, full_train_acc, full_val_acc, train_auc, val_auc, train_acc, val_acc)
            log_roc_data(roc_file, val_fpr, val_tpr, val_thresholds, epoch)
        
        if global_rank == 0:
            print(f"Epoch {epoch+1}/{args.epoch}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {full_train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {full_val_acc:.4f}")
            if args.mode in ['generator', 'all']:
                print(f"Val Gen Loss: {val_gen_loss:.4f}")
            if args.mode == 'all':
                print(f"Val Cls Smear Loss: {val_cls_smear_loss:.4f}")
                print(f"Val Reg Smear Loss: {val_reg_smear_loss:.4f}")
                
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_head_state_dict': optimizer_head.state_dict(),
                'scheduler_head_state_dict': scheduler_head.state_dict(),
                'optimizer_body_state_dict': optimizer_body.state_dict(),
                'scheduler_body_state_dict': scheduler_body.state_dict(),
                'best_val_loss': best_val_loss,
            }, args.checkpoint)
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_head_state_dict': optimizer_head.state_dict(),
                    'scheduler_head_state_dict': scheduler_head.state_dict(),
                    'optimizer_body_state_dict': optimizer_body.state_dict(),
                    'scheduler_body_state_dict': scheduler_body.state_dict(),
                    'best_val_loss': best_val_loss,
                }, f"checkpoint_best.pth")
    
    cleanup()

if __name__ == "__main__":
    args = parse_args()
    train(args)
    

#print(f"PyTorch version: {torch.__version__}")
#print(f"CUDA version: {torch.version.cuda}")