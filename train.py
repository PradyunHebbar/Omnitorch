import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from pet_model import PET, get_logsnr_alpha_sigma
from jetclass_dataset import get_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="Train the PET model on JetClass dataset.")
    parser.add_argument("--folder", type=str, default="/global/cfs/cdirs/m3246/phebbar/Work/Datasets/omnilearn/PET/", help="Folder containing input files")
    parser.add_argument("--batch", type=int, default=250, help="Batch size")
    parser.add_argument("--epoch", type=int, default=200, help="Max epoch")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--projection_dim", type=int, default=128, help="Projection dimension")
    parser.add_argument("--drop_probability", type=float, default=0.0, help="Drop probability")
    parser.add_argument("--feature_drop", type=float, default=0.1, help="Feature drop probability")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--local", action="store_true", help="Use local embedding")
    parser.add_argument("--simple", action="store_true", help="Use simple head model")
    parser.add_argument("--talking_head", action="store_true", help="Use talking head attention")
    parser.add_argument("--layer_scale", action="store_true", help="Use layer scale")
    parser.add_argument("--layer_scale_init", type=float, default=1e-5, help="Layer scale init value")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    return parser.parse_args()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    train_loader = get_dataloader(os.path.join(args.folder, 'JetClass', 'train'),
                                  args.batch, rank, world_size)
    val_loader = get_dataloader(os.path.join(args.folder, 'JetClass', 'val'),
                                args.batch, rank, world_size)
    
    model = PET(num_feat=13,
                num_jet=4,
                num_classes=10,
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
                mode='all',
                num_diffusion=3,
                dropout=args.dropout,
               device=device).to(device)
    
    model = DDP(model, device_ids=[rank])
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)
    scaler = GradScaler()
    
    
    from torch.nn import BCEWithLogitsLoss
    criterion_cls = BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epoch):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for batch in train_loader:
            inputs, labels = batch
            inputs['input_label'] = labels  # Add labels to the inputs dictionary
            inputs = {k: v.to(rank) for k, v in inputs.items()}
            print(f"Input dtypes after to(rank): {[(k, v.dtype) for k, v in inputs.items()]}")
            labels = labels.to(rank)
            #labels = labels.float().to(rank)
            
            #DEBUG
            print(f"Input keys: {inputs.keys()}")
            print(f"Labels shape: {labels.shape}")
        
            #for k,v in inputs.items():
             #   print(f"key:{k}"," ",f"shape:{v.shape}")
            
            optimizer.zero_grad()
            
            with autocast():
                
                print("Model parameters:")
                for name, param in model.named_parameters():
                    print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")

                print("\nInputs:")
                for key, value in inputs.items():
                    print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                
                classifier_output, generator_output = model(inputs)
                cls_pred, reg_pred = classifier_output
                
                # Classification loss
                labels = labels.float()
                loss_cls = criterion_cls(cls_pred, labels)
                
                # Regression loss
                loss_reg = criterion_reg(reg_pred, inputs['input_jet'])
                
                # Generator loss
                t = torch.rand(inputs['input_features'].shape[0], 1, device=rank)
                _, alpha, sigma = get_logsnr_alpha_sigma(t)
                
                eps = torch.randn_like(inputs['input_features']) * inputs['input_mask'].unsqueeze(-1)
                mask_diffusion = torch.cat([
                    torch.ones_like(eps[:, :, :model.module.num_diffusion], dtype=torch.bool),
                    torch.zeros_like(eps[:, :, model.module.num_diffusion:], dtype=torch.bool)
                ], dim=-1)
                
                eps = torch.where(mask_diffusion, eps, torch.zeros_like(eps))
                perturbed_x = alpha.unsqueeze(1) * inputs['input_features'] + eps * sigma.unsqueeze(1)
                perturbed_x = torch.where(mask_diffusion, perturbed_x, torch.zeros_like(perturbed_x))
                
                v_pred = generator_output
                v_pred = v_pred[:, :, :model.module.num_diffusion].reshape(v_pred.shape[0], -1)
                
                v = alpha.unsqueeze(1) * eps - sigma.unsqueeze(1) * inputs['input_features']
                v = v[:, :, :model.module.num_diffusion].reshape(v.shape[0], -1)
                
                loss_gen = torch.sum(torch.square(v - v_pred)) / (model.module.num_diffusion * torch.sum(inputs['input_mask']))
                
                # Total loss
                loss = loss_cls + loss_reg + loss_gen
                print(f"Loss components: cls={loss_cls.item():.4f}, reg={loss_reg.item():.4f}, gen={loss_gen.item():.4f}")
                
            #Testing gradients (All gradients will be zero before the backward calculation)
            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(f"Parameter {name} has no gradient")
            #     elif param.grad.abs().sum() == 0:
            #         print(f"Parameter {name} has zero gradient")
            # print("-----FIRST ONE------")
        
            scaler.scale(loss).backward()
            print("-----SECOND ONE------")
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"Parameter {name} has no gradient")
                elif param.grad.abs().sum() == 0:
                    print(f"Parameter {name} has zero gradient")
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_acc += (cls_pred.argmax(dim=1) == labels.argmax(dim=1)).float().mean().item()
        
        scheduler.step()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Validation step
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs = {k: v.to(rank) for k, v in inputs.items()}
                labels = labels.to(rank)
                
                with autocast():
                    classifier_output, generator_output = model(inputs)
                    cls_pred, reg_pred = classifier_output
                    
                    loss_cls = criterion_cls(cls_pred, labels)
                    loss_reg = criterion_reg(reg_pred, inputs['input_jet'])
                    
                    # Simplified generator loss for validation
                    loss_gen = criterion_reg(generator_output, inputs['input_features'])
                    
                    loss = loss_cls + loss_reg + loss_gen
                
                val_loss += loss.item()
                val_acc += (cls_pred.argmax(dim=1) == labels.argmax(dim=1)).float().mean().item()
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epoch}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                }, f"checkpoint_best.pth")
    
    cleanup()

if __name__ == "__main__":
    args = parse_args()
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size, args), nprocs=world_size, join=True)

#print(f"PyTorch version: {torch.__version__}")
#print(f"CUDA version: {torch.version.cuda}")