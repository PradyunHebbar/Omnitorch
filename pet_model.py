import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_layers import StochasticDepth, RandomDrop, TalkingHeadAttention, LayerScale

class PET(nn.Module):
    def __init__(self,
                 num_feat,
                 num_jet,
                 num_classes=10,
                 num_keep=7,
                 feature_drop=0.1,
                 projection_dim=128,
                 local=True,
                 K=10,
                 num_local=2,
                 num_layers=8,
                 num_class_layers=2,
                 num_gen_layers=2,
                 num_heads=4,
                 drop_probability=0.0,
                 simple=False,
                 layer_scale=True,
                 layer_scale_init=1e-5,
                 talking_head=False,
                 mode='all',
                 num_diffusion=3,
                 dropout=0.0,
                 # no class activation option
                device='cuda'):
        super().__init__()
        self.device = device
        self.num_feat = num_feat
        self.num_jet = num_jet
        self.num_classes = num_classes
        self.num_keep = num_keep
        self.feature_drop = feature_drop
        self.drop_probability = drop_probability
        self.dropout = dropout
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.layer_scale_init = layer_scale_init
        self.mode = mode
        self.num_diffusion = num_diffusion
        self.num_local = num_local
        
        self.random_drop = RandomDrop(feature_drop if 'all' in self.mode else 0.0, num_keep)
        self.feature_embedding = nn.Sequential(
            nn.Linear(num_feat, 2*projection_dim),
            nn.GELU(approximate='none'),
            nn.Linear(2*projection_dim, projection_dim),
            nn.GELU(approximate='none')
        )
        
        self.time_embedding = self.FourierProjection(projection_dim).to(self.device)
        self.time_embed_linear = nn.Linear(self.projection_dim, 2*self.projection_dim, bias=False,device=self.device)
        
        if local:
            self.local_embedding = self.LocalEmbedding(K)
        
        self.transformer_blocks = nn.ModuleList([
            self.TransformerBlock(projection_dim, num_heads, dropout, talking_head, layer_scale, layer_scale_init, drop_probability)
            for _ in range(num_layers)
        ])
        
        self.classifier_head = self.ClassifierHead(
            projection_dim, num_jet, num_classes, num_class_layers, simple,
            num_heads, dropout, talking_head, layer_scale, layer_scale_init, drop_probability,
            self.ClassifierTransformerBlock
        )
        self.generator_head = self.GeneratorHead(
            projection_dim, num_jet, num_classes, num_feat, num_gen_layers, simple,
            num_heads, dropout, talking_head, layer_scale, layer_scale_init, drop_probability,
            feature_drop, self.GeneratorTransformerBlock, self.FourierProjection
        )
        # self.generator_head = self.GeneratorHead(
        #     projection_dim, num_jet, num_classes, num_feat, num_gen_layers, simple,
        #     num_heads, dropout, talking_head, layer_scale, layer_scale_init,
        #     drop_probability, feature_drop, self.GeneratorTransformerBlock
        # )

    def forward(self, x, mode='all'):
        input_features = x['input_features'].to(self.device)
        input_points = x['input_points'].to(self.device)
        mask = x['input_mask'].to(self.device)
        jet = x['input_jet'].to(self.device)
        
        # the time is only important for the diffusion model, where we perturb the data using a time-dependent function. During training, we sample time from an uniform distribution and determine the perturbation parameters to be applied to the data: https://github.com/ViniciusMikuni/OmniLearn/blob/main/scripts/PET.py#L153
        time = x.get('input_time', torch.zeros(input_features.shape[0], 1, device=self.device))
        
        label = x.get('input_label', None)
        #print(f"In PET forward, label: {label}")
    
        encoded = self.random_drop(input_features)
        encoded = self.feature_embedding(encoded)
    
        time_emb = self.time_embedding(time)
        #print(f"time_emb shape after time_embedding: {time_emb.shape}")
    
        # Correct reshaping of time_emb
        time_emb = time_emb.squeeze(1).unsqueeze(1).repeat(1, encoded.shape[1], 1)
        #print(f"time_emb shape after reshape and repeat: {time_emb.shape}")
    
        time_emb = time_emb * mask.unsqueeze(-1)
        #print(f"time_emb shape after applying mask: {time_emb.shape}")
    
        time_emb = self.time_embed_linear(time_emb)
        scale, shift = torch.chunk(time_emb, 2, dim=-1)
        encoded = encoded * (1.0 + scale) + shift
        
        if hasattr(self, 'local_embedding'):
            coord_shift = 999.0 * (mask == 0).float().unsqueeze(-1)
            points = input_points[:, :, :2] + coord_shift  # Shape: (batch_size, num_points, 2)
            local_features = input_features  # Initial features
            for _ in range(self.num_local):
                local_features = self.local_embedding(points, local_features)
                points = local_features  # Update points with the new features
            encoded = local_features+encoded  # Combine with original features
        
        skip_connection = encoded
        for transformer_block in self.transformer_blocks:
            encoded = transformer_block(encoded, mask)
        
        encoded = encoded + skip_connection  #END OF PET BODY
        
        if mode in ['classifier', 'all']:
            #print(f"Before classifier_head, shapes: features: {features.shape}, jet: {jet.shape}, mask: {mask.shape}")
            classifier_output = self.classifier_head(encoded, jet, mask)
        
        if mode in ['generator', 'all']:
            #print(f"Before generator_head, shapes: features: {features.shape}, jet: {jet.shape}, mask: {mask.shape}, time: {time.shape}, label: {label.shape if label is not None else None}")
            #print(f"Before generator_head, label dtype: {label.dtype if label is not None else None}")
            generator_output = self.generator_head(encoded, jet, mask, time, label)
            
        #print(f"Classifier output shapes: {classifier_output[0].shape}, {classifier_output[1].shape}")
        #print(f"Generator output shape: {generator_output.shape}")
        
        if mode == 'classifier':
            return classifier_output
        elif mode == 'generator':
            return generator_output
        else:
            return classifier_output, generator_output
        
# LOCAL EMBEDDING START  ---------------------------------------------------------------------------------------------------------------------------------      
    
    def LocalEmbedding(self, K):
        class LocalEmbeddingLayer(nn.Module): 
            def __init__(self, projection_dim, K):
                super().__init__()
                self.K = K
                self.projection_dim = projection_dim
                # self.mlp = nn.Sequential(    # Shift MLP to forward pass to use C as input
                #     nn.Linear(2*C, 2 * projection_dim),
                #     nn.GELU(),
                #     nn.Linear(2 * projection_dim, projection_dim),
                #     nn.GELU()
                #     )
            def pairwise_distance(self, points):
                r = torch.sum(points * points, dim=2, keepdim=True)  # Shape: (N, P, 1)
                m = torch.bmm(points, points.transpose(1, 2))        # Shape: (N, P, P)
                D = r - 2 * m + r.transpose(1, 2) + 1e-5            # Shape: (N, P, P)
                return D

            def forward(self, points, features):
                distances = self.pairwise_distance(points) #uses custom pairwise function, not torch.cdist
                _, indices = torch.topk(-distances, k=self.K + 1, dim=-1)
                indices = indices[:, :, 1:]  # Exclude self
                # indices Shape: (N, P, 10)
                
                batch_size, num_points, _ = features.shape # (B, P, num_feats)
                batch_indices = torch.arange(batch_size, device=features.device).view(-1, 1, 1)
                batch_indices = batch_indices.repeat(1, num_points, self.K)
                indices = torch.stack([batch_indices, indices], dim=-1)  # Shape: (N, P, K, 2)
                # concat indices torch.Size([N, P, K, 2])
            
                # Gather neighbor features
                neighbors = features[indices[:, :, :, 0], indices[:, :, :, 1]] # Shape: (N, P, K, C)
                # neighbors: torch.Size([64, 150, 10, 13])
                knn_fts_center = features.unsqueeze(2).expand_as(neighbors)      # Shape: (N, P, K, C)
                # knn fts center: torch.Size([64, 150, 10, 13])
                local_features = torch.cat([neighbors - knn_fts_center, knn_fts_center], dim=-1)
                # local_features: torch.Size([N, P, K, 26])
                # local_features.shape[-1] Shape : 2*C
                
                mlp = nn.Sequential(
                    nn.Linear(local_features.shape[-1], 2 * self.projection_dim),
                    nn.GELU(approximate='none'),
                    nn.Linear(2 * self.projection_dim, self.projection_dim),
                    nn.GELU(approximate='none')
                    ).to(local_features.device)
                
                local_features = mlp(local_features)
                local_features = torch.mean(local_features, dim=2)

                return local_features

        return LocalEmbeddingLayer(self.projection_dim, K)
# LOCAL EMBEDDING END  ---------------------------------------------------------------------------------------------------------------------------------  
# PET BODY TRANSFORMER START  --------------------------------------------------------------------------------------------------------------------------
    def TransformerBlock(self, projection_dim, num_heads, dropout, talking_head, layer_scale, layer_scale_init, drop_probability):
        class TransformerBlockModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.LayerNorm(projection_dim)
                self.norm2 = nn.LayerNorm(projection_dim)
                
                
                if talking_head:
                    self.attn = TalkingHeadAttention(projection_dim, num_heads, dropout)
                else:
                    self.attn = nn.MultiheadAttention(projection_dim, num_heads, dropout, batch_first=True)
                
                self.mlp = nn.Sequential(
                    nn.Linear(projection_dim, 2 * projection_dim),
                    nn.GELU(approximate='none'),
                    nn.Dropout(dropout),
                    nn.Linear(2 * projection_dim, projection_dim),
                )
                
                self.drop_path = StochasticDepth(drop_probability)
                
                if layer_scale:
                    self.layer_scale1 = LayerScale(layer_scale_init, projection_dim)
                    self.layer_scale2 = LayerScale(layer_scale_init, projection_dim)

            def forward(self, x, mask=None):
               # print(f"TransformerBlock input shapes: x: {x.shape}, mask: {mask.shape if mask is not None else None}")
                if talking_head:
                    attn_output, _ = self.attn(self.norm1(x), int_matrix=None, mask=mask)
                else:
                    attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=~mask.bool() if mask is not None else None)
                
                if layer_scale:
                    x = x + self.drop_path(self.layer_scale1(attn_output, mask))
                    x = x + self.drop_path(self.layer_scale2(self.mlp(self.norm2(x)), mask))
                else:
                    x = x + self.drop_path(attn_output)
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                    
                if mask is not None:
                    x = x * mask.unsqueeze(-1)
             
                return x

        return TransformerBlockModule()
# PET BODY TRANSFORMER END  ------------------------------------------------------------------------------------------------------------------------------    
# CLASSIFIER HEAD START  --------------------------------------------------------------------------------------------------------------------------------- 

    def ClassifierTransformerBlock(self, projection_dim, num_heads, dropout, talking_head, layer_scale, layer_scale_init, drop_probability):
        class ClassifierTransformerBlockModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.LayerNorm(projection_dim)
                self.norm2 = nn.LayerNorm(projection_dim)
                self.norm3 = nn.LayerNorm(projection_dim)
                
                
               
                self.attn = nn.MultiheadAttention(projection_dim, num_heads, dropout, batch_first=True)
                
                self.mlp = nn.Sequential(
                    nn.Linear(projection_dim, 2 * projection_dim),
                    nn.GELU(approximate='none'),
                    nn.Dropout(dropout),
                    nn.Linear(2 * projection_dim, projection_dim),
                )
                
                if layer_scale:
                    self.layer_scale1 = LayerScale(layer_scale_init, projection_dim)
                    self.layer_scale2 = LayerScale(layer_scale_init, projection_dim)

            def forward(self, x, class_token, mask=None):
               # print(f"TransformerBlock input shapes: x: {x.shape}, mask: {mask.shape if mask is not None else None}")
                
                x1 = self.norm1(x)
                query = x1[:, 0].unsqueeze(1)  # Only use the class token as query
    
                updates, _ = self.attn(query, x1, x1, key_padding_mask=~mask.bool() if mask is not None else None) 
                updates = self.norm2(updates)
            
                if layer_scale:
                    updates = self.layer_scale1(updates, mask[:,:1]) # Apply layer scale only to class token
                x2 = updates + class_token
                
                x3 = self.norm3(x2)
                x3 = self.mlp(x3)
                
                if layer_scale:
                    x3 = self.layer_scale2(x3, mask[:,:1]) # Apply layer scale only to class token
                else:
                    x3 = x3
                cls_token = x3 + x2
             
                return x, cls_token

        return ClassifierTransformerBlockModule()

    def ClassifierHead(self, projection_dim, num_jet, num_classes, num_layers, simple,
                       num_heads, dropout, talking_head, layer_scale, layer_scale_init,
                       drop_probability, ClassifierTransformerBlock):
        class ClassifierHeadModule(nn.Module):
            def __init__(self):
                super().__init__()
                
                if simple:
                    self.jet_embedding = nn.Sequential(
                        nn.Linear(num_jet, 2 * projection_dim),
                        nn.GELU(approximate='none'),
                        nn.Linear(2 * projection_dim, projection_dim),
                        nn.GELU(approximate='none')
                            )
                else:
                    self.jet_embedding = nn.Sequential(
                        nn.Linear(num_jet, 2 * projection_dim),
                        nn.GELU(approximate='none')
                            )
                
                if simple:
                    self.classifier = nn.Sequential(
                        nn.LayerNorm(projection_dim),
                        nn.Linear(projection_dim, num_classes)
                    )
                    self.regressor = nn.Linear(projection_dim, num_jet)
                else:
                    self.class_token = nn.Parameter(torch.zeros(1, 1, projection_dim))
                    self.clf_transformer_blocks = nn.ModuleList([
                        ClassifierTransformerBlock(projection_dim, num_heads, dropout, talking_head,
                                         layer_scale, layer_scale_init, drop_probability)
                        for _ in range(num_layers)
                    ])
                    self.classifier = nn.Linear(projection_dim, num_classes)
                    self.regressor = nn.Linear(projection_dim, num_jet)

            def forward(self, x, jet, mask):
                #print(f"ClassifierHead input shapes: x: {x.shape}, jet: {jet.shape}, mask: {mask.shape}")
                
                jet_emb = self.jet_embedding(jet)  # jet: torch.Size([B, num_jet]) --> jet_emb: torch.Size([B, 2*proj_dim])  
                
                if hasattr(self, 'simple'):
                    x = F.layer_norm(x, [x.size(-1)]) #Group Norm
                    x = torch.mean(x, dim=1) #Average Pooling
                    x = x + jet_emb   #concat
                    class_output = self.classifier(x)  #Out Dense
                    reg_output = self.regressor(x)    #Out Dense
                else:
                    conditional = jet_emb.unsqueeze(1).expand(-1, x.shape[1], -1) # conditional: torch.Size([B, num_part, 2*proj_dim])
                   
                    scale, shift = torch.chunk(conditional, 2, dim=-1)
                    x = x * (1.0 + scale) + shift # x : torch.Size([B, num_part, proj_dim])
                    
                
                    B = x.shape[0] # B : B
                    # cls Before:  torch.Size([1, 1, proj_dim])
                    class_tokens = self.class_token.expand(B, -1, -1)  
                    # Class tiling | cls After: torch.Size([B, 1, proj_dim])

                    
                    # Updated mask to include class token
                    mask = torch.cat([torch.ones(B, 1, device=mask.device, dtype=mask.dtype), mask], dim=1) 
                    
                    # initial input to transformer concatenated_0 : torch.Size([B, num_part+1, proj_dim])
                    # class token : torch.Size([B, 1, proj_dim]) , x : torch.Size([B, num_part, proj_dim])
                    for transformer_block in self.clf_transformer_blocks:
                        concatenated = torch.cat([class_tokens, x], dim=1) #concat at particle level
                        out_x , class_tokens = transformer_block(concatenated, class_tokens, mask)
                   # output after transformer class_token : torch.Size([B, 1, proj_dim])
                
                    
                    class_tokens = F.layer_norm(class_tokens, [class_tokens.size(-1)])
                    class_output = self.classifier(class_tokens[:, 0])
                    reg_output = self.regressor(class_tokens[:, 0])
                
                return class_output, reg_output

        return ClassifierHeadModule()
    
# CLASSIFIER HEAD END  ---------------------------------------------------------------------------------------------------------------------------------

# GENERATOR HEAD START  ---------------------------------------------------------------------------------------------------------------------------------
    
    def GeneratorTransformerBlock(self, projection_dim, num_heads, dropout, talking_head, layer_scale, layer_scale_init, drop_probability):
        class GeneratorTransformerBlockModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.LayerNorm(projection_dim)
                self.norm3 = nn.LayerNorm(projection_dim)
            
                self.attn = nn.MultiheadAttention(projection_dim, num_heads, dropout, batch_first=True)
            
                self.mlp = nn.Sequential(
                    nn.Linear(projection_dim, 2 * projection_dim),
                    nn.GELU(approximate='none'),
                    nn.Dropout(dropout),
                    nn.Linear(2 * projection_dim, projection_dim),
                )
            
                if layer_scale:
                    self.layer_scale1 = LayerScale(layer_scale_init, projection_dim)
                    self.layer_scale2 = LayerScale(layer_scale_init, projection_dim)

            def forward(self, x, cond_token, mask=None):
                x1 = self.norm1(x)
                updates, _ = self.attn(x1, x1, x1, key_padding_mask=~mask.bool() if mask is not None else None)
            
                if layer_scale:
                    updates = self.layer_scale1(updates, mask)
                x2 = updates+cond_token
                x3 = self.norm3(x2)
                x3 = self.mlp(x3)
            
                if layer_scale:
                    x3 = self.layer_scale2(x3, mask)
                cond_token = x2 + x3
            
                return x, cond_token

        return GeneratorTransformerBlockModule()
    
    
    def GeneratorHead(self, projection_dim, num_jet, num_classes, num_feat, num_layers, simple,
                            num_heads, dropout, talking_head, layer_scale, layer_scale_init,
                            drop_probability, feature_drop, GeneratorTransformerBlock, FourierProjection):

        class GeneratorHeadModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.jet_embedding = nn.Sequential(
                    nn.Linear(num_jet, projection_dim),
                    nn.GELU(approximate='none')
                )
                self.time_embedding = FourierProjection(projection_dim)
                self.cond_token = nn.Sequential(
                    nn.Linear(2 * projection_dim, 2 * projection_dim),
                    nn.GELU(approximate='none'),
                    nn.Linear(2 * projection_dim, projection_dim),
                    nn.GELU(approximate='none')
                )
                #self.label_embedding = nn.Embedding(num_classes, projection_dim)
                self.label_dense = nn.Linear(num_classes, projection_dim, bias=False)
                self.feature_drop = feature_drop
                self.stochastic_depth = StochasticDepth(feature_drop)
            
                if simple:
                    self.cond_dense = nn.Linear(projection_dim, 2 * projection_dim)
                    self.generator = nn.Sequential(
                        nn.LayerNorm(projection_dim),
                        nn.Linear(projection_dim, 2 * projection_dim),
                        nn.GELU(approximate='none'),
                        nn.Dropout(dropout),
                        nn.Linear(2 * projection_dim, num_feat)
                    )
                else:
                    self.gen_transformer_blocks = nn.ModuleList([
                        GeneratorTransformerBlock(projection_dim, num_heads, dropout, talking_head,
                                                  layer_scale, layer_scale_init, drop_probability)
                        for _ in range(num_layers)
                    ])
                    self.generator = nn.Linear(projection_dim, num_feat)

            def forward(self, x, jet, mask, time, label):
                #print(f"GeneratorHead input shapes: x: {x.shape}, jet: {jet.shape}, mask: {mask.shape}, time: {time.shape}, label: {label.shape if label is not None else None}")
                #print(f"GeneratorHead input dtypes: x: {x.dtype}, jet: {jet.dtype}, mask: {mask.dtype}, time: {time.dtype}, label: {label.dtype if label is not None else None}")
                
                jet_emb = self.jet_embedding(jet) # jet_emb shape after embedding: torch.Size([B, proj_dim])
                time_emb = self.time_embedding(time) # time_emb shape after embedding: torch.Size([B, 1, proj_dim])
                
                # Squeeze out the extra dimension from time_emb
                time_emb = time_emb.squeeze(1) # time_emb shape after squeezing: torch.Size([B, proj_dim])
                
                cond_token = self.cond_token(torch.cat([time_emb, jet_emb], dim=-1)) # After MLP, cond_token shape: torch.Size([B, proj_dim])
                
                
                if label is not None:
                    #label_emb = self.label_embedding(label)
                    label_emb = self.label_dense(label.float())  
                    label_emb = self.stochastic_depth(label_emb)
                    cond_token = cond_token + label_emb
                else:
                    print("ERROR: In Generation Head, Label is None, skipping label embedding")
            
                cond_token = cond_token.unsqueeze(1).expand(-1, x.shape[1], -1) * mask.unsqueeze(-1)
            
                if hasattr(self, 'simple'):
                    cond_token = self.cond_dense(cond_token)
                    cond_token = F.gelu(cond_token)
                    scale, shift = torch.chunk(cond_token, 2, dim=-1)
                    x = F.layer_norm(x, [x.size(-1)])
                    x = x * (1.0 + scale) + shift
                    x = self.generator(x)
                else:
                    for transformer_block in self.gen_transformer_blocks:
                        concatenated = cond_token + x
                        out_x, cond_token = transformer_block(concatenated, cond_token, mask)
                    x = cond_token + x
                    x = F.layer_norm(x, [x.size(-1)])
                    x = self.generator(x)
            
                return x * mask.unsqueeze(-1)

        return GeneratorHeadModule()

    def FourierProjection(self, projection_dim, num_embed=64):
        #print(f"FourierProjection called with projection_dim: {projection_dim}, num_embed: {num_embed}")
        class FourierEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.W = nn.Parameter(torch.randn(num_embed // 2) * 1000.0)

            def forward(self, x):
                x_proj = x[:, None] * self.W[None, :]
                return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        return nn.Sequential(
            FourierEmbedding(),
            nn.Linear(num_embed, 2 * projection_dim),
            nn.GELU(),
            nn.Linear(2 * projection_dim, projection_dim),
            nn.GELU()
             )
# GENERATOR HEAD END  ---------------------------------------------------------------------------------------------------------------------------------
    
def get_logsnr_alpha_sigma(time):
    def logsnr_schedule_cosine(t, logsnr_min=-20., logsnr_max=20.):
        b = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max)))
        a = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min))) - b
        return -2. * torch.log(torch.tan(a * t + b))

    logsnr = logsnr_schedule_cosine(time)
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))
    return logsnr, alpha, sigma

#print(f"PyTorch version: {torch.__version__}")