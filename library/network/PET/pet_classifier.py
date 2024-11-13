import torch
import torch.nn as nn
import torch.nn.functional as F

from ..transformers import ClassifierTransformerBlockModule

class ClassifierHead(nn.Module):
    def __init__(self, projection_dim, num_jet, num_classes, num_class_layers, simple,
                 num_heads, dropout, talking_head, layer_scale, layer_scale_init, drop_probability):
        super().__init__()
        self.simple = simple
        self.projection_dim = projection_dim

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
                ClassifierTransformerBlockModule(projection_dim, num_heads, dropout, talking_head,
                                                 layer_scale, layer_scale_init, drop_probability)
                for _ in range(num_class_layers)
            ])
            self.classifier = nn.Linear(projection_dim, num_classes)
            self.regressor = nn.Linear(projection_dim, num_jet)

    def forward(self, x, jet, mask):
        jet_emb = self.jet_embedding(jet) # jet: torch.Size([B, num_jet]) --> jet_emb: torch.Size([B, 2*proj_dim])

        if self.simple:
            x = F.layer_norm(x, [x.size(-1)]) # Group Norm
            x = torch.mean(x, dim=1) # Average Pooling
            x = x + jet_emb # Concat
            class_output = self.classifier(x) # Out Dense
            reg_output = self.regressor(x) # Out Dense
        else:
            conditional = jet_emb.unsqueeze(1).expand(-1, x.shape[1], -1) # conditional: torch.Size([B, num_part, 2*proj_dim])
            scale, shift = torch.chunk(conditional, 2, dim=-1)
            x = x * (1.0 + scale) + shift

            B = x.shape[0]
            # cls Before:  torch.Size([1, 1, proj_dim])
            class_tokens = self.class_token.expand(B, -1, -1)
            # Class tiling | cls After: torch.Size([B, 1, proj_dim])
            
            # Updated mask to include class token
            mask = torch.cat([torch.ones(B, 1, device=mask.device, dtype=mask.dtype), mask], dim=1)

            # Initial input to transformer concatenated_0 : torch.Size([B, num_part+1, proj_dim])
            # Class token : torch.Size([B, 1, proj_dim]) , x : torch.Size([B, num_part, proj_dim])
            for transformer_block in self.clf_transformer_blocks:
                concatenated = torch.cat([class_tokens, x], dim=1)
                out_x, class_tokens = transformer_block(concatenated, class_tokens, mask)
            # Output after transformer class_token : torch.Size([B, 1, proj_dim])

            class_tokens = F.layer_norm(class_tokens, [class_tokens.size(-1)])
            class_output = self.classifier(class_tokens[:, 0])
            reg_output = self.regressor(class_tokens[:, 0])

        return class_output, reg_output
