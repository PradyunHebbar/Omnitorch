import torch
import torch.nn as nn
import torch.nn.functional as F

from ..transformers import GeneratorTransformerBlockModule
from ..embedding import FourierEmbedding
from ..utils import StochasticDepth

class GeneratorHead(nn.Module):
    def __init__(self, projection_dim, num_jet, num_classes, num_feat, num_layers, simple,
                 num_heads, dropout, talking_head, layer_scale, layer_scale_init,
                 drop_probability, feature_drop):
        super().__init__()
        self.simple = simple
        self.projection_dim = projection_dim
        self.num_diffusion = 3  # Adjust this based on your settings

        self.jet_embedding = nn.Sequential(
            nn.Linear(num_jet, projection_dim),
            nn.GELU(approximate='none')
        )
        self.time_embedding = FourierEmbedding(projection_dim)
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
                GeneratorTransformerBlockModule(projection_dim, num_heads, dropout, talking_head,
                                                layer_scale, layer_scale_init, drop_probability)
                for _ in range(num_layers)
            ])
            self.generator = nn.Linear(projection_dim, num_feat)

    def forward(self, x, jet, mask, time, label):
        jet_emb = self.jet_embedding(jet) # jet_emb shape after embedding: torch.Size([B, proj_dim])
        time_emb = self.time_embedding(time) # time_emb shape after embedding: torch.Size([B, 1, proj_dim])
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

        if self.simple:
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
