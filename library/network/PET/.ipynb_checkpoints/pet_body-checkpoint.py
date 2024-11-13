import torch
import torch.nn as nn
import torch.nn.functional as F

from ..embedding import LocalEmbeddingLayer, FourierEmbedding
from ..transformers import TransformerBlockModule
from ..utils import RandomDrop

class PETBody(nn.Module):
    def __init__(self, num_feat, num_keep, feature_drop, projection_dim, local, K, num_local,
                 num_layers, num_heads, drop_probability, talking_head, layer_scale,
                 layer_scale_init, dropout, device, mode):
        super().__init__()
        self.device = device
        self.num_keep = num_keep
        self.feature_drop = feature_drop
        self.projection_dim = projection_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_local = num_local
        self.drop_probability = drop_probability
        self.layer_scale = layer_scale
        self.layer_scale_init = layer_scale_init
        self.dropout = dropout
        self.mode = mode

        self.random_drop = RandomDrop(feature_drop if 'all' in self.mode else 0.0, num_keep)
        self.feature_embedding = nn.Sequential(
            nn.Linear(num_feat, 2*projection_dim),
            nn.GELU(approximate='none'),
            nn.Linear(2*projection_dim, projection_dim),
            nn.GELU(approximate='none')
        )

        self.time_embedding = FourierEmbedding(projection_dim).to(self.device)
        self.time_embed_linear = nn.Linear(projection_dim, 2*projection_dim, bias=False, device=self.device)

        if local:
            self.local_embedding = LocalEmbeddingLayer(projection_dim, K)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlockModule(projection_dim, num_heads, dropout, talking_head, layer_scale, layer_scale_init, drop_probability)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        input_features = x['input_features'].to(self.device)
        input_points = x['input_points'].to(self.device)
        mask = x['input_mask'].to(self.device)

        # time is only important for the diffusion model, where we perturb the data using a time-dependent function. During training, we sample time from an uniform distribution and determine the perturbation parameters to be applied to the data
        time = x.get('input_time', torch.zeros(input_features.shape[0], 1, device=self.device))

        encoded = self.random_drop(input_features)
        encoded = self.feature_embedding(encoded)

        time_emb = self.time_embedding(time)
        time_emb = time_emb.squeeze(1).unsqueeze(1).repeat(1, encoded.shape[1], 1)
        time_emb = time_emb * mask.unsqueeze(-1)
        time_emb = self.time_embed_linear(time_emb)
        scale, shift = torch.chunk(time_emb, 2, dim=-1)

        encoded = torch.add(torch.mul(encoded, (1.0 + scale)), shift)

        if hasattr(self, 'local_embedding'):
            coord_shift = 999.0 * (mask == 0).float().unsqueeze(-1)
            points = input_points[:, :, :2] + coord_shift # Shape: (batch_size, num_points, 2)
            local_features = input_features # Initial features
            for _ in range(self.num_local):
                local_features = self.local_embedding(points, local_features)
                points = local_features # Update points with the new features
            encoded = local_features + encoded # Combine with original features

        skip_connection = encoded
        for transformer_block in self.transformer_blocks:
            encoded = transformer_block(encoded, mask)

        return torch.add(encoded, skip_connection)
