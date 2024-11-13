import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.custom_layers import LayerScale

class GeneratorTransformerBlockModule(nn.Module):
    def __init__(self, projection_dim, num_heads, dropout, talking_head, layer_scale, layer_scale_init, drop_probability):
        super().__init__()
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.talking_head = talking_head
        self.layer_scale_flag = layer_scale
        self.drop_probability = drop_probability

        self.norm1 = nn.LayerNorm(projection_dim)
        self.norm3 = nn.LayerNorm(projection_dim)

        self.attn = nn.MultiheadAttention(projection_dim, num_heads, dropout, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, 2 * projection_dim),
            nn.GELU(approximate='none'),
            nn.Linear(2 * projection_dim, projection_dim),
        )

        if layer_scale:
            self.layer_scale1 = LayerScale(layer_scale_init, projection_dim)
            self.layer_scale2 = LayerScale(layer_scale_init, projection_dim)

    def forward(self, x, cond_token, mask=None):
        x1 = self.norm1(x)
        updates, _ = self.attn(x1, x1, x1, key_padding_mask=~mask.bool() if mask is not None else None)

        if self.layer_scale_flag:
            updates = self.layer_scale1(updates, mask)
        x2 = updates + cond_token
        x3 = self.norm3(x2)
        x3 = self.mlp(x3)

        if self.layer_scale_flag:
            x3 = self.layer_scale2(x3, mask)
        cond_token = x2 + x3

        return x, cond_token
