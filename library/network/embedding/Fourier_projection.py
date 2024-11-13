import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierEmbedding(nn.Module):
    def __init__(self, projection_dim, num_embed=64):
        super().__init__()
        self.projection_dim = projection_dim
        self.num_embed = num_embed
        self.half_dim = num_embed // 2

        # Calculate frequencies
        emb = torch.log(torch.tensor(10000.0)) / (self.half_dim - 1)
        self.freq = torch.exp(-emb * torch.arange(self.half_dim, dtype=torch.float32))

        self.dense1 = nn.Linear(num_embed, 2 * projection_dim, bias=False)
        self.dense2 = nn.Linear(2 * projection_dim, projection_dim, bias=False)

    def forward(self, x):
        # To Ensure x is 2D: (batch_size, 1)
        if x.dim() == 1:
            x = x.unsqueeze(1)

        angle = x * self.freq.to(x.device) * 1000.0
        embedding = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1) * x

        embedding = self.dense1(embedding)
        embedding = F.silu(embedding) # SiLU is equivalent to Swish
        embedding = self.dense2(embedding)
        embedding = F.silu(embedding)

        return embedding
