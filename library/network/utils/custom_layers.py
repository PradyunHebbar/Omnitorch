import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineDecayWithWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, decay_steps, warmup_start_lr, warmup_target_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.warmup_start_lr = warmup_start_lr
        self.warmup_target_lr = warmup_target_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps: # self.last_epoch is actually last_step
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [self.warmup_start_lr + alpha * (self.warmup_target_lr - self.warmup_start_lr) for _ in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / (self.decay_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.warmup_target_lr * cosine_decay for _ in self.base_lrs]


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor)  # binarize
        return torch.div(x, keep_prob) * random_tensor

class RandomDrop(nn.Module):
    def __init__(self, drop_prob: float, num_skip: int):
        super().__init__()
        self.drop_prob = drop_prob
        self.num_skip = num_skip

    def forward(self, x):
        if not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0], 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize [ IN-PLACE OPERATION ]
        
        # Create a new tensor instead of modifying in-place
        output = x.clone()
        output[:, :, self.num_skip:] = x[:, :, self.num_skip:] * random_tensor.unsqueeze(2)
        return output

class TalkingHeadAttention(nn.Module):
    def __init__(self, projection_dim: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        
        head_dim = self.projection_dim // self.num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(projection_dim, projection_dim * 3)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(projection_dim, projection_dim)
        self.proj_l = nn.Linear(self.num_heads, self.num_heads)
        self.proj_w = nn.Linear(self.num_heads, self.num_heads)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x, int_matrix=None, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))
        if int_matrix is not None:
            attn += int_matrix

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn += (1.0 - mask) * -1e9

        attn = F.softmax(attn, dim=-1)
        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class LayerScale(nn.Module):
    def __init__(self, init_values, projection_dim):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(projection_dim))

    def forward(self, x, mask=None):
        if mask is not None:
            return x * self.gamma * mask.unsqueeze(-1)
        else:
            return x * self.gamma

# Note: Using PyTorch's native LayerNorm instead of GroupNormalization since it exists
#print("Note: Using PyTorch's native LayerNorm instead of GroupNormalization")