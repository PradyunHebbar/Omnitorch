import torch
import torch.nn as nn

class LocalEmbeddingLayer(nn.Module):
    def __init__(self, projection_dim, K):
        super().__init__()
        self.K = K
        self.projection_dim = projection_dim

    def pairwise_distance(self, points):
        r = torch.sum(points * points, dim=2, keepdim=True)
        m = torch.bmm(points, points.transpose(1, 2))
        D = r - 2 * m + r.transpose(1, 2) + 1e-5
        return D

    def forward(self, points, features):
        distances = self.pairwise_distance(points) # uses custom pairwise function, not torch.cdist
        _, indices = torch.topk(-distances, k=self.K + 1, dim=-1)
        indices = indices[:, :, 1:] # Exclude self
        # indices Shape: (N, P, 10)
        
        batch_size, num_points, _ = features.shape
        batch_indices = torch.arange(batch_size, device=features.device).view(-1, 1, 1)
        batch_indices = batch_indices.repeat(1, num_points, self.K)
        indices = torch.stack([batch_indices, indices], dim=-1)
        # concat indices torch.Size([N, P, K, 2])

        # Gather neighbor features
        neighbors = features[indices[:, :, :, 0], indices[:, :, :, 1]] # Shape: (N, P, K, C) | neighbors: torch.Size([64, 150, 10, 13])
        knn_fts_center = features.unsqueeze(2).expand_as(neighbors) # Shape: (N, P, K, C) | knn fts center: torch.Size([64, 150, 10, 13])
        local_features = torch.cat([neighbors - knn_fts_center, knn_fts_center], dim=-1) # local_features: torch.Size([N, P, K, 26]) local_features.shape[-1] Shape : 2*C

        # try to put it in __init__, by specifying "if" in the for loop
        mlp = nn.Sequential(
            nn.Linear(local_features.shape[-1], 2 * self.projection_dim),
            nn.GELU(approximate='none'),
            nn.Linear(2 * self.projection_dim, self.projection_dim),
            nn.GELU(approximate='none')
        ).to(local_features.device)

        local_features = mlp(local_features)
        local_features = torch.mean(local_features, dim=2)

        return local_features
