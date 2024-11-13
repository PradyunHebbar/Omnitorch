import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch
from torch.utils.data import Dataset, DataLoader
import h5py as h5
import numpy as np
import gc

class TopDataset(Dataset):
    """Dataset for top data."""

    def __init__(self, path, batch_size=512,nevts=None):
        self.path = path
        self.batch_size = batch_size
        self.nevts = nevts

        # Mean and std for preprocessing
        self.mean_part = np.array([0.0, 0.0, -0.0278,
                                   1.8999407, -0.027, 2.244736, 0.0,
                                   0.0, 0.0,  0.0,  0.0,  0.0, 0.0])
        self.std_part = np.array([0.215, 0.215,  0.070,
                                  1.2212526, 0.069, 1.2334691, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        self.mean_jet = np.array([618.22492, 0.0, 120.64709, 39.4133173])
        self.std_jet = np.array([106.71761, 0.88998157, 40.196922, 15.096386])

        self.num_pad = 6
        self.num_jet = 4
        self.num_feat = 13
        self.num_classes = 2

        # Load data indices
        with h5.File(self.path, 'r') as f:
            self.total_events = f['data'].shape[0]
            self.num_part = f['data'].shape[1]
            self.num_feat = f['data'].shape[2] + self.num_pad
            if self.nevts is None or self.nevts > self.total_events:
                self.nevts = self.total_events
            self.indices = list(range(self.nevts))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the index
        index = self.indices[idx]

        # Load data from h5 file
        with h5.File(self.path, 'r') as f:
            data = f['data'][index]
            label = f['pid'][index]
            jet = f['jet'][index]

        # Mask where data[:, 2] != 0
        mask = data[:, 2] != 0

        # Preprocess data
        data = self.preprocess(data, mask)
        data = self.pad(data, num_pad=self.num_pad)

        # Preprocess jet
        jet = self.preprocess_jet(jet)

        # Prepare points_chunk
        points = data[:, :2]

        # Convert label to one-hot encoding
        label = np.identity(2)[int(label)]

        # Convert to torch tensors
        data = torch.tensor(data, dtype=torch.float32)
        points = torch.tensor(points, dtype=torch.float32)
        mask = torch.tensor(mask.astype(np.float32), dtype=torch.float32)
        jet = torch.tensor(jet, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        # Prepare input dict
        input_dict = {
            'input_features': data,
            'input_points': points,
            'input_mask': mask,
            'input_jet': jet
        }

        return input_dict, label

    def pad(self, x, num_pad):
        return np.pad(x, pad_width=((0, 0), (0, num_pad)),
                      mode='constant', constant_values=0)

    def preprocess(self, x, mask):
        num_feat = x.shape[-1]
        return mask[:, None] * (x - self.mean_part[:num_feat]) / self.std_part[:num_feat]

    def preprocess_jet(self, x):
        return (x - self.mean_jet) / self.std_jet





def get_top_dataloader(path, batch_size, num_workers=16, distributed=False, shuffle=False, dist=None):
    dataset = TopDataset(path)
    
    if distributed:
        if dist is not None:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle_loader = False
    else:
        sampler = None
        shuffle_loader = shuffle
        
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle_loader,
        num_workers=num_workers,
        pin_memory=True)