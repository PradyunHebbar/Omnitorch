import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import random

class JetClassDataset(Dataset):
    def __init__(self, path):
        self.path = path
        
        self.files = [os.path.join(self.path, f) for f in os.listdir(path) if f.endswith('.h5')]
        self.files.sort()
        #self.files = self.files[rank::world_size]  # Distribute files among processes handled by DS
        
        self.mean_part = [0.0, 0.0, -0.0278, 1.8999407, -0.027, 2.244736, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.std_part = [0.215, 0.215, 0.070, 1.2212526, 0.069, 1.2334691, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        
        self.mean_jet = [6.18224920e+02, 0.0, 1.2064709e+02, 3.94133173e+01]
        self.std_jet = [106.71761, 0.88998157, 40.196922, 15.096386]
        
        self.num_classes = 10
        self.num_feat = 13
        self.num_jet = 4
        
    def __len__(self):
        return len(self.files) * 100000  # Assuming each file has 100,000 events
    
    def __getitem__(self, idx):  #Generalize data loading 
        file_idx = idx // 100000
        event_idx = idx % 100000
        
        with h5py.File(self.files[file_idx], 'r') as f:
            data = f['data'][event_idx]
            jet = f['jet'][event_idx]
            pid = f['pid'][event_idx]
        
        mask = data[:, 2] != 0
        data = self.preprocess(data, mask)
        jet = self.preprocess_jet(jet)
        
        return {
            'input_features': torch.FloatTensor(data),
            'input_points': torch.FloatTensor(data[:, :2]),
            'input_mask': torch.FloatTensor(mask),
            'input_jet': torch.FloatTensor(jet)
        }, torch.LongTensor(pid)
    
    def preprocess(self, x, mask):
        return mask[:, None] * (x - self.mean_part) / self.std_part
    
    def preprocess_jet(self, x):
        return (x - self.mean_jet) / self.std_jet
# use num_workers to load data with multiple processes. Rule of thumb = num_workers = 4 * num_GPUs
# Each process (GPU) will create its own DataLoader with num_workers workers.
# def get_dataloader(path, batch_size, rank, world_size, shuffle=True, num_workers=4):
#     dataset = JetClassDataset(path, rank, world_size)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
def get_dataloader(path, batch_size, shuffle=True, num_workers=4):
    dataset = JetClassDataset(path)
    sampler = DistributedSampler(dataset, shuffle=shuffle)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=num_workers, pin_memory=True)