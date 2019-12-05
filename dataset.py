from __future__ import absolute_import, division, print_function

import os
import pdb

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import TensorDataset

class CoNLLGloveDataset(Dataset):
    def __init__(self, path):
        x,y = [],[]
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                items = line.split('\t')
                yi = [int(d) for d in items[0].split()]
                xi = [int(d) for d in items[1].split()]
                x.append(xi)
                y.append(yi)
        
        self.x = torch.tensor(x).long()
        self.y = torch.tensor(y).long()
 
    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class CoNLLBertDataset(Dataset):
    def __init__(self, path):
        # load features from file
        features = torch.load(path)
        # convert to tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_id = torch.tensor([f.label_id for f in features], dtype=torch.long)

        self.x = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        self.y = all_label_id
 
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
