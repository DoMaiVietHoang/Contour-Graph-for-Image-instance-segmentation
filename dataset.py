import os 
import numpy as np
import pandas as pd 
import torch
from torch_geometric.data import Data, Dataset

class TreeCrownGraphDataset(Dataset):
    def __init__(self, adjacent_dir, lpips_dir, shape_dir):
        """
        Init dataset from adjacent_dir, lpips_dir, shape_dir
        Args:
            adjacent_dir: str, path to adjacent matrix
            lpips_dir: str, path to lpips features
            shape_dir: str, path to shape features
        """
        super(TreeCrownGraphDataset, self).__init__()
        self.adjacent_dir = adjacent_dir
        self.lpips_dir = lpips_dir
        self.shape_dir = shape_dir
        self.adjacent_files = sorted([f for f in os.listdir(adjacent_dir) if f.endswith('.txt')])
        self.imager_ids = [f.split('.')[0] for f in self.adjacent_files]
    def len(self):
        return len(self.adjacent_files)
    def get(self, idx):
        """
        Create the Data object for a single tree
        """
        