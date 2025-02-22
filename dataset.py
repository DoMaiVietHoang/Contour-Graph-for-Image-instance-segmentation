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
        image_id = self.imager_ids[idx]
        adj_file = os.path.join(self.adjacent_dir, f"{image_id}.txt")
        with open(adj_file, 'r') as f:
            adj_data = f.read().strip().split()
            adj_data = np.array([int(x) for x in adj_data], dtype=np.int32)
            num_nodes = int(np.sqrt(len(adj_data)))
            A = adj_data.reshape(num_nodes, num_nodes)
        #Create the edge indx from the adjacent matrix
        edge_index = np.where(A>0)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        #Construct the shape featrues
        shape_file = os.path.join(self.shape_dir, f"{image_id}_shape_features.csv")
        shape_df   = pd.read_csv(shape_file)
        shape_df   = shape_df.sort_values('Region ID')
        node_features = shape_df[['Area', 'Extent', 'Aspect Ratio', 'Solidity', 'Deviation']].values
        x = torch.tensor(node_features, dtype=torch.float)
        #Construct the similarity LPIPS
        lpips_file = os.path.join(self.lpips_dir, f"{image_id}_LPIPS.csv")
        lpips_df   = pd.read_csv(lpips_file)
        edge_attr  = []
        for i, (src, dst) in enumerate(zip(edge_index[0].numpy(), edge_index[1].numpy())):
            row = lpips_df[((lpips_df['Region ID'] == src + 1) & 
                           (lpips_df['Adjacent Region ID'] == dst + 1)) |
                          ((lpips_df['Region ID'] == dst + 1) & 
                           (lpips_df['Adjacent Region ID'] == src + 1))]
            if not row.empty:
                similarity = row['Similarity'].values[0]
            else:
                similarity = 0.0  
        edge_attr.append([similarity])
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

# adjacent_dir = '/Users/admin/Desktop/Computer Vision/Contour-Graph-for-Image-instance-segmentation/Data/Adjacent/'
# lpips_dir = '/Users/admin/Desktop/Computer Vision/Contour-Graph-for-Image-instance-segmentation/Data/LPIPS/'
# shape_dir = '/Users/admin/Desktop/Computer Vision/Contour-Graph-for-Image-instance-segmentation/Data/Shape_features/'

# dataset = TreeCrownGraphDataset(adjacent_dir, lpips_dir, shape_dir)
# sample = dataset[1]
# print("Node features shape:", sample.x.shape)
# print("Edge index shape:", sample.edge_index.shape)
# print("Edge attributes shape:", sample.edge_attr.shape)



            



