import copy

import torch_geometric

class Transform():
    def __init__(self, lift):
        self.lift = lift
        
    def transform(self, list_of_data):
        data_lifted = []
        for i in range(len(list_of_data)):
            list_of_data[i] = self.lift(list_of_data[i])
        return list_of_data

class LiftedDataset(torch_geometric.data.Dataset):
    def __init__(self, list_of_data):
        super().__init__()
        self.list_of_data = list_of_data
        
    def get(self, idx):
        return self.list_of_data[idx]
    
    def len(self):
        return len(self.list_of_data)