import copy

import torch_geometric


class LiftedDataset(torch_geometric.data.Dataset):
    def __init__(self, data, lift):
        super().__init__()
        self.data = copy.copy(data)
        self.fields = lift.added_fields
        self.lift = lift
        self.lifted_data = self.apply_lift(data)
        
    def apply_lift(self, data):
        lifting = {}
        for field in self.fields:
            lifting[field] = []
            
        for i in range(len(data)):
            d = data.get(i)
            d_lifted = self.lift(d)
            for field in self.fields:
                lifting[field].append(d_lifted[field])
        return lifting
    
    def get(self, idx):
        data = self.data[idx]
        for field in self.fields:
            data[field] = self.lifted_data[field][idx]
        return data
    
    def len(self):
        return len(self.data)