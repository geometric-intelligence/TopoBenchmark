import torch
import torch_geometric


class ProjectionLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()

    def lift_features(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        keys = sorted([key.split("_")[1] for key in data.keys() if "incidence" in key])
        for elem in keys:
            if f"x_{elem}" not in data:
                idx_to_project = 0 if elem == "hyperedges" else int(elem) - 1
                data["x_" + elem] = torch.matmul(
                    data["incidence_" + elem].t(),
                    data[f"x_{idx_to_project}"],
                )
        return data

    def forward(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        data = self.lift_features(data)
        return data

    def __call__(self, data):
        return self.forward(data)
    
class ConcatentionLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()

    def lift_features(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        
        def non_zero_positions(tensor):
            positions = []
            for i in range(tensor.size(0)):
                non_zero_indices = torch.nonzero(tensor[i]).squeeze()
                
                # Check if non_zero_indices is empty
                if non_zero_indices.numel() > 0:
                    positions.append(non_zero_indices)
            # Sort the positions such respecting the node index order
            positions = torch.stack(positions).sort()[0]
            return positions
        
        keys = sorted([key.split("_")[1] for key in data.keys() if "incidence" in key])
        for elem in keys:
            if f"x_{elem}" not in data:
                idx_to_project = 0 if elem == "hyperedges" else 0
                dense_incidence = data["incidence_" + elem].T.to_dense()
                n, _ = dense_incidence.shape
                
                if n != 0:
                    positions = non_zero_positions(dense_incidence)
                    
                    # Obtain the node representations, so it can be combined for higher order features.
                    for i,_ in enumerate(range(int(elem), 1, -1)):
                        dense_incidence = abs(data["incidence_" + str(int(elem) - 1 - i)].T.to_dense())
                        dense_incidence = dense_incidence[positions].sum(dim=1)
                        positions = non_zero_positions(dense_incidence)
                            
                    values = data[f"x_{idx_to_project}"][positions].view(n, -1)
                else:
                    values = torch.tensor([])
                
                data["x_" + elem] = values
        return data

    def forward(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        data = self.lift_features(data)
        return data

    def __call__(self, data):
        return self.forward(data)

class SetLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()

    def lift_features(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        
        def non_zero_positions(tensor):
            positions = []
            for i in range(tensor.size(0)):
                non_zero_indices = torch.nonzero(tensor[i]).squeeze()
                
                # Check if non_zero_indices is empty
                if non_zero_indices.numel() > 0:
                    positions.append(non_zero_indices)
            # Sort the positions such respecting the node index order
            positions = torch.stack(positions).sort()[0]
            return positions
        
        keys = sorted([key.split("_")[1] for key in data.keys() if "incidence" in key])
        for elem in keys:
            if f"x_{elem}" not in data:
                idx_to_project = 0 if elem == "hyperedges" else 0
                dense_incidence = data["incidence_" + elem].T.to_dense()
                n, _ = dense_incidence.shape
                
                if n != 0:
                    positions = non_zero_positions(dense_incidence)
                    
                    # Obtain the node representations, so it can be combined for higher order features.
                    for i,_ in enumerate(range(int(elem), 1, -1)):
                        dense_incidence = abs(data["incidence_" + str(int(elem) - 1 - i)].T.to_dense())
                        dense_incidence = dense_incidence[positions].sum(dim=1)
                        positions = non_zero_positions(dense_incidence)
        
                    values = positions
                else:
                    values = torch.tensor([])
                
                data["x_" + elem] = values
        return data

    def forward(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        data = self.lift_features(data)
        return data

    def __call__(self, data):
        return self.forward(data)