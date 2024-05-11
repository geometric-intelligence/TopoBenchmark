import numpy as np
import os
import torch
import torch_geometric
import urllib.request


def load_heterophilic_data(name, path):
    file_name = f'{name}.npz'

    data = np.load(os.path.join(path, file_name))

    x = torch.tensor(data['node_features'])
    y = torch.tensor(data['node_labels'])
    edge_index = torch.tensor(data['edges']).T

    # Make edge_index undirected
    edge_index = torch_geometric.utils.to_undirected(edge_index)

    # Remove self-loops
    edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
    
    data = torch_geometric.data.Data(x=x, y=y, edge_index=edge_index)
    return data

def download_hetero_datasets(name, path):
    url = 'https://github.com/OpenGSL/HeterophilousDatasets/raw/main/data/'
    name = f'{name}.npz'
    try:
        print(f'Downloading {name}')
        path2save = os.path.join(path, name)
        urllib.request.urlretrieve(url + name, path2save)
        print('Done!')
    except Exception as e:
        raise Exception('''Download failed! Make sure you have stable Internet connection and enter the right name''') from e