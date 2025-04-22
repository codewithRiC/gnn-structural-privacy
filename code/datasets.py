import os
from functools import partial
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, Actor, Amazon
from torch_geometric.transforms import ToSparseTensor, RandomNodeSplit
from torch_geometric.utils import to_undirected

from transforms import Normalize, FilterTopClass


class KarateClub(InMemoryDataset):
    url = 'https://raw.githubusercontent.com/benedekrozemberczki/karateclub/master/dataset/node_level'
    available_datasets = {
        'twitch',
        'facebook',
        'github',
        'deezer',
        'lastfm',
        'wikipedia'
    }

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in self.available_datasets

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return ['edges.csv', 'features.csv', 'target.csv']

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for part in ['edges', 'features', 'target']:
            download_url(f'{self.url}/{self.name}/{part}.csv', self.raw_dir)

    def process(self):
        target_file = os.path.join(self.raw_dir, self.raw_file_names[2])
        y = pd.read_csv(target_file)['target']
        y = torch.from_numpy(y.to_numpy(dtype=int))
        num_nodes = len(y)

        edge_file = os.path.join(self.raw_dir, self.raw_file_names[0])
        edge_index = pd.read_csv(edge_file)
        edge_index = torch.from_numpy(edge_index.to_numpy()).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes)  # undirected edges

        feature_file = os.path.join(self.raw_dir, self.raw_file_names[1])
        x = pd.read_csv(feature_file).drop_duplicates()
        x = x.pivot(index='node_id', columns='feature_id', values='value').fillna(0)
        x = x.reindex(range(num_nodes), fill_value=0)
        x = torch.from_numpy(x.to_numpy()).float()

        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'KarateClub-{self.name}()'


supported_datasets = {
    'cornell': partial(WebKB, name='cornell'),
    'photo': partial(Amazon, name='photo'),
    'wisconsin': partial(WebKB, name='wisconsin'),
    'actor': partial(Actor),
    'chameleon': partial(WikipediaNetwork, name='chameleon'),
    'cora': partial(Planetoid, name='cora'),
    'citeseer': partial(Planetoid, name='citeseer'),
    'pubmed': partial(Planetoid, name='pubmed'),
    'facebook': partial(KarateClub, name='facebook'),
    'lastfm': partial(KarateClub, name='lastfm', transform=FilterTopClass(10)),
}

class BitcoinAlpha(InMemoryDataset):
    available_datasets = {'bitcoinalpha'}

    def __init__(self, root, name='bitcoinalpha', transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in self.available_datasets

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return ['bitcoinalpha.csv']

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Skip downloading since the file is already available locally
        print("Skipping download. Ensure the 'bitcoinalpha.csv' file is present in the raw directory.")

    def process(self):
        import networkx as nx
        import numpy as np
        import pickle
        import json

        # Load the single CSV file
        file_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        df = pd.read_csv(file_path)

        # Create a directed graph from the edge list
        G = nx.from_pandas_edgelist(df, source='SOURCE', target='TARGET', edge_attr='RATING', create_using=nx.DiGraph())

        # Relabel nodes to ensure they are sequentially numbered
        mapping = {node: idx for idx, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)

        # Extract edge attributes
        rating = nx.get_edge_attributes(G, 'RATING')
        max_rating = max(rating.values())
        degree_sequence_in = [d for _, d in G.in_degree()]
        dmax_in = max(degree_sequence_in)
        degree_sequence_out = [d for _, d in G.out_degree()]
        dmax_out = max(degree_sequence_out)

        # Generate labels for nodes
        label_mapping = {}
        rate_mapping = {}
        decision_threshold = 0.3
        number_of_in_nodes_threshold = 3

        for node in G.nodes():
            in_edges_list = G.in_edges(node)
            if len(in_edges_list) < number_of_in_nodes_threshold:
                total_rate = 0
                label = 0
                rate_mapping[node] = 0
                label_mapping[node] = label
            else:
                total_rate = 0
                for source, _ in in_edges_list:
                    total_rate += G.get_edge_data(source, node)['RATING'] / abs(G.get_edge_data(source, node)['RATING'])
                average_rate = total_rate / len(in_edges_list)

                label = 1 if average_rate >= decision_threshold else 0
                rate_mapping[node] = average_rate
                label_mapping[node] = label

        # Generate features for nodes
        feature_length = 8
        feat_dict = {}
        for node in G.nodes():
            out_edges_list = G.out_edges(node)

            if len(out_edges_list) == 0:
                features = np.ones(feature_length, dtype=float) / 1000
            else:
                features = np.zeros(feature_length, dtype=float)
                w_pos = 0
                w_neg = 0
                for _, target in out_edges_list:
                    w = G.get_edge_data(node, target)['RATING']
                    if w >= 0:
                        w_pos += w
                    else:
                        w_neg -= w

                abstotal = w_pos + w_neg
                average = (w_pos - w_neg) / len(out_edges_list) / max_rating

                features[0] = w_pos / max_rating / len(out_edges_list)  # average positive vote
                features[1] = w_neg / max_rating / len(out_edges_list)  # average negative vote
                features[2] = w_pos / abstotal
                features[3] = average
                features[4] = features[0] * G.in_degree(node) / dmax_in
                features[5] = features[1] * G.in_degree(node) / dmax_in
                features[6] = features[0] * G.out_degree(node) / dmax_out
                features[7] = features[1] * G.out_degree(node) / dmax_out

                features = features / 1.01 + 0.001

            feat_dict[node] = features

        # Convert features and labels to tensors
        num_nodes = len(G.nodes())
        x = torch.tensor([feat_dict[node] for node in range(num_nodes)], dtype=torch.float)
        y = torch.tensor([label_mapping[node] for node in range(num_nodes)], dtype=torch.long)

        # Convert edges to edge_index
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t()

        # Create the PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Save the processed data
        torch.save(self.collate([data]), self.processed_paths[0])
        
        
        #------#TODO: Modified to save the original raw data-----------------------------
        # Create train, test, and validation masks 
        train_size = int(num_nodes * 0.5)  # 50% for training
        test_size = int(num_nodes * 0.25)  # 25% for testing
        val_size = num_nodes - train_size - test_size  # Remaining 25% for validation

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[:train_size] = True
        test_mask[train_size:train_size + test_size] = True
        val_mask[train_size + test_size:] = True

        # Shuffle the masks to ensure randomness
        perm = torch.randperm(num_nodes)
        train_mask = train_mask[perm]
        test_mask = test_mask[perm]
        val_mask = val_mask[perm]

        # Save the processed dataset as raw files
        processed_dir = self.processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        prefix = os.path.join(processed_dir, f"ind.{self.name}")

        # Save node features
        with open(f"{prefix}.x", "wb") as f:
            pickle.dump(x[train_mask], f)  # Training node features
        with open(f"{prefix}.tx", "wb") as f:
            pickle.dump(x[test_mask], f)  # Test node features
        with open(f"{prefix}.allx", "wb") as f:
            pickle.dump(x[train_mask | val_mask], f)  # All node features (train + validation)

        # Save node labels
        with open(f"{prefix}.y", "wb") as f:
            pickle.dump(y[train_mask], f)  # Training node labels
        with open(f"{prefix}.ty", "wb") as f:
            pickle.dump(y[test_mask], f)  # Test node labels
        with open(f"{prefix}.ally", "wb") as f:
            pickle.dump(y[train_mask | val_mask], f)  # All node labels

        # Save graph structure
        graph = {}
        for i in range(num_nodes):
            neighbors = edge_index[1, edge_index[0] == i].tolist()
            graph[i] = neighbors
        with open(f"{prefix}.graph", "w") as f:
            json.dump(graph, f, indent=4)

        # Save test indices
        test_indices = torch.nonzero(test_mask, as_tuple=True)[0].tolist()
        with open(f"{prefix}.test.index", "w") as f:
            f.write("\n".join(map(str, test_indices)))
            
        # Save the entire graph as a .pt file
        graph_data = {
            "node_features": x,
            "node_labels": y,
            "edge_index": edge_index,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
        }
        torch.save(graph_data, os.path.join(processed_dir, "graph_data.pt"))    

        print(f"Processed dataset saved in {processed_dir}")
        
        #------------------------------------------------------------------------------
            
    def __repr__(self):
        return f'BitcoinAlpha-{self.name}()'


# Add BitcoinAlpha to the supported datasets
supported_datasets['bitcoinalpha'] = partial(BitcoinAlpha, name='bitcoinalpha')

def load_dataset(
        dataset: dict(help='name of the dataset', option='-d', choices=supported_datasets) = 'cora',
        data_dir: dict(help='directory to store the dataset') = './datasets',
        data_range: dict(help='min and max feature value', nargs=2, type=float) = (0, 1),
        val_ratio: dict(help='fraction of nodes used for validation') = .25,
        test_ratio: dict(help='fraction of nodes used for test') = .25,
):
    data = supported_datasets[dataset](root=os.path.join(data_dir, dataset))
    data = RandomNodeSplit(split='train_rest', num_val=val_ratio, num_test=test_ratio)(data[0])
    data = ToSparseTensor()(data)
    data.name = dataset
    data.num_classes = int(data.y.max().item()) + 1

    if data_range is not None:
        low, high = data_range
        data = Normalize(low, high)(data)

    return data


def get_edge_sets(data, random_order=False):
    # Takes a pyg data.Data dataset, computes and return
    # existing_edges = [(idx, idx),...], non_existing_edges = [(idx, idx),...].

    dense_adj = data.adj_t.to_dense()
    existing_edges = dense_adj.nonzero()
    non_existing_edges = (dense_adj == 0).nonzero()

    if random_order:
        existing_edges = existing_edges[torch.randperm(existing_edges.size()[0])]
        non_existing_edges = non_existing_edges[torch.randperm(existing_edges.size()[0])]

    return existing_edges, non_existing_edges


def generate_random_edge_sets(data, perc_ones=0.1):
    dense_adj = data.adj_t.to_dense()
    dense_adj = (torch.rand(size=dense_adj.size()) < perc_ones).int()
    existing_edges = dense_adj.nonzero()
    non_existing_edges = (dense_adj == 0).nonzero()

    return existing_edges, non_existing_edges


def compare_adjacency_matrices(data, non_sp_data):
    dense = data.adj_t.to_dense()
    non_sp_dense = non_sp_data.adj_t.to_dense()
    # print(dense)
    diff = int(torch.sum(torch.abs(dense - non_sp_dense)))
    print(f"Comparing datasets: the two adjacency matrices have {diff}/{torch.numel(dense)} different entries.")

    print("Number of edges:")
    print(f"Perturbed: {int(torch.sum(dense))} edges, {int(torch.numel(dense) - torch.sum(dense))} non-edges")
    print(
        f"Original: {int(torch.sum(non_sp_dense))} edges, {int(torch.numel(non_sp_dense) - torch.sum(non_sp_dense))} non-edges")

    # getting edge lists from data
    existing_edges, non_existing_edges = get_edge_sets(data)
    non_sp_existing_edges, non_sp_non_existing_edges = get_edge_sets(non_sp_data)

    # computing list intersections
    l1 = existing_edges.tolist()
    l2 = non_sp_existing_edges.tolist()
    common_edges = len([list(x) for x in set(tuple(x) for x in l1).intersection(set(tuple(x) for x in l2))])
    l1 = non_existing_edges.tolist()
    l2 = non_sp_non_existing_edges.tolist()
    common_non_edges = len([list(x) for x in set(tuple(x) for x in l1).intersection(set(tuple(x) for x in l2))])

    print(f"Common edges: {common_edges}")
    print(f"Common non-edges: {common_non_edges}")
