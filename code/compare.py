import torch

# Load the Graph_data.pt files
ap_data = torch.load('./results/AP/graph_data.pt')
bp_data = torch.load('./results/BP/graph_data.pt')

# Extract the node feature matrices
ap_node_features = ap_data['node_features']
bp_node_features = bp_data['node_features']

# Ensure the node feature matrices have the same shape
assert ap_node_features.shape == bp_node_features.shape, "Node feature matrices have different shapes!"

# Compare the zeroes in the matrices for each node
num_nodes = ap_node_features.shape[0]

for node in range(num_nodes):
    ap_zeroes = (ap_node_features[node] == 0).sum().item()
    bp_zeroes = (bp_node_features[node] == 0).sum().item()
    print(f"Node {node}: AP zeroes = {ap_zeroes}, BP zeroes = {bp_zeroes}")