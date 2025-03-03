# import torch

# # Replace 'model.pt' with the path to your .pt file
# file_path = "./code/results/private_structure_cd2c9da4-d367-11ef-9aa9-54b20380cf4d/node_0.pt"

# # Load the .pt file
# data = torch.load(file_path)

# # Print its keys or structure
# if isinstance(data, dict):
#     print("Keys in the .pt file:")
#     for key in data.keys():
#         print(key)
# else:
#     print("Content of the .pt file:")
#     print(data)


import torch
import pandas as pd

# Iterate through files
for i in range(2707):
    file_path = f"./code/results/FeatureP_e_0.1/graph_data.pt"
    # file_path = f"./code/datasets/cora/cora/processed/data.pt"


    try:
        # Load the .pt file
        data = torch.load(file_path)

        # If data is a dictionary, convert it to a table
        if isinstance(data, dict):
            print(f"Node {i} - Keys in the .pt file:")
            for key, value in data.items():
                if torch.is_tensor(value):  # Convert tensor data to a table
                    # Move the tensor to CPU if it's on CUDA
                    if value.is_cuda:
                        value = value.cpu()

                    # Convert tensor to a DataFrame
                    df = pd.DataFrame(value.numpy())  
                    print(f"\nKey: {key}")
                    print(df)
                else:
                    print(f"Key: {key}, Value: {value}")  # For non-tensor data
        else:
            # For non-dict data, print it directly
            print(f"Node {i} - Content of the .pt file:")
            if torch.is_tensor(data):
                # Move the tensor to CPU if it's on CUDA
                if data.is_cuda:
                    data = data.cpu()

                # Convert tensor to a DataFrame
                df = pd.DataFrame(data.numpy())  
                print(df)
            else:
                print(data)

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
