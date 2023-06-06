import numpy as np
import torch
import torchvision.transforms as transforms



# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the input data
])

# Define the normalization process
normalization = {
    "pos": {
        "x": {"data_range": [-8.3, 8.3], "norm_range": [-1, 1], "idx": 0},
        "y": {"data_range": [-5.5, 5.5], "norm_range": [-1, 1], "idx": 1},
        "z": {"data_range": [0, 6.5], "norm_range": [-1, 1], "idx": 2},
    },
    "rot": {
        "ql_1": {"data_range": [-np.pi, np.pi], "norm_range": [-1, 1], "idx": 0},
        "ql_2": {"data_range": [-np.pi, np.pi], "norm_range": [-1, 1], "idx": 1},
        "ql_3": {"data_range": [-np.pi, np.pi], "norm_range": [-1, 1], "idx": 2},
    }
}


def reverse_normalize(value, data_range, norm_range):
    normalized_value = (value - norm_range[0]) / (norm_range[1] - norm_range[0])
    return normalized_value * (data_range[1] - data_range[0]) + data_range[0]

def calculate_distance(predictions, targets):
    distances = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=1))
    return distances