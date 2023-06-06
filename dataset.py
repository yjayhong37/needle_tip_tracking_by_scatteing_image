import csv
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Define the custom dataset
class CustomDataset(Dataset):
    def __init__(self, img_folder, csv_folder, transform=None, normalization=None):
        self.img_folder = img_folder
        self.csv_folder = csv_folder
        self.transform = transform
        self.normalization = normalization
        self.file_list = sorted([f for f in os.listdir(img_folder) if f.endswith('.png')])

    def __len__(self):
        return len(self.file_list)

    def _normalize(self, value, data_range, norm_range):
        normalized_value = (value - data_range[0]) / (data_range[1] - data_range[0])
        return normalized_value * (norm_range[1] - norm_range[0]) + norm_range[0]

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.img_folder, img_name)
        csv_name = 'Trak_' + str(idx) + '.csv'
        csv_path = os.path.join(self.csv_folder, csv_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            row = next(reader)
            target = [float(val) for val in row]
            if self.normalization is not None:
                for key, value in self.normalization.items():
                    for k, v in value.items():
                        data_range = v['data_range']
                        norm_range = v['norm_range']
                        target[v['idx']] = self._normalize(target[v['idx']], data_range, norm_range)

        target = torch.tensor(target).unsqueeze(0).float()  # Add batch dimension

        return image, target