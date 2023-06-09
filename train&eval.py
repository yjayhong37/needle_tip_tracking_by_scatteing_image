# -*- coding: utf-8 -*-
"""Needle_Tip_Tracking_By_Scatteing_image.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14ZZBtujKw1X9nzh5VGhcXTvJJiVjQ236

applied Early Stopping, L2 Regularization and reduced model complexity
"""
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transform
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model import MyModel
from dataset import CustomDataset
from tools import transform, normalization, reverse_normalize, calculate_distance





img_folder = 'directory to img'
csv_folder = 'directory to csv'

# Create a custom dataset
dataset = CustomDataset(img_folder, csv_folder, transform=transform, normalization=normalization)

# Split the dataset into training and testing sets
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Create data loaders for training and testing
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model and move it to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel()
model = model.to(device)




# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

# Define early stopping parameters
patience = 10  # Number of epochs to wait for improvement
best_loss = float('inf')
best_epoch = 0

# Training loop with early stopping
num_epochs = 300
losses = []

for epoch in range(num_epochs):
    running_loss = 0.0

    # Use tqdm to create a progress bar
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)

    for images, targets in progress_bar:
        images = images.to(device)
        targets = targets.to(device).squeeze(1).float()  # Squeeze the target tensor

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Update the progress bar
        progress_bar.set_postfix({'Loss': loss.item()})

    average_loss = running_loss / len(train_dataloader)
    losses.append(average_loss)  # Append the average loss to the list
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, average_loss))

    # Check if the current loss is the best so far
    if average_loss < best_loss:
        best_loss = average_loss
        best_epoch = epoch
        torch.save(model.state_dict(), './best_model.pt')  # Save the best model

    # Check if early stopping criterion is met
    if epoch - best_epoch >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

# Plot the loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()





# Prediction and evaluation

model.eval()

total_distances = torch.zeros(3, device=device)
num_samples = 0
distances_list = []

for images, targets in test_dataloader:
    images = images.to(device)
    targets = targets.clone().detach().to(device).float()

    outputs = model(images)

    # Reverse normalization for each dimension of the target
    for dim, value in normalization['pos'].items():
        data_range = value['data_range']
        norm_range = value['norm_range']
        outputs[:, value['idx']] = reverse_normalize(outputs[:, value['idx']], data_range, norm_range)
        targets[:, 0, value['idx']] = reverse_normalize(targets[:, 0, value['idx']], data_range, norm_range)

    distances = calculate_distance(outputs, targets)
    total_distances += torch.sum(distances, dim=0)
    num_samples += distances.size(0)
    distances_list.append(distances)

average_distances = total_distances / num_samples
average_distance = torch.mean(average_distances)

# Calculate standard deviation for each axis
distances_tensor = torch.cat(distances_list, dim=0)
std_x = torch.std(distances_tensor[:, 0])
std_y = torch.std(distances_tensor[:, 1])
std_z = torch.std(distances_tensor[:, 2])

# Calculate L2 norm
l2_norm = torch.norm(average_distances, p=2)

# Calculate standard deviation of L2 norm
l2_norm_tensor = torch.norm(distances_tensor, p=2, dim=1)
std_l2_norm = torch.std(l2_norm_tensor)

print('Average distance in x-axis: {:.4f} mm'.format(average_distances[0] * 10))
print('Average distance in y-axis: {:.4f} mm'.format(average_distances[1] * 10))
print('Average distance in z-axis: {:.4f} mm'.format(average_distances[2] * 10))
print('Accuracy : {:.4f} mm'.format(l2_norm * 10))
print("="*50)
print('Standard deviation in x-axis: {:.4f} mm'.format(std_x * 10))
print('Standard deviation in y-axis: {:.4f} mm'.format(std_y * 10))
print('Standard deviation in z-axis: {:.4f} mm'.format(std_z * 10))
print('Standard deviation of Accuracy: {:.4f} mm'.format(std_l2_norm*10))