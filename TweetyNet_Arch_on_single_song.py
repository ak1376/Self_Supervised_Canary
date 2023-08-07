#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 09:26:09 2023

@author: AnanyaKapoor
"""

# %% Housekeeping

from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# import umap
# Parameters we set
num_spec = 1
window_size = 100
stride = 10


folderpath_song = '/Users/ananyakapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/Canary_SSL_Repo/Song_0/'
# plt.figure()
# plt.pcolormesh(times, frequencies, spec, cmap='jet')
# plt.show()


# For each spectrogram we will extract
# 1. Each timepoint's syllable label
# 2. The spectrogram itself
stacked_labels = []
stacked_specs = []
# Extract the data within the numpy file. We will use this to create the spectrogram
dat = np.load(f'{folderpath_song}synthetic_data.npz')
spec = dat['s']
times = dat['t']
frequencies = dat['f']
labels = dat['labels']
labels.shape = (1, labels.shape[0])
labels = labels.T


# Let's get rid of higher order frequencies
mask = (frequencies < 4000) & (frequencies > 600)
masked_frequencies = frequencies[mask]

subsetted_spec = spec[mask.reshape(mask.shape[0],), :]

stacked_labels.append(labels)
stacked_specs.append(subsetted_spec)


stacked_specs = np.concatenate((stacked_specs), axis=1)
stacked_labels = np.concatenate((stacked_labels), axis=0)

# Get a list of unique categories (syllable labels)
unique_categories = np.unique(stacked_labels)

# Create a dictionary that maps categories to random colors
category_colors = {category: np.random.rand(
    3,) for category in unique_categories}

spec_for_analysis = stacked_specs.T
window_labels_arr = []
embedding_arr = []
# Find the exact sampling frequency (the time in miliseconds between one pixel [timepoint] and another pixel)
dx = np.diff(times)[0]

# We will now extract each mini-spectrogram from the full spectrogram
stacked_windows = []
# Find the syllable labels for each mini-spectrogram
stacked_labels_for_window = []
# Find the mini-spectrograms onset and ending times
stacked_window_times = []

# The below for-loop will find each mini-spectrogram (window) and populate the empty lists we defined above.
for i in range(0, spec_for_analysis.shape[0] - window_size + 1, stride):
    # Find the window
    window = spec_for_analysis[i:i + window_size, :]
    # Get the window onset and ending times
    window_times = dx*np.arange(i, i + window_size)
    # We will flatten the window to be a 1D vector
    window = window.reshape(1, window.shape[0]*window.shape[1])
    # Extract the syllable labels for the window
    labels_for_window = stacked_labels[i:i+window_size, :]
    # Reshape the syllable labels for the window into a 1D array
    labels_for_window = labels_for_window.reshape(
        1, labels_for_window.shape[0]*labels_for_window.shape[1])
    # Populate the empty lists defined above
    stacked_windows.append(window)
    stacked_labels_for_window.append(labels_for_window)
    stacked_window_times.append(window_times)

# Convert the populated lists into a stacked numpy array
stacked_windows = np.stack(stacked_windows, axis=0)
stacked_windows = np.squeeze(stacked_windows)

stacked_labels_for_window = np.stack(stacked_labels_for_window, axis=0)
stacked_labels_for_window = np.squeeze(stacked_labels_for_window)

stacked_window_times = np.stack(stacked_window_times, axis=0)

# Calculate the row-wise sum
# row_sums = stacked_windows.sum(axis=1)

# Perform row normalization
# stacked_windows  = stacked_windows  / row_sums[:, np.newaxis]


# Z-score
row_means = np.mean(stacked_windows, axis=1)
row_stds = np.std(stacked_windows, axis=1)

stacked_windows = (stacked_windows -
                   row_means[:, np.newaxis]) / row_stds[:, np.newaxis]

if torch.cuda.is_available():
  device = "cuda:0"
else:
  device = "cpu"


# %% Define neural network


class TweetyNetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=(5, 5), stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(8, 1), stride=(8, 1))
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(8, 1), stride=(8, 1))
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64*1*36, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.fc3(x)
        return x


cnn_model = TweetyNetCNN()
cnn_model = cnn_model.double().to(device)

optimizer = optim.Adam(cnn_model.parameters(), lr=0.001, weight_decay=1e-5)
# criterion = nn.CrossEntropyLoss()

# %% Train Neural network

stacked_windows_tensor = stacked_windows.copy()
stacked_windows_tensor.shape = (stacked_windows.shape[0], 1, 100, 40)
stacked_windows_tensor = torch.tensor(stacked_windows_tensor).double()

# pil_image = Image.fromarray(stacked_windows_tensor)

# # Define the transformation pipeline
# transform = transforms.Compose([
#     transforms.ToTensor(),                # Convert PIL image to PyTorch tensor
#     transforms.Normalize((0.5,), (0.5,))  # Normalize the data
# ])

# Apply the transformations
# transformed_data = transform(pil_image)

# print(transformed_data.shape)
# transformed_data = transformed_data.reshape(transformed_data.shape[1], 1, 100, 40)


# stacked_windows_tensor = transformed_data.double()

actual_labels = np.max(stacked_labels_for_window, axis=1)
actual_labels = torch.tensor(actual_labels)

batch_size = 64

# Create a TensorDataset to combine features and labels
dataset = TensorDataset(stacked_windows_tensor, actual_labels)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # Iterate through batches
# loss_epoch_list = []
# num_epochs = 100
# for epoch in np.arange(num_epochs):
#     total_loss_epoch = 0
#     for batch_idx, (data, targets) in enumerate(train_loader):
#         preds = cnn_model(data).float()
#         loss = criterion(preds, targets.long())
#         total_loss_epoch+=loss.item()
#         optimizer.zero_grad()
#         loss.backward()
#         # loss.backward()
#         optimizer.step()
#     loss_epoch_list.append(total_loss_epoch)
#     print(total_loss_epoch)


# # x = stacked_windows[0:10,:]
# # x.shape = (10, 1, 100, 40)
# # x = torch.tensor(x).double().to(device)

# # a = cnn_model(x)

# # optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)


# x = cnn_model.conv1(stacked_windows_tensor)
# x = cnn_model.relu(x)
# x = cnn_model.pool1(x)
# x = cnn_model.conv2(x)
# x = cnn_model.relu(x)
# x = cnn_model.pool2(x)
# x = torch.flatten(x, 1)
# x = cnn_model.fc1(x)


# import umap
# reducer = umap.UMAP()
# embedding = reducer.fit_transform(x.detach().numpy())

# %% Contrastive Loss Function (Chopra 2005)


# def contrastive_loss(embedding_1, embedding_2, label_1, label_2, epsilon):
#     if label_1 == label_2:
#         loss = torch.norm(embedding_1 - embedding_2)**2
#     else:
#         # if the distance between embeddings < epsilon then our loss will be bigger. This encourages our loss function to represent dissimilar syllables distinctly (greater than margin which means the loss component there will be 0)
#         loss = torch.max(torch.tensor(0), epsilon -
#                          (torch.norm(embedding_1 - embedding_2)))**2

#     return loss


# criterion = contrastive_loss
# total_loss_for_batch_list = []
# for batch_idx, (data, targets) in enumerate(train_loader):
#     # Get the number of rows and columns in the tensor
#     num_rows = data.shape[0]

#     # Create a list of all possible pairs of row indices
#     row_indices = list(range(num_rows))
#     pairs = list(product(row_indices, repeat=2))

#     total_loss_for_batch = 0
#     for training_example in np.arange(len(pairs)):
#         sample_1_index = pairs[training_example][0]
#         sample_2_index = pairs[training_example][1]

#         if sample_1_index == sample_2_index:
#             continue

#         # Extract the ground truth label for each sample
#         sample_1_label = targets[sample_1_index]
#         sample_2_label = targets[sample_2_index]

#         # Extract the spectrogram slice for each sample
#         sample_1_slice = data[sample_1_index, :, :, :].unsqueeze(1)
#         sample_2_slice = data[sample_2_index, :, :, :].unsqueeze(1)

#         # Run each sample through our Siamese network
#         sample_1_embedding = cnn_model(sample_1_slice)
#         sample_2_embedding = cnn_model(sample_2_slice)

#         loss = criterion(sample_1_embedding, sample_2_embedding,
#                          sample_1_label, sample_2_label, 3)
#         total_loss_for_batch += loss

#     total_loss_for_batch_list.append(total_loss_for_batch.item())
#     optimizer.zero_grad()
#     total_loss_for_batch.backward()
#     optimizer.step()

# embedding_arr = np.empty((0, 1000))

# for batch_idx, (data, targets) in enumerate(train_loader):
#     embed = cnn_model(data).detach().numpy()
#     embedding_arr = np.concatenate((embedding_arr, embed))

# np.save('/Users/AnanyaKapoor/Desktop/Contrastive_Loss_TweetyNet_Embed.npy', embedding_arr)

# %% Noise Contrastive Estimation

class TweetyNetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=(5, 5), stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(8, 1), stride=(8, 1))
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(8, 1), stride=(8, 1))
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64*1*36, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


cnn_model = TweetyNetCNN()
cnn_model = cnn_model.double().to(device)

optimizer = optim.Adam(cnn_model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.BCELoss()


positive_sample_index_list = []
negative_samples_indices_list = []


unique_syllables = torch.unique(actual_labels)

indices_dict = {int(element): np.where(actual_labels == element)[0] for element in unique_syllables}
total_batch_loss_list = []
for batch_idx, (data, targets) in enumerate(train_loader):
    total_batch_loss = 0
    for anchor_index in np.arange(data.shape[0]):
        anchor_label = targets[anchor_index]
        # Sample a positive sample from our total dataset
        indices_of_positve_samples = torch.where(actual_labels == anchor_label)[0]
        positive_sample_index = torch.randint(0, indices_of_positve_samples.shape[0], size=(1,))
        positive_sample_index_list.append(positive_sample_index.item())
        
        # Sample negative samples from our total dataset. We will use a 
        # weighted probability distribution. There will be a 0.0001 probability
        # that we will sample from the positive class and a (1-0.001)/(K-1)
        # probability that we sample from the remaining K-1 classes
        
        # Create a tensor with custom probabilities
        epsilon = 0.0001
        probs = torch.zeros((1, unique_categories.shape[0]))
        probs[:,:] = (1-epsilon)/(probs.shape[1]-1)
        probs[:,int(anchor_label)] = epsilon

        
        # Number of samples to generate
        num_samples = unique_categories.shape[0] - 1
        
        # Sample indices based on the custom probabilities
        sampled_labels = torch.multinomial(probs, num_samples, replacement=False)
        
        # Now let's randomly sample an index value from each sampled label
        random_samples = {key: np.random.choice(values) for key, values in indices_dict.items() if key in sampled_labels}
        indices_of_negative_samples = np.array(list(random_samples.values()))
        negative_samples_indices_list.append(indices_of_negative_samples)
        
        # Now let's extract the positive and negative samples' spectrogram 
        # slices
        
        positive_sample = stacked_windows_tensor[positive_sample_index, :,:,:]
        negative_samples = stacked_windows_tensor[indices_of_negative_samples, :,:,:]
        
        dat = torch.concatenate((positive_sample, negative_samples))
        
        artificial_labels = torch.zeros((1,unique_categories.shape[0]))
        artificial_labels[:,0] = 1
        
        # Get the number of rows in the tensors
        num_rows = dat.shape[0]
        
        # Generate a random permutation of indices
        shuffled_indices = torch.randperm(num_rows)
        
        # Shuffle both tensors based on the same indices
        dat = dat[shuffled_indices,:,:,:]
        artificial_labels = artificial_labels[:,shuffled_indices]
        
        pred_probs = cnn_model(dat)
        
        # h_t = 1/(1+torch.exp(-1*(torch.log(pred_probs) - torch.log(probs).T)))
        # torch.sum(artificial_labels.T*torch.log(h_t) + (1-artificial_labels.T)*torch.log(h_t))
        
        loss = criterion(pred_probs, artificial_labels.T.double())
        total_batch_loss+=loss
        
    total_batch_loss_list.append(total_batch_loss.item())
    optimizer.zero_grad()
    total_batch_loss.backward()
    optimizer.step()

    
        

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
