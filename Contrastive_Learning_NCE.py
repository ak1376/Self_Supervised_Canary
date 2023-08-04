#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 10:35:05 2023

@author: AnanyaKapoor
"""

# %% Housekeeping

import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
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
mask = (frequencies<4000)&(frequencies>600)
masked_frequencies = frequencies[mask]

subsetted_spec = spec[mask.reshape(mask.shape[0],),:]

stacked_labels.append(labels)
stacked_specs.append(subsetted_spec)

    
stacked_specs = np.concatenate((stacked_specs), axis = 1)
stacked_labels = np.concatenate((stacked_labels), axis = 0)

# Get a list of unique categories (syllable labels)
unique_categories = np.unique(stacked_labels)

# Create a dictionary that maps categories to random colors
category_colors = {category: np.random.rand(3,) for category in unique_categories}

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
    labels_for_window = labels_for_window.reshape(1, labels_for_window.shape[0]*labels_for_window.shape[1])
    # Populate the empty lists defined above
    stacked_windows.append(window)
    stacked_labels_for_window.append(labels_for_window)
    stacked_window_times.append(window_times)

# Convert the populated lists into a stacked numpy array
stacked_windows = np.stack(stacked_windows, axis = 0)
stacked_windows = np.squeeze(stacked_windows)

stacked_labels_for_window = np.stack(stacked_labels_for_window, axis = 0)
stacked_labels_for_window = np.squeeze(stacked_labels_for_window)

stacked_window_times = np.stack(stacked_window_times, axis = 0)

# Calculate the row-wise sum
# row_sums = stacked_windows.sum(axis=1)

# Perform row normalization
# stacked_windows  = stacked_windows  / row_sums[:, np.newaxis]


# Z-score
row_means = np.mean(stacked_windows, axis = 1)
row_stds = np.std(stacked_windows, axis = 1)

# stacked_windows = (stacked_windows - row_means[:, np.newaxis]) / row_stds[:,np.newaxis]

if torch.cuda.is_available():  
  device = "cuda:0" 
else:  
  device = "cpu" 
  

# What if I applied this to real data

# stacked_windows = np.load('/Users/AnanyaKapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Analysis/UMAP_Analysis/Num_Spectrograms_10_Window_Size_100_Stride_50/stacked_windows.npy')
# stacked_labels_for_window = np.load('/Users/AnanyaKapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Analysis/UMAP_Analysis/Num_Spectrograms_10_Window_Size_100_Stride_50/labels_for_window.npy')




# %% Setting up a simple 3-layer CNN

# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc = nn.Linear(16*50*20, 1)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x


class TweetyNetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(5,5), stride=(5,5))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(5,5), stride=(5,5))
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(5440, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()
        
        

    def forward(self, x):
        x = x.double().to(device)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x.double())
        x = self.fc2(self.relu(x)) 
        x = self.fc3(self.relu(x))
        x = self.fc4(self.relu(x))
        # x = self.sigmoid(x)
        return x
    
    
# Initialize the CNN model
cnn_model = TweetyNetCNN()
# cnn_model = SimpleCNN()
cnn_model = cnn_model.type(torch.double)

# def convert_model_weights_to_float(model):
#     for param in model.parameters():
#         param.data = param.data.float()

# # Example usage:
# # Assuming you have a CNN model called 'cnn_model'
# convert_model_weights_to_float(cnn_model)



# x = stacked_windows[0,:]
# x.shape = (1, 100, 40)
# x = torch.tensor(x)
# cnn_model.forward(x)

# %% Noise Contrastive Estimation 

# Let's try this out on a toy example with a batch of 32 windows. We will
# randomly select a positive and set of negative samples for each anchor in the
# batch. 


optimizer = optim.SGD(cnn_model.parameters(), lr=0.01)

criterion = nn.BCEWithLogitsLoss()

# Generate 32 random indices
actual_labels = np.max(stacked_labels_for_window, axis = 1)

random_indices = np.random.randint(0, stacked_windows.shape[0], 32)

anchor_labels = actual_labels[random_indices]

unique_syllables = np.unique(actual_labels)

indices_of_positive_samples = [np.random.choice(np.where(actual_labels == int(y))[0]) for y in anchor_labels]

indices_dict = {int(element): np.where(actual_labels == element)[0] for element in unique_syllables}

negative_samples_indices_list = []
for i in np.arange(anchor_labels.shape[0]):
    y = anchor_labels[i]
    exclude_key = int(y)
    
    # Randomly sample a value from each key in the dictionary, except for 'key2'
    random_samples = {key: np.random.choice(values) for key, values in indices_dict.items() if key != exclude_key}
    indices_of_negative_samples = np.array(list(random_samples.values()))
    
    negative_samples_indices_list.append(list(indices_of_negative_samples))


positive_samples_specs = stacked_windows[indices_of_positive_samples,:]

negative_samples_specs = np.empty((0, 4000))

for i in np.arange(len(negative_samples_indices_list)):
    neg_indices = negative_samples_indices_list[i]
    neg_specs = stacked_windows[neg_indices, :]
    negative_samples_specs = np.vstack((negative_samples_specs, neg_specs))

dat = np.vstack((positive_samples_specs, negative_samples_specs))
dat.shape = (dat.shape[0], 1, 100, 40)
dat = torch.tensor(dat).double()

artificial_labels = np.zeros((dat.shape[0], 1))
artificial_labels[0:32] = 1
artificial_labels_torch = torch.tensor(artificial_labels)

# Step 1: Generate a random permutation of indices
num_rows = artificial_labels_torch.size(0)
shuffled_indices = torch.randperm(num_rows)

# Step 2: Shuffle both tensors based on the shuffled indices
dat = dat[shuffled_indices,:,:,:]
artificial_labels_torch = artificial_labels_torch[shuffled_indices]

cnn_model.train()
num_epochs = 500
loss_list = []
for i in np.arange(num_epochs):

    # Now let's push our data through our CNN model 
    a = cnn_model(dat)
    
    # Below line is my implementation, which gives the same result as PyTorch's BCE
    # loss function. 
    # loss = -1 * (torch.sum(torch.log(a[artificial_labels == 1])) + torch.sum(torch.log(1-a[artificial_labels == 0])))
        
    optimizer.zero_grad() 
    output = criterion(cnn_model(dat).view(-1), artificial_labels_torch.view(-1))
    output.backward()
    optimizer.step()
    loss_list.append(output.item())
    # print(output.item())
    print(f'Epoch {i}, Loss Value: {output.item()}')


# Now we will push through our anchors to get a representation for them 
# cnn_model.eval()
# with torch.no_grad():
#     anchor_specs = stacked_windows[random_indices, :]
#     anchor_specs.shape = (32, 1, 100, 40)
    
#     anchor_specs = torch.tensor(anchor_specs).double()
    
#     representation = cnn_model.pool1(cnn_model.relu1(cnn_model.conv1(anchor_specs)))
#     representation = cnn_model.pool2(cnn_model.relu2(cnn_model.conv2(representation)))
#     representation = representation.view(-1, 32 * 25 * 10)
#     representation = cnn_model.relu3(cnn_model.fc1(representation))
#     representation = cnn_model.relu4(cnn_model.fc2(representation))
#     representation = cnn_model.relu5(cnn_model.fc3(representation))
#     representation = cnn_model.relu6(cnn_model.fc4(representation))
#     rep = representation.detach().numpy()
#     reducer = umap.UMAP()
#     embedding = reducer.fit_transform(rep)


# %% Loss Function: Noise Contrastive Estimation 

# We need to define the probability distribution for the negative samples. This
# will be done with respect to each positive sample. In other words, if we have
# 10 syllables then we will have one positive sample and 9 negative samples
# drawn from a weighted probability distribution 


# actual_labels = np.max(stacked_labels_for_window, axis = 1)

# from torch.utils.data import DataLoader, TensorDataset

# # Assuming you have your input data as tensors (X) and corresponding labels (y)
# # If you don't have labels, you can create a dummy tensor with zeros as the labels.

# # Assuming you have a total of 1000 data points and batch size of 32
# total_data_points = actual_labels.shape[0]
# batch_size = 32

# # Create a TensorDataset from your input data and labels
# stacked_windows_reshaped = stacked_windows.copy()
# stacked_windows_reshaped.shape = (stacked_windows_reshaped.shape[0], 1, 100, 40)
# stacked_windows_tensor = torch.tensor(stacked_windows_reshaped)
# actual_labels_tensor = torch.tensor(actual_labels)
# dataset = TensorDataset(stacked_windows_tensor, actual_labels_tensor)

# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(cnn_model.parameters(), lr=0.01)

# total_batch_loss = []
# positive_sample_index_list = []
# negative_samples_indices_list = []
# for batch_X, batch_y in data_loader:
#     batch_loss = 0
#     optimizer.zero_grad() 
#     batch_loss_list = []
#     for i in np.arange(batch_X.size(0)):
#         x = batch_X[i,:,:,:]
#         y = batch_y[i]
#         # Randomly choose a positive sample 
#         indices_of_positive_samples = np.where(actual_labels == int(y))[0]
#         rand_index_positive = np.random.choice(indices_of_positive_samples)
#         positive_sample_index_list.append(rand_index_positive)
        
#         positive_sample = stacked_windows_reshaped[rand_index_positive ,:, :, :] 
#         positive_sample = torch.tensor(positive_sample)
#         positive_sample = torch.unsqueeze(positive_sample, 0)
        
#         # Randomly choose negative samples for every other syllable type
#         # indices_of_negative_samples = np.where(actual_labels != int(y))[0]
        
#         unique_elements, counts = np.unique(actual_labels, return_counts=True)
#         indices_dict = {int(element): np.where(actual_labels == element)[0] for element in unique_elements}
        
#         # Assuming you want to exclude 'key2' when sampling
#         exclude_key = int(y)
        
#         # Randomly sample a value from each key in the dictionary, except for 'key2'
#         random_samples = {key: np.random.choice(values) for key, values in indices_dict.items() if key != exclude_key}
#         indices_of_negative_samples = np.array(list(random_samples.values()))
#         negative_samples_indices_list.append(list(indices_of_negative_samples))
        
#         negative_samples = stacked_windows_reshaped[indices_of_negative_samples, :, : ,: ]
#         negative_samples = torch.tensor(negative_samples)
        
#         # Dataset to pass through CNN: positive and negative samples
#         artificial_labels = torch.zeros(10)
#         artificial_labels[int(y)] = 1
        
#         x = torch.cat((positive_sample, negative_samples), dim=0)
        
#         # We need to define the probability distribution for the negative 
#         # samples. This will be done with respect to each positive sample. In
#         # other words, if we have 10 syllables then we will have one positive
#         # sample and 9 negative samples drawn from a weighted probability
#         # distribution 
        
#         # Filter out the specific element from the array
#         filtered_arr = actual_labels
        
#         # Find unique elements and their counts in the filtered array
#         unique_elements, counts = np.unique(filtered_arr, return_counts=True)
        
#         counts[unique_elements == int(y)] = 1
        
#         negative_sample_prob_dist = counts / np.sum(counts)
#         negative_sample_prob_dist = torch.tensor(negative_sample_prob_dist, requires_grad = False)

#         # Run the positive sample through the network
#         positive_sample_logit = cnn_model.forward(positive_sample)
        
#         log_odds = torch.tensor(torch.log(negative_sample_prob_dist)) 
#         # log_odds[int(y)] = positive_sample_logit
#         ai = positive_sample_logit - log_odds
        
        
        
        
#         # For each negative sample, find the q probability
#         # q_probabilities = negative_sample_prob_dist[np.array(list(random_samples.keys()))]

#         # ai = positive_sample_logit - torch.tensor(torch.log(negative_sample_prob_dist))  
        
#         # pi = cnn_model.sigmoid1(ai)
        
#         # Instantiate the BCEWithLogitsLoss        
#         artificial_labels = artificial_labels.reshape(1, 10)
        
#         # Calculate the loss
#         loss = criterion(ai, artificial_labels)
                
#         batch_loss += loss
#         # batch_loss += loss.item()
#         batch_loss_list.append(loss.item())
        
#     batch_loss.backward()
#     optimizer.step()
#     total_batch_loss.append(batch_loss.item()/batch_X.size(0))

# # The loss stagnates for some reason. Not improving. Must be an issue with the
# # loss function implementation. 

# # Model evaluation
# cnn_model.eval()
# total_loss_testing = 0

# embedding_arr = np.zeros((stacked_windows_tensor.shape[0], 84))

# for i in np.arange(stacked_windows_tensor.shape[0]):
#     positive_sample = stacked_windows_tensor[positive_sample_index_list[i], :, :, :]
#     positive_sample = positive_sample.unsqueeze(1)
    
#     negative_samples = stacked_windows_tensor[negative_samples_indices_list[i]]
    
#     reference_label = actual_labels[i]
    
#     artificial_labels = torch.zeros(10)
#     artificial_labels[int(reference_label)] = 1
    
#     # Filter out the specific element from the array
#     filtered_arr = actual_labels
    
#     # Find unique elements and their counts in the filtered array
#     unique_elements, counts = np.unique(filtered_arr, return_counts=True)
    
#     counts[unique_elements == int(reference_label)] = 1
    
#     negative_sample_prob_dist = counts / np.sum(counts)
#     negative_sample_prob_dist = torch.tensor(negative_sample_prob_dist, requires_grad = False)

#     # Run the positive sample through the network
#     positive_sample_logit = cnn_model.forward(positive_sample)
    
#     log_odds = torch.tensor(torch.log(negative_sample_prob_dist)) 
#     log_odds[int(reference_label)] = positive_sample_logit
#     ai = positive_sample_logit - log_odds
    
#     # Instantiate the BCEWithLogitsLoss        
#     artificial_labels = artificial_labels.reshape(1, 10)
    
#     # Calculate the loss
#     loss = criterion(ai, artificial_labels)
#     total_loss_testing += loss.item()
    
#     embedding_arr[i, :] = cnn_model.fc3.weight.detach().numpy()
    
# total_loss_testing = total_loss_testing/stacked_windows_tensor.shape[0]  











# Now let's pass our         
        
        
        
                
        
        
        
    
    
    






















