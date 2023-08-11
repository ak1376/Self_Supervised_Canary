
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:32:59 2023

@author: AnanyaKapoor
"""

# Load libraries

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# import umap
import warnings
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import HoverTool, ColumnDataSource
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from PIL import Image
# from torchvision import transforms
import random
import torch.nn.functional as F

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
# warnings.filterwarnings("ignore")

# Set parameters
bird_dir = '/Users/AnanyaKapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb3/'
audio_files = bird_dir+'llb3_songs'
directory = bird_dir+ 'llb3_data_matrices/Python_Files'
analysis_path = '/Users/AnanyaKapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/Canary_SSL_Repo/'

# Parameters we set
num_spec = 10
window_size = 100
stride = 10

# Define the folder name
folder_name = f'{analysis_path}Num_Spectrograms_{num_spec}_Window_Size_{window_size}_Stride_{stride}'

# Create the folder if it doesn't already exist
if not os.path.exists(folder_name+"/Plots/Window_Plots"):
    os.makedirs(folder_name+"/Plots/Window_Plots")
    print(f'Folder "{folder_name}" created successfully.')
else:
    print(f'Folder "{folder_name}" already exists.')

# =============================================================================
# # If you are loading the results from a previous analysis, run the following lines of code
# =============================================================================

stacked_windows = np.load(folder_name+'/stacked_windows.npy') # An array of all the mini-spectrograms
stacked_labels_for_window = np.load(folder_name+'/stacked_labels_for_window.npy') # The syllable labels for each time point in each mini-spectrogram
embedding = np.load(folder_name+'/UMAP_Embedding.npy') # The pre-computed UMAP embedding (2 dimensional)
masked_frequencies = np.load(analysis_path+'/masked_frequencies_lowthresh_500_highthresh_7000.npy') # The frequencies we want to use for analysis. Excluding unnecessarily low and high frequencies
stacked_window_times = np.load(folder_name+'/stacked_window_times.npy') # The onsets and ending of each mini-spectrogram
    
# open the file for reading in binary mode
with open(folder_name+'/category_colors.pkl', 'rb') as f:
    # load the dictionary from the file using pickle.load()
    category_colors = pickle.load(f)   
    
# Each syllable is given a unique color. Each mini-spectrogram will have an average syllable color associated with it. This is the average RGB value across all unique syllables in the mini-spectrogram
mean_colors_per_minispec = np.load(folder_name+'/mean_colors_per_minispec.npy')

# %% TweetyNet Architecture 

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
        self.fc1 = nn.Linear(64*1*147, 4000)
        self.fc2 = nn.Linear(4000, 1000)
        # self.fc3 = nn.Linear(100, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # Encoder f
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

# %% Data processing 

class Batcher():
    def __init__(self, num_positive_samples, num_negative_samples, stacked_windows, stacked_window_labels, anchor_label, anchor_index, pixel_length_time, pixel_length_frequency):
        self.num_positive = num_positive_samples
        self.num_negative = num_negative_samples
        self.batch_size = self.num_positive + self.num_negative
        self.full_dataset = stacked_windows
        self.actual_labels = np.max(stacked_window_labels, axis=1)
        self.anchor_label = anchor_label
        self.anchor_index = anchor_index
        self.pixel_length_time = pixel_length_time
        self.pixel_length_frequency = pixel_length_frequency
    
    def sampler(self):
        # We will want to sample positive and negative samples from our dataset
        # In the future we will be using the UMAP representations to do this.
        # For now let's just use the ground truth labels to do supervised
        # contrastive learning. 
        
        unique_syllables = np.unique(self.actual_labels)
        indices_dict = {int(element): np.where(self.actual_labels == element)[0] for element in unique_syllables}
        num_negative_samples_each = self.num_negative
        
        # Sample a positive sample from our total dataset
        indices_of_positive_samples = np.where(self.actual_labels == self.anchor_label)[0]
        positive_sample_index = np.random.choice(indices_of_positive_samples)
        anchor_sample_index = np.array(self.anchor_index).reshape(1,)
        positive_sample_index = np.array(positive_sample_index).reshape(1,)
        positive_sample_index = np.concatenate((positive_sample_index, anchor_sample_index))
        
        # Sample the negative indices from our total dataset
        sampled_value_list = []
        exclude_key = self.anchor_label
        
        random_samples = {key: np.random.choice(values, num_negative_samples_each) for key, values in indices_dict.items() if key != exclude_key}

        a = np.array(list(random_samples.values()))

        negative_samples_indices = a.reshape(a.shape[0]*a.shape[1],)
        
        # for key, values in indices_dict.items():
        #     if key != exclude_key:
        #         sampled_value = random.choice(values)
                
        #     sampled_value_list.append(sampled_value)
        
        # negative_samples_indices = np.array(sampled_value_list)
        
        indices_to_grab = np.concatenate((positive_sample_index, negative_samples_indices))
          
        # Now let's extract the spectrogram slices for the positive and 
        # negative samples 
        
        batch_data = self.full_dataset[indices_to_grab, :]
        
        # Process the batch_data to be a Pytorch tensor with the shape
        # (batch_data.shape[0], 1, batch_data.shape[1], batch_data.shape[2])
        
        batch_data_tensor = torch.tensor(batch_data)
        batch_data_tensor = batch_data_tensor.reshape(batch_data.shape[0], 1, self.pixel_length_time, self.pixel_length_frequency)
        
        # We will also return an indicator vector telling us which samples were
        # positive and which were negative
        
        artificial_labels = np.zeros((1, batch_data.shape[0]))
        artificial_labels[:,:self.num_positive+1] = 1
        
        return batch_data_tensor, artificial_labels, positive_sample_index, negative_samples_indices


# anchor_label = 3
# batch_obj = Batcher(2, 64, stacked_windows,stacked_labels_for_window, anchor_label, 100, 151)
# batch_data, artificial_labels_batch, positive_sample_index_batch, negative_samples_indices_batch = batch_obj.sampler()


# %% InfoNCE (from SimCLR)

cnn_model = TweetyNetCNN()
cnn_model = cnn_model.double()

optimizer = optim.AdamW(cnn_model.parameters(), 
                        lr=5e-4, 
                        weight_decay=1e-4)

lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                    T_max=500,
                                                    eta_min=5e-4/50)
nll_list = []

num_epoch = 100

# Shuffle our stacked windows and the labels respectively

# Let's shuffle the stacked_windows and stacked_labels_for_window

# Shuffle the indices of stacked_windows
shuffled_indices = np.random.permutation(stacked_windows.shape[0])

# Shuffle array1 using the shuffled indices
stacked_windows = stacked_windows[shuffled_indices,:]

# Shuffle array2 using the same shuffled indices
stacked_labels_for_window = stacked_labels_for_window[shuffled_indices,:]

mean_colors_per_minispec = mean_colors_per_minispec[shuffled_indices, :]


actual_labels = np.max(stacked_labels_for_window, axis=1)

# I want to remove the spectrogram slices with just silence (for now)

silence_labels = np.where(actual_labels == 0)

stacked_windows_no_silence = np.delete(stacked_windows, silence_labels, axis=0)

actual_labels_no_silence = np.delete(actual_labels, silence_labels, axis = 0)

stacked_labels_for_window_no_silence = np.delete(stacked_labels_for_window, silence_labels, axis = 0)

# Let's z-score each spectrogram slice

# row_means = np.mean(stacked_windows_no_silence, axis=1)
# row_stds = np.std(stacked_windows_no_silence, axis=1)

# stacked_windows_no_silence = (stacked_windows_no_silence -
#                    row_means[:, np.newaxis]) / row_stds[:, np.newaxis]

# Higher order batches

from torch.utils.data import DataLoader, TensorDataset

stacked_windows_tensor = torch.tensor(stacked_windows_no_silence)

stacked_windows_tensor = stacked_windows_tensor.reshape(stacked_windows_tensor.shape[0], 1, 100, 151)
dataset = TensorDataset(stacked_windows_tensor)

# Define batch size
batch_size = 64

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

num_epochs = 500

epoch_loss_list = []

for epoch in np.arange(num_epochs):
    batch_loss_list = []
    
    for batch_data in dataloader:
        batch_loss = 0
        X = batch_data[0]
        for i in np.arange(X.shape[0]):
            print(i)
            anchor_label = actual_labels_no_silence[i]
            batch_obj = Batcher(1, 2, stacked_windows_no_silence,stacked_labels_for_window_no_silence, anchor_label, i, 100, 151)
            batch_data, artificial_labels_batch, positive_sample_index_batch, negative_samples_indices_batch = batch_obj.sampler()
    
            embedding = cnn_model(batch_data)
            
            # Pairwise cosine similarity for the entire batch. We are only interested in 
            cos_sim = F.cosine_similarity(embedding.unsqueeze(1), embedding.unsqueeze(0), dim=2)
            
            
            self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
            cos_sim = cos_sim.masked_fill_(self_mask, -9e15)
            temperature=0.07
            
            cos_sim = cos_sim / temperature
            nll = -cos_sim[0,1]+ torch.logsumexp(cos_sim, dim=-1)
            nll = nll.mean()
            batch_loss+=nll
            
        
        batch_loss_list.append(batch_loss.item())
        # batch_loss = np.mean(batch_loss_list)
        print(batch_loss.item())
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        lr_scheduler.step()
    epoch_loss_list.append(np.mean(batch_loss_list))

        
        
        








# for epoch in np.arange(num_epoch):
#     batch_loss = []
#     for i in np.arange(10):
#         anchor_label = actual_labels[i]
#         batch_obj = Batcher(1, 2, stacked_windows,stacked_labels_for_window, anchor_label, i, 100, 151)
#         batch_data, artificial_labels_batch, positive_sample_index_batch, negative_samples_indices_batch = batch_obj.sampler()

#         embedding = cnn_model(batch_data)
        
#         # Pairwise cosine similarity for the entire batch. We are only interested in 
#         cos_sim = F.cosine_similarity(embedding.unsqueeze(1), embedding.unsqueeze(0), dim=2)
        
        
#         self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
#         cos_sim = cos_sim.masked_fill_(self_mask, -9e15)
#         temperature=0.07
        
#         cos_sim_w_temp = cos_sim / temperature
#         nll = -cos_sim_w_temp[0,1]+ torch.logsumexp(cos_sim_w_temp, dim=-1)
#         nll = nll.mean()
#         print(nll.item())
#         batch_loss.append(nll.item())
        
#         optimizer.zero_grad()
#         nll.backward()
#         optimizer.step()
#         lr_scheduler.step()
    
    
#     nll_list.append(np.mean(batch_loss))
#     print(f'Epoch {epoch}, Mean Batch Loss: {np.mean(batch_loss):.4f}')
    
    

    
    
# a = cnn_model(batch_data).detach().numpy()

# indices_of_specs = np.concatenate((positive_sample_index_batch, negative_samples_indices_batch))

# batch_colors = mean_colors_per_minispec[indices_of_specs,:]

# np.save('/Users/AnanyaKapoor/Desktop/embedding_infonce.npy', a)
# np.save('/Users/AnanyaKapoor/Desktop/batch_colors_infonce.npy', batch_colors)






# import umap

# reducer = umap.UMAP()
# embedding_umap = reducer.fit_transform(a)

























# class SimCLR(nn.Module):
    
#     def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs = 500):
#         super().__init__()
#         self.cnn_encoder = TweetyNetCNN()
#         # The MLP for g(.) consists of Linear->ReLU->Linear 
#         self.cnn_encoder.fc = nn.Sequential(
#             self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
#             nn.ReLU(inplace=True),
#             nn.Linear(4*hidden_dim, hidden_dim)
#         )
        
#     def InfoNCE_Loss(self, batch):
#         imgs, _ = batch
#         imgs = torch.cat(imgs, dim=0)
        
#         # Encode all images
#         feats = self.cnn_encoder(imgs)
#         # Calculate cosine similarity
#         cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
#         # Mask out cosine similarity to itself
#         self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
#         cos_sim.masked_fill_(self_mask, -9e15)
        
#         # InfoNCE loss
#         cos_sim = cos_sim / self.hparams.temperature
#         nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)


   









     
        
        
        







