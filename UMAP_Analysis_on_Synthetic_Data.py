#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:31:02 2023

Probable bug in this code.



@author: AnanyaKapoor
"""

import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as signal
import sounddevice as sd  
from scipy.io.wavfile import write
import pandas as pd
import seaborn as sns 
import umap
import os 
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import HoverTool, ColumnDataSource

# %% Wow let's try UMAP now

plt.close('all')
plt.ioff()

# Parameters we set
num_spec = 10
window_size = 100
stride = 10
phrase_repeats = 5
num_songs = 10
radius_value = 0.01
num_syllables = 10

folderpath = '/Users/ananyakapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/Canary_SSL_Repo/'
songpath = f'{folderpath}num_songs_{num_songs}_num_syllables_{num_syllables}_phrase_repeats_{phrase_repeats}_radius_{radius_value}/'


files = os.listdir(folderpath)
all_songs_data = [element for element in files if 'num_songs' in element] # Get the file paths of each numpy file from Yarden's data
all_songs_data = os.listdir(f'{folderpath}{all_songs_data[0]}')
all_songs_data = [element for element in all_songs_data if 'Song' in element] # Get the file paths of each numpy file from Yarden's data

all_songs_data.sort()

# For each spectrogram we will extract
# 1. Each timepoint's syllable label
# 2. The spectrogram itself
stacked_labels = [] 
stacked_specs = []
for i in np.arange(num_spec):
    # Extract the data within the numpy file. We will use this to create the spectrogram
    dat = np.load(f'{songpath}{all_songs_data[i]}/synthetic_data.npz')
    spec = dat['s']
    times = dat['t']
    times.shape = (1, times.shape[0])
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
dx = np.diff(times)[0,0]

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

# For each mini-spectrogram, find the average color across all unique syllables
mean_colors_per_minispec = np.zeros((stacked_labels_for_window.shape[0], 3))
for i in np.arange(stacked_labels_for_window.shape[0]):
    list_of_colors_for_row = [category_colors[x] for x in stacked_labels_for_window[i,:]]
    all_colors_in_minispec = np.array(list_of_colors_for_row)
    mean_color = np.mean(all_colors_in_minispec, axis = 0)
    mean_colors_per_minispec[i,:] = mean_color
    
# Perform a UMAP embedding on the dataset of mini-spectrograms
reducer = umap.UMAP()
embedding = reducer.fit_transform(stacked_windows)







# yarden_data = np.load('/Users/AnanyaKapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb3/llb3_data_matrices/Python_Files/llb3_0002_2018_04_23_14_18_03.wav.npz')


# spec_yarden = yarden_data['s']
# times_yarden = yarden_data['t']
# frequencies_yarden = yarden_data['f']
# labels_yarden = yarden_data['labels']
# labels = labels.T





















    
# Let's save all the numpy arrays
np.save(f'{songpath}stacked_windows.npy', stacked_windows)
np.save(f'{songpath}labels_for_window.npy', labels_for_window)
np.save(f'{songpath}masked_frequencies_lowthresh_600_highthresh_4000.npy', masked_frequencies)
np.save(f'{songpath}stacked_window_times.npy', stacked_window_times)
np.save(f'{songpath}mean_colors_per_minispec.npy', mean_colors_per_minispec)

# Perform a UMAP embedding on the dataset of mini-spectrograms
reducer = umap.UMAP()
embedding = reducer.fit_transform(stacked_windows)
np.save(f'{songpath}UMAP_Embedding_of_spec.npy', embedding)

# The below function will save an image for each mini-spectrogram. This will be used for understanding the UMAP plot.

if not os.path.exists(f'{songpath}Plots/Window_Plots/'):
    # Create the directory
    os.makedirs(f'{songpath}Plots/Window_Plots/')
    print("Directory created successfully.")
else:
    print("Directory already exists.")

def embeddable_image(data, window_times, iteration_number):
    
    data.shape = (window_size, int(data.shape[0]/window_size))
    data = data.T 
    window_times = window_times.reshape(1, window_times.shape[0])
    plt.pcolormesh(window_times, masked_frequencies, data, cmap='jet')
    # let's save the plt colormesh as an image.
    plt.savefig(f'{songpath}Plots/Window_Plots/Window_{iteration_number}.png')
    plt.close()

# for i in np.arange(stacked_windows.shape[0]):
#     if i%10 == 0:
#         print(f'Iteration {i} of {stacked_windows.shape[0]}')
#     data = stacked_windows[i,:]
#     window_times = stacked_window_times[i,:]
#     embeddable_image(data, window_times, i)

# Specify an HTML file to save the Bokeh image to.
output_file(filename=f'{songpath}Plots/umap.html')

# Convert the UMAP embedding to a Pandas Dataframe
spec_df = pd.DataFrame(embedding, columns=('x', 'y'))


# Create a ColumnDataSource from the data. This contains the UMAP embedding components and the mean colors per mini-spectrogram
source = ColumnDataSource(data=dict(x = embedding[:,0], y = embedding[:,1], colors=mean_colors_per_minispec))


# Create a figure and add a scatter plot
p = figure(width=800, height=600, tools=('pan, box_zoom, hover, reset'))
p.scatter(x='x', y='y', size = 7, color = 'colors', source=source)

hover = p.select(dict(type=HoverTool))
hover.tooltips = """
    <div>
        <h3>@x, @y</h3>
        <div>
            <img
                src="@image" height="100" alt="@image" width="100"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
        </div>
    </div>
"""

p.add_tools(HoverTool(tooltips="""
"""))


# Set the image path for each data point
# source.data['image'] = []
# for i in np.arange(spec_df.shape[0]):
#     source.data['image'].append(f'{folderpath_song}/Plots/Window_Plots/Window_{i}.png')

show(p)

save(p)


# # %% Visualization of the UMAP (Ethan's code)

# arr = np.array([])
# arr = np.vstack((stacked_window_times[:,0], stacked_window_times[:,-1]))

# np.savez('/Users/ananyakapoor/Desktop/arr_for_plotting.npz', embStartEnd = arr,behavioralArr  = spec, embVals = embedding,neuroArr = spec)


# # %% Implementation of the NCC classifier 

# actual_labels = np.max(stacked_labels_for_window, axis = 1)

# unique_labels = np.unique(actual_labels)

# avg_representation = np.zeros((unique_labels.shape[0], 2)) # 2nd dimension is UMAP embedding size



# for lab in unique_labels:
#     lab = int(lab)
#     embedding_rows = np.where(actual_labels == lab)
#     embedding_subset = np.squeeze(embedding[embedding_rows, :])
#     avg_representation[lab, :] = np.mean(embedding_subset, axis = 0)
    
# pred_labels = []
# for i in np.arange(embedding.shape[0]):
#     dist_metric = np.sum((embedding[i,:] - avg_representation)**2, axis = 1)
#     pred_labels.append(np.argmin(dist_metric))
    

# acc_value = np.mean(actual_labels == pred_labels)

# # This shows that the representations of syllables form centroid like geometry
# # in representation space. 

# from sklearn.metrics.cluster import v_measure_score


# v_measure_score(actual_labels, np.array(pred_labels))

# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score

# # Create an instance of SVC with a linear kernel
# svc_classifier = SVC(kernel='linear')

# # Train the classifier on the training data
# svc_classifier.fit(embedding, actual_labels)

# # Make predictions on the test data
# y_pred = svc_classifier.predict(embedding)

# # Evaluate the accuracy of the classifier
# accuracy = accuracy_score(actual_labels, y_pred)
# print(f"Accuracy: {accuracy:.2f}")


