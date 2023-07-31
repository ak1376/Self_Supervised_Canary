#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:31:02 2023

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


# # The below function will save an image for each mini-spectrogram. This will be used for understanding the UMAP plot.
# def embeddable_image(data, window_times, iteration_number):
    
#     data.shape = (window_size, int(data.shape[0]/window_size))
#     data = data.T 
#     window_times = window_times.reshape(1, window_times.shape[0])
#     plt.pcolormesh(window_times, masked_frequencies, data, cmap='jet')
#     # let's save the plt colormesh as an image.
#     plt.savefig(folderpath_song+'/Plots/Window_Plots/'+f'Window_{iteration_number}.png')
#     plt.close()
    
    
# for i in np.arange(stacked_windows.shape[0]):
#     if i%10 == 0:
#         print(f'Iteration {i} of {stacked_windows.shape[0]}')
#     data = stacked_windows[i,:]
#     window_times = stacked_window_times[i,:]
#     embeddable_image(data, window_times, i)

# Specify an HTML file to save the Bokeh image to.
output_file(filename=f'{folderpath_song}Plots/umap.html')

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
source.data['image'] = []
for i in np.arange(spec_df.shape[0]):
    source.data['image'].append(f'{folderpath_song}/Plots/Window_Plots/Window_{i}.png')

show(p)

save(p)


# %% Visualization of the UMAP 

arr = np.array([])
arr = np.vstack((stacked_window_times[:,0], stacked_window_times[:,-1]))

np.savez('/Users/ananyakapoor/Desktop/arr_for_plotting.npz', embStartEnd = arr,behavioralArr  = spec, embVals = embedding,neuroArr = spec)


# %% Implementation of the NCC classifier 

actual_labels = np.max(stacked_labels_for_window, axis = 1)

unique_labels = np.unique(actual_labels)

avg_representation = np.zeros((unique_labels.shape[0], 2)) # 2nd dimension is UMAP embedding size



for lab in unique_labels:
    lab = int(lab)
    embedding_rows = np.where(actual_labels == lab)
    embedding_subset = np.squeeze(embedding[embedding_rows, :])
    avg_representation[lab, :] = np.mean(embedding_subset, axis = 0)
    
pred_labels = []
for i in np.arange(embedding.shape[0]):
    dist_metric = np.sum((embedding[i,:] - avg_representation)**2, axis = 1)
    pred_labels.append(np.argmin(dist_metric))
    

acc_value = np.mean(actual_labels == pred_labels)

# This shows that the representations of syllables form centroid like geometry
# in representation space. 

from sklearn.metrics.cluster import v_measure_score


v_measure_score(actual_labels, np.array(pred_labels))

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create an instance of SVC with a linear kernel
svc_classifier = SVC(kernel='linear')

# Train the classifier on the training data
svc_classifier.fit(embedding, actual_labels)

# Make predictions on the test data
y_pred = svc_classifier.predict(embedding)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(actual_labels, y_pred)
print(f"Accuracy: {accuracy:.2f}")



# # Create a meshgrid to plot the decision boundaries
# h = 0.02  # Step size in the meshgrid
# x_min, x_max = embedding[:, 0].min() - 1, embedding[:, 0].max() + 1
# y_min, y_max = embedding[:, 1].min() - 1, embedding[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# # Obtain the predicted class labels for each point in the meshgrid
# Z = svc_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# # Plot the decision boundaries and the data points
# plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
# plt.scatter(embedding[:, 0], embedding[:, 1], c=actual_labels, cmap=plt.cm.Paired, edgecolors='k')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Decision Boundaries from SVC')
# plt.show()












# import hdbscan

# # Actual labels 

# actual_labels = np.max(stacked_labels_for_window, axis = 1)
# unique_syllables, counts = np.unique(actual_labels, return_counts = True)

# clusterer = hdbscan.HDBSCAN(min_cluster_size=234)
# # clusterer = hdbscan.HDBSCAN()
# clusterer.fit(embedding)



# labels_pred = clusterer.labels_

# v_measure_score(actual_labels, labels_pred)

# plt.figure()
# plt.scatter(embedding[:, 0], embedding[:, 1], c=labels_pred, s=40, cmap='tab20');
# plt.show()


# len(np.unique(clusterer.labels_[clusterer.labels_ != -1]))





