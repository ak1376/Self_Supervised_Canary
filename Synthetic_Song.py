#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 10:54:56 2023

This code will simulate a canary song but with no random walk and simulating 
each syllable separately. Each unique syllable will be defined by a 
10-dimensional multivariate Gaussian. Therefore, we will have a 10d parameter
value for each syllable occurrence, NOT each syllable-phrase occurrence


@author: AnanyaKapoor
"""

import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as signal
import sounddevice as sd  
from scipy.io.wavfile import write

folderpath = '/Users/ananyakapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/Canary_SSL_Repo/'

sampling_freq = 44100

# =============================================================================
# # Toy implementation 
# =============================================================================

# # Let's define a multivariate Gaussian distribution for one syllable 

# mean_phi_0 = float(np.random.uniform(0, 2*np.pi, 1))
# mean_delta_phi = float(np.random.uniform(-3*np.pi/2, 3*np.pi/2, 1)) # In radians
# mean_B = float(np.random.uniform(0, 750, 1)) # In Hz
# mean_c = float(np.random.uniform(40, 70, 1))
# mean_f_0 = float(np.random.uniform(800, 1500, 1)) # In Hz
# mean_T = float(np.random.uniform(.20, .100, 1))
# mean_Z_1 = float(np.random.uniform(0.88, 0.93, 1))
# mean_Z_2 = float(np.random.uniform(0.88, 0.93, 1))
# mean_theta_1 = float(np.random.uniform(0.01, np.pi/2, 1))
# mean_theta_2 = float(np.random.uniform(0.01, np.pi/2, 1))

# mean_vector = np.array([mean_phi_0, mean_delta_phi, mean_B, mean_c, mean_f_0, mean_T, mean_Z_1, mean_Z_2, mean_theta_1, mean_theta_2])
# covariance_matrix = 0.000000005*np.eye(10) # If we increase the variance values then we need to find a way to ensure that the Z_1, Z_2, theta_1, theta_2 are within the ranges. This is important

# phi_0_vector = []
# delta_phi_vector = []
# B_vector = []
# c_vector = []
# f_0_vector = []
# T_vector = []
# Z_1_vector = []
# Z_2_vector = []
# theta_1_vector = []
# theta_2_vector = []

# # Initializing empty arrays that will hold our signal wave, filtered wave, and enveloped wave

# total_signal_wave = np.array([])
# total_filtered = np.array([])
# total_envelope = np.array([])

# num_repeats = 50

# for i in np.arange(num_repeats):

#     # Each syllable repeat will have different values for each parameter
#     acoustic_params = np.random.multivariate_normal(mean_vector, covariance_matrix)
    
#     phi_0 = acoustic_params[0]
#     phi_0_vector.append(phi_0)
    
#     delta_phi = acoustic_params[1]
#     delta_phi_vector.append(delta_phi)
    
#     B = acoustic_params[2]
#     B_vector.append(B)
    
#     c = acoustic_params[3]
#     c_vector.append(c)
    
#     f_0 = acoustic_params[4]
#     f_0_vector.append(f_0)
    
#     T = acoustic_params[5]
#     T_vector.append(T)
#     print(T)
    
#     Z_1 = acoustic_params[6]
#     Z_1_vector.append(Z_1)
    
#     Z_2 = acoustic_params[7]
#     Z_2_vector.append(Z_2)
    
#     theta_1 = acoustic_params[8]
#     theta_1_vector.append(theta_1)
    
#     theta_2 = acoustic_params[9]
#     theta_2_vector.append(theta_2)
    
#     num_samples = int((T)*sampling_freq)
#     t = np.linspace(0, ((T)), num_samples) 

#     # Calculate the fundamental frequency across time
#     f = f_0 + B*np.cos(phi_0 + delta_phi*t/T)
    
#     if np.min(f)<700:
#         low_frequency_check = 1
        
#     if np.max(f)>3000:
#         high_frequency_check = 1
            
    
#     # It's the B*np.cos(phi_0_values + delta_phi_values*t/T) that gives the fundamental frequency its wavy shape. f_0 just shifts it up
    
#     #     # Now let's calculate the harmonics 
#     num_harmonics = 12
#     theta_arr = np.zeros((num_harmonics, t.shape[0]))
#     for k in np.arange(num_harmonics):
#         # val = 2*np.pi*(k+1)*f.reshape(f.shape[0],)
#         val = 2*np.pi*(k+1)*f_0*t + (2*np.pi*(k+1)*B*T/(delta_phi))*(np.sin((phi_0)+(delta_phi)/T*t) - np.sin((phi_0)))
#         theta_arr[k, :] = val
        
#     ## coefficients
    
#     A_list = [1]
#     for k in np.arange(2, (num_harmonics + 1)):
#         coef = 1/(1+c*2**(k-1))
#         # coef = 1
#         A_list.append(coef)
        
#     #     # Raw signal
        
#     s_t_arr = np.zeros_like(t)
    
#     for k in np.arange(len(A_list)):
#         signal_val = A_list[k]*np.sin(theta_arr[k,:])
#         s_t_arr += signal_val
    
    
#     total_signal_wave = np.concatenate((total_signal_wave, s_t_arr))
        
#     #     # Filtered signal

#     r1_roots = Z_1 * np.exp(1j*theta_1)
#     r2_roots = Z_2 * np.exp(1j*theta_2)
#     roots = [r1_roots, np.conjugate(r1_roots), r2_roots, np.conjugate(r2_roots)]
    
#     # Convert the roots to zeros, poles, and gain representation
#     zeros = []
#     poles = roots
#     gain = 1.0

#     # Convert zeros, poles, and gain to filter coefficients
#     b, a = signal.zpk2tf(zeros, poles, gain)

#     # Apply the all-pole filter to the input signal
#     y_arr = signal.lfilter(b, a, s_t_arr)

#     total_filtered = np.concatenate((total_filtered, y_arr))
        
#     #     # Enveloped signal 
    
#     # W_t = (0.42 + 0.5*np.cos(np.pi * t/T) + 0.08*np.cos(2*np.pi * t/T))
#     W_t = 0.5 * (1 - np.cos(2 * np.pi * t / T))
        
#     waveform_filtered_envelope = y_arr * W_t
    
#     total_envelope = np.concatenate((total_envelope, waveform_filtered_envelope))


# =============================================================================
# # Actual Implementation
# =============================================================================

num_syllables = 10
num_short = 7
num_long = 3

mean_phi_0 = (np.random.uniform(0, 2*np.pi, num_syllables)).reshape(1, num_syllables)
mean_delta_phi = (np.random.uniform(-3*np.pi/2, 3*np.pi/2, num_syllables)).reshape(1, num_syllables) # In radians
mean_B = (np.random.uniform(0, 750, num_syllables)).reshape(1, num_syllables) # In Hz
mean_c = (np.random.uniform(40, 70, num_syllables)).reshape(1, num_syllables)
mean_f_0 = (np.random.uniform(800, 1500, num_syllables)).reshape(1, num_syllables) # In Hz

short_durations = np.random.uniform(20/1000, 70/1000, num_short)
long_durations = np.random.uniform(100/1000, 200/1000, num_long)
short_repeats = np.random.randint(30, 50, num_short)
long_repeats = np.random.randint(3, 5, num_long)

mean_T = np.concatenate((short_durations, long_durations))
num_repeats = np.concatenate((short_repeats, long_repeats))

permutation = np.random.permutation(len(mean_T))

mean_T = mean_T[permutation]
mean_T.shape = (1, num_syllables)
num_repeats = num_repeats[permutation]

mean_Z_1 = (np.random.uniform(0.88, 0.93, num_syllables)).reshape(1, num_syllables)
mean_Z_2 = (np.random.uniform(0.88, 0.93, num_syllables)).reshape(1, num_syllables)
mean_theta_1 = (np.random.uniform(0.01, np.pi/2, num_syllables)).reshape(1, num_syllables)
mean_theta_2 = (np.random.uniform(0.01, np.pi/2, num_syllables)).reshape(1, num_syllables)

# num_repeats = 50*np.ones((1, num_syllables)) # Simple example

mean_matrix = np.concatenate((mean_phi_0, mean_delta_phi, mean_B, mean_c, mean_f_0, mean_T, mean_Z_1, mean_Z_2, mean_theta_1, mean_theta_2), axis = 0)

covariance_matrix = 0.00005*np.eye(mean_matrix.shape[0]) 

# Let's find a random order of syllable phrases to simulate 
unique_syllables = np.arange(num_syllables)
syllable_phrase_order = unique_syllables.copy()
np.random.shuffle(syllable_phrase_order) # ex: 0, 2, 1 means that we will simulate syllable 0 first, followed by 2 and then followed by 1

phrase_repeats = 3
syllable_phrase_order = np.repeat(syllable_phrase_order, phrase_repeats)
np.random.shuffle(syllable_phrase_order) # The bird will sing a random arrangement of syllable phrases (random transition statistics)

phi_0_vector = []
delta_phi_vector = []
B_vector = []
c_vector = []
f_0_vector = []
T_vector = []
Z_1_vector = []
Z_2_vector = []
theta_1_vector = []
theta_2_vector = []

# Initializing empty arrays that will hold our signal wave, filtered wave, and enveloped wave

total_signal_wave = np.array([])
total_filtered = np.array([])
total_envelope = np.array([])


# Sample parameters
window_duration_seconds = 0.02  # 40 ms window
window_size = int(sampling_freq * window_duration_seconds)
overlap_fraction = 0.9       # 90 percent overlap           
overlap = int(window_size * overlap_fraction) 

labels_per_sample = np.array([])


# Double for loop: one over the syllable phrase and the other over the number of repeats of syllable
phrase_duration = []
for syl in syllable_phrase_order:
    for i in np.arange(int(num_repeats[syl])):
        
        # Draw acoustic parameters with respect to the mean vector corresponding to the syllable we are simulating
        acoustic_params = np.random.multivariate_normal(mean_matrix[:,syl], covariance_matrix)
        
        phi_0 = acoustic_params[0]
        phi_0_vector.append(phi_0)
        
        delta_phi = acoustic_params[1]
        delta_phi_vector.append(delta_phi)
        
        B = acoustic_params[2]
        B_vector.append(B)
        
        c = acoustic_params[3]
        c_vector.append(c)
        
        f_0 = acoustic_params[4]
        f_0_vector.append(f_0)
        
        T = acoustic_params[5]
        T_vector.append(T)
        # print(T)
        
        Z_1 = acoustic_params[6]
        Z_1_vector.append(Z_1)
        
        Z_2 = acoustic_params[7]
        Z_2_vector.append(Z_2)
        
        theta_1 = acoustic_params[8]
        theta_1_vector.append(theta_1)
        
        theta_2 = acoustic_params[9]
        theta_2_vector.append(theta_2)
        
        num_samples = int((T)*sampling_freq)
        t = np.linspace(0, ((T)), num_samples) 

        # Calculate the fundamental frequency across time
        f = f_0 + B*np.cos(phi_0 + delta_phi*t/T)
        syllable_labels = np.repeat(syl, t.shape[0])
        labels_per_sample = np.concatenate((labels_per_sample, syllable_labels))
        
        if np.min(f)<700:
            low_frequency_check = 1
            
        if np.max(f)>3000:
            high_frequency_check = 1
                
        
        # It's the B*np.cos(phi_0_values + delta_phi_values*t/T) that gives the fundamental frequency its wavy shape. f_0 just shifts it up
        
        #     # Now let's calculate the harmonics 
        num_harmonics = 12
        theta_arr = np.zeros((num_harmonics, t.shape[0]))
        for k in np.arange(num_harmonics):
            # val = 2*np.pi*(k+1)*f.reshape(f.shape[0],)
            val = 2*np.pi*(k+1)*f_0*t + (2*np.pi*(k+1)*B*T/(delta_phi))*(np.sin((phi_0)+(delta_phi)/T*t) - np.sin((phi_0)))
            theta_arr[k, :] = val
            
        ## coefficients
        
        A_list = [1]
        for k in np.arange(2, (num_harmonics + 1)):
            coef = 1/(1+c*2**(k-1))
            # coef = 1
            A_list.append(coef)
            
        #     # Raw signal
            
        s_t_arr = np.zeros_like(t)
        
        for k in np.arange(len(A_list)):
            signal_val = A_list[k]*np.sin(theta_arr[k,:])
            s_t_arr += signal_val
        
        
        total_signal_wave = np.concatenate((total_signal_wave, s_t_arr))
            
        #     # Filtered signal

        r1_roots = Z_1 * np.exp(1j*theta_1)
        r2_roots = Z_2 * np.exp(1j*theta_2)
        roots = [r1_roots, np.conjugate(r1_roots), r2_roots, np.conjugate(r2_roots)]
        
        # Convert the roots to zeros, poles, and gain representation
        zeros = []
        poles = roots
        gain = 1.0

        # Convert zeros, poles, and gain to filter coefficients
        b, a = signal.zpk2tf(zeros, poles, gain)

        # Apply the all-pole filter to the input signal
        y_arr = signal.lfilter(b, a, s_t_arr)

        total_filtered = np.concatenate((total_filtered, y_arr))
            
        #     # Enveloped signal 
        
        # W_t = (0.42 + 0.5*np.cos(np.pi * t/T) + 0.08*np.cos(2*np.pi * t/T))
        W_t = 0.5 * (1 - np.cos(2 * np.pi * t / T))
            
        waveform_filtered_envelope = y_arr * W_t
        
        total_envelope = np.concatenate((total_envelope, waveform_filtered_envelope))
        
    phrase_duration.append(waveform_filtered_envelope.shape[0]/44100)

normalized_signal = np.zeros_like(total_envelope)

for i in range(0, len(total_envelope) - window_size + 1, window_size - overlap):
    window = total_envelope[i:i + window_size]  # Extract a window of the signal
    scaling_factor = 1.0 / np.max(np.abs(window))  # Calculate the scaling factor
    normalized_signal[i:i + window_size] = window * scaling_factor  # Normalize the window

# Compute the spectrogram
frequencies, times, spectrogram = signal.spectrogram(
    normalized_signal,
    fs=sampling_freq,
    window='hamming',
    nperseg=int(window_duration_seconds * sampling_freq),
    noverlap=int(window_duration_seconds * sampling_freq * overlap_fraction)
)

plt.figure()
plt.pcolormesh(times, frequencies, spectrogram, cmap='jet')
plt.show()

# Calculate the number of samples and pixels
num_samples = len(normalized_signal)
num_pixels = spectrogram.shape[1]

# Create an array to store labels per pixel
labels_per_pixel = np.zeros(num_pixels)

# Calculate the mapping between samples and pixels
samples_per_pixel = (window_size - overlap)
mapping = np.arange(0, num_samples - window_size + 1, samples_per_pixel)

# Map each label to the corresponding time pixel in the spectrogram using majority voting
for i in range(num_pixels):
    start_sample = mapping[i]
    end_sample = start_sample + samples_per_pixel
    labels_in_window = labels_per_sample[start_sample:end_sample]
    labels_per_pixel[i] = np.bincount(labels_in_window.astype('int')).argmax()

dat = {
        's': spectrogram,
        't': times, 
        'f':frequencies, 
        'labels':labels_per_pixel
        }

# np.savez(f'{folderpath}synthetic_data.npz', **dat)
write(f'{folderpath}enveloped_filtered_signal.wav', sampling_freq, normalized_signal)


syllables = np.array([])
for syl in syllable_phrase_order:
    repeats = num_repeats[syl]
    repeated_syllable = np.repeat(syl, repeats)
    syllables = np.concatenate((syllables, repeated_syllable))
    


syllables = syllables.astype('int')


df_dict = {
    'Syllable': syllables.tolist(), 
    'f_0': f_0_vector, 
    'B' : B_vector,
    'phi_0': phi_0_vector, 
    'delta_phi': delta_phi_vector, 
    'c': c_vector, 
    'Z1': Z_1_vector,
    'Z2': Z_2_vector, 
    'theta_1': theta_1_vector, 
    'theta_2': theta_2_vector,
    'T_flattened': T_vector
    }

    
import pandas as pd
import seaborn as sns 
import umap
df = pd.DataFrame(df_dict)

# plt.figure(figsize=(35, 35))
# sns.pairplot(df, hue = 'Syllable')
# # Adjust the layout to prevent clipping
# plt.tight_layout()
# plt.show()

grouped_df = df.groupby('Syllable').mean()
print(grouped_df.T)


reducer = umap.UMAP()
X  = df.iloc[:, 1:]
X = X.values
y = df.Syllable
y = y.values
embedding = reducer.fit_transform(X)

plt.figure()
# plt.scatter(embedding[:,0], embedding[:,1], c=y, cmap='viridis', s=50)

categories = y 

# Create separate scatter plots for each category
for category in np.unique(categories):
    mask = categories == category
    plt.scatter(embedding[mask,0], embedding[mask,1], label=category, s=50)

# Set plot labels and title
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP Embedding of the Parameter Regimes for Each Syllable')

# Show the legend
plt.legend()

plt.show()

# %% Now I want to plot the phrase durations across all phrases in our song

# plt.figure()
# plt.hist(phrase_duration)
# plt.show()

phrase_duration_arr = np.array(phrase_duration)

from scipy.stats import gaussian_kde

# Create a histogram of the data
hist, bins = np.histogram(phrase_duration_arr, bins=10, density=True)

# Calculate the density curve using KDE
density_curve = gaussian_kde(phrase_duration_arr)

# Generate x values for the density curve
x = np.linspace(phrase_duration_arr.min(), phrase_duration_arr.max(), 100)

# Calculate the y values (density) for the density curve
y = density_curve(x)

plt.figure()

# Plot the histogram
plt.hist(phrase_duration_arr, bins=10, density=True, alpha=0.5, label='Histogram')

# Plot the density curve
plt.plot(x, y, color='red', label='Density Curve')

# Set plot labels and title
plt.xlabel('Data')
plt.ylabel('Density')
plt.title('Histogram with Density Curve')

# Show the legend
plt.legend()

plt.show()










