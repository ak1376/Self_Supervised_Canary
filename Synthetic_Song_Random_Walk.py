#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 09:38:41 2023

This code generates synthetic canary song that we will apply downstream machine
learning to. Canary syllables are generated using 10 acoustic parameters. This
code simulates the acoustic parameters per syllable, with respect to a random
walk process. 

The code is adapted from Gardner et. al 2005's supplementary methods. However,
we make the following modifications:
    1. A similar but modified envelope 
    2. Fixed syllable durations per syllable.
    3. Modifying the parameters slightly from syllable repeat to syllable
       repeat within a phrase 
       
This code will create n_songs number of songs, of which we will be:
    1. Computing a spectrogram of each song 
    2. Extracting the audio of each song 
    3. For the same syllable, varying the duration from song to song.

@author: AnanyaKapoor
"""

import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as signal
import sounddevice as sd  
from scipy.io.wavfile import write

folderpath = '/Users/ananyakapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/Canary_SSL_Repo/'

# We will initialize some parameters and the range of possible values the paraters could take

sampling_freq = 44100
phi_0_arr = np.arange(0,2*np.pi, 0.8) # In radians
delta_phi_arr = np.arange(-3*np.pi/2, 3*np.pi/2, 0.8) # In radians
B_arr = np.arange(0, 3000, 200) # In Hz
c_arr = np.arange(40, 70, 5)
f_0_arr = np.arange(800, 1500, 100) # In Hz
# T_current = 50; T_arr = np.arange(40, 300, 20+0.33*T_current) # Need to figure out what T_current means. I think it means the duration of the current syllable. 
Z_1_arr = np.arange(0.88, 0.93, 0.02)
Z_2_arr = np.arange(0.88, 0.93, 0.02)
theta_1_arr = np.arange(0.01, np.pi/2, 0.2)
theta_2_arr = np.arange(0.01, np.pi/2, 0.2)

# Initializing empty arrays that will hold our signal wave, filtered wave, and enveloped wave

total_signal_wave = np.array([])
total_filtered = np.array([])
total_envelope = np.array([])


# We want to specify a library of syllables that can be rearranged in each song. 

num_syllables = 2
num_short = 1
num_long = 1

short_durations = np.random.uniform(20/1000, 70/1000, num_short)
long_durations = np.random.uniform(100/1000, 200/1000, num_long)

short_repeats = np.random.randint(20, 30, num_short)
long_repeats = np.random.randint(3, 5, num_long)

T_arr = np.concatenate((short_durations, long_durations))
num_repeats_arr = np.concatenate((short_repeats, long_repeats))


permutation = np.random.permutation(len(T_arr))

T_arr_shuffled = T_arr[permutation]
num_repeats_arr_shuffled = num_repeats_arr[permutation]

T_list = T_arr_shuffled
num_repeats_list = num_repeats_arr_shuffled

syllable_number = np.arange(num_syllables)


# It would be very convenient to have a dictionary that stores all this info

syllable_dict = {
    'syllable' : syllable_number, 
    'duration' : T_list,
    'num_repeats' : num_repeats_list
    }



# We should have some repeats of syllable phrases

num_phrase_repeats = 10
T_list = np.repeat(T_list, num_phrase_repeats)
num_repeats_list = np.repeat(num_repeats_list, num_phrase_repeats)
syllable_number = np.repeat(syllable_number, num_phrase_repeats)

permutation = np.random.permutation(len(T_list))

T_list = T_list[permutation]
num_repeats_list = num_repeats_list[permutation]
syllable_number = syllable_number[permutation]

# Let's check to make sure that the shuffling was done correctly

unique_categories, indices = np.unique(syllable_number, return_inverse=True)
averages = [np.mean(num_repeats_list[indices == i]) for i in range(len(unique_categories))]

averages == syllable_dict['num_repeats']

# Shuffling was done correctly. We now have "num_phrase_repeats" of each 
# syllable phrase in a random order.

# We are going to specify some parameters that will be modified through a
# random walk. Our simulation will have the following features:
    #
    # 1. The parameter values will be have slight jitter from syllable phrase
    #    to syllable phrase. There should be 1%-5% jitter to start with. This
    #    requires us storing the parameter values for each phrase in the 
    #    syllable_dict dictionary we created
    #
    # 2. All parameters but the syllable duration will be updated through a 
    #    random walk based on the values from the previous phrase. However, the
    #    duration of the syllable within each syllable phrase should be a 
    #    random walk based on the syllable duration from the previous 
    #    occurrence of the SAME SYLLABLE PHRASE. 

f_0 = 2000
phi_0 = np.pi
delta_phi = np.pi/6
B = 1000
c = 40
Z1 = 0.88
Z2 = 0.92
theta_1 = np.pi/4
theta_2 = np.pi/6

f_0_list = []
B_list = []
phi_0_list =[]
delta_phi_list = []
c_list = []
Z_1_list = []
Z_2_list = []
theta_1_list = []
theta_2_list = []

# Define the number of inner lists (30 in this case)
num_inner_lists = len(syllable_dict['syllable'])

# Define the size of each inner list (you can modify this as per your requirements)
inner_list_size = num_phrase_repeats

f_0_list = [[] * inner_list_size for _ in range(num_inner_lists)]
B_list = [[] * inner_list_size for _ in range(num_inner_lists)]
phi_0_list = [[] * inner_list_size for _ in range(num_inner_lists)]
delta_phi_list = [[] * inner_list_size for _ in range(num_inner_lists)]
c_list = [[] * inner_list_size for _ in range(num_inner_lists)]
Z_1_list = [[] * inner_list_size for _ in range(num_inner_lists)]
Z_2_list = [[] * inner_list_size for _ in range(num_inner_lists)]
theta_1_list = [[] * inner_list_size for _ in range(num_inner_lists)]
theta_2_list = [[] * inner_list_size for _ in range(num_inner_lists)]
T_values_list = [[] * inner_list_size for _ in range(num_inner_lists)]

low_frequency_check = 0
high_frequency_check = 0
initial_indicator = 1 
for syl_index in np.arange(syllable_number.shape[0]):
    syl = syllable_number[syl_index]
    
    if syl_index == 0 and initial_indicator == 1 : # Initial condition (start of song). Not resetting the random walk
        random_walk_indicator = 0
        T = syllable_dict['duration'][syl]
        
    else:
        random_walk_indicator = 1
        
    if random_walk_indicator == 1:
            
        random_num = np.random.uniform(-1, 1)
        # B block
        if low_frequency_check == 1:
            B+=20
        elif high_frequency_check == 1:
            B-= 20   
        
        elif len(B_list[syl])==0:
            B = syllable_dict['duration'][syl]
            
        
        
        
        
        
        elif B + random_num*200 < 0 or B + random_num*200 > 3000:
            B-=random_num*200
        else:
            B+=random_num*200
        # if B + random_num*200 < 0 or B + random_num*200 > 3000:
        #     B-=random_num*200
        # else:
        #     B+=random_num*200
        
        
        # f_0 block
        if low_frequency_check == 1:
            f_0 += 50
        elif high_frequency_check == 1:
            f_0 -= 50
        elif f_0 + random_num*100<800 or f_0 + random_num*100>4000:
            f_0-=random_num*100
        else:
            f_0 += random_num*100

        # Phi_0 block
        if phi_0 + random_num*0.8 < 0 or phi_0 + random_num*0.8 > 2*np.pi:
            phi_0-= random_num*0.8
        else:
            phi_0+=random_num*0.8
        
        # delta_phi block
        if delta_phi + random_num*0.8 < -3*np.pi/2 or delta_phi + random_num*0.8 > 3*np.pi/2:
            delta_phi-= random_num*0.8
        else:
            delta_phi+=random_num*0.8
        
        # # c block
        if c + random_num*5 < 40 or c + random_num*5 > 70:
            c-= random_num*5
        else:
            c+=random_num*5
        
        # Z1 block
        if Z1 + random_num*0.02 < 0.88 or Z1 + random_num*0.02 > 0.93:
            Z1 -= random_num*0.02
        else:
            Z1 += random_num*0.02
        
        # Z2 Block
        if Z2 + random_num*0.02 < 0.88 or Z2 + random_num*0.02 > 0.93:
            Z2 -= random_num*0.02
        else:
            Z2 += random_num*0.02


        # theta_1 block
        if theta_1 + random_num*0.2 < 0.01 or theta_1 + random_num*0.2 > np.pi/2:
            theta_1 -= random_num*0.2
        else:
            theta_1 += random_num*0.2
        
        # theta_2 block
        if theta_2 + random_num*0.2 < 0.01 or theta_2 + random_num*0.2 > np.pi/2:
            theta_2 -= random_num*0.2
        else:
            theta_2 += random_num*0.2
            
        if len(T_values_list[syl]) == 0:
            T = syllable_dict['duration'][syl]
            
        else:
            T = T_values_list[syl][-1] +random_num/100

    num_repeats = syllable_dict['num_repeats'][syl]
    f_0_list[syl].append(f_0)
    B_list[syl].append(B)
    phi_0_list[syl].append(phi_0)
    delta_phi_list[syl].append(delta_phi)
    c_list[syl].append(c)
    Z_1_list[syl].append(Z1)
    Z_2_list[syl].append(Z2)
    theta_1_list[syl].append(theta_1)
    theta_2_list[syl].append(theta_2)
    T_values_list[syl].append(T)  
    

    num_samples = int((T)*sampling_freq)
    t = np.linspace(0, ((T)), num_samples) 

    # Calculate the fundamental frequency across time
    f = f_0 + B*np.cos(phi_0 + delta_phi*t/T)
    
    if np.min(f)<700:
        low_frequency_check = 1
        
    if np.max(f)>3000:
        high_frequency_check = 1
            
    
    # It's the B*np.cos(phi_0_values + delta_phi_values*t/T) that gives the fundamental frequency its wavy shape. f_0 just shifts it up
    
    # =============================================================================
    #     # Now let's calculate the harmonics 
    # =============================================================================
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
        
    # =============================================================================
    #     # Raw signal
    # =============================================================================
        
    s_t_arr = np.zeros_like(t)
    
    for k in np.arange(len(A_list)):
        signal_val = A_list[k]*np.sin(theta_arr[k,:])
        s_t_arr += signal_val
    
    
    total_signal_wave = np.concatenate((total_signal_wave, s_t_arr))
        
    # =============================================================================
    #     # Filtered signal
    # =============================================================================

    r1_roots = Z1 * np.exp(1j*theta_1)
    r2_roots = Z2 * np.exp(1j*theta_2)
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
        
    # =============================================================================
    #     # Enveloped signal 
    # =============================================================================
    # W_t = (0.42 + 0.5*np.cos(np.pi * t/T) + 0.08*np.cos(2*np.pi * t/T))
    W_t = 0.5 * (1 - np.cos(2 * np.pi * t / T))

    phrase_waveform = np.array([])
    for i in np.arange(num_repeats):
        
        waveform_filtered_envelope = y_arr * W_t
        phrase_waveform = np.concatenate((phrase_waveform, waveform_filtered_envelope))
    
    total_envelope = np.concatenate((total_envelope, phrase_waveform))
        

        
# Sample parameters
window_duration_seconds = 0.02  # 40 ms window
window_size = int(sampling_freq * window_duration_seconds)
overlap_fraction = 0.9          # 90 percent overlap           
overlap = int(window_size * overlap_fraction) 

# =============================================================================
# # Raw signal
# =============================================================================
# write('/Users/ananyakapoor/Desktop/raw_signal.wav', sampling_freq, total_signal_wave)


# frequencies, times, spectrogram = signal.spectrogram(total_signal_wave, fs=sampling_freq,
#                                                     window='hamming', nperseg=256,
#                                                     noverlap=128, nfft=512)

# Compute the spectrogram
# frequencies, times, spectrogram = signal.spectrogram(
#     total_signal_wave,
#     fs=sampling_freq,
#     window='hamming',
#     nperseg=int(window_duration_seconds * sampling_freq),
#     noverlap=int(window_duration_seconds * sampling_freq * overlap_fraction)
# )



# # Plot the spectrogram
# plt.figure()
# plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), shading='auto', cmap='inferno')
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# plt.title("Raw")
# plt.show()

# =============================================================================
# # Filtered signal
# =============================================================================

# write('/Users/ananyakapoor/Desktop/filtered_only_signal.wav', sampling_freq, total_filtered)

# frequencies, times, spectrogram = signal.spectrogram(total_filtered, fs=sampling_freq,
#                                                     window='hamming', nperseg=256,
#                                                     noverlap=128, nfft=512)



# # Plot the spectrogram
# plt.figure()
# plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), shading='auto', cmap='inferno')
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# plt.title("Filtered Only")
# plt.show()


# =============================================================================
# # Enveloped signal
# =============================================================================

# Perform running amplitude normalization
normalized_signal = np.zeros_like(total_envelope)

for i in range(0, len(total_envelope) - window_size + 1, window_size - overlap):
    window = total_envelope[i:i + window_size]  # Extract a window of the signal
    scaling_factor = 1.0 / np.max(np.abs(window))  # Calculate the scaling factor
    normalized_signal[i:i + window_size] = window * scaling_factor  # Normalize the window


# Compute the spectrogram
frequencies, times, spectrogram = signal.spectrogram(
    normalized_signal,
    fs=sampling_freq,
    window='bartlett',
    nperseg=int(window_duration_seconds * sampling_freq),
    noverlap=int(window_duration_seconds * sampling_freq * overlap_fraction)
)


# frequencies, times, spectrogram = signal.spectrogram(total_envelope, fs=sampling_freq,
#                                                     window='hamming', nperseg=256,
#                                                     noverlap=128, nfft=512)


# # Plot the spectrogram
plt.figure()
plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), shading='auto', cmap='inferno')
plt.colorbar(label='Power Spectral Density (dB/Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title("Enveloped and Filtered")
plt.show()


# Save the following data
# 1. The audio representation
# 2. a structure containing the times, frequencies, and spectrogram data

dat = {
       's': spectrogram,
       't': times, 
       'f':frequencies
       }

np.savez(f'{folderpath}synthetic_data.npz', **dat)
write(f'{folderpath}enveloped_filtered_signal.wav', sampling_freq, normalized_signal)

# %% Let's plot the parameter regime 

# Paired (for two syllables) bar plot of parameters 

import pandas as pd

f_0_flattened = [item for sublist in f_0_list for item in (sublist if isinstance(sublist, list) else [sublist])]
B_flattened = [item for sublist in B_list for item in (sublist if isinstance(sublist, list) else [sublist])]
phi_0_flattened = [item for sublist in phi_0_list for item in (sublist if isinstance(sublist, list) else [sublist])] 
delta_phi_flattened = [item for sublist in delta_phi_list for item in (sublist if isinstance(sublist, list) else [sublist])]
c_flattened = [item for sublist in c_list for item in (sublist if isinstance(sublist, list) else [sublist])]
Z1_flattened = [item for sublist in Z_1_list for item in (sublist if isinstance(sublist, list) else [sublist])]
Z2_flattened = [item for sublist in Z_2_list for item in (sublist if isinstance(sublist, list) else [sublist])]
theta_1_flattened = [item for sublist in theta_1_list for item in (sublist if isinstance(sublist, list) else [sublist])]
theta_2_flattened = [item for sublist in theta_2_list for item in (sublist if isinstance(sublist, list) else [sublist])]
T_flattened = [item for sublist in T_values_list for item in (sublist if isinstance(sublist, list) else [sublist])]

syllables_flattened = np.array([])
for i in np.arange(np.unique(syllable_number).shape[0]):
    syllables_flattened = np.concatenate((syllables_flattened, i*np.ones(num_phrase_repeats)))
    
df_dict = {
    'Syllable': syllables_flattened.tolist(), 
    'f_0': f_0_flattened, 
    'B' : B_flattened,
    'phi_0': phi_0_flattened, 
    'delta_phi': delta_phi_flattened, 
    'c': c_flattened, 
    'Z1': Z1_flattened,
    'Z2': Z2_flattened, 
    'theta_1': theta_1_flattened, 
    'theta_2': theta_2_flattened,
    'T_flattened': T_flattened
    }

df = pd.DataFrame(df_dict)


grouped_df = df.groupby('Syllable').mean()

import matplotlib.pyplot as plt

# Assuming 'grouped_df' is the DataFrame with the mean values for each group
groups_to_plot = grouped_df.index
categories = grouped_df.columns
num_groups = len(groups_to_plot)
num_categories = len(categories)
width = 0.35

fig, ax = plt.subplots()

# Position of the bars for each category label
x = range(num_categories)

# Plotting the bars for each group
for i, group in enumerate(groups_to_plot):
    values = grouped_df.loc[group].values
    ax.bar([j + (i - 0.5) * width for j in x], values, width, label=group)

# Set the x-axis tick positions and labels
ax.set_xticks(x)
ax.set_xticklabels(categories)

# Set the labels and title
ax.set_xlabel('Categories')
ax.set_ylabel('Mean Values')
ax.set_title('Paired Bar Plot')
ax.legend()

# Show the plot
plt.show()

print(grouped_df.T)

print(grouped_df.loc[0]-grouped_df.loc[1])

# All the parameters are within the step size range. But the s







