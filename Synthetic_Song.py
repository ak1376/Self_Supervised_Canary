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

num_syllables = 10
num_short = 5
num_long = 5

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


# Let's randomly choose a parameter set for the first occurrence of each unique
# syllable phrase. 

# f_0_values = np.array([2000, 950, 800])
f_0_values = np.linspace(800, 3000, num_syllables)
np.random.shuffle(f_0_values)
B_values = np.linspace(0, 800, num_syllables)
np.random.shuffle(B_values)
# B_values = np.array([1000, 300, 400])
phi_0_values = np.random.choice(phi_0_arr, (num_syllables))
delta_phi_values = np.random.choice(delta_phi_arr, (num_syllables))
c_values = np.random.choice(c_arr, (num_syllables))
Z1_values = np.random.choice(Z_1_arr, (num_syllables))
Z2_values = np.random.choice(Z_2_arr, (num_syllables))
theta_1_values = np.random.choice(theta_1_arr, (num_syllables))
theta_2_values = np.random.choice(theta_2_arr, (num_syllables))

# Let's add these initial parameters to the dictionary we created earlier


syllable_dict['initial_f_0'] = f_0_values
syllable_dict['initial_B'] = B_values
syllable_dict['initial_phi_0'] = phi_0_values
syllable_dict['initial_delta_phi'] = delta_phi_values
syllable_dict['initial_c'] = c_values
syllable_dict['initial_Z_1'] = Z1_values
syllable_dict['initial_Z_2'] = Z2_values
syllable_dict['initial_theta_1'] = theta_1_values
syllable_dict['initial_theta_2'] = theta_2_values

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

labels_per_sample = np.array([])
labels_per_pixel = np.array([])
low_frequency_check = 0
high_frequency_check = 0
initial_indicator = 1 

# Sample parameters
window_duration_seconds = 0.02  # 40 ms window
window_size = int(sampling_freq * window_duration_seconds)
overlap_fraction = 0.9       # 90 percent overlap           
overlap = int(window_size * overlap_fraction) 

for syl_index in np.arange(syllable_number.shape[0]):
    syl = syllable_number[syl_index]
    
    if len(f_0_list[syl]) == 0: 
        initial_indicator = 1
    else:
        initial_indicator = 0
    
    if initial_indicator == 1 : # Initial condition (start of song). Not resetting the random walk
        random_walk_indicator = 0
        T = syllable_dict['duration'][syl]
        B = syllable_dict['initial_B'][syl]
        f_0 = syllable_dict['initial_f_0'][syl]
        phi_0 = syllable_dict['initial_phi_0'][syl]
        delta_phi = syllable_dict['initial_delta_phi'][syl]
        c = syllable_dict['initial_c'][syl]
        Z1 = syllable_dict['initial_Z_1'][syl]
        Z2 = syllable_dict['initial_Z_2'][syl]
        theta_1 = syllable_dict['initial_theta_1'][syl]
        theta_2 = syllable_dict['initial_theta_2'][syl]
        
    else:
        random_walk_indicator = 1
        
    if random_walk_indicator == 1:
        B = B_list[syl][-1]
        f_0 = f_0_list[syl][-1]
        phi_0 = phi_0_list[syl][-1]
        delta_phi = delta_phi_list[syl][-1]
        c = c_list[syl][-1]
        Z1 = Z_1_list[syl][-1]
        Z2 = Z_2_list[syl][-1]
        theta_1 = theta_1_list[syl][-1]
        theta_2 = theta_2_list[syl][-1] 
        T = T_values_list[syl][-1]

        random_num = np.random.uniform(-1, 1)
        # B block
        if low_frequency_check == 1:
            B+=20
        elif high_frequency_check == 1:
            B-= 20   
        
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
        
        # Alter the duration of the syllable in this phrase
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
    labels_within_phrase = np.array([])
    for i in np.arange(num_repeats):
        
        waveform_filtered_envelope = y_arr * W_t
        phrase_waveform = np.concatenate((phrase_waveform, waveform_filtered_envelope))
        
        # We will denote the last 10 ms of the syllable as silence
        # silence_samples = t[np.where(t>=np.max(t)-0.01)].shape[0]
        # syllable_labels = np.concatenate((np.repeat(syl, t.shape[0]-silence_samples), np.repeat(999, silence_samples)))
        syllable_labels = np.repeat(syl, t.shape[0])
        labels_within_phrase = np.concatenate((labels_within_phrase, syllable_labels))
    
    total_envelope = np.concatenate((total_envelope, phrase_waveform))
    labels_per_sample = np.concatenate((labels_per_sample, labels_within_phrase))


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

np.savez(f'{folderpath}synthetic_data.npz', **dat)
write(f'{folderpath}enveloped_filtered_signal.wav', sampling_freq, normalized_signal)

# %% Let's plot the parameter regime 

import pandas as pd
import seaborn as sns

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
    'Syllable': syllables_flattened, 
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

# plt.figure(figsize=(35, 35))
sns.pairplot(df, hue = 'Syllable')
# Adjust the layout to prevent clipping
plt.tight_layout()
plt.show()


grouped_df = df.groupby('Syllable').mean()
print(grouped_df.T)

# Let's get the labels at every time point

labels = np.array([])
total_syllable_duration_in_phrase = []
syllable_counter = np.zeros((1,syllable_dict['syllable'].shape[0]))
for i in np.arange(syllable_number.shape[0]):
    syl = syllable_number[i]
    labels_for_phrase = np.repeat(syl, syllable_dict['num_repeats'][syl])
    labels = np.concatenate((labels, labels_for_phrase))   
    syllable_duration_in_phrase = np.repeat(T_values_list[syl][int(syllable_counter[:,syl])], syllable_dict['num_repeats'][syl]).tolist()
    total_syllable_duration_in_phrase.append(syllable_duration_in_phrase)
    syllable_counter[:,syl]+=1
    
    
import umap

reducer = umap.UMAP()
X  = df.iloc[:, 1:]
X = X.values
y = df.Syllable
y = y.values
embedding = reducer.fit_transform(X)

plt.figure()
plt.scatter(embedding[:,0], embedding[:,1], c=y, cmap='viridis', s=50)

    
    
    
    
    
    
    
    
    
    
    
    
    