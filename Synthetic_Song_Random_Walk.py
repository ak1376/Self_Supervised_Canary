#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 08:14:34 2023

Code to create synthetic canary song following the methods of Gardner et. al
2005. Code is set up to generate synthetic song of an arbitrary number of 
syllables. The user just needs to manually provide values for (1. syllable 
duration, (2. center frequency and (3. amplitude modulation


@author: AnanyaKapoor
"""


folderpath = '/Users/AnanyaKapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/Canary_SSL_Repo/'


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parameters
sampling_freq = 44100

# Syllable-specific parameters
num_syllables = 5
num_repeats = 50

syllable_profile = np.array([])
for syl in np.arange(num_syllables):
  tiled = np.tile(np.array([(syl + 1), 0]), 100)
  syllable_profile = np.concatenate((syllable_profile, tiled))


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


syllable_names = ['0', '1', '2', '3', '4', '5']


# Pick acoustic parameter values
duration_values = [0.05, 0.3, 0.2, 0.28, 0.25, 0.15]
f_0_values = [1, 1000, 400, 800, 600, 300, 500]
B_values = [0, 500, 200, 20, 700, 200, 400]
phi_0_values = np.random.choice(phi_0_arr, (num_syllables+1)).tolist()
delta_phi_values = np.random.choice(delta_phi_arr, (num_syllables+1)).tolist()
c_values = np.random.choice(c_arr, (num_syllables+1)).tolist()
Z1_values = np.random.choice(Z_1_arr, (num_syllables+1)).tolist()
Z2_values = np.random.choice(Z_2_arr, (num_syllables+1)).tolist()
theta_1_values = np.random.choice(theta_1_arr, (num_syllables+1)).tolist()
theta_2_values = np.random.choice(theta_2_arr, (num_syllables+1)).tolist()


duration_dict = dict(zip(syllable_names, duration_values))
f_0_dict = dict(zip(syllable_names, f_0_values))
B_dict = dict(zip(syllable_names, B_values))
phi_0_dict = dict(zip(syllable_names, phi_0_values))
delta_phi_dict = dict(zip(syllable_names, delta_phi_values))
c_dict = dict(zip(syllable_names, c_values))
Z1_dict = dict(zip(syllable_names, Z1_values))
Z2_dict = dict(zip(syllable_names, Z2_values))
theta_1_dict = dict(zip(syllable_names, theta_1_values))
theta_2_dict = dict(zip(syllable_names, theta_2_values))
num_harmonics = 3
total_signal_wave = np.array([])
total_envelope = np.array([])


# Simulate synthetic song

for i in np.arange(syllable_profile.shape[0]):
    key_val = str(int(syllable_profile[i]))
    
# =============================================================================
#     # Specify acoustic parameters
# =============================================================================

    T = duration_dict[key_val]
    f_0 = f_0_dict[key_val]
    B = B_dict[key_val]
    phi_0 = phi_0_dict[key_val]
    delta_phi = delta_phi_dict[key_val]
    c = c_dict[key_val]
    Z1 = Z1_dict[key_val]
    Z2 = Z2_dict[key_val]
    theta_1 = theta_1_dict[key_val]
    theta_2 = theta_2_dict[key_val]
    
# =============================================================================
#     # Calculate fundamental frequency
# =============================================================================
    total_samples = int(sampling_freq * duration_dict[key_val])
    t = np.linspace(0, (duration_dict[key_val]), total_samples, endpoint=False)
    
    fundamental_freq = f_0 + B*np.cos(phi_0 + delta_phi*(t)/T)
    fundamental_freq.shape = (t.shape[0], 1)
    
# =============================================================================
#     # Harmonics
# =============================================================================
    theta_arr = np.zeros((num_harmonics, t.shape[0]))
    for k in np.arange(num_harmonics):
        # val = 2*np.pi*(k+1)*fundamental_freq.reshape(fundamental_freq.shape[0],)
        val = 2*np.pi*(k+1)*f_0*t + (2*np.pi*(k+1)*B*T/delta_phi)*(np.sin(phi_0+delta_phi/T*t) - np.sin(phi_0))
        theta_arr[k, :] = val
        
    a1 = 1
    a2 = 1/(1+c*2**1)
    a3 = 1/(1+c*2**2)

# =============================================================================
#     # Calculate signal
# =============================================================================
    s_t_arr = a1*np.sin(theta_arr[0,:]) + a2 * np.sin(theta_arr[1,:]) + a3*np.sin(theta_arr[2,:])
    
    signal_wave = s_t_arr
    total_signal_wave = np.concatenate((total_signal_wave, signal_wave))
    
# =============================================================================
#     # Let's calculate the envelope for each syllable
# =============================================================================
    W_t = 0.42 + 0.5 * np.cos(np.pi*t/(T)) + 0.08 * np.cos(2*np.pi*t/(T))
    waveform_with_envelope = signal_wave * W_t
    total_envelope = np.concatenate((total_envelope, waveform_with_envelope))
    
# Spectrogram of the signal
frequencies, times, spectrogram = signal.spectrogram(total_signal_wave, fs=sampling_freq,
                                                    window='hamming', nperseg=256,
                                                    noverlap=128, nfft=512)

# Plot the spectrogram
plt.figure()
plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram))
plt.colorbar(label='Power Spectral Density (dB/Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title("Spectrogram of Signal")
plt.show()

import sounddevice as sd  
from scipy.io.wavfile import write


write(f'{folderpath}raw_signal.wav', sampling_freq, total_signal_wave)



# =============================================================================
# # Find spectrogram of enveloped signal
# =============================================================================


# Spectrogram of the signal
frequencies, times, spectrogram = signal.spectrogram(total_envelope, fs=sampling_freq,
                                                    window='hamming', nperseg=256,
                                                    noverlap=128, nfft=512)

# Plot the spectrogram
plt.figure()
plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram))
plt.colorbar(label='Power Spectral Density (dB/Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title("Enveloped Spectrogram of Signal")
plt.show()

# write(f'{folderpath}normalized_signal.wav', sampling_freq, W_t)





