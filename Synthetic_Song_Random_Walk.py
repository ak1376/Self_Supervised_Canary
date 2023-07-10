#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:14:03 2023

@author: AnanyaKapoor
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os

folderpath = '/Users/AnanyaKapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/Canary_SSL_Repo/'

# Let's only simulate one syllable for now
sampling_freq = 44100
T = 44100
t = np.linspace(0,1,T, endpoint = False)
f_0 = 2000
B = 1000
phi_0 = np.pi/3
delta_phi = np.pi/2
c = 50
Z1 = 0.92
Z2 = 0.885
theta_1 = 1
theta_2 = 1.25

num_iterations = 10

B_list = []
phi_0_list = []
delta_phi_list = []
c_list = []
Z_1_list = []
Z_2_list = []
theta_1_list = []
theta_2_list = []

total_syllables = np.array([])
total_signal_wave = np.array([])
total_filtered = np.array([])
total_normalized = np.array([])
silence = np.zeros((40, 1))
for i in np.arange(num_iterations):
    
# =============================================================================
#     # Code block for parameters
# =============================================================================
    if i == 0: # Initial conditions (automatically meet parameter criteria)
        B_list.append(B)
        phi_0_list.append(phi_0)
        delta_phi_list.append(delta_phi)
        c_list.append(c)
        Z_1_list.append(Z1)
        Z_2_list.append(Z2)
        theta_1_list.append(theta_1)
        theta_2_list.append(theta_2)
        
        
    else:
        random_num = np.random.uniform(-1, 1)
        
        # B block 
        if B + random_num*200 < 0 or B + random_num*200 > 3000:
            B-=random_num*200
        else:
            B+=random_num*200
        
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
        
        # _0 block
        if f_0 + random_num*100<800 or f_0 + random_num*100>4000:
            f_0-=random_num*100
        else:
            f_0 += random_num*100
            
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
    
# =============================================================================
#     # Calculating fundamental frequency from parameters        
# =============================================================================
    fundamental_freq = f_0 + B*np.cos(phi_0 + delta_phi*t/T)
    fundamental_freq.shape = (t.shape[0], 1)
    syllable_freqs = np.concatenate((fundamental_freq, silence), axis = 0)
    syllable_freqs.shape = (syllable_freqs.shape[0],)
    total_syllables = np.concatenate((total_syllables, syllable_freqs))
    
# =============================================================================
#     # Calculate harmonics
# =============================================================================
    theta_arr = np.zeros((12, t.shape[0]))
    for k in np.arange(12):
        val = 2*np.pi*(k+1)*f_0*t + (2*np.pi*(k+1)*B/delta_phi)*(np.sin(phi_0+delta_phi*t) - np.sin(phi_0))
        theta_arr[k, :] = val
    
    # # let's compute the amplitudes

    A_arr = np.ones((12, t.shape[0]))

    for k in np.arange(1, 12):
        A_arr[k,:] = 1/(1+c*2**(k-1))
        
    s_t_arr = np.zeros_like(t)
    for k in np.arange(12):
        s_t_arr+= A_arr[k,:]*np.sin(theta_arr[k,:])
        
    signal_wave = s_t_arr
    total_signal_wave = np.concatenate((total_signal_wave, signal_wave))
    
# =============================================================================
#     # Calculate the filtered signal
# =============================================================================
    
    r1_roots = Z1 * np.exp(1j*theta_1)
    r2_roots = Z2 * np.exp(1j*theta_2)
    roots = [r1_roots, np.conjugate(r1_roots), r2_roots, np.conjugate(r2_roots)]
    
    coefs = np.poly(roots)

    a0 = coefs[0]
    a1 = coefs[1]
    a2 = coefs[2]
    a3 = coefs[3]
    a4 = coefs[4]
    
    y_arr = np.zeros_like(signal_wave)
    for t_val in np.arange(1,signal_wave.shape[0]):
        
        if t_val == 1: 
            filtered_val = (signal_wave[t_val] - a1*y_arr[t_val-1])/a0
        elif t_val == 2: 
            filtered_val = (signal_wave[t_val] - a1*y_arr[t_val-1] - a2*y_arr[t_val-2])/a0
        elif t_val == 3: 
            filtered_val = (signal_wave[t_val] - a1*y_arr[t_val-1] - a2*y_arr[t_val-2] - a3* y_arr[t_val-3])/a0
        else:
            filtered_val = (signal_wave[t_val] - a1*y_arr[t_val-1] - a2*y_arr[t_val-2] - a3* y_arr[t_val-3] - a4*y_arr[t_val-4])/a0
        
        y_arr[t_val] = filtered_val
    
    total_filtered = np.concatenate((total_filtered, y_arr))
    
    W_t = np.zeros_like(signal_wave)
    for t_val in np.arange(signal_wave.shape[0]):
        W_t[t_val] = 0.42+0.5*np.cos(np.pi*signal_wave[t_val]) + 0.08*np.cos(2*np.pi*signal_wave[t_val])
        
    total_normalized = np.concatenate((total_normalized, W_t))
    B_list.append(B)
    phi_0_list.append(phi_0)
    delta_phi_list.append(delta_phi)
    c_list.append(c)
    Z_1_list.append(Z1)
    Z_2_list.append(Z2)
    theta_1_list.append(theta_1)
    theta_2_list.append(theta_2)



# Plot of frequency
plt.figure()
plt.plot(total_syllables)
plt.title("Plot of Frequency [f(t)]")
plt.show()

# Spectrogram of signal

from scipy import signal
frequencies, times, spectrogram = signal.spectrogram(total_signal_wave, fs=sampling_freq, window='hann', nperseg=256, noverlap=128, nfft=512)

# Plot the spectrogram
plt.figure()
plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram))
plt.colorbar(label='Power Spectral Density (dB/Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title("Spectrogram of Signal")
plt.show()
  

# Spectrogram of filtered signal
frequencies_filtered, times_filtered, spectrogram_filtered = signal.spectrogram(total_filtered, fs=sampling_freq, window='hann', nperseg=256, noverlap=128, nfft=512)

 # nperseg_val = 50
 # nfft_val = 100
 # frequencies, times, spectrogram = signal.spectrogram(signal_wave, fs=sampling_freq, nperseg = nperseg_val, nfft = nfft_val, noverlap = 0)

 # Plot the spectrogram
plt.figure()
plt.pcolormesh(times_filtered, frequencies_filtered, 10 * np.log10(spectrogram_filtered))
plt.colorbar(label='Power Spectral Density (dB/Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title("Spectrogram of Filtered Signal")
# plt.title(f'Spectrogram Using nperseg = {nperseg_val}, nfft = {nfft_val}')
plt.show()


# Spectrogram of normalized signal
frequencies_W, times_W, spectrogram_W = signal.spectrogram(total_normalized, fs=sampling_freq, window='hann', nperseg=256, noverlap=128, nfft=512)

 # nperseg_val = 50
 # nfft_val = 100
 # frequencies, times, spectrogram = signal.spectrogram(signal_wave, fs=sampling_freq, nperseg = nperseg_val, nfft = nfft_val, noverlap = 0)

 # Plot the spectrogram
plt.figure()
plt.pcolormesh(times_W, frequencies_W, 10 * np.log10(spectrogram_W))
plt.colorbar(label='Power Spectral Density (dB/Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title("Spectrogram of Normalized Filtered Signal ")
# plt.title(f'Spectrogram Using nperseg = {nperseg_val}, nfft = {nfft_val}')
plt.show()

     
import sounddevice as sd  
from scipy.io.wavfile import write


write(f'{folderpath}raw_signal.wav', sampling_freq, total_signal_wave)
# sd.play(total_signal_wave, samplerate=44100)


