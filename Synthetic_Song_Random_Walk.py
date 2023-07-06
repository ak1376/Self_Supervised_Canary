#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 07:13:00 2023

@author: AnanyaKapoor
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

phi_0_arr = np.arange(0, 2*np.pi, 0.8)
delta_phi_arr = np.arange(-3*np.pi/2, 3*np.pi/2, 0.8)
B = 0.05
f_0 = 1000
duration = 3.0
sampling_freq = 44100

T = duration*sampling_freq

t_arr = np.linspace(0, duration, int(duration*sampling_freq), endpoint=False)

# Randomly initialize parameters
phi_0 = np.random.choice(phi_0_arr)
delta_phi = np.random.choice(delta_phi_arr)
fundamental_freq = f_0 + B*np.cos(phi_0 + delta_phi*0/T)

phi_0_list = [phi_0]
delta_phi_list = [delta_phi]
fundamental_freq_list = []
fundamental_freq_list.append(fundamental_freq)

for t in np.arange(1, t_arr.shape[0]):
    random_num = np.random.uniform(-0.01, 0.01)
    
    # phi_0 block
    if phi_0 + random_num*0.8 < 0 or phi_0 + random_num*0.8 > 2*np.pi:
        phi_0-= random_num*0.8
    else:
        phi_0+=random_num*0.8
        
    # delta_phi block
    if delta_phi + random_num*0.8 < -3*np.pi/2 or delta_phi + random_num*0.8 > 3*np.pi/2:
        delta_phi-= random_num*0.8
    else:
        delta_phi+=random_num*0.8
    
    fundamental_freq = f_0 + B*np.cos(phi_0 + delta_phi*0/T)
    fundamental_freq_list.append(fundamental_freq)
    phi_0_list.append(phi_0)
    delta_phi_list.append(delta_phi)
    
    
theta_arr = np.zeros((12, t_arr.shape[0]))

for k in np.arange(12):
    for t in np.arange(t_arr.shape[0]):
        val = 2*np.pi*(k+1)*f_0*t_arr[t] + (2*np.pi*(k+1)*B*T/delta_phi_list[t])*(np.sin(phi_0_list[t]+delta_phi_list[t]*t_arr[t]/T) - np.sin(phi_0_list[t]))
        theta_arr[k, t] = val
        
# let's initialize the amplitudes
A_list = [1]
for k in np.arange(2,13):
    A_list.append(1/(1+3*2**(k-1)))

# A_list = np.linspace(1, 0.0001, 12).tolist()

s_t_arr = np.zeros_like(t_arr)
for k in np.arange(12):
    s_t_arr+= A_list[k]*np.sin(theta_arr[k,:])
    
signal_wave = s_t_arr

from scipy import signal
frequencies, times, spectrogram = signal.spectrogram(signal_wave, fs=sampling_freq, window='hann', nperseg=256, noverlap=128, nfft=512)

# nperseg_val = 50
# nfft_val = 100
# frequencies, times, spectrogram = signal.spectrogram(signal_wave, fs=sampling_freq, nperseg = nperseg_val, nfft = nfft_val, noverlap = 0)

# Plot the spectrogram
plt.figure()
plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram))
plt.colorbar(label='Power Spectral Density (dB/Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
# plt.title(f'Spectrogram Using nperseg = {nperseg_val}, nfft = {nfft_val}')
plt.show()


# W_t = np.zeros_like(signal_wave)
# for t in np.arange(t_arr.shape[0]):
#     W_t[t] = 0.42+0.5*np.cos(np.pi*signal_wave[t]/T) + 0.08*np.cos(2*np.pi*signal_wave[t]/T)

import sounddevice as sd        
sd.play(signal_wave, samplerate=44100)


        
        
    