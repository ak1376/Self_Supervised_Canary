#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 09:38:41 2023

This code creates a synthetic song for any arbitrary number of syllables. This
code was modeled from the supplemental methods of Gardner et. al 2005. The
resulting synthetic song will be used as a testbed for ML modeling validation. 

@author: AnanyaKapoor
"""

import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as signal
import sounddevice as sd  
from scipy.io.wavfile import write

folderpath = '/Users/ananyakapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/Canary_SSL_Repo/'

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

total_signal_wave = np.array([])
total_envelope = np.array([])
total_filtered = np.array([])


num_syllables = 30
num_short = 20
num_long = 10

short_durations = np.random.uniform(20/1000, 70/1000, num_short)
long_durations = np.random.uniform(100/1000, 200/1000, num_long)

short_repeats = np.random.randint(20, 30, num_short)
long_repeats = np.random.randint(3, 5, num_long)


T_arr = np.concatenate((short_durations, long_durations))
num_repeats_arr = np.concatenate((short_repeats, long_repeats))


permutation = np.random.permutation(len(T_arr))

T_arr_shuffled = T_arr[permutation]
num_repeats_arr_shuffled = num_repeats_arr[permutation]

T_list = T_arr_shuffled.tolist()
num_repeats_list = num_repeats_arr_shuffled.tolist()

# num_syllables = 14
# num_repeats_list = [25,
#                     4,
#                     20,
#                     20,
#                     3,
#                     30,
#                     2,
#                     2, 
#                     25,
#                     22,
#                     28,
#                     3,
#                     2,
#                     3]

# T_list = [40/1000,
#           150/1000,
#           80/1000,
#           90/1000, 
#           200/1000, 
#           30/1000, 
#           200/1000,
#           180/1000, 
#           50/1000, 
#           50/1000, 
#           70/1000,
#           170/1000, 
#           180/1000,
#           80/1000]

# f_0_list = [2000,
#             900,
#             1320,
#             1400,
#             800, 
#             1300, 
#             850,
#             1500, 
#             1200,
#             755,
#             1350,
#             1000,
#             800,
#             1500]

# B_list = [800,
#           200,
#           900,
#           850,
#           100, 
#           150, 
#           800, 
#           900, 
#           1000, 
#           1000, 
#           200, 
#           350, 
#           150, 
#           1700]


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
low_frequency_check = 0
high_frequency_check = 0

for syl in np.arange(num_syllables):
    for i in np.arange(num_repeats_list[syl]):
    
        if syl == 0 and i ==0 : # Initial condition (start of song)
            f_0_list.append(f_0)
            B_list.append(B)
            phi_0_list.append(phi_0)
            delta_phi_list.append(delta_phi)
            c_list.append(c)
            Z_1_list.append(Z1)
            Z_2_list.append(Z2)
            theta_1_list.append(theta_1)
            theta_2_list.append(theta_2)
            
        elif syl !=0 and i == 0:
                
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
            
            f_0_list.append(f_0)
            B_list.append(B)
            phi_0_list.append(phi_0)
            delta_phi_list.append(delta_phi)
            c_list.append(c)
            Z_1_list.append(Z1)
            Z_2_list.append(Z2)
            theta_1_list.append(theta_1)
            theta_2_list.append(theta_2)
            
        
        T = T_list[syl]
        # f_0 = f_0_list[syl]
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
    
    
        # coefs = np.poly(roots)
    
        # a1 = coefs[0]
        # a2 = coefs[1]
        # a3 = coefs[2]
        # a4 = coefs[3]
        # a5 = coefs[4]
        
        # y_arr = signal.lfilter([1], coefs, s_t_arr)
        
        
        # y_arr = np.zeros_like(waveform_with_envelope)
        # for t_val in np.arange(1,waveform_with_envelope.shape[0]): 
    
        #     if t_val == 1: 
        #         filtered_val = (waveform_with_envelope[t_val] - a2*y_arr[t_val-1])/a1
        #     elif t_val == 2: 
        #         filtered_val = (waveform_with_envelope[t_val] - a2*y_arr[t_val-1] - a3*y_arr[t_val-2])/a1
        #     elif t_val == 3: 
        #         filtered_val = (waveform_with_envelope[t_val] - a2*y_arr[t_val-1] - a3*y_arr[t_val-2] - a4* y_arr[t_val-3])/a1
        #     else:
        #         filtered_val = (waveform_with_envelope[t_val] - a2*y_arr[t_val-1] - a3*y_arr[t_val-2] - a4* y_arr[t_val-3] - a5*y_arr[t_val-4])/a1
        
        #     y_arr[t_val] = filtered_val
    
        total_filtered = np.concatenate((total_filtered, y_arr))
            
        # =============================================================================
        #     # Enveloped signal 
        # =============================================================================
        # W_t = (0.42 + 0.5*np.cos(np.pi * t/T) + 0.08*np.cos(2*np.pi * t/T))
        W_t = 0.5 * (1 - np.cos(2 * np.pi * t / T))
    
        waveform_filtered_envelope = y_arr * W_t
        
        total_envelope = np.concatenate((total_envelope, waveform_filtered_envelope))
        

        
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



# =============================================================================
# # GROUND TRUTH SIMULATION WAVEFORM
# =============================================================================


# import wave
# import numpy as np
# import matplotlib.pyplot as plt

# # Open the .wav file
# wav_file = wave.open('/Users/AnanyaKapoor/Downloads/1108214s_sound/gardnersound4.wav', 'r')

# # Get the audio file parameters
# sample_width = wav_file.getsampwidth()
# sample_rate = wav_file.getframerate()
# num_frames = wav_file.getnframes()

# # Read the audio data
# audio_data = wav_file.readframes(num_frames)

# # Convert the audio data to a numpy array
# audio_array = np.frombuffer(audio_data, dtype=np.int16)

# # Close the .wav file
# wav_file.close()

# # Generate the time axis
# duration = num_frames / sample_rate
# t_groundtruth = np.linspace(0, duration, num_frames)

# # # Plot the waveform
# # plt.figure()
# # plt.plot(t_groundtruth, audio_array)
# # plt.xlabel('Time (s)')
# # plt.ylabel('Amplitude')
# # plt.title('Ground Truth Waveform')
# # plt.show()

# from scipy import signal

# frequencies, times, spectrogram = signal.spectrogram(audio_array, fs=sample_rate,
#                                                     window='hamming', nperseg=256,
#                                                     noverlap=128, nfft=512)


# # spectrogram[frequencies>3000] = 10**(-30)

# # Plot the spectrogram
# plt.figure()
# plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), shading='auto', cmap='inferno')
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# plt.title("Spectrogram of Signal")
# plt.show()


# import wave
# import numpy as np
# import matplotlib.pyplot as plt

# wav_file = wave.open('smb://ion-nas.uoregon.edu/glab/Rose/sample_songs/USA5199_45108.62802931_7_1_17_26_42.wav')

# # Get the audio file parameters
# sample_width = wav_file.getsampwidth()
# sample_rate = wav_file.getframerate()
# num_frames = wav_file.getnframes()

# # Read the audio data
# audio_data = wav_file.readframes(num_frames)

# # Convert the audio data to a numpy array
# audio_array = np.frombuffer(audio_data, dtype=np.int16)

# # Close the .wav file
# wav_file.close()

# # Generate the time axis
# duration = num_frames / sample_rate
# t_groundtruth = np.linspace(0, duration, num_frames)

# # # Plot the waveform
# # plt.figure()
# # plt.plot(t_groundtruth, audio_array)
# # plt.xlabel('Time (s)')
# # plt.ylabel('Amplitude')
# # plt.title('Ground Truth Waveform')
# # plt.show()

# from scipy import signal

# frequencies, times, spectrogram = signal.spectrogram(audio_array, fs=sample_rate,
#                                                     window='hamming', nperseg=256,
#                                                     noverlap=128, nfft=512)


# # spectrogram[frequencies>3000] = 10**(-30)

# # Plot the spectrogram
# plt.figure()
# plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), shading='auto', cmap='inferno')
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# plt.title("Spectrogram of Signal")
# plt.show()



