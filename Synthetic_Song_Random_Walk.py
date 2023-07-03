#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 08:38:08 2023

This script will create a synthetic canary song following the supplemental
methods from Gardner et. al 2005

@author: AnanyaKapoor
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import integrate

# Initialize dictionaries for: 
    # 1. parameter's low range
    # 2. parameter's high range
    # 3. parameter's step size 
    
param_names = ['phi_0', 'delta_phi', 'B', 'c', 'f_0', 'Z_1', 'Z_2', 'theta_1', 'theta_2', 'T']
T_lower = 40
T_upper = 300

param_low_range = [
    0, 
    -3*math.pi/2, 
    0, 
    40,
    800, 
    0.88, 
    0.88, 
    0.01, 
    0.01, 
    T_lower]


param_high_range = [
    2*math.pi,
    3*math.pi/2,
    3000,
    70,
    1500,
    0.93,
    0.93,
    math.pi/2, 
    math.pi/2, 
    T_upper]

param_step_size = [
    0.8, 
    0.8, 
    200,
    5,
    100,
    0.02,
    0.02, 
    0.2, 
    0.2,
    999] # 999 will be replaced at each time step


param_low_range_dict = {k: v for k, v in zip(param_names, param_low_range)}
param_high_range_dict = {k: v for k, v in zip(param_names, param_high_range)}
param_step_size_dict = {k: v for k, v in zip(param_names, param_step_size)}


# # Initialize the parameters
phi_0_arr = np.arange(param_low_range_dict['phi_0'],param_high_range_dict['phi_0'], param_step_size_dict['phi_0']) # In radians
delta_phi_arr = np.arange(-3*math.pi/2, 3*math.pi/2, 0.8) # In radians
B_arr = np.arange(0, 3000, 200) # In Hz
c_arr = np.arange(40, 70, 5)
f_0_arr = np.arange(800, 1500, 100) # In Hz
# T_current = 50; T_arr = np.arange(40, 300, 20+0.33*T_current) # Need to figure out what T_current means. I think it means the duration of the current syllable. 
Z_1_arr = np.arange(0.88, 0.93, 0.02)
Z_2_arr = np.arange(0.88, 0.93, 0.02)
theta_1_arr = np.arange(0.01, math.pi/2, 0.2)
theta_2_arr = np.arange(0.01, math.pi/2, 0.2)

# # Let's pick a random value from each list
# # all_params_list = [phi_0_arr, delta_phi_arr, B_arr, c_arr, f_0_arr, T_arr, Z_1_arr, Z_2_arr, theta_1_arr, theta_2_arr]

# initial_params = {}
# initial_params['phi_0'] = np.random.choice(phi_0_arr)
# initial_params['delta_phi'] = np.random.choice(delta_phi_arr)
# initial_params['B'] = np.random.choice(B_arr)
# initial_params['c'] = np.random.choice(c_arr)
# initial_params['f_0'] = np.random.choice(f_0_arr)
# # initial_params['T'] = np.random.choice(T_arr)
# initial_params['Z_1'] = np.random.choice(Z_1_arr)
# initial_params['Z_2'] = np.random.choice(Z_2_arr)
# initial_params['theta_1'] = np.random.choice(theta_1_arr)
# initial_params['theta_2'] = np.random.choice(theta_2_arr)

# Let's say we want to generate a synthetic song that has two syllables: A and B. 
# Syllable A: center pitch  = 1700 Hz, duration = 40 ms 
# Syllable B: center pitch  = 400 Hz, duration = 300 ms

syllable_profile = np.concatenate((np.repeat(0, 40), np.repeat(1,300)), axis = 0)

fundamental_freq = []
song_duration = syllable_profile.shape[0]

# Initialization step
initial_params = []
for i in range(len(param_names)):
    key_name = param_names[i]
    arr = np.arange(param_low_range_dict[key_name],param_high_range_dict[key_name],param_step_size_dict[key_name])
    init_param = np.random.choice(arr)
    initial_params.append(init_param)

if syllable_profile[0] == 0:
    T_current = 40
else:
    T_current = 300
    
param_step_size_dict['T'] = 20+0.33*T_current
T_arr = np.arange(param_low_range_dict['T'], param_high_range_dict['T'], param_step_size_dict['T'] )
T = np.random.choice(T_arr)

initial_param_dict = {k: v for k, v in zip(param_names, initial_params)}
initial_param_dict['T'] = T

f_val = initial_param_dict['f_0'] + initial_param_dict['B']*np.cos(initial_param_dict['phi_0']+initial_param_dict['delta_phi']*0/initial_param_dict['T'])
fundamental_freq.append(f_val)

phi_0 = initial_param_dict['phi_0']
delta_phi = initial_param_dict['delta_phi']
B = initial_param_dict['B']
c = initial_param_dict['c']
f_0 = initial_param_dict['f_0']
Z_1 = initial_param_dict['Z_1']
Z_2 = initial_param_dict['Z_2']
theta_1 = initial_param_dict['theta_1']
theta_2 = initial_param_dict['theta_2']
T = initial_param_dict['T']

param_values = np.zeros((song_duration, 10))
param_values[0,:] = np.array([phi_0, delta_phi, B, c, f_0, Z_1, Z_2, theta_1, theta_2, T])


# The bottom code does not check to see if the parameter values are out of range

for t in np.arange(1, song_duration):
    if syllable_profile[t] == 0:
        T_current = 40
    else:
        T_current = 300
        
    param_step_size_dict['T'] = 20+0.33*T_current

    # If t>0 then we will update params as param += runif(0, 1)*each parameter's respective step size 
    rand_num = np.random.uniform(-1, 1)
    
    step_size_list = []
    
    for key, value in param_step_size_dict.items():
        step_size = rand_num*param_step_size_dict[key]
        step_size_list.append(step_size)

    step_size_arr = np.array(step_size_list)
        
    param_list = []
    for i in np.arange(param_values.shape[1]):
        key_name = param_names[i]
        param_val = param_values[t-1, i]
        if param_val+step_size_arr[i]<param_low_range[i]:
            param_val-=step_size_arr[i]
        elif param_val + step_size_arr[i]>param_high_range[i]:
            param_val-=step_size_arr[i]
        else:
            param_val+=step_size_arr[i]
        
        param_list.append(param_val)
    param_values[t,:] = np.array(param_list)
    
    phi_0 = param_values[t,0]
    delta_phi = param_values[t, 1]
    B = param_values[t,2]
    c = param_values[t,3]
    f_0 = param_values[t, 4]
    Z_1 = param_values[t, 5]
    Z_2 = param_values[t, 6]
    theta_1 = param_values[t, 7]
    theta_2 = param_values[t, 8]
    T = param_values[t, 9]
    
    f_val = f_0 + B*np.cos(phi_0+delta_phi*t/T)
    if f_val>0:
        print("All good")
    else:
        print("Negative Fundamental Frequency")

    fundamental_freq.append(f_val)



import matplotlib.pyplot as plt

# Create a figure and an array of subplots
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 20))

# Plotting data on each subplot
for i, ax in enumerate(axes.flatten()):
    ax.axhline(y=np.min(param_values[:,i]), color='red', linestyle='dashed')
    ax.axhline(y=np.max(param_values[:,i]), color='red', linestyle='dashed')
    ax.plot(param_values[:,i])
    ax.set_title(f'{param_names[i]}')
    
    
plt.figure()
plt.plot(fundamental_freq)
plt.xlabel("Time (t)")
plt.ylabel("Fundamental Frequency (Hz)")
plt.show()
        
# Now let's create the harmonics jawn


## Theta jawn

from scipy import integrate

# Define the function to integrate with parameters that change at each timestep
def f(x, t, f_0, B, phi_0, delta_phi, T):
    return  f_0 + B*np.cos(phi_0+delta_phi*t/T)

# Define the time steps and corresponding parameter values
# time_steps = [0, 1, 2, 3]  # Example time steps
# parameter_a = [1, 2, 3, 4]  # Example parameter a values at each timestep
# parameter_b = [0.5, 0.5, 0.5, 0.5]  # Example parameter b values at each timestep

time_steps = np.arange(song_duration)
# Initialize arrays to store the integral results
results = []
errors = []

# Perform numerical integration at each timestep with varying parameters
for i, t in enumerate(time_steps):
    result, error = integrate.quad(f, 0, song_duration, args=(t, param_values[t, 4], 
                                                  param_values[t, 2], 
                                                  param_values[t, 0], 
                                                  param_values[t,1], 
                                                  param_values[t, 9]))
    results.append(result)
    errors.append(error)
    
theta_arr = np.array(results)
theta_arr = theta_arr*2*math.pi

theta_harmonics_arr = np.zeros((12, theta_arr.shape[0]))

for k in np.arange(1, 12+1):
    theta_harmonics_arr[k-1,:] = k*theta_arr
    

# let's initialize the amplitudes
A_list = [1]
for k in np.arange(1,12):
    A_list.append(1/(1+40*2**(k-1)))
    
s_t_list = []
for t in np.arange(song_duration):
    s_t = 0
    for k in np.arange(1,12+1):
        s_t+=A_list[k-1]*np.sin(theta_harmonics_arr[k-1,t])
    
    s_t_list.append(s_t)
    
    
    
import librosa

D = librosa.stft(np.array(s_t_list), n_fft = 10)      
S_db = librosa.amplitude_to_db(np.abs(D), ref = np.max)


fig, ax = plt.subplots()
img = librosa.display.specshow(S_db, x_axis = 'time', y_axis = 'log', ax = ax)










    