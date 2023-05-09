import pandas as pd
import csv
import numpy as np
from datetime import datetime
from matplotlib.colors import ListedColormap
from ahrs.filters import Mahony, Madgwick
from ahrs.common import orientation
import os
import pandas as pd
import csv
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import resample
import math
from scipy.signal import savgol_filter
from scipy.signal import freqz, butter, lfilter
from scipy import interpolate
from scipy.spatial.transform import Rotation
from scipy.stats import circmean
from scipy.interpolate import CubicSpline
from math import nan
from shimmer_utilities_functions import *


def calculate_activity_counts(accel_magnitudes, epoch_size_seconds, resampling_frequency_leuenberger, threshold):
    """
    Calculate the activity count (AC) for each epoch of a given accelerometer magnitude data.

    Parameters:
        accel_magnitudes (ndarray): 1D numpy array of accelerometer magnitudes
        epoch_size_seconds (int): duration of epoch in seconds

    Returns:
        ndarray: 1D numpy array of ACs values for each epoch (number_epoch,)
    """

    # Split the data into epochs using the window_data function
    epochs = window_data(accel_magnitudes, resampling_frequency_leuenberger, epoch_size_seconds)
    
    # Initialize an array to store the AC of each epoch
    activity_counts = np.zeros(epochs.shape[0])
    
    # Calculate the AC for each epoch using the get_activity_count_per_epoch function
    for i in range(epochs.shape[0]):
        activity_counts[i] = get_activity_count_per_epoch(epochs[i], epoch_size_seconds, resampling_frequency_leuenberger, threshold)
    
    return activity_counts


def get_activity_count_per_epoch(epoch_accel_magnitudes, epoch_size_seconds, resampling_frequency_leuenberger, threshold):
    """
    Calculate the activity count (AC) for a given epoch of acceleration magnitude data.

    Parameters:
        epoch_accel_magnitudes (ndarray): 1D numpy array of acceleration magnitudes for a single epoch
        epoch_size_seconds (int): duration of epoch in seconds
        threshold (float): threshold value for counting accelerations above the threshold (default = 0.05)

    Returns:
        float: AC value for the given epoch
    """
    # Sum all the acceleration magnitudes over the epoch length
    acceleration_sum = np.sum(epoch_accel_magnitudes)
    
    # Calculate the activity count per epoch by dividing the sum of the magnitudes by the product of epoch duration in minutes
    # and the sampling frequency
    activity_count_per_epoch = acceleration_sum / ((epoch_size_seconds/60) * resampling_frequency_leuenberger)
    
    # Using the threshold of 0.05g/count 
    if activity_count_per_epoch < convert_g_to_ms2(threshold):
        return 0
    else:
        count = int(activity_count_per_epoch / convert_g_to_ms2(threshold))
        return count


def plot_activity_counts_with_events(activity_counts, total_time, events):
    """
    Plot activity counts over time with timeline events.

    Args:
        activity_counts (array-like): Array of activity counts.
        total_time (float): Total recording time in seconds.
        events (list): List of tuples with the event time in seconds and the event label.
    """

    # Calculate the time interval between each epoch
    epoch_time = total_time / len(activity_counts)

    # Determine the number of seconds per tick label
    sec_per_tick = round((total_time / 10) / 3600) * 3600

    # Calculate the number of ticks and tick positions
    num_ticks = int(np.ceil(total_time / sec_per_tick))
    tick_positions = np.arange(num_ticks) * sec_per_tick

    # Set up the figure with a larger plot size
    fig, ax = plt.subplots(figsize=(16, 8))

    # Create the bar plot
    ax.bar(x=range(len(activity_counts)), height=activity_counts, width=3, align='edge')

    # Set the x-axis ticks and labels to be at the center of each epoch
    ax.set_xticks([i + 0.5 for i in range(len(activity_counts))])
    ax.set_xticklabels([f'{i+1}' for i in range(len(activity_counts))])

    # Set the x-axis tick positions and labels with hours modulo 24
    ax.set_xticks(tick_positions / epoch_time)
    ax.set_xticklabels([f'{int((tick_pos/3600 + 10) % 24)}:00' for tick_pos in tick_positions])
    
    # Add the events to the plot
    if events:
        for i, (event_time, event_label) in enumerate(events):
            event_pos = event_time / epoch_time
            ax.axvline(x=event_pos, linestyle='--', color='black', linewidth=1)
            bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
            y_pos = 100 + i*25 + 50 # Random y value between 200 and 450
            ax.text(event_pos, y_pos, event_label, fontsize=10, ha='center', va='center', bbox=bbox_props)

    # Set the x-axis label and the y-axis label
    ax.set_xlabel('Time')
    ax.set_ylabel('Activity Counts')

    # Set the plot title
    ax.set_title('Activity Counts over time for left wrist using a 0.05g threshold')

    # Show the plot
    plt.show()


def plot_activity_counts_superposition(activity_counts1, activity_counts2, total_time):
    """
    Plot two sets of activity counts over time.

    Args:
        activity_counts1 (array-like): Array of activity counts for first set of data.
        activity_counts2 (array-like): Array of activity counts for second set of data.
        total_time (float): Total recording time in seconds.
    """

    font_size = 12 
    
    # Calculate the time interval between each epoch
    epoch_time = total_time / len(activity_counts1)

    # Determine the number of seconds per tick label
    sec_per_tick = round((total_time / 10) / 3600) * 3600

    # Calculate the number of ticks and tick positions
    num_ticks = int(np.ceil(total_time / sec_per_tick))
    tick_positions = np.arange(num_ticks) * sec_per_tick

    # Set up the figure with a larger plot size
    fig, ax = plt.subplots(figsize=(16, 8))

    # Create the bar plots with transparency
    ax.bar(x=range(len(activity_counts1)), height=activity_counts1, width=3, align='edge', alpha=0.4, color='red', label='Left wrist')
    ax.bar(x=range(len(activity_counts2)), height=activity_counts2, width=3, align='edge', alpha=0.4, color='blue', label='Right wrist')

    # Set the x-axis ticks and labels to be at the center of each epoch
    ax.set_xticks([i + 0.5 for i in range(len(activity_counts1))])
    ax.set_xticklabels([f'{i+1}' for i in range(len(activity_counts1))])

    # Set the x-axis tick positions and labels with hours modulo 24
    ax.set_xticks(tick_positions / epoch_time)
    ax.set_xticklabels([f'{int((tick_pos/3600 + 10) % 24)}:00' for tick_pos in tick_positions], fontsize=font_size)

    ax.set_yticklabels([f'{tick_label}' for tick_label in ax.get_yticks()], fontsize=font_size)
    ax.set_xlabel('Time', fontsize=font_size)
    ax.set_ylabel('Activity Counts', fontsize=font_size)
    
    # Set the plot title
    ax.set_title('Activity Counts over time for both wrists using a 0.05g threshold', fontsize=font_size)

    # Add a legend on the right side of the figure
    ax.legend(loc='upper right', fontsize=font_size)

    # Show the plot
    plt.show()


def get_acceleration_magnitude_per_second(accel_leuenberger_LW, accel_leuenberger_RW, 
                                           resampling_frequency_leuenberger, resampling_frequency_laterality):
    """
    Get one value of magnitude per second using 1Hz resampling and apply the 0.05g threshold for activity.

    Args:
        accel_leuenberger_LW (array-like): Array of acceleration values for left wrist.
        accel_leuenberger_RW (array-like): Array of acceleration values for right wrist.
        resampling_frequency_leuenberger (int): Sampling frequency of the accelerometer data in Hz.
        resampling_frequency_laterality (int): Desired sampling frequency of the output data in Hz.

    Returns:
        tuple: A tuple of two array-like objects with the resampled acceleration magnitudes for each wrist.
    """
    # Reshape the data to match the resampling function
    accel_leuenberger_LW = accel_leuenberger_LW.reshape(-1, 1)
    accel_leuenberger_RW = accel_leuenberger_RW.reshape(-1, 1)
    
    # Resample the acceleration data at 1Hz (initially at 50Hz after resampling)
    LW_accel_res = resample_data_cubic_spline(accel_leuenberger_LW, resampling_frequency_leuenberger, 
                                             resampling_frequency_laterality)
    RW_accel_res = resample_data_cubic_spline(accel_leuenberger_RW, resampling_frequency_leuenberger, 
                                             resampling_frequency_laterality)
    
    # Check if it's enough to be an active epoch using the 0.05g threshold 
    # If the acceleration magnitude for the 1 second epoch is less than the threshold defined for activity,
    # set it to 0.0 to remove low excitation
    threshold_ms2 = convert_g_to_ms2(activity_threshold_litterature)
    LW_accel_res[LW_accel_res < threshold_ms2] = 0
    RW_accel_res[RW_accel_res < threshold_ms2] = 0
    
    return LW_accel_res, RW_accel_res


def get_active_use_duration(accel_leuenberger_LW, accel_leuenberger_RW, threshold, resampling_frequency_leuenberger,
                             resampling_frequency_laterality, epoch_size=1):
    
    LW_accel_res, RW_accel_res = get_acceleration_magnitude_per_second(accel_leuenberger_LW, accel_leuenberger_RW,
                                                                        resampling_frequency_leuenberger, resampling_frequency_laterality)
    
    active_duration_LW = sum(1 for accel in LW_accel_res if accel >= convert_g_to_ms2(threshold))
    active_duration_RW = sum(1 for accel in RW_accel_res if accel >= convert_g_to_ms2(threshold))
            
    return active_duration_LW, active_duration_RW


def get_active_ratio(active_duration_time, total_recording_time):
    active_ratio = active_duration_time / total_recording_time
    active_percentage = active_ratio * 100
    return math.ceil(active_percentage)


def compute_pitch_from_dcm(dcms):
    """
    Computes pitch angles from an array of direction cosine matrices (DCMs).

    Args:
        dcms (numpy.ndarray): Array of DCMs of shape (n_samples, 3, 3).

    Returns:
        numpy.ndarray: Array of pitch angles in degrees of shape (n_samples,).
    """
    forearm_vec = np.array([1, 0, 0])
    elevations = []
    for dcm in dcms:
        # Compute earth frame vector
        earth_vec = dcm @ forearm_vec
        # Compute angle between earth_vec and horizontal plane
        horizontal_proj = np.sqrt(earth_vec[0]**2 + earth_vec[1]**2)
        elevation = np.arctan2(earth_vec[2], horizontal_proj) * 180 / np.pi
        elevations.append(elevation)
    return np.array(elevations)


def compute_yaw_from_dcms(dcms):
    """
    Calculate yaw angles from a given array of direction cosine matrices (DCMs).
    
    Args:
        dcms (np.ndarray): Array of direction cosine matrices (3x3 matrices).
        
    Returns:
        np.ndarray: Array of yaw angles in degrees, ranging from -180 to 180 degrees.
    """
    num_headings = 12  # Number of heading vectors
    phi = np.radians(180) / num_headings  # Calculate spacing between heading vectors
    heading_vectors = np.array([[np.cos(i * phi), np.sin(i * phi), 0]
                                 for i in range(num_headings)])  # Set of heading vectors
    yaw_angles = np.zeros((dcms.shape[0],))  # Initialize array to store yaw angles

    for i in range(dcms.shape[0]):
        # Rotate heading vectors into earth frame using DCM
        rotated_heading_vectors = np.dot(dcms[i], heading_vectors.T).T

        # Find the heading vector with the smallest ze-component
        min_ze_index = np.argmin(rotated_heading_vectors[:, 2])

        # Calculate yaw angle from the chosen heading vector
        yaw_angle = np.degrees(np.arctan2(rotated_heading_vectors[min_ze_index, 1],
                                          rotated_heading_vectors[min_ze_index, 0]))

        # Correct for initial offset n * phi
        yaw_angle -= min_ze_index * np.degrees(phi)

        # Take into account the sign of ze-component for number of turns
        if rotated_heading_vectors[min_ze_index, 2] < 0:
            yaw_angle += 180

        # Adjust yaw angle to be within the range of -180 to 180 degrees
        if yaw_angle < -180:
            yaw_angle += 360
        elif yaw_angle > 180:
            yaw_angle -= 360

        # Store the calculated yaw angle in the result array
        yaw_angles[i] = yaw_angle

    return yaw_angles


def segment_data_gm(data, window_size=2, overlap=0.75, sampling_rate=50):
    # Calculate the number of samples in a window
    window_samples = int(window_size * sampling_rate)
    
    # Calculate the number of overlapping samples
    overlap_samples = int(window_samples * overlap)
    
    # Calculate the number of windows
    num_windows = int(np.ceil(len(data) / (window_samples - overlap_samples)))
    
    # Pad the data with NaN values if necessary
    padding_size = (window_samples - len(data) % window_samples) % window_samples
    padded_data = np.concatenate([data, np.full(padding_size, np.nan)])
    
    # Create a list to hold the segmented data arrays
    segmented_data_list = []
    
    # Fill in the segmented data array with the sliding window approach
    for i in range(num_windows):
        start_index = i * (window_samples - overlap_samples)
        end_index = start_index + window_samples
        segmented_data_list.append(padded_data[start_index:end_index])
    
    # Convert the list of arrays to a numpy array of arrays
    segmented_data = np.array(segmented_data_list)
    
    return segmented_data


def compute_GM(theta, psi): 
    
    epoch_size_leuenberger = 2 #s
    overlap_leuenberger = 0.75 #75% 
    resample_gm = 1 #Hz
   
    #epoch of 2s 75%overlapping 
    segmented_theta = segment_data_gm(theta)
    segmented_yaw = segment_data_gm(psi)
    
    # Compute the number of epochs
    num_epochs = segmented_theta.shape[0]
    
    # Initialize an empty array to hold the GM for each epoch
    gm_epochs = np.zeros(num_epochs)
    
    for i in range(num_epochs):
        
        # Extract the theta and yaw epochs
        epoch_thetas = segmented_theta[i]
        epoch_psis = segmented_yaw[i]
        
        # Compute the overall absolute change in yaw and pitch angles inside the epoch
        delta_psi = np.abs(np.nanmax(epoch_psis) - np.nanmin(epoch_psis))
        delta_theta = np.abs(np.nanmax(epoch_thetas) - np.nanmin(epoch_thetas))

        # Compute the absolute pitch of the forearm
        pitch = np.nanmean(epoch_thetas)
        
        # Check if the criteria for GM are met
        if ((delta_psi + delta_theta) > 30) and (np.abs(pitch) < 30):
            gm_epochs[i] = 1
    return gm_epochs


def get_gm_per_epoch(gm_scores, epoch_size):
    sampling_freq = 2  # samples per second

    # Calculate the number of samples per epoch
    samples_per_epoch = epoch_size * sampling_freq

    # Calculate the number of epochs
    n_epochs = len(gm_scores) // samples_per_epoch

    # Reshape the array into epochs
    epochs = gm_scores[:n_epochs * samples_per_epoch].reshape(n_epochs, samples_per_epoch)
    
    # Sum the GM scores for each epoch
    gm_per_epoch = np.sum(epochs, axis=1)
    
    return gm_per_epoch


def plot_gm_epoch(gm, total_time, events=None):
    
    # Calculate the time interval between each epoch
    epoch_time = total_time / len(gm)

    # Determine the number of seconds per tick label
    sec_per_tick = round((total_time/10) / 3600) * 3600

    # Calculate the number of ticks and tick positions
    num_ticks = int(np.ceil(total_time / sec_per_tick))
    tick_positions = np.arange(num_ticks) * sec_per_tick

    # Set up the figure with a larger plot size
    fig, ax = plt.subplots(figsize=(12, 6)) 

    # Create the bar plot
    ax.bar(x=range(len(gm)), height=gm, width=3, align='edge')

    # Set the x-axis ticks and labels to be at the center of each epoch
    ax.set_xticks([i + 0.5 for i in range(len(gm))])
    ax.set_xticklabels([f'{i+1}' for i in range(len(gm))])

    # Set the x-axis tick positions and labels with hours modulo 24
    ax.set_xticks(tick_positions / epoch_time)
    ax.set_xticklabels([f'{int((tick_pos/3600 + 10) % 24)}:00' for tick_pos in tick_positions])
    

    # Add the events to the plot
    if events:
        for i, (event_time, event_label) in enumerate(events):
            event_pos = event_time / epoch_time
            ax.axvline(x=event_pos, linestyle='--', color='black', linewidth=1)
            bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
            y_pos = 150 + i*15
            ax.text(event_pos, y_pos, event_label, fontsize=10, ha='center', va='center', bbox=bbox_props)

    # Set the x-axis label and the y-axis label
    ax.set_xlabel('Time')
    ax.set_ylabel('GM score')

    # Set the plot title
    ax.set_title('GM Score per Epoch')

    # Show the plot
    plt.show()


def plot_gm_superposed(gm1, gm2, total_time, events=None):
    """
    Plot two sets of GM scores over time.

    Args:
        gm1 (array-like): Array of GM scores for first set of data.
        gm2 (array-like): Array of GM scores for second set of data.
        total_time (float): Total recording time in seconds.
        events (list of tuples, optional): List of events with the format [(event_time, event_label), ...].
    """
    
    font_size = 12 
    
    # Calculate the time interval between each epoch
    epoch_time = total_time / len(gm1)

    # Determine the number of seconds per tick label
    sec_per_tick = round((total_time/10) / 3600) * 3600

    # Calculate the number of ticks and tick positions
    num_ticks = int(np.ceil(total_time / sec_per_tick))
    tick_positions = np.arange(num_ticks) * sec_per_tick

    # Set up the figure with a larger plot size
    fig, ax = plt.subplots(figsize=(16, 8))  # Set figsize as desired

    # Create the bar plots with transparency
    ax.bar(x=range(len(gm1)), height=gm1, width=3, align='edge', alpha=0.4, color='red', label='Left wrist')
    ax.bar(x=range(len(gm2)), height=gm2, width=3, align='edge', alpha=0.4, color='blue', label='Right wrist')

    # Set the x-axis ticks and labels to be at the center of each epoch
    ax.set_xticks([i + 0.5 for i in range(len(gm1))])
    ax.set_xticklabels([f'{i+1}' for i in range(len(gm1))])

    # Set the x-axis tick positions and labels with hours modulo 24
    ax.set_xticks(tick_positions / epoch_time)
    ax.set_xticklabels([f'{int((tick_pos/3600 + 10) % 24)}:00' for tick_pos in tick_positions],  fontsize=font_size)
    
    ax.set_yticklabels([f'{tick_label}' for tick_label in ax.get_yticks()], fontsize=font_size)
    

    # Add the events to the plot
    if events:
        for i, (event_time, event_label) in enumerate(events):
            event_pos = event_time / epoch_time
            ax.axvline(x=event_pos, linestyle='--', color='black', linewidth=1)
            bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
            y_pos = 100 + i*25 + 50 # Random y value between 200 and 450
            ax.text(event_pos, y_pos, event_label, fontsize=10, ha='center', va='center', bbox=bbox_props)

    # Set the x-axis label and the y-axis label
    ax.set_xlabel('Time',fontsize=font_size)
    ax.set_ylabel('GM score',fontsize=font_size)

    # Add a legend on the right side of the figure
    ax.legend(loc='upper right')

    # Show the plot
    plt.show()


def compute_GMAC(theta, accel_leuenberger, recording_time, sampling_frequency_UL, resampling_frequency_leuenberger, threshold):
    
    epoch_size = 1 #s
    
    #resample theta at 50Hz
    res_theta_LW = resample_data_cubic_spline(theta,sampling_frequency_UL,resampling_frequency_leuenberger)
    
    #compute window of 1 seconds 
    theta_per_epoch = window_data(res_theta_LW, sampling_frequency_UL, epoch_size)
    
    #number of epochs
    num_epoch = len(theta_per_epoch)
  
    # Initialize an array to store the mean pitch per epoch
    theta_mean_per_epoch = np.zeros(num_epoch)
    # Iterate over each epoch and compute the mean pitch angle
    for i in range(num_epoch):
        theta_mean_per_epoch[i] = np.mean(theta_per_epoch[i])
    
    #Compute AC count per 1s epoch
    AC_per_epoch = calculate_activity_counts(accel_leuenberger,epoch_size, resampling_frequency_leuenberger, threshold)
    
    # Initialize an array to store GMAC score 
    GMAC = np.zeros(num_epoch)
    
    for i in range(num_epoch):
        if ((np.abs(theta_mean_per_epoch[i]) < 30) and (AC_per_epoch[i] > 0)):
            GMAC[i] = 1
    return GMAC  


def compute_gmac_per_epoch(gmac_scores, epoch_size):
    
    sampling_freq = 1  # samples per second

    # Calculate the number of samples per epoch
    samples_per_epoch = epoch_size * sampling_freq

    # Calculate the number of epochs
    n_epochs = len(gmac_scores) // samples_per_epoch

    # Reshape the array into epochs
    epochs = gmac_scores[:n_epochs * samples_per_epoch].reshape(n_epochs, samples_per_epoch)
    
    gmac_per_epoch = np.zeros(n_epochs)
    
    #sum all the elements per epoch
    for i in range(n_epochs): 
        gmac_per_epoch[i] = np.sum(epochs[i])
        
    return gmac_per_epoch


def plot_gmac_epoch(gmac_scores, total_time, events=None):
    """
    Plot GMAC scores over time.

    Args:
        gmac_scores (array-like): Array of GMAC scores for each epoch.
        total_time (float): Total recording time in seconds.
    """
    
    # Calculate the time interval between each epoch
    epoch_time = total_time / len(gmac_scores)

    # Determine the number of seconds per tick label
    sec_per_tick = round((total_time/10) / 3600) * 3600

    # Calculate the number of ticks and tick positions
    num_ticks = int(np.ceil(total_time / sec_per_tick))
    tick_positions = np.arange(num_ticks) * sec_per_tick

    # Set up the figure with a larger plot size
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create the bar plot
    ax.bar(x=range(len(gmac_scores)), height=gmac_scores, width=3, align='edge')

    # Set the x-axis ticks and labels to be at the center of each epoch
    ax.set_xticks([i + 0.5 for i in range(len(gmac_scores))])
    ax.set_xticklabels([f'{i+1}' for i in range(len(gmac_scores))])

    # Set the x-axis tick positions and labels with hours modulo 24
    ax.set_xticks(tick_positions / epoch_time)
    ax.set_xticklabels([f'{int((tick_pos/3600 + 10) % 24)}:00' for tick_pos in tick_positions])
    
    # Add the events to the plot
    if events:
        for i, (event_time, event_label) in enumerate(events):
            event_pos = event_time / epoch_time
            ax.axvline(x=event_pos, linestyle='--', color='black', linewidth=1)
            bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
            y_pos = 100 + i*25 + 50 # Random y value between 200 and 450
            ax.text(event_pos, y_pos, event_label, fontsize=10, ha='center', va='center', bbox=bbox_props)

    # Set the x-axis label and the y-axis label
    ax.set_xlabel('Time')
    ax.set_ylabel('GMAC score')

    # Set the plot title
    ax.set_title('')

    # Show the plot
    plt.show()


def plot_gmac_superposed(gmac1, gmac2, total_time, events=None):
    """
    Plot two sets of GMAC scores over time.

    Args:
        gmac1 (array-like): Array of GMAC scores for first set of data.
        gmac2 (array-like): Array of GMAC scores for second set of data.
        total_time (float): Total recording time in seconds.
        events (list of tuples): List of events with the format [(event_time, event_label), ...].
    """
    # Calculate the time interval between each epoch
    epoch_time = total_time / len(gmac1)

    # Determine the number of seconds per tick label
    sec_per_tick = round((total_time / 10) / 3600) * 3600

    # Calculate the number of ticks and tick positions
    num_ticks = int(np.ceil(total_time / sec_per_tick))
    tick_positions = np.arange(num_ticks) * sec_per_tick

    # Set up the figure with a larger plot size
    fig, ax = plt.subplots(figsize=(16, 8))

    # Create the bar plots with transparency
    ax.bar(x=range(len(gmac1)), height=gmac1, width=3, align='edge', alpha=0.4, color='red', label='Left wrist')
    ax.bar(x=range(len(gmac2)), height=gmac2, width=3, align='edge', alpha=0.4, color='blue', label='Right wrist')

    # Set the x-axis ticks and labels to be at the center of each epoch
    ax.set_xticks([i + 0.5 for i in range(len(gmac1))])
    ax.set_xticklabels([f'{i+1}' for i in range(len(gmac1))])

    # Set the x-axis tick positions and labels with hours modulo 24
    ax.set_xticks(tick_positions / epoch_time)
    ax.set_xticklabels([f'{int((tick_pos/3600 + 10) % 24)}:00' for tick_pos in tick_positions])

    # Add the punctual events to the plot
    if events:
        for event_time, event_label in events:
            event_pos = event_time / epoch_time
            ax.scatter(x=event_pos, y=0, marker='x', s=100, color='black')
            ax.text(event_pos, 30, event_label, fontsize=16, ha='center', va='top')

    # Set the x-axis and y-axis labels
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('GMAC', fontsize=12)

    # Add a legend on the right side of the figure
    ax.legend(loc='upper right')

    # Show the plot
    plt.show()


def get_acceleration_per_second_laterality(accel_leuenberger_LW, accel_leuenberger_RW, threshold,
 resampling_frequency_leuenberger, resampling_frequency_laterality):
    
    # Reshape to match the resampling function
    accel_leuenberger_LW = accel_leuenberger_LW.reshape(-1, 1)
    accel_leuenberger_RW = accel_leuenberger_RW.reshape(-1, 1)
    
    # Resample at 1Hz (initially at 50Hz after resampling)
    LW_accel_res = resample_data_cubic_spline(accel_leuenberger_LW, resampling_frequency_leuenberger, resampling_frequency_laterality)
    RW_accel_res = resample_data_cubic_spline(accel_leuenberger_RW, resampling_frequency_leuenberger, resampling_frequency_laterality)
    
    # Check if it's enough to be an active epoch using the threshold 
    # If the acceleration magnitude for the 1 second epoch is less than the threshold defined for activity,
    # set it to 0.0 to remove low excitation
    LW_accel_res[LW_accel_res < convert_g_to_ms2(threshold)] = 0
    RW_accel_res[RW_accel_res < convert_g_to_ms2(threshold)] = 0
    
    LW_accel_res_count = np.zeros(len(LW_accel_res))
    LW_accel_res_count = LW_accel_res_count.reshape(-1)
    RW_accel_res_count = np.zeros(len(RW_accel_res))
    RW_accel_res_count = RW_accel_res_count.reshape(-1)
    
    for i in range(len(LW_accel_res_count)): 
        LW_accel_res_count[i] = int(LW_accel_res[i] / convert_g_to_ms2(threshold))
        RW_accel_res_count[i] = int(RW_accel_res[i] / convert_g_to_ms2(threshold))
    
    return LW_accel_res_count, RW_accel_res_count


def get_bilateral_magnitude(accel_leuenberger_LW,accel_leuenberger_RW, threshold,
    resampling_frequency_leuenberger, resampling_frequency_laterality):
    
    LW_accel_res, RW_accel_res = get_acceleration_per_second_laterality(accel_leuenberger_LW,accel_leuenberger_RW, threshold,
     resampling_frequency_leuenberger, resampling_frequency_laterality)
    
    bilateral_magnitude = np.zeros(len(LW_accel_res))
    #sum the left and right epochs for each epoch
    for i in range(len(LW_accel_res)):
        bilateral_magnitude[i] = LW_accel_res[i]+RW_accel_res[i]
        
    return bilateral_magnitude


def get_magnitude_ratio(accel_leuenberger_LW, accel_leuenberger_RW, BM, threshold,
    resampling_frequency_leuenberger, resampling_frequency_laterality):
    
    LW_accel_res, RW_accel_res = get_acceleration_per_second_laterality(accel_leuenberger_LW, accel_leuenberger_RW, threshold,
     resampling_frequency_leuenberger, resampling_frequency_laterality)

    # Create masks
    bm_nonzero_mask = (BM != 0.0)
    lw_zero_mask = (LW_accel_res == 0.0)
    rw_zero_mask = (RW_accel_res == 0.0)
    
    # Compute the vector magnitude ratio for each second for valid division values
    mag_ratio = np.divide(LW_accel_res, RW_accel_res, out=np.ones_like(LW_accel_res), where=(RW_accel_res != 0) & (LW_accel_res != 0))
    
    # Transform the ratio values using a natural logarithm only for the good candidates
    mag_ratio_log = np.log(mag_ratio)
    
    # Handle cases with LW or RW = 0
    mag_ratio_log[lw_zero_mask] = -7 #left is at 0 ie full use of the dominant ie left side 
    mag_ratio_log[rw_zero_mask] = 7
    
    # Handle cases with BM=0 and with 0/0 ratio
    mag_ratio_log[~bm_nonzero_mask] = np.nan
    mag_ratio_log[lw_zero_mask & rw_zero_mask] = np.nan
    
    return mag_ratio_log


def plot_distribution_BM(data):
    plt.hist(data, bins=10, color='skyblue', range=(0, 25)) # Set x-axis range from -5 to 5
    plt.xlabel('Bilateral Magnitude [a.u.]')
    plt.ylabel('Frequency')
    plt.title('Distribution of the Bilateral Magnitude')
    plt.show()


def plot_distribution_ratio(data):
    data = np.array(data)
    data = data[~np.isnan(data)] # remove NaN values

    plt.hist(data, bins=15, color='#1f77b4')
    plt.xlabel('Ratio [a.u.]')
    plt.ylabel('Frequency')
    plt.text(0.1, -0.15, 'Dominant side', transform=plt.gca().transAxes, horizontalalignment='center')
    plt.text(0.98, -0.15, 'Non Dominant Side', transform=plt.gca().transAxes, horizontalalignment='center', ha='right')
    
    # Add vertical line for median
    median_val = np.nanmedian(data)
    plt.axvline(median_val, linestyle='--', color='red', label='Median')
    plt.legend(loc='upper center')
    
    plt.show()


def get_tendency_ratio(ratio_array):
    """Calculate the percentage of values in an array of ratios between -7 and 0 (excluded),
       and between 0 (excluded) and +7. NaN values are excluded from the analysis.

    Args:
        ratio_array (array-like): Array of ratio values between -7 and 7.

    Returns:
        tuple: A tuple containing the percentage of values between -7 and 0 excluded, 
               and the percentage of values between 0 excluded and +7.
    """
    ratio_array = np.array(ratio_array)
    ratio_array = ratio_array[~np.isnan(ratio_array)] # remove NaN values
    
    neg_count = 0
    pos_count = 0
    total_count = len(ratio_array)
    
    for ratio in ratio_array:
        if ratio > 0:
            pos_count += 1
        elif ratio < 0:
            neg_count += 1
    
    neg_pct = round(neg_count / total_count * 100, 2)
    pos_pct = round(pos_count / total_count * 100, 2)
    zer_pct = round(100 - neg_pct - pos_pct, 2)
    
    return (neg_pct, pos_pct, zer_pct)


def plot_ratio_tendency(ratio_array):
    """
    Plot the percentage of values in an array of ratios between -7 and 0 (excluded),
    and between 0 (excluded) and +7. NaN values are excluded from the analysis.
    
    Args:
        ratio_array (array-like): Array of ratio values between -7 and 7.
        
    Returns:
        None
    """
    # Remove NaN values
    ratio_array = np.array(ratio_array)
    ratio_array = ratio_array[~np.isnan(ratio_array)]
    
    # Calculate the percentages
    neg_pct, pos_pct, zer_pct = get_tendency_ratio(ratio_array)
    
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(["Right UL Domination", "Left UL domination", "Symmetrical use"], [neg_pct, pos_pct, zer_pct], color=['#1f77b4', '#ff7f0e', '#2ca02c'], width=0.4)

    # Format the percentage values and display them on top of the bars
    for i, pct in enumerate([neg_pct, pos_pct, zer_pct]):
        ax.text(i, pct+2, f"{round(pct)}%", ha='center', fontsize=12)

    # Set the y axis limit to 60%
    ax.set_ylim([0, 60])
    
    ax.set_ylabel("Percentage")
    plt.show()


def plot_density(BM, ratio):
    # Define the range of the ratio values
    ratio_range = (-8, 8)
    
    ratio = np.squeeze(ratio)
    # Remove NaN values from the ratio and BM arrays
    nan_mask = np.isnan(ratio)
    BM = BM[~nan_mask]
    ratio = ratio[~nan_mask]
    
    # Determine the duration of each ratio value
    ratio_duration = np.zeros(len(ratio))
    unique_ratios = np.unique(ratio)
    for r in unique_ratios:
        mask = (ratio == r)
        ratio_duration[mask] = np.sum(mask)
    max_duration = np.max(ratio_duration)
    
    # Create the colormap for the density plot
    cmap = ListedColormap(plt.cm.get_cmap('bwr')(np.linspace(0, 1, 256)))
    
    # Create the density plot
    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.scatter(ratio, BM, c=ratio_duration, cmap=cmap, edgecolors='none', alpha=0.75)
    ax.set_xlim(ratio_range)
    ax.set_xlabel('Ratio')
    ax.set_ylabel('Bilateral Magnitude')
    #ax.set_title('Density Plot of Ratio and Bilateral Magnitude')
    plt.text(0.1, -0.15, 'Dominant side', transform=plt.gca().transAxes, horizontalalignment='center')
    plt.text(0.98, -0.15, 'Non Dominant Side', transform=plt.gca().transAxes, horizontalalignment='center', ha='right')
    
    # Create the color bar
    cbar = fig.colorbar(img)
    cbar.ax.set_ylabel('Duration (s)', rotation=270, labelpad=15)
    cbar.set_ticks(np.linspace(0, max_duration, 5, dtype=int))
    
    plt.show()