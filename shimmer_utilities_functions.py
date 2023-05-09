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

def convert_g_to_ms2(g_data):
    """
    Convert acceleration data in units of "g" to units of "m/s^2".
    
    Parameters:
    g_data (list or numpy array): A list or array of acceleration data in units of "g".
    
    Returns:
    An array of acceleration data in units of "m/s^2".
    """
    # Convert g to m/s^2 using the conversion factor of 9.81 m/s^2 per g
    ms2_data = g_data * 9.81
    return ms2_data


def convert_ms2_to_g(ms2_data):
    """
    Convert acceleration data in units of "m/s^2" to units of "g".
    
    Parameters:
    ms2_data (list or numpy array): A list or array of acceleration data in units of "m/s^2".
    
    Returns:
    An array of acceleration data in units of "g".
    """
    # Convert m/s^2 to g using the conversion factor of 1 g per 9.81 m/s^2
    g_data = ms2_data / 9.81 
    return g_data


def get_recording_time(timestamps_array):
    """
    Calculate the total recording time from an array of timestamps.

    Parameters:
    -----------
    timestamps_array : numpy.ndarray
        Array of timestamps in datetime format.

    Returns:
    --------
    float
        Total recording time in seconds.
    """
    min_time = np.min(timestamps_array)
    max_time = np.max(timestamps_array)
    total_time = (max_time - min_time).total_seconds()
    return total_time


def get_statistics(data):
    """
    Calculate various statistics for a given array of data.
    
    Parameters:
    data (list or numpy array): An array of data to compute statistics for.
    
    Returns:
    A dictionary containing the computed statistics.
    """
    statistics = {}
    statistics['mean'] = np.nanmean(data)
    statistics['median'] = np.nanmedian(data)
    statistics['iqr'] = np.nanpercentile(data, 75) - np.nanpercentile(data, 25)
    statistics['range'] = np.nanmax(data) - np.nanmin(data)
    statistics['std'] = np.nanstd(data)
    statistics['max'] = np.nanmax(data)
    statistics['min'] = np.nanmin(data)
    statistics['num_elements'] = len(data)
    return statistics


def get_sampling_frequency(timestamp_array):
    recording_time = get_recording_time(timestamp_array)
    samples_number = timestamp_array.shape[0]
    sampling_frequency = samples_number / recording_time
    return sampling_frequency


def preprocessing_leuenberger(raw_9DOF_data, sampling_frequency, resampling_frequency_leuenberger):
    """
    This function preprocesses 9-DOF (degrees of freedom) sensor data collected by the Leuenberger
    IMU sensor. The preprocessing consists of resampling the data to 50Hz, removing gravity from the
    acceleration data, and concatenating the processed data into a single array.
    
    Parameters:
    raw_9DOF_data (numpy array): The raw 9-DOF sensor data.
    sampling_frequency (float): The sampling frequency of the raw data.
    
    Returns:
    A numpy array of preprocessed sensor data.
    """
    # Step 1: Resample the data to 50Hz using cubic spline interpolation
    resampled_leuenberger_data = resample_data_cubic_spline(raw_9DOF_data, sampling_frequency, resampling_frequency_leuenberger)
    
    # Get the resampled 9 DOF data
    acc = resampled_leuenberger_data[:, 0:3]
    gyro = resampled_leuenberger_data[:, 3:6]
    mag = resampled_leuenberger_data[:, 6:9]
    
    # Step 2: Remove gravity from the acceleration data
    filtered_accel_data = filter_gravity(acc, resampling_frequency_leuenberger)
    
    # Get the full preprocessed data
    filtered_leuenberger_data = np.hstack((filtered_accel_data, gyro, mag))
    
    return filtered_leuenberger_data

def filter_gravity(raw_accel_data, fs):
    """
    This function removes the gravity component from accelerometer data using a lowpass filter.
    
    Parameters:
    raw_accel_data (numpy array): Array of raw accelerometer data of shape (num_samples, 3) for x, y, and z axes.
    fs (float): Sampling frequency of the data.
    
    Returns:
    filtered_accel_data (numpy array): Array of filtered accelerometer data with gravity component removed.
    """
    # Lowpass filter the acceleration data along all axes with a cutoff frequency of 0.25 Hz
    cutoff_freq = 0.25  # Hz
    nyquist_freq = 0.5 * fs
    norm_cutoff_freq = cutoff_freq / nyquist_freq
    b, a = butter(1, norm_cutoff_freq, btype='lowpass')
    
    # Apply the filter forwards and backwards to avoid phase distortion
    # using filtfilt (+phase shift then - phase shift)
    x_gravity_offset = filtfilt(b, a, raw_accel_data[:, 0])
    y_gravity_offset = filtfilt(b, a, raw_accel_data[:, 1])
    z_gravity_offset = filtfilt(b, a, raw_accel_data[:, 2])
    
    # Subtract the gravity component to get the dynamic acceleration
    x_filtered = raw_accel_data[:, 0] - x_gravity_offset
    y_filtered = raw_accel_data[:, 1] - y_gravity_offset
    z_filtered = raw_accel_data[:, 2] - z_gravity_offset
    
    # Copy the raw data and replace the gravity component with the filtered dynamic acceleration
    filtered_accel_data = np.copy(raw_accel_data)
    filtered_accel_data[:, 0] = x_filtered
    filtered_accel_data[:, 1] = y_filtered
    filtered_accel_data[:, 2] = z_filtered
    
    return filtered_accel_data

def resample_data_cubic_spline(raw_data, fs, fdesired):
    """
    Resamples the given data to the desired frequency using cubic spline interpolation.

    Parameters:
    raw_data (ndarray): The raw data to resample. Should have shape (num_samples, num_channels).
    fs (float): The sampling frequency of the raw data.
    fdesired (float): The desired resampling frequency.

    Returns:
    ndarray: The resampled data with shape (num_resampled_samples, num_channels).
    """

    # Reshape the input array if it has shape (n_samples,)
    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(-1, 1)

    # Calculate the resampling factor
    resampling_factor = fs / fdesired

    # Define the time points for the original signal
    time_points = np.arange(raw_data.shape[0]) / fs

    # Define the time points for the resampled signal
    resampled_time_points = np.arange(0, time_points[-1], 1 / fdesired)

    # Initialize an empty array for the resampled data
    resampled_data = np.zeros((len(resampled_time_points), raw_data.shape[1]))

    # Loop over each column of the data and resample using cubic spline interpolation
    for i in range(raw_data.shape[1]):
        # Create a cubic spline interpolator object for this column
        interpolator = interpolate.interp1d(time_points, raw_data[:, i], kind='cubic')

        # Evaluate the interpolator at the resampled time points
        resampled_data[:, i] = interpolator(resampled_time_points)

    return resampled_data


def apply_median_filter(data):
    """
    Apply fifth-order median filter to remove sharp edges and outliers from input data.
    
    Parameters:
    -- data (np.ndarray): Input data of any shape.
    
    Returns:
    -- filtered_data (np.ndarray): Filtered data of the same shape as input data.
    """
    # Ensure input data is a numpy array
    data = np.array(data)
    
    # Get number of columns in input data
    num_columns = data.shape[-1]
    
    # Apply median filter along each column separately
    filtered_data = np.zeros_like(data)
    for i in range(num_columns):
        filtered_data[..., i] = medfilt(data[..., i], kernel_size=5)
    
    return filtered_data

def get_accel_mag(filtered_accel_data, threshold):
    """
    Calculates the magnitude of the acceleration vector for each sample in the filtered raw accelerometer data.

    Args:
        filtered_accel_data (np.ndarray): NumPy array of shape `(num_samples, 3)` containing the filtered raw accelerometer data
                                           in units of g (i.e., acceleration due to gravity).
        threshold (float): Threshold value in g for minimum acceleration magnitude required for activity.

    Returns:
        np.ndarray: Magnitude of the acceleration vector for each sample. If the acceleration magnitude is less than the
                    threshold defined for activity, it is set to 0.0 to remove low excitation.
    """
    # Calculate the magnitude of the acceleration vector for each sample
    accel_mag = np.linalg.norm(filtered_accel_data, axis=1)
    
    # If the acceleration magnitude is less than the threshold defined for activity, set it to 0.0 to remove low excitation
    accel_mag[accel_mag < convert_g_to_ms2(threshold)] = 0.0

    return accel_mag


def plot_acceleration_over_time(acceleration, recording_period):
    """
    Plot the magnitude of acceleration over time.

    Args:
        acceleration (array-like): Array of acceleration values in g.
        recording_period (float): Total recording period in seconds.
    """
    # Create time array
    time = np.linspace(0, recording_period, len(acceleration))

    # Create the plot
    plt.plot(time, acceleration)
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (g)')
    plt.title('Acceleration vs Time')
    plt.grid(True)
    plt.show()


def get_active_percentage(values):
    """Calculate the percentage of values in the array that are greater than 0."""
    active_count = np.sum(values > 0)
    return (active_count / len(values)) * 100 if np.any(values) else 0.0


def window_data(data, sampling_frequency, window_size, overlap=0):
    """
    Window the input data into epochs of a specified size with optional overlap.
    
    Parameters:
        data (ndarray): Input data as a numpy array with shape (n_samples,).
        window_size (float): Size of the window in seconds.
        overlap (float, optional): Proportion of overlap between windows (between 0 and 1).
    
    Returns:
        ndarray: Windowed data as a numpy array with shape (n_epochs, window_size * sampling_freq).
    """
    
    # Calculate the number of samples per window
    window_samples = int(window_size * sampling_frequency)
    
    # Calculate the number of overlapping samples
    overlap_samples = int(window_samples * overlap)
    
    # Calculate the total number of epochs
    n_epochs = int(np.floor((data.shape[0] - window_samples) / (window_samples - overlap_samples)) + 1)
    
    # Truncate the input data to ensure an even number of epochs
    #not the best solutin because we are losing some data
    data = data[:n_epochs * (window_samples - overlap_samples) + window_samples]
    
    # Reshape the data into epochs with overlap
    windowed_data = np.lib.stride_tricks.as_strided(
        data,
        shape=(n_epochs, window_samples),
        strides=((window_samples - overlap_samples) * data.itemsize, data.itemsize)
    )
    
    return windowed_data


def get_normalized_acceleration_histogram(acceleration_data):
    # Convert acceleration from m/s^2 to g
    acceleration_g = acceleration_data / 9.8  # Acceleration due to gravity in m/s^2

    # Define histogram bins and compute histogram
    bins = np.arange(0, 9, 1)  # Bins from 0 to 8 g
    histogram, _ = np.histogram(acceleration_g, bins=bins)

    # Normalize histogram
    normalized_histogram = histogram / np.sum(histogram)

    # Compute mean of the array
    mean_acceleration = np.mean(acceleration_g)

    # Plot normalized histogram
    fig, axis = plt.subplots()
    axis.bar(bins[:-1], normalized_histogram, width=0.8)
    axis.axvline(x=mean_acceleration, color='red', linestyle='dashed', label='Mean')
    axis.set_xlabel('Acceleration [g]')
    axis.set_ylabel('Normalized probability')
    axis.set_title('Normalized histogram of acceleration magnitudes')

    # Set x-axis tick labels smaller
    axis.tick_params(axis='x', labelsize=8)

    axis.legend()
    plt.show()


def compute_quaternions_madgwick(filtered_data):
    """
    Compute quaternions using the Madgwick algorithm for MARG sensor fusion.

    Args:
        filtered_data: A 2D numpy array of size n x 9, where n is the number of samples.
            The first three columns represent the accelerometer data, the next three
            represent the gyroscope data, and the last three represent the magnetometer data.

    Returns:
        A 1D numpy array of size 4 representing the quaternion values.
    """
    # Extract the accelerometer, gyroscope and magnetometer data from the filtered data
    acc_data = filtered_data[:, :3].astype(float)
    gyro_data = np.radians(filtered_data[:, 3:6].astype(float))
    mag_data = (filtered_data[:, 6:] * 100).astype(float)

    # Compute quaternions using Madgwick algorithm
    madgwick = Madgwick(gyr=gyro_data, acc=acc_data, mag=mag_data)
    q = madgwick.Q

    return q


def compute_dcms(quaternions):
    """
    Computes direction cosine matrices (DCMs) for each orientation from a quaternion array.

    Args:
        quaternions (numpy.ndarray): Array of quaternions of shape (n_samples, 4).

    Returns:
        numpy.ndarray: Array of DCMs of shape (n_samples, 3, 3).
    """
    n_samples = quaternions.shape[0]
    dcms = np.zeros((n_samples, 3, 3))

    for i in range(n_samples):
        q0, q1, q2, q3 = quaternions[i, :]
        dcms[i, 0, 0] = q0**2 + q1**2 - q2**2 - q3**2
        dcms[i, 0, 1] = 2 * (q1*q2 - q0*q3)
        dcms[i, 0, 2] = 2 * (q0*q2 + q1*q3)
        dcms[i, 1, 0] = 2 * (q1*q2 + q0*q3)
        dcms[i, 1, 1] = q0**2 - q1**2 + q2**2 - q3**2
        dcms[i, 1, 2] = 2 * (q2*q3 - q0*q1)
        dcms[i, 2, 0] = 2 * (q1*q3 - q0*q2)
        dcms[i, 2, 1] = 2 * (q0*q1 + q2*q3)
        dcms[i, 2, 2] = q0**2 - q1**2 - q2**2 + q3**2

    return dcms


def plot_angle(angle_deg, total_time):
    num_epochs = len(angle_deg)
    epoch_duration = total_time / num_epochs
    time_array = np.arange(epoch_duration / 2, total_time, epoch_duration)

    plt.plot(time_array, angle_deg)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.show()


def get_normalized_histogram(elevations):
    # Define histogram bins and compute histogram
    bins = np.arange(-90, 91, 1)
    hist, _ = np.histogram(elevations, bins=bins)

    # Normalize histogram
    norm_hist = hist / np.sum(hist)

    # Compute mean of the array
    mean_elevation = np.mean(elevations)

    # Plot normalized histogram
    fig, ax = plt.subplots()
    ax.bar(bins[:-1], norm_hist, width=3)
    ax.axvline(x=mean_elevation, color='red', linestyle='dashed', label='Mean')
    ax.set_xlabel('Elevation [degrees]')
    ax.set_ylabel('Normalized probability')
    ax.set_title('Normalized histogram of forearm elevation angle')
    ax.legend()
    plt.show()

def get_normalized_histogram_yaw(headings):
    # Define histogram bins and compute histogram
    bins = np.arange(-180, 181, 1)
    hist, _ = np.histogram(headings, bins=bins)

    # Normalize histogram
    norm_hist = hist / np.sum(hist)

    # Compute mean of the array
    mean_heading = np.mean(headings)

    # Plot normalized histogram
    fig, ax = plt.subplots()
    ax.bar(bins[:-1], norm_hist, width=6)
    ax.axvline(x=mean_heading, color='red', linestyle='dashed', label='Mean')
    ax.set_xlabel('Heading [degrees]')
    ax.set_ylabel('Normalized probability')
    ax.set_title('Normalized histogram of forearm heading angle')
    ax.legend()
    plt.show()


def get_normalized_histogram_superpose_elevation(elevations1, elevations2):
    # Define histogram bins and compute histogram
    bins = np.arange(-90, 91, 1)
    hist1, _ = np.histogram(elevations1, bins=bins)
    hist2, _ = np.histogram(elevations2, bins=bins)

    # Normalize histograms
    norm_hist1 = hist1 / np.sum(hist1)
    norm_hist2 = hist2 / np.sum(hist2)

    # Compute means of the arrays
    mean_elevation1 = np.mean(elevations1)
    mean_elevation2 = np.mean(elevations2)

    # Plot normalized histograms
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(bins[:-1], norm_hist1, width=3, alpha=0.5, color='red', label='Left wrist')
    ax.bar(bins[:-1], norm_hist2, width=3, alpha=0.5, color='blue', label='Right wrist')
    ax.axvline(x=mean_elevation1, color='red', linestyle='dashed', label='Mean Left wrist')
    ax.axvline(x=mean_elevation2, color='blue', linestyle='dashed', label='Mean Right wrist')
    ax.set_xlabel('Elevation [degrees]', fontsize=12)
    ax.set_ylabel('Normalized probability', fontsize=12)
    ax.legend(loc='upper right')
    plt.show()


def get_normalized_histogram_compare_elevation(elevations1, elevations2):
    
    fontsize = 16

    # Define histogram bins and compute histogram
    bins = np.arange(-90, 91, 1)
    hist1, _ = np.histogram(elevations1, bins=bins)
    hist2, _ = np.histogram(elevations2, bins=bins)

    # Normalize histograms
    norm_hist1 = hist1 / np.sum(hist1)
    norm_hist2 = hist2 / np.sum(hist2)

    # Compute means of the arrays
    mean_elevation1 = np.mean(elevations1)
    mean_elevation2 = np.mean(elevations2)

    # Plot normalized histograms
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(bins[:-1], norm_hist1, width=3, alpha=0.5, color='red', label='Elevation from sensor')
    ax.bar(bins[:-1], norm_hist2, width=3, alpha=0.5, color='blue', label='Elevation using Madgwick')
    ax.axvline(x=mean_elevation1, color='red', linestyle='dashed', label='Sensor')
    ax.axvline(x=mean_elevation2, color='blue', linestyle='dashed', label='Madgwick')
    ax.set_xlabel('Elevation [degrees]',fontsize=fontsize)
    ax.set_ylabel('Normalized probability',fontsize=fontsize)
    #ax.set_title('Normalized histogram of forearm elevation angle')
    ax.legend(loc='upper right')
    ax.tick_params(axis='both', labelsize=fontsize)
    plt.show()


def get_ratio_active_zone(elevation_angles):
    # Initialize counters for angles within active zone and total angles
    angles_within_active_zone = 0
    total_angles = len(elevation_angles)

    # Iterate through each elevation angle
    for angle in elevation_angles:
        # Check if angle is within -30 to +30 degrees range
        if -30 <= angle <= 30:
            angles_within_active_zone += 1

    # Calculate percentage of angles within active zone
    ratio_active_zone = (angles_within_active_zone / total_angles) * 100

    return ratio_active_zone


def plot_array_distribution(arr):
    plt.hist(arr, bins=50)
    plt.title('Distribution of Array Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()


def plot_active_scores_comparison(left_wrist_scores, right_wrist_scores):
    # Create the figure and the axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Set the tick labels
    ax.set_xticks([0.25, 1.25])
    ax.set_xticklabels(['Left wrist', 'Right wrist'])

    # Set the y axis limit to 100%
    ax.set_ylim([0, 50])

    # Define the colors for the different scores
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Plot the scores for the left wrist
    for i, score in enumerate(left_wrist_scores[:3]):
        x = 0 + i * 0.25
        ax.bar(x, score, width=0.2, color=colors[i], label=['AC', 'GMAC', 'GM'][i])
        ax.text(x, score+1, f"{round(score)}%", ha='center', va='bottom')

    # Move the x-axis for the right wrist a bit further
    ax.spines['right'].set_position(('axes', 1))

    # Plot the scores for the right wrist
    for i, score in enumerate(right_wrist_scores[:3]):
        x = 1 + i * 0.25
        ax.bar(x, score, width=0.2, color=colors[i])
        ax.text(x, score+1, f"{round(score)}%", ha='center', va='bottom')

    # Add a legend
    ax.legend(loc='upper right')

    # Add labels for the x and y axes
    ax.set_ylabel('Percentage of Active Score')
    ax.set_xlabel('Wrist')

    # Show the plot
    plt.show()


