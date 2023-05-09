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


def get_synch_data(data_LW, data_RW, data_chest):
    """
    Synchronizes the data from the left, right, and chest sensors based on their timestamps.

    Args:
        data_LW (numpy array): Data from the left wrist sensor
        data_RW (numpy array): Data from the right wrist sensor
        data_chest (numpy array): Data from the chest sensor

    Returns:
        Tuple of synchronized data arrays from left, right, and chest sensors
    """
    
    # Set the auxiliary data to the chest data
    data_AUX = data_chest
    
    # Extract timestamps from all data arrays
    timestamp_LW = data_LW[:, 0]
    timestamp_RW = data_RW[:, 0]
    timestamp_AUX = data_AUX[:, 0]

    # Compute the start and stop times for the data analysis
    start = max(min(timestamp_LW), min(timestamp_RW), min(timestamp_AUX))
    stop = min(max(timestamp_LW), max(timestamp_RW), max(timestamp_AUX))

    # Create boolean masks for the rows that fall within the start and stop times
    mask_LW = (timestamp_LW >= start) & (timestamp_LW <= stop)
    mask_RW = (timestamp_RW >= start) & (timestamp_RW <= stop)
    mask_AUX = (timestamp_AUX >= start) & (timestamp_AUX <= stop)

    # Select only the rows that fall within the start and stop times
    data_LW_selected = data_LW[mask_LW]
    data_RW_selected = data_RW[mask_RW]
    data_AUX_selected = data_AUX[mask_AUX]

    # Ensure all arrays have the same size
    n_LW, n_RW, n_AUX = data_LW_selected.shape[0], data_RW_selected.shape[0], data_AUX_selected.shape[0]
    min_len = min(n_LW, n_RW, n_AUX)
    if n_LW > min_len:
        difference = n_LW - min_len
        data_LW_selected = data_LW_selected[:-difference]
    if n_RW > min_len:
        difference = n_RW - min_len
        data_RW_selected = data_RW_selected[:-difference]
    if n_AUX > min_len:
        difference = n_AUX - min_len
        data_AUX_selected = data_AUX_selected[:-difference]

    # Ensure all arrays have the same number of elements
    n_LW, n_RW, n_AUX = data_LW_selected.shape[0], data_RW_selected.shape[0], data_AUX_selected.shape[0]
    if (n_LW != n_RW) or (n_LW != n_AUX):
        raise ValueError("Arrays have different lengths after truncation")

    # Return the synchronized data arrays
    print('Synchronization done')
    return data_LW_selected, data_RW_selected, data_AUX_selected


def get_raw_data(input_file):
    # Load the data from the input file, skipping the first 3 lines
    data = np.genfromtxt(input_file, skip_header=3, dtype=object, delimiter=',')

    # Delete the last column, which is always empty
    data = data[:, :-1]

    # Convert the first column to string to get the timestamp
    data[:, 0] = data[:, 0].astype(str)

    # Convert all other columns to float
    data[:, 1:] = data[:, 1:].astype(float)

    return data


def extract_from_raw_data(raw_data_IMU):
    """
    Extract the data from the sensor, different number of features can be selected on the Shimmer interface

    """
    
    print('Start data extraction')
    num_columns = raw_data_IMU.shape[1]
    
    if num_columns == 10:

        # Extract timestamp: need to be Timestamp_FormattedUnix_CAL on Consensys
        timestamp = np.array([datetime.strptime(ts, '%Y/%m/%d %H:%M:%S.%f') for ts in raw_data_IMU[:, 0]])
        
        # reshape the one-dimensional array to have the same number of rows as the other arrays
        timestamp = timestamp.reshape((timestamp.shape[0], 1))

        # Extract the wide range accelerometer data (WR data)
        raw_accel_data = raw_data_IMU[:, 1:4]

        # Extract the gyroscope data
        raw_gyro_data = raw_data_IMU[:, 4:7]

        # Extract the magnetometer data
        raw_mag_data = raw_data_IMU[:, 7:10]

        return np.concatenate((timestamp, raw_accel_data, raw_gyro_data, raw_mag_data), axis=1)

    elif num_columns == 13:
        
        # Extract timestamp
        timestamp = np.array([datetime.strptime(ts, '%Y/%m/%d %H:%M:%S.%f') for ts in raw_data_IMU[:, 0]])
        # reshape the one-dimensional array to have the same number of rows as the other arrays
        timestamp = timestamp.reshape((timestamp.shape[0], 1))

        # Extract the wide range accelerometer data (WR data)
        raw_accel_data = raw_data_IMU[:, 4:7]

        # Extract the gyroscope data
        raw_gyro_data = raw_data_IMU[:, 7:10]

        # Extract the magnetometer data
        raw_mag_data = raw_data_IMU[:, 10:13]

        return np.concatenate((timestamp, raw_accel_data, raw_gyro_data, raw_mag_data), axis=1)
    
    elif num_columns == 14:
        
        # Extract timestamp
        timestamp = np.array([datetime.strptime(ts, '%Y/%m/%d %H:%M:%S.%f') for ts in raw_data_IMU[:, 0]])
        
        # reshape the one-dimensional array to have the same number of rows as the other arrays
        timestamp = timestamp.reshape((timestamp.shape[0], 1))

        # Extract the wide range accelerometer data (WR data)
        raw_accel_data = raw_data_IMU[:, 1:4]

        # Extract the gyroscope data
        raw_gyro_data = raw_data_IMU[:, 4:7]

        # Extract the magnetometer data
        raw_mag_data = raw_data_IMU[:, 7:10]
        
        # Extract the orientation
        quaternions = raw_data_IMU[:, 10:14]

        return np.concatenate((timestamp, raw_accel_data, raw_gyro_data, raw_mag_data, quaternions), axis=1)


    elif num_columns == 17:

        # Extract timestamp
        timestamp = np.array([datetime.strptime(ts, '%Y/%m/%d %H:%M:%S.%f') for ts in raw_data_IMU[:, 0]])
        
        # reshape the one-dimensional array to have the same number of rows as the other arrays
        timestamp = timestamp.reshape((timestamp.shape[0], 1))

        # Extract the wide range accelerometer data (WR data)
        raw_accel_data = raw_data_IMU[:, 1:4]
        
        # Extract the Euler angles
        euler_angles = raw_data_IMU[:, 4:7]

        # Extract the gyroscope data
        raw_gyro_data = raw_data_IMU[:, 7:10]

        # Extract the magnetometer data
        raw_mag_data = raw_data_IMU[:, 10:13]

        # Extract the orientation
        quaternions = raw_data_IMU[:, 13:17]

        return np.concatenate((timestamp, raw_accel_data, raw_gyro_data, raw_mag_data, quaternions, euler_angles), axis=1)

    else:
        # Handle invalid number of columns
        print("Invalid number of columns")


def plot_imu_data(data, fs):
    # Separate the raw data into acceleration, gyroscope, and magnetometer components
    accel_data, gyro_data, mag_data = data[:, 0:3], data[:, 3:6], data[:, 6:9]

    # Compute the time vector based on the sampling frequency
    t = np.arange(len(data)) / fs

    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(15, 10))

    # Plot the acceleration data
    ax1.plot(t, accel_data[:, 0], label='X')
    ax1.plot(t, accel_data[:, 1], label='Y')
    ax1.plot(t, accel_data[:, 2], label='Z')
    ax1.set_ylabel('Acceleration (m/s^2)')
    ax1.legend()

    # Plot the gyroscope data
    ax2.plot(t, gyro_data[:, 0], label='X')
    ax2.plot(t, gyro_data[:, 1], label='Y')
    ax2.plot(t, gyro_data[:, 2], label='Z')
    ax2.set_ylabel('Angular velocity (deg/s)')
    ax2.legend()

    # Plot the magnetometer data
    ax3.plot(t, mag_data[:, 0], label='X')
    ax3.plot(t, mag_data[:, 1], label='Y')
    ax3.plot(t, mag_data[:, 2], label='Z')
    ax3.set_ylabel('Magnetic field (Ga)')
    ax3.set_xlabel('Time (s)')
    ax3.legend()

    # Adjust the plot layout
    fig.tight_layout()
    plt.show()

    
def get_timestamp(raw_data_IMU):
    timestamp = raw_data_IMU[:, 0].reshape((raw_data_IMU[:, 0].shape[0], 1))
    return timestamp


def get_acc_WR(raw_data_IMU):
    return raw_data_IMU[:, 1:4]


def get_gyro(raw_data_IMU):
    return raw_data_IMU[:, 4:7]


def get_mag(raw_data_IMU):
    return raw_data_IMU[:, 7:10]


def get_quaternion(raw_data_IMU):
    return raw_data_IMU[:, 10:14]


def get_euler_angles(raw_data_IMU):
    euler_angles = raw_data_IMU[:, 14:17]
    euler_angles = euler_angles.astype(float)
    return euler_angles


def get_9_DOF(raw_data_IMU):
    return np.hstack((get_acc_WR(raw_data_IMU), get_gyro(raw_data_IMU), get_mag(raw_data_IMU)))