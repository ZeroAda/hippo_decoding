import numpy as np
from scipy.io import loadmat
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import umap

from torch_geometric.data import Data
from torch_geometric.data import DataLoader

def read_data(data_path, session_name):
    session_path = data_path.get(session_name)
    with h5py.File(session_path['raw_signal_path'], 'r') as file:
        print("Variables in file:")
        # List all groups (MATLAB variables are stored as groups in HDF5)
        for var_name in file:
            print(var_name)
        data = file[var_name][()]
    
    label_path = data_path.get('public')['label_path']
    sheet_name = session_path['sheet_name']
    label = pd.read_excel(label_path, sheet_name=sheet_name)
        
    return data, label
    
def read_map(json_path):
    with open(json_path, 'r') as file:
        channel_map = json.load(file)
        print(channel_map)
    return channel_map

def normalize(data):
    normalized_data = []
    for i in range(data.shape[0]):
        normalized_data.append((data[i] - np.mean(data[i])) / np.std(data[i]))
    return np.array(normalized_data)

def label_data(label, channel_map):
    # reorganize channel index - label mapping 
    # constrcuct an array of string, each channel index has a string label
    channel_index_label = []
    for i in range(1024):
        # find shank
        shank = i // 128 
        # column is 2* shank th column
        column = 2 * shank + 1
        row = 128 - i % 128 - 1
        # # find the label
        a = label.iloc[row, column]
        # replace label with channel mapped label
        if a in channel_map:
            a = channel_map[a]
        
        channel_index_label.append(a)

    # unique of the label
    unique_label = np.unique(channel_index_label)
    return channel_index_label, unique_label


def dimension_reduct(data, method, n_components):
    if method=="pca":
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(data)

    elif method=="umap":
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        transformed_data = reducer.fit_transform(data)
        
    return transformed_data



def calculate_noise_level(data):
    """Estimate the noise level based on the median absolute deviation."""
    return np.median(np.abs(data - np.median(data))) / 0.6745

def extract_spikes_simple(data):
    # Parameters
    threshold_factor = 3  # How many times the noise level to set the threshold
    min_spike_interval = 10  # Minimum number of indices between spikes, a simple way to avoid detecting multiple peaks for one spike
    
    # Estimate noise level and set threshold
    noise_level = calculate_noise_level(data)
    threshold = noise_level * threshold_factor
    
    # Detect spikes: indices where data crosses the threshold
    spikes_indices = np.where(data > threshold)[0]
    
    # Filter out spikes too close to each other
    spikes_diff = np.diff(spikes_indices)
    if spikes_indices.size > 0:
        valid_spikes = spikes_indices[np.insert(spikes_diff > min_spike_interval, 0, True)]
    else:
        valid_spikes = np.array([])
    
    return valid_spikes



def calculate_isi(spike_times):
    """
    Calculate the interspike intervals (ISIs) from spike times.
    
    Parameters:
    - spike_times: Indices of detected spikes.
    
    Returns:
    - isis: The interspike intervals.
    """
    isis = np.diff(spike_times)
    return isis

def isi_analysis(data, threshold_factor=3):

    spike_times = extract_spikes_simple(data)
    

    # divide them into 10 chunks, each contain 10% of the data
    chunk_size = len(data) // 10
    isis = []
    isis_total = []
    for i in range(10):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        isis.append(calculate_isi(spike_times[(spike_times >= start) & (spike_times < end)]))
        isis_total.extend(isis[i])
    # isis = np.array(isis)
    minISI = 10
    maxISI = 10000
    bin_num = 129
    bins = 10** np.linspace(np.log10(minISI), np.log10(maxISI), bin_num)
    normalized_counts = []
    for i in range(10):
        counts, _ = np.histogram(isis[i], bins=bins)
        total_IEIs = np.sum(counts)
        if total_IEIs == 0:
            normalized_counts.append(np.zeros_like(counts))
        else:
            normalized_counts.append(counts / total_IEIs)

    # print("shape", np.array(normalized_counts).shape)
    return np.array(normalized_counts)

