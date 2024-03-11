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
    for i in range(data.shape[0]):
        data[i] = (data[i] - np.mean(data[i])) / np.std(data[i])        
    return data

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