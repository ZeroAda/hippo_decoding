import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
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


from lxml import etree
from matplotlib.lines import Line2D
###############oldprocessing
# def read_data(data_path, session_name):
#     session_path = data_path.get(session_name)
#     with h5py.File(session_path['raw_signal_path'], 'r') as file:
#         print("Variables in file:")
#         # List all groups (MATLAB variables are stored as groups in HDF5)
#         for var_name in file:
#             print(var_name)
#         data = file[var_name][()]
    
#     label_path = data_path.get('public')['label_path']
#     sheet_name = session_path['sheet_name']
#     label = pd.read_excel(label_path, sheet_name=sheet_name)
#     print("check read data shapes", data.shape, label.shape)
        
#     return data, label
    
# def read_map(json_path):
#     with open(json_path, 'r') as file:
#         channel_map = json.load(file)
#         print(channel_map)
#     return channel_map

# def normalize(data):
#     normalized_data = []
#     for i in range(data.shape[0]):
#         normalized_data.append((data[i] - np.mean(data[i])) / np.std(data[i]))
#     return np.array(normalized_data)

# def label_data(label, channel_map):
#     # reorganize channel index - label mapping 
#     # constrcuct an array of string, each channel index has a string label
#     channel_index_label = []
#     print("label shape", label.shape)
#     for i in range(1024):
#         # find shank
#         shank = i // 128 
#         # column is 2* shank th column
#         column = 2 * shank + 1
#         row = 128 - i % 128 - 1
#         # # find the label
#         a = label.iloc[row, column]
#         # replace label with channel mapped label
#         if a in channel_map:
#             a = channel_map[a]
        
#         channel_index_label.append(a)

#     # unique of the label
#     unique_label = np.unique(channel_index_label)
#     return channel_index_label, unique_label
#######################


def load_session_data(json_file_path, session_name):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        return data.get(session_name)
    

def load_data(raw_signal_path, label_path, xml_path, sheet_name):

    # load raw data from mat file
    with h5py.File(raw_signal_path, 'r') as file:
        raw_signal = file['data'][:]
    
    # load region labels from excel
    df = pd.read_excel(label_path, sheet_name=sheet_name)

    # load xml file to find skipped channels
    tree = etree.parse(xml_path)
    root = tree.getroot()
    element = root.find('anatomicalDescription')

    # find all channels to be skipped (1 = skipped, 0 = not skip)
    skip_arr = np.zeros((1024, ))

    for channel in element.findall('.//channel'):
        skip_value = channel.attrib['skip']
        c = int(channel.text)
        skip_arr[c] = skip_value

    skipped_channels = np.where(skip_arr == 1)[0]

    return raw_signal, df, skipped_channels

def load_spike_data(spike_signal_path, label_path, xml_path, sheet_name):

    # load raw data from mat file
    # with h5py.File(raw_signal_path, 'r') as file:
    #     raw_signal = file['data'][:]
    spike_mat = loadmat(spike_signal_path)
    dataframes = pd.DataFrame(spike_mat['spikes'][0])
    spike_times = dataframes['times'][0][0]
    spike_channel = dataframes['cluID'][0][0]
    spike_neuron = np.arange(0, spike_channel.shape[0])

    # neuron-channel map dic
    # neuron_channel_map = dict(zip(spike_neuron, spike_channel))
    
    
    # load region labels from excel
    df = pd.read_excel(label_path, sheet_name=sheet_name)

    # load xml file to find skipped channels
    tree = etree.parse(xml_path)
    root = tree.getroot()
    element = root.find('anatomicalDescription')

    # find all channels to be skipped (1 = skipped, 0 = not skip)
    skip_arr = np.zeros((1024, ))

    for channel in element.findall('.//channel'):
        skip_value = channel.attrib['skip']
        c = int(channel.text)
        skip_arr[c] = skip_value

    skipped_channels = np.where(skip_arr == 1)[0]

    return spike_times, spike_channel, df, skipped_channels

def process_labels(df, mapping_path, skipped_channels):
    """
    Here we obtain a channel-region map with bad channels skipped
    """
    with open(mapping_path) as json_file:
        mapping = json.load(json_file)
    
    channels = pd.concat([df.iloc[:, i] for i in range(0, len(df.columns), 2)], ignore_index=True)
    regions = pd.concat([df.iloc[:, i+1] for i in range(0, len(df.columns), 2)], ignore_index=True)

    output = pd.DataFrame({"channels": channels, "regions":regions})
    output["channels"] -= 1

    # fill in NaN channels with UNK
    for idx, row in output.iterrows():
        if pd.isna(row["channels"]):
            output.at[idx, "channels"] = output.at[idx-1, "channels"]-1
        if pd.isna(row["regions"]):
            output.at[idx, "regions"] = "UNK"

    # remove bad channels in skipped_channels
    processed_channels = np.array(output["channels"]).astype(int)
    mask = ~output["channels"].isin(skipped_channels)
    output = output[mask]

    # fix diverse labels
    output = output.replace(mapping)

    # fix shuffled channel number
    corrected_channels = np.array([list(range(shank*128+127, shank*128-1, -1)) for shank in range(8)]).flatten()
    channel_channel_map = dict(zip(processed_channels, corrected_channels))
    output = output.replace(channel_channel_map)

    output["channels"] = output["channels"].astype(int)

    # update skipped_channels based on shuffled channel number
    vectorized_func = np.vectorize(channel_channel_map.get)
    skipped_channels = vectorized_func(skipped_channels) if len(skipped_channels) > 0 else skipped_channels

    return output, skipped_channels, channel_channel_map

def process_spike_labels(df, mapping_path, skipped_channels, spike_channel):
    """
    Here we obtain a channel-region map with bad channels skipped
    """
    with open(mapping_path) as json_file:
        mapping = json.load(json_file)
    
    channels = pd.concat([df.iloc[:, i] for i in range(0, len(df.columns), 2)], ignore_index=True)
    regions = pd.concat([df.iloc[:, i+1] for i in range(0, len(df.columns), 2)], ignore_index=True)

    output = pd.DataFrame({"channels": channels, "regions":regions})
    output["channels"] -= 1

    # fill in NaN channels with UNK
    for idx, row in output.iterrows():
        if pd.isna(row["channels"]):
            output.at[idx, "channels"] = output.at[idx-1, "channels"]-1
        if pd.isna(row["regions"]):
            # print("UNK")
            output.at[idx, "regions"] = "UNK"

    # remove bad channels in skipped_channels
    processed_channels = np.array(output["channels"]).astype(int)
    mask = ~output["channels"].isin(skipped_channels)
    output = output[mask]

    # fix diverse labels
    output = output.replace(mapping)

    # fix shuffled channel number
    corrected_channels = np.array([list(range(shank*128+127, shank*128-1, -1)) for shank in range(8)]).flatten()
    channel_channel_map = dict(zip(processed_channels, corrected_channels))
    output = output.replace(channel_channel_map)

    # fixed spike channel: neuron-spike corresponding
    spike_channel = pd.DataFrame({"channels": spike_channel})
    spike_channel['channels'] -= 1
    spike_channel = spike_channel.replace(channel_channel_map)
    print("unique spike channels", spike_channel["channels"].unique())

    # add neuron column
    spike_channel["neuron"] = np.arange(0, spike_channel.shape[0])


    output["channels"] = output["channels"].astype(int)
    print("channel length", len(output["channels"]))
        # create neuron-region dataframe, take region of output based on channel of spike_channel
    spike_region = pd.merge(spike_channel, output, on="channels", how="left")
    spike_region = spike_region.drop(columns=["channels"])
    # find nan region and store 'neuron' as skipped_neurons
    skipped_neurons = spike_region[spike_region["regions"].isna()]["neuron"].to_numpy()
    # drop na regions rows
    # spike_region = spike_region.dropna()

    # update skipped_channels based on shuffled channel number
    vectorized_func = np.vectorize(channel_channel_map.get)
    skipped_channels = vectorized_func(skipped_channels) if len(skipped_channels) > 0 else skipped_channels

    return output, skipped_channels, channel_channel_map, spike_channel, spike_region, skipped_neurons


def process_signals(raw_signal, channel_channel_map):
    # n_channel * n_samples (30 s*20000 Hz)
    raw_signal = raw_signal.astype('float32')
    processed_signal = np.zeros_like(raw_signal)

    for i in range(len(raw_signal)):
        j = channel_channel_map[i]
        if np.std(raw_signal[i]) != 0:
            processed_signal[j] = (raw_signal[i] - np.mean(raw_signal[i])) / np.std(raw_signal[i])

    return processed_signal


def visualize_raw_signals(raw_signal, channel_region_map, skipped_channels, T = 1000, sr = 20000):
    """
    Visualize raw signals colored by region (skipping bad channels)
    """
    offset = 5
    sr_converted = int(sr/1000) #samples/ms
    t_offset = T + 100

    time = np.arange(0, T, 1/sr_converted)

    colors = ["red", "orange", "green", "blue", "magenta", "black"]
    regions = ["cortex", "CA1", "CA2", "CA3", "DG", "UNK"]
    regions_to_colors = dict(zip(regions, colors))
    
    skip_arr = np.zeros(len(raw_signal))
    skip_arr[skipped_channels] = 1


    plt.figure(figsize=(18, 16))
    for shank in range(8):
        for i in range(128): 
            channel = shank*128+i

            row = channel_region_map[channel_region_map["channels"] == channel]
            region = row.iloc[0]["regions"] if len(row) > 0 else "UNK"

            if not skip_arr[channel]:
                color = regions_to_colors[region]
                plt.plot(time+t_offset*shank, raw_signal[channel, :T*sr_converted]+offset*i, c=color)

    legend_handles = [Line2D([0], [0], color=color, lw=2, label=list(regions_to_colors.keys())[i]) for i, color in enumerate(colors)]

    plt.legend(handles=legend_handles)

    plt.xlabel("time(ms)")
    plt.ylabel("channel")
    plt.title("Raw signal detected by 8 shanks in 1000 ms")
    plt.savefig("figs/visualize.png")

def visualize_spike_signals(raw_signal, channel_region_map, skipped_channels, spike_channel, spike_times, T = 1000, sr = 20000):
    """
    Visualize spike signals colored by region (skipping bad channels)
    """

    ### process spike
    # take 12min-12.5min
    



    ###

    offset = 5
    sr_converted = int(sr/1000) #samples/ms
    t_offset = T + 100

    time = np.arange(0, T, 1/sr_converted)
    print("time", time.shape)

    colors = ["red", "orange", "green", "blue", "magenta", "black"]
    regions = ["cortex", "CA1", "CA2", "CA3", "DG", "UNK"]
    regions_to_colors = dict(zip(regions, colors))
    
    skip_arr = np.zeros(len(raw_signal))
    skip_arr[skipped_channels] = 1


    plt.figure(figsize=(18, 16))
    for shank in range(8):
        for i in range(128): 
            channel = shank*128+i
            

            row = channel_region_map[channel_region_map["channels"] == channel]
            region = row.iloc[0]["regions"] if len(row) > 0 else "UNK"

            if not skip_arr[channel]:
                color = regions_to_colors[region]
                neuron_indices = spike_channel[spike_channel['channels'] == channel]['neuron'].to_numpy()
                
                plt.plot(time+t_offset*shank, raw_signal[channel, :T*sr_converted]+offset*i, c=color)
                for neuron in neuron_indices:
                    spike_time = spike_times[neuron][(spike_times[neuron] >= 660) & (spike_times[neuron] < 661)] - 660
                    spike_timestep = spike_time * 1000
                    print("range of spike time", spike_timestep)

                    # print("sptime", spike_timestep)
                    y_pos = np.ones_like(spike_timestep) * offset*i
                    plt.scatter(spike_timestep+t_offset*shank, y_pos, c=color, marker='x')

    legend_handles = [Line2D([0], [0], color=color, lw=2, label=list(regions_to_colors.keys())[i]) for i, color in enumerate(colors)]

    plt.legend(handles=legend_handles)

    plt.xlabel("time(ms)")
    plt.ylabel("channel")
    plt.title("Raw signal detected by 8 shanks in 1000 ms")
    plt.savefig("figs_spike/visualize.png")



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
    """
    Extract spikes from the data using a simple thresholding method.

    Parameters:
    - data: 1D time series data

    Returns:
    - spikes_indices: Indices of detected spikes
    """
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


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass filter to the data.
    
    Parameters:
    - data: The data to filter.
    - lowcut: The low cutoff frequency.
    - highcut: The high cutoff frequency.
    - fs: The sampling frequency.
    - order: The order of the filter.
    
    Returns:
    - filtered_data: The filtered data.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def isi_analysis(data):
    """
    original data
    1. Bandpass filter the data.
    2. Extract spikes.
    3. Calculate ISIs.
    4. Compute the histogram of ISIs.

    Parameters:
    - data: raw data

    Returns:
    - normalized_counts: Normalized histogram of ISIs
    - bins: bins of the histogram
    """

    ## band pass filter
    filtered_data = bandpass_filter(data, lowcut=200, highcut=3000, fs=20000)
    # filtered_data = data

    ## extract spikes
    spike_times = extract_spikes_simple(filtered_data)

    # divide them into 10 chunks, each contain 10% of the data
    chunk_size = len(filtered_data) // 10
    isis = []
    isis_total = []
    for i in range(10):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        isis.append(calculate_isi(spike_times[(spike_times >= start) & (spike_times < end)]))
        isis_total.extend(isis[i])
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

    return np.array(normalized_counts), bins


def visualize_isi(data, label, session_name):
    """
    visualize isi data for each label - original data
    """
    
    label_t = ['CA1', 'CA2', 'CA3', 'DG', 'cortex']
    color_map = {
            "cortex": "red",
            "CA1": "orange",
            "CA3": "blue",
            "CA2": "green",
            "DG": "pink",
            "UNK": "black"
        }
    # find one data for each label_t

    fig, axs = plt.subplots(5, 10, figsize=(20, 10))
    fig.suptitle(session_name)
    for i in range(5):
        if np.where(np.array(label) == label_t[i])[0].shape[0] == 0:
            continue
        index = np.where(np.array(label) == label_t[i])[0][0]
        data_i = data[index]
        counts, bins= isi_analysis(data_i)
        bin_midpoints = 0.5 * (bins[:-1] + bins[1:])
        for j in range(10):
            axs[i, j].bar(bin_midpoints, counts[j], width=np.diff(bins), edgecolor=color_map[label_t[i]], color=color_map[label_t[i]])
            # log scale x
            axs[i, j].set_xscale('log')
            # Annotate only the first column with the brain region name
            if j == 0:
                axs[i, j].annotate(label_t[i], xy=(-0.5, 0.5), xycoords='axes fraction',
                                textcoords='offset points', size='large', ha='right', va='center')

            # Setting titles for each subplot in the top row to indicate chunks
            if i == 0:
                axs[i, j].set_title(f"chunk {j+1}")
            # axs[i, j].set_xlabel("ISI (ms)")

    # save fig
    # make figure wider
    plt.tight_layout()
    title_name = f"figs/isi_analysis_{session_name}.pdf"
    plt.savefig(title_name)


def isi_spike_analysis(spike_times):
    """
    spike data
    3. Calculate ISIs.
    4. Compute the histogram of ISIs.

    Parameters:
    - spike_times: spike times

    Returns:
    - normalized_counts: Normalized histogram of ISIs
    - bins: bins of the histogram
    """

    # ## band pass filter
    # filtered_data = bandpass_filter(data, lowcut=200, highcut=3000, fs=20000)
    # # filtered_data = data

    # ## extract spikes
    # spike_times = extract_spikes_simple(filtered_data)

    # divide them into 10 chunks, each contain 10% of the data
    chunk_size = 3600 // 100
    isis = []
    isis_total = []
    for i in range(10):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        isis.append(calculate_isi(spike_times[(spike_times >= start) & (spike_times < end)]))
        isis_total.extend(isis[i])
    minISI = 1e-3
    maxISI = 100
    bin_num = 500
    bins = 10** np.linspace(np.log10(minISI), np.log10(maxISI), bin_num)
    normalized_counts = []
    for i in range(10):
        counts, _ = np.histogram(isis[i], bins=bins)
        total_IEIs = np.sum(counts)
        if total_IEIs == 0:
            normalized_counts.append(np.zeros_like(counts))
        else:
            normalized_counts.append(counts / total_IEIs)

    return np.array(normalized_counts), bins        


def visualize_spike_isi(data, label, session_name):
    """
    visualize isi data for each label - original data
    """
    
    label_t = ['CA1', 'CA2', 'CA3', 'DG', 'cortex']
    color_map = {
            "cortex": "red",
            "CA1": "orange",
            "CA3": "blue",
            "CA2": "green",
            "DG": "pink",
            "UNK": "black"
        }
    # find one data for each label_t

    fig, axs = plt.subplots(5, 10, figsize=(20, 10))
    fig.suptitle(session_name)
    for i in range(5):
        if np.where(np.array(label) == label_t[i])[0].shape[0] == 0:
            continue
        index = np.where(np.array(label) == label_t[i])[0][0]
        data_i = data[index]
        counts, bins= isi_spike_analysis(data_i)
        bin_midpoints = 0.5 * (bins[:-1] + bins[1:])
        for j in range(10):
            axs[i, j].bar(bin_midpoints, counts[j], width=np.diff(bins), edgecolor=color_map[label_t[i]], color=color_map[label_t[i]])
            # log scale x
            axs[i, j].set_xscale('log')
            # Annotate only the first column with the brain region name
            if j == 0:
                axs[i, j].annotate(label_t[i], xy=(-0.5, 0.5), xycoords='axes fraction',
                                textcoords='offset points', size='large', ha='right', va='center')

            # Setting titles for each subplot in the top row to indicate chunks
            if i == 0:
                axs[i, j].set_title(f"chunk {j+1}")
            # axs[i, j].set_xlabel("ISI (ms)")

    # save fig
    # make figure wider
    plt.tight_layout()
    title_name = f"figs_spike/isi_analysis_{session_name}.pdf"
    plt.savefig(title_name)
