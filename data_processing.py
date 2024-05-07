import h5py
from lxml import etree
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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
    plt.show()