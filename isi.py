import numpy as np
from scipy.signal import butter, filtfilt


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