import numpy as np
import random

def jitter(signal, noise_level=0.05):
    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise

def scaling(signal, scale_range=(0.9, 1.1)):
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return signal * scale_factor

def window_slicing(signal, window_size=1000):
    if signal.shape[0] < window_size:
        return signal  # Return original if slicing is not possible
    start = np.random.randint(0, signal.shape[0] - window_size)
    return signal[start:start + window_size]

def generate_orthogonal_matrix(size):
    # Generate a random matrix
    H = np.random.randn(size, size)
    # Use QR decomposition to obtain an orthogonal matrix Q
    Q, R = np.linalg.qr(H)
    return Q

def rotation(signal):
    if signal.shape[0] < 2:
        return signal  # Not enough features to rotate
    rotation_matrix = generate_orthogonal_matrix(2)
    rotated_features = np.dot(signal[:2], rotation_matrix)
    signal[:2] = rotated_features
    return signal

def random_augmentation(signal):
    augmentations = [jitter, scaling, window_slicing, rotation]
    aug_func = random.choice(augmentations)
    return aug_func(signal)


def create_positive_pairs(raw_signal, channel_ridx_map):
    n = raw_signal.shape[0]
    new_raw_signal = np.zeros((2 * n, raw_signal.shape[1]))
    new_channel_ridx_map = np.zeros((2 * n, channel_ridx_map.shape[1]))

    for idx in range(n):
        original_signal = raw_signal[idx]
        augmented_signal = random_augmentation(original_signal)

        new_raw_signal[2 * idx] = original_signal
        new_raw_signal[2 * idx + 1] = augmented_signal

        new_channel_ridx_map[2 * idx] = channel_ridx_map[idx]
        new_channel_ridx_map[2 * idx + 1] = channel_ridx_map[idx]

    return new_raw_signal, new_channel_ridx_map


def create_augmented_signal(raw_signal, channel_ridx_map):
    n = raw_signal.shape[0]
    augmented_signals = np.zeros_like(raw_signal) 
    augmented_map = np.zeros_like(channel_ridx_map) 

    for idx in range(n):
        original_signal = raw_signal[idx]
        augmented_signal = random_augmentation(original_signal)

        augmented_signals[idx] = augmented_signal

        augmented_map[idx] = channel_ridx_map[idx]

    return augmented_signals, augmented_map