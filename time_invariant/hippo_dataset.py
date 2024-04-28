from sklearn.model_selection import train_test_split
import os
import numpy as np

from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

import numpy as np
import torch
from torch.utils.data import Dataset

class HippoDataset(Dataset):
    def __init__(self, data, sequence_length, step_size=100):
        concatenated_data = np.concatenate(data)
        self.data = torch.from_numpy(concatenated_data.astype(np.float32)).unsqueeze(1)  # Make it two-dimensional
        self.sequence_length = sequence_length
        self.step_size = step_size

    def __len__(self):
        # Adjust the total length to account for the step size
        return max(0, (self.data.size(0) - self.sequence_length) // self.step_size + 1)

    def __getitem__(self, idx):
        start_idx = idx * self.step_size
        end_idx = start_idx + self.sequence_length
        seq_pad = max(0, end_idx + 1 - self.data.size(0))

        x = self.data[start_idx:end_idx, :]
        y = self.data[start_idx + 1:end_idx + 1, :]

        if seq_pad > 0:
            x = F.pad(x, (0, 0, 0, seq_pad - 1), 'constant', 0)  # Padding for x
            y = F.pad(y, (0, 0, 0, seq_pad), 'constant', 0)  # Same padding for y

            print(f"Start Index: {start_idx}, End Index: {end_idx}, Data size: {self.data.size(0)}, Seq Pad: {seq_pad}")
            print(f"Padded X shape: {x.shape}, Padded Y shape: {y.shape}")

        return x, y
    
class HippoDatasetEmbeddings(Dataset):
    def __init__(self, data, sequence_length=100, step_size=100):
        # No concatenation of the entire dataset, handle each item individually
        self.data = [torch.from_numpy(d.astype(np.float32)).unsqueeze(1) for d in data]
        self.sequence_length = sequence_length
        self.step_size = step_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Process each sequence separately
        sequence = self.data[idx]
        total_len = sequence.size(0)
        start_idx = 0
        end_idx = min(start_idx + self.sequence_length, total_len)
        x = sequence[start_idx:end_idx]

        if end_idx - start_idx < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.sequence_length - (end_idx - start_idx)), 'constant', 0)

        return x, x  # Assuming you want to use x for both input and target for simplification

def create_dataset(data_dir, sequence_length=100, test_split=0.2, random_state=42, train=False):
    data_files = [f for f in os.listdir(data_dir) if f.startswith('processed_data_')]
     # data_list to be a list concatenating all data on the first axis
    data_list = []

    for file in data_files:
        data_path = os.path.join(data_dir, file)
        data = np.load(data_path)
        print(data.shape)
        data_list.extend(data)

    data_train, data_test = train_test_split(
        data_list, test_size=test_split, random_state=random_state
    )

    if train:
        train_dataset = HippoDataset(data_train, sequence_length=sequence_length)
        test_dataset = HippoDataset(data_test, sequence_length=sequence_length)
    else:
        train_dataset = HippoDatasetEmbeddings(data_train, sequence_length=sequence_length)
        test_dataset = HippoDatasetEmbeddings(data_test, sequence_length=sequence_length)

    return train_dataset, test_dataset

# class HippoDatasetSession(Dataset):
#     def __init__(self, file_list, sequence_length=100):
#         self.data_files = file_list
#         self.sequence_length = sequence_length

#     def __len__(self):
#         # This approach assumes all files are loaded to calculate the total length
#         total_length = 0
#         for file_path in self.data_files:
#             data = np.load(file_path)
#             total_length += data.shape[1] - self.sequence_length
#         return total_length

#     def __getitem__(self, global_idx):
#         # This approach finds the appropriate file and local index for global_idx
#         cumulative_length = 0
#         for file_path in self.data_files:
#             data = torch.from_numpy(np.load(file_path)).float()
#             if cumulative_length + data.size(1) - self.sequence_length > global_idx:
#                 local_idx = global_idx - cumulative_length
#                 return (data[:, local_idx:local_idx + self.sequence_length],
#                         data[:, local_idx + 1:local_idx + self.sequence_length + 1])
#             cumulative_length += data.size(1) - self.sequence_length
#         raise IndexError("Index out of bounds")

# def split_dataset(data_dir, file_prefix, train_sessions, sequence_length=100):
#     all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith(file_prefix + 'processed_data_')]

#     train_files = [f for f in all_files if any(session in f for session in train_sessions)]
#     test_files = [f for f in all_files if f not in train_files]

#     train_dataset = HippoDatasetSession(train_files, sequence_length=sequence_length)
#     test_dataset = HippoDatasetSession(test_files, sequence_length=sequence_length)

#     return train_dataset, test_dataset
