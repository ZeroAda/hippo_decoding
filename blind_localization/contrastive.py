import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


class RawSignalDataset(Dataset):
    def __init__(self, signals, labels, transform=None):
        self.signals = signals
        self.labels = labels
        self.transform = transform
        self.label_to_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}

    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]

        # Positive sample: randomly choose within the same label
        if len(self.label_to_indices[label]) > 1:
            positive_idx = idx
            while positive_idx == idx:
                positive_idx = np.random.choice(self.label_to_indices[label])
        else:
            positive_idx = np.random.choice(len(self.labels))

        # Negative sample: randomly choose from a different label
        negative_label = np.random.choice(list(self.label_to_indices.keys() - {label}))
        negative_idx = np.random.choice(self.label_to_indices[negative_label])

        pos_signal = self.signals[positive_idx]
        neg_signal = self.signals[negative_idx]

        if self.transform:
            signal = self.transform(signal)
            pos_signal = self.transform(pos_signal)
            neg_signal = self.transform(neg_signal)

        signal = torch.tensor(signal, dtype=torch.float32)
        pos_signal = torch.tensor(pos_signal, dtype=torch.float32)
        neg_signal = torch.tensor(neg_signal, dtype=torch.float32)
            
        return (signal, pos_signal, neg_signal), label


class ContrastiveEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(ContrastiveEncoder, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return x
    

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        z = torch.cat([z_i, z_j], dim=0)

        # compute similarity for all pairs of z_i, z_j
        sim = self.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        mask = torch.ones_like(sim).fill_diagonal_(0)
        sim = sim * mask

        # exponentiate positive pair similarities between z_i, z_j
        pos_sim_ij = torch.diag(sim, z_i.size(0))
        pos_sim_ji = torch.diag(sim, -z_i.size(0))
        exp_pos_sim = torch.exp(torch.cat([pos_sim_ij, pos_sim_ji], dim=0))

        # sum exponentials of non-self positive similarities
        sum_exp_reg = torch.sum(torch.exp(sim), dim=1, keepdim=True)

        loss = -torch.log(exp_pos_sim / sum_exp_reg)
        return loss.mean()