import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np
import os
import math

class TransformerModel(nn.Module):
    def __init__(self, n_features, n_heads, n_hidden, n_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_features, 
            nhead=n_heads, 
            dim_feedforward=n_hidden, 
            dropout=dropout,
            batch_first=True  # Set batch_first to True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pos_encoder = PositionalEncoding(n_features).to(self.device)
        self.output_layer = nn.Linear(n_features, 1)

    def forward(self, src, src_mask=None):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        output = self.output_layer(output)
        return output

    def extract_embeddings(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('encodings', self.encoding)  # register as a buffer to keep on the same device

    def forward(self, x):
        return x + self.encodings[:x.size(0), :x.size(1), :].detach()
