import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import argparse
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from hippo_dataset import HippoDatasetEmbeddings
from encoder import TransformerModel, PositionalEncoding
from classifier import MLP

def parse_args():
    parser = argparse.ArgumentParser(description="Extract embeddings from a Transformer model.")
    parser.add_argument('--batch_size', type=int, default=128, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--sequence_length', type=int, default=100, help='Length of each sequence')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=1024, help='Dimension of the model')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of transformer encoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--save_path', type=str, default='./time_invariant/model_checkpoints/', help='Path to save the best model and embeddings')
    return parser.parse_args()

def get_embeddings(data, model, device):
    """ Get embeddings from a given dataset.
        Args:
        data (np.ndarray): The dataset to extract embeddings from.

        Returns:
        np.ndarray: The embeddings extracted from the dataset.
    """ 
    dataset = HippoDatasetEmbeddings(data)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    embeddings = []
    print('Extracting embeddings...')
    with torch.no_grad():
        for idx, (data, _) in enumerate(tqdm(loader)):
            data = data.to(device)
            embedding = model.extract_embeddings(data)
            # print(embedding.shape)
            # torch.Size([1, 100, 1024])
            embeddings_mean = torch.mean(embedding, dim=1)
            # print(embeddings_mean.shape)
            embeddings.append(embeddings_mean.cpu().numpy().squeeze())
            # print(len(embeddings[0]))
            # exit()

    return embeddings

def plot_pca_comparison(data_pts_reduced, embeddings_reduced, labels):
    fig = plt.figure(figsize=(16, 8))

    # Plotting the original data points
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(data_pts_reduced[:, 0], data_pts_reduced[:, 1], data_pts_reduced[:, 2], c=labels, cmap='viridis')
    ax1.set_title('Original Data Points')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    # Plotting the embeddings
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1], embeddings_reduced[:, 2], c=labels, cmap='viridis')
    ax2.set_title('Embeddings')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')

    fig.colorbar(scatter2, ax=[ax1, ax2], shrink=0.5, location='right', label='Class Labels')

    plt.savefig('pca_comparison.png')

data_dir = 'processed_dataset'

data_pts = []
embeddings = []
labels = []

data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('processed_data_')])
label_files = sorted([f for f in os.listdir(data_dir) if f.startswith('channel_index_label_')])

args = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerModel(n_features=args.d_model, n_heads=args.nhead, n_hidden=args.dim_feedforward,
                            n_layers=args.num_encoder_layers, dropout=args.dropout)
model.to(device)
model.load_state_dict(torch.load(args.save_path + 'best_model.pth'))

for data_file, lbl_file in zip(data_files, label_files):
    data_pt = np.load(os.path.join(data_dir, data_file))
    data_pts.extend(data_pt)
    embeddings.extend(get_embeddings(data_pt, model, device))
    label = np.load(os.path.join(data_dir, lbl_file))
    labels.extend(label)
    print(len(data_pts), len(embeddings), len(labels))
    # break

# embeddings = np.vstack(embeddings)
# labels = np.array(labels)
# labels = np.concatenate(labels)

vis = False
classify = True

if vis:
    scaler = StandardScaler()
    data_pts_scaled = scaler.fit_transform(data_pts)
    embeddings_scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=3)
    data_pts_reduced = pca.fit_transform(data_pts)
    embeddings_reduced = pca.fit_transform(embeddings)
    plot_pca_comparison(data_pts_reduced, embeddings_reduced, labels)

if classify:
    embeddings = np.array(embeddings)
    print(embeddings.shape)
    X_train_val, X_test, y_train_val, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    print(X_train_val.shape, X_test.shape, len(y_train_val), len(y_test))
    input_size = embeddings.shape[1]
    num_classes = len(np.unique(labels))

    mlp = MLP(X=X_train_val, y=y_train_val, input_size=input_size, hidden_size=64, num_classes=num_classes, batch_size=32, learning_rate=0.001, num_epochs=100, model_save_path='mlp_model.pth')
    mlp.train()
    
    test_accuracy = mlp.test(X_test, y_test)
    print(f'Test accuracy: {test_accuracy}%')

    