import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

from hippo_dataset import create_dataset
from encoder import TransformerModel, PositionalEncoding

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer model on time series data.")
    parser.add_argument('--cross_session', action='store_true', help='Use cross-session validation')
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
    parser.add_argument('--extract_embeddings', action='store_true', help='Extract embeddings from the model')
    return parser.parse_args()

def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            mask = create_key_padding_mask(data_length=data.size(1), batch_size=data.size(0), sequence_length=data.size(1), device=device)
            output = model(data, src_mask=mask)
            loss = criterion(output.view(-1), target.view(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

def create_key_padding_mask(data_length, batch_size, sequence_length, device):
    """ Create a mask to hide padding and future words.
        Args:
        data_length (int): Total length of data in one batch.
        batch_size (int): Number of sequences per batch.
        sequence_length (int): Length of each sequence.

        Returns:
        torch.Tensor: A key padding mask of shape [batch_size, sequence_length].
    """
    mask = torch.zeros(batch_size, sequence_length, device=device, dtype=torch.bool)
    return mask

def custom_collate_fn(batch):
    # Extract all x and y tensors
    xs, ys = zip(*batch)

    # Stack x and y tensors ensuring all are of the same size
    xs = torch.stack(xs, dim=0)
    ys = torch.stack(ys, dim=0)

    return xs, ys

def extract_and_save_embeddings(dataset, model, device, save_path, folder_name):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)  # batch_size=1 for individual processing
    os.makedirs(os.path.join(save_path, folder_name), exist_ok=True)

    model.eval()
    embeddings = []
    with torch.no_grad():
        for idx, (data, _) in enumerate(loader):
            data = data.to(device)
            embedding = model.extract_embeddings(data)
            np.save(os.path.join(folder_name, f'embedding_{idx}.npy'), embedding.cpu().numpy())

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # if args.cross_session:
    #     train_sessions = ['AD_HF01_1', 'AD_HF02_2']
    #     train_dataset, test_dataset = split_dataset('processed_dataset', 'processed_data_', train_sessions)
    # else:
    train_dataset, test_dataset = create_dataset(
        data_dir='processed_dataset',
        sequence_length=args.sequence_length,
        test_split=0.2
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    model = TransformerModel(n_features=args.d_model, n_heads=args.nhead, n_hidden=args.dim_feedforward,
                             n_layers=args.num_encoder_layers, dropout=args.dropout)
    model.to(device)
    
    if args.extract_embeddings:
        model.load_state_dict(torch.load(args.save_path + 'best_model.pth'))
        # embeddings = []
        # model.eval()
        # with torch.no_grad():
        #     for data, _ in train_loader:
        #         data = data.to(device)
        #         embedding = model.extract_embeddings(data)
        #         embeddings.append(embedding.cpu().numpy())
        # np.save(os.path.join(args.save_path, 'embeddings.npy'), np.concatenate(embeddings, axis=0))
        extract_and_save_embeddings(train_dataset, model, device, args.save_path, 'train_embeddings')
        extract_and_save_embeddings(test_dataset, model, device, args.save_path, 'test_embeddings')
        exit()

    
    # Create mask for sequence length
    # mask = create_mask(train_loader.dataset.sequence_length, device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in tqdm(range(args.epochs), desc='Training'):
        model.train()
        train_loss = 0
        for data, labels in tqdm(train_loader, desc='Epoch', leave=True):
            data, labels = data.to(device), labels.to(device)
            mask = create_key_padding_mask(data_length=data.size(1), batch_size=data.size(0), sequence_length=data.size(1), device=device)
            optimizer.zero_grad()
            output = model(data, src_mask=mask)
            # print(output.size(), labels.size())
            # print(output.view(-1).size(), labels.view(-1).size())
            loss = loss_function(output.view(-1), labels.view(-1))
            # print("loss: ", loss)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        val_loss = evaluate(model, test_loader, loss_function, device)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path + 'best_model.pth')
            
    plot_losses(train_losses, val_losses)

if __name__ == '__main__':
    main()