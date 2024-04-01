import pandas as pd
import argparse

from data_preprocess import *
from mlp import *
from trans import *
from dataset import *
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch_geometric.data import DataLoader
from lolcat import *


def run(model, reduction, dimension, session_name):
    data_path_json = '/scratch/cl7201/hippo_decoding/data_path.json'
    with open(data_path_json, 'r') as file:
            data_path = json.load(file)
            
    channel_map = read_map(data_path.get('public')['mapping_path'])
    data, label = read_data(data_path, session_name)
    print("data", data[:1])

    normalized_data = normalize(data)
    print(normalized_data.shape)

    channel_index_label, unique_label = label_data(label, channel_map)
    print(len(channel_index_label))
    print(unique_label)
    print("normal data", normalized_data[:1])
    if reduction == "autoencoder":
        input_dim = 600000  # Dimensionality of each sample
        num_classes = 5  # Number of target classes

        X = []
        y = []
        label_index = {'CA1':0,
                        'CA2':1,
                        'CA3':2,
                        'DG':3,
                        'cortex':4}
        for i in range(len(channel_index_label)):
            if pd.isna(channel_index_label[i]):
                continue  
            label = str(channel_index_label[i]).strip().lower() 
            if label == "unk" or label == "nan":
                continue
            X.append(normalized_data[i])
            y.append(label_index[channel_index_label[i]])
            
        print(len(y))
        print(np.unique(y, return_counts=True))

        encoding_dim = 1024  # Dimensionality of the encoded representation
        batch_size = 4  # Adjust based on your system's memory capacity
        learning_rate = 0.001
        num_epochs = 10  # For demonstration, use a small number of epochs

        combined_framework = CombinedFramework(X, y, input_dim, encoding_dim, num_classes,
                                            batch_size=batch_size, learning_rate=learning_rate, num_epochs=num_epochs)
        
        print("Starting training...")
        combined_framework.train()
        print("Evaluating model...")
        accuracy = combined_framework.evaluate()
    else:
        # extract spike
        # if exist "processed_data.npy", load it, otherwise, compute
        if os.path.exists('processed_data.npy'):
            print("Loading data isi...")
            processed_data = list(np.load('processed_data.npy'))
        else:
            processed_data = []
            print("Processing data isi...")
            for i in tqdm(range(normalized_data.shape[0])): 
                processed_data.append(isi_analysis(normalized_data[i]))
        
        label_index = {'CA1':0,
                        'CA2':1,
                        'CA3':2,
                        'DG':3,
                        'cortex':4}

        if model == "lolcat":
            # print(np.unique(y, return_counts=True)

            lolcat_trainer = LOLCARTrainer(processed_data, channel_index_label, label_index)
            lolcat_trainer.train()
            accuracy = lolcat_trainer.evaluate()
        
        elif model == "mlp":
            mlp = MLP(X, y, input_size=100)
            mlp.train()
            accuracy = mlp.evaluate()
        elif model == "transformer":
            X = torch.tensor(X)
            y = torch.tensor(y)

            batch_size = 32
            learning_rate = 0.001
            num_epochs = 100

            transformer_framework = TransformerFramework(X=X, y=y, input_dim=dimension, num_classes=5,
                                                        batch_size=batch_size, learning_rate=learning_rate,
                                                        num_epochs=num_epochs)

            print("Starting training...")
            transformer_framework.train()
            print("Evaluating model...")
            accuracy = transformer_framework.evaluate()
        # elif model == "lolcat":

            
    return accuracy

def run_dim(model, session_name):
    dimensions = range(3, 11)
    accuracy_results = { "umap": [], "pca": [] }

    for dimension in dimensions:
        for method in ["umap", "pca"]:
            print(f"Running: Dimension: {dimension}, Method: {method}")
            accuracy = run(model, method, dimension, session_name)
            print(f"Finished: Dimension: {dimension}, Method: {method}, Accuracy: {accuracy}")
            accuracy_results[method].append((dimension, accuracy))

    with open('accuracy_results.json', 'w') as file:
        json.dump(accuracy_results, file)
    plot_results(accuracy_results)
    
    return accuracy_results

def plot_results(accuracy_results):
    plt.figure(figsize=(10, 6))
    for method, results in accuracy_results.items():
        dimensions, accuracies = zip(*results)
        plt.plot(dimensions, accuracies, label=method.upper())

    plt.xlabel('Dimension')
    plt.ylabel('Accuracy')
    plt.title('Abalation Study on AD_HF01_1')
    plt.legend()
    # save the plot
    plt.savefig('abalation_AD_HF01_1.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select model and session name')
    parser.add_argument('--model', type=str, default='lolcat', help='Select model')
    parser.add_argument('--reduction', type=str, default='umap', help='Select dimensionality reduction method')
    parser.add_argument('--dimension', type=int, default=10, help='Select dimensionality of the encoded representation')
    parser.add_argument('--session', type=str, default='AD_HF01_1', help='Select session name')
    parser.add_argument('--plot', action='store_true', help='Plot results for accuracy by dimension and reduction method')
    args = parser.parse_args()
    model, reduction, dimension, session_name, plot_acc = args.model, args.reduction, args.dimension, args.session, args.plot
    if not plot_acc:
        run(model, reduction, dimension, session_name)
    else:
        print("Running dimensionality comparison for both UMAP and PCA...")
        run_dim(model, session_name)

    # with open('AD_HF01_1_results.json', 'r') as file:
    #     accuracy_results = json.load(file)
    # # plot the results
    # plot_results(accuracy_results)