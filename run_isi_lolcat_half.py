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
import time


def run(model, dimension, session_name):
    time_start = time.time()
    data_path_json = '/scratch/cl7201/hippo_decoding/data_path.json'
    with open(data_path_json, 'r') as file:
            data_path = json.load(file)
            
    channel_map = read_map(data_path.get('public')['mapping_path'])
    data, label = read_data(data_path, session_name)

    normalized_data = normalize(data)
    print(normalized_data.shape)

    channel_index_label, unique_label = label_data(label, channel_map)
    print(len(channel_index_label))
    print(unique_label)
    
    
    # extract spike
    # if exist "processed_data.npy", load it, otherwise, compute
    
    processed_data = []
    print("Processing data isi...")
    # visualize isi
    visualize_isi(normalized_data, channel_index_label, session_name)
    print("visualize finished")
    for i in tqdm(range(normalized_data.shape[0])): 
        normalized_counts, _ = isi_analysis(normalized_data[i])
        processed_data.append(normalized_counts)
    
    label_index = {'CA1':0,
                    'CA2':1,
                    'CA3':2,
                    'DG':3,
                    'cortex':4}
    labels = ['CA1', 'CA2', 'CA3', 'DG', 'cortex']

    if model == "lolcat":
        # print(np.unique(y, return_counts=True)
        model_save_path = f"half_models/lolcat_head{dimension}_{session_name}.pt"
        lolcat_trainer = LOLCARTrainer_v2(processed_data, channel_index_label, label_index, heads=dimension, model_save_path=model_save_path, num_epochs=200)
        loss_values = lolcat_trainer.train()
        accuracy, cm = lolcat_trainer.evaluate(best_model=True)
        # plot heatmap of confusion matrix
        fig, ax = plt.subplots()

        im, cbar = heatmap(cm, labels, labels, ax=ax,
                        cmap="YlGn", cbarlabel="Accuracy")
        texts = annotate_heatmap(im, valfmt="{x:.1f}")

        fig.tight_layout()
        title_name = f"half_figs/cm_head{dimension}_{session_name}.pdf"
        plt.savefig(title_name)
    time_end = time.time()
    print("Time elapsed: ", time_end - time_start)
    

    return accuracy, loss_values

def run_heads(model):
    time_start = time.time()
    dimensions = [4]
    session_names = ['AD_HF01_1', 'AD_HF02_2', 'AD_HF02_4', 'AD_HF03_1', 'AD_HF03_2', 'NN_syn_01', 'NN_syn_02']


    # Collect data
    all_accuracies = np.zeros((len(dimensions), len(session_names)))
    all_loss_values = np.zeros((len(dimensions), len(session_names), 200))  

    for i, dimension in enumerate(dimensions):
        for j, session_name in enumerate(session_names):
            print("Running heads ", dimension, "for session ", session_name)
            accuracy, loss_values = run(model, dimension, session_name)
            all_accuracies[i, j] = accuracy
            all_loss_values[i, j, :] = loss_values

    # Calculate mean accuracies and loss
    mean_accuracies = np.mean(all_accuracies, axis=1)
    std_accuracies = np.std(all_accuracies, axis=1)
    mean_loss_values = np.mean(all_loss_values, axis=1)
    std_loss_values = np.std(all_loss_values, axis=1)
    # save as npy
    np.save('mean_accuracies.npy', mean_accuracies)
    np.save('std_accuracies.npy', std_accuracies)
    np.save('mean_loss_values.npy', mean_loss_values)
    np.save('std_loss_values.npy', std_loss_values)

    # load
    # mean_accuracies = np.load('mean_accuracies.npy')
    # std_accuracies = np.load('std_accuracies.npy')
    # mean_loss_values = np.load('mean_loss_values.npy')
    # std_loss_values = np.load('std_loss_values.npy')
    
    plt.figure(figsize=(10, 5))
    # Plotting Accuracy vs. Parameter
    plt.errorbar(dimensions, mean_accuracies, yerr=std_accuracies, marker='o', linestyle='-', color='b', label='Mean Accuracy ± Std')
    plt.title('Accuracy vs Heads')
    plt.xlabel('# of Heads')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig('half_figs/accuracy_vs_heads.pdf')

    plt.figure(figsize=(10, 6))
    # Plotting Loss Curves for each Parameter
    for i in range(len(dimensions)):
        plt.plot(range(1, 201), mean_loss_values[i], marker='.', linestyle='-', label=f'Heads {dimensions[i]}')
        plt.fill_between(range(1, 201), mean_loss_values[i] - std_loss_values[i], mean_loss_values[i] + std_loss_values[i], alpha=0.2)
    plt.title('Mean Loss Curves Across Different Heads')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('half_figs/loss_curves.pdf')
    time_end = time.time()
    print("Time elapsed: ", time_end - time_start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select model and session name')
    parser.add_argument('--model', type=str, default='lolcat', help='Select model')
    parser.add_argument('--dimension', type=int, default=1, help='Select dimensionality of the encoded representation')
    parser.add_argument('--session', type=str, default='AD_HF01_1', help='Select session name')
    parser.add_argument('--plot', action='store_true', help='Plot results for accuracy by dimension and reduction method')
    args = parser.parse_args()
    model, dimension, session_name, plot_acc = args.model, args.dimension, args.session, args.plot
    if not plot_acc:
        run(model, dimension, session_name)
    else:
        print("Running heads from 1 to 4")
        run_heads(model)