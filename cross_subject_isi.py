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


def run(model, dimension, session_name, train_session, train_dimension):
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
    
    
    # extract spike
    # if exist "processed_data.npy", load it, otherwise, compute
    
    processed_data = []
    print("Processing data isi...")
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

        lolcat_trainer = LOLCARTrainer(processed_data, channel_index_label, label_index, heads=dimension)
        # _, _ = lolcat_trainer.evaluate_test(normalized_data, processed_data, channel_index_label, session_name, train_session, train_dimension)
        accuracy, cm = lolcat_trainer.evaluate_accuracy(normalized_data, processed_data, channel_index_label, session_name, train_session, train_dimension)


        # plot heatmap of confusion matrix
        fig, ax = plt.subplots()

        im, cbar = heatmap(cm, labels, labels, ax=ax,
                        cmap="YlGn", cbarlabel="Accuracy")
        texts = annotate_heatmap(im, valfmt="{x:.1f}")

        fig.tight_layout()
        plot_title = f"figs/cm_head{dimension}_{session_name}_cross_{train_session}.pdf"
        plt.savefig(plot_title)

    return accuracy

def run_all(model="lolcat", dimension=1, train_session="AD_HF01_1"):
    session_names = ['AD_HF01_1', 'AD_HF02_2', 'AD_HF02_4', 'AD_HF03_1', 'AD_HF03_2', 'NN_syn_01', 'NN_syn_02']
    accuracy_list = []
    
    for j, session_name in enumerate(session_names):
        print("test for session", session_name, "==============")
        accuracy_list.append(run(model, dimension, session_name, train_session, dimension))
    
    # plot bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(session_names, accuracy_list)
    plt.xlabel('Session')
    plt.ylabel('Accuracy')
    plt.title(f'Cross-Subject on Model {train_session} head={dimension}')
    for index, value in enumerate(accuracy_list):
        value = round(value, 2)
        plt.text(index, value + 1, str(value))
    plot_title = f'figs/cross_subject_{train_session}_head{dimension}.pdf'
    plt.savefig(plot_title)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select model and session name')
    parser.add_argument('--model', type=str, default='lolcat', help='Select model')
    parser.add_argument('--dimension', type=int, default=1, help='Select dimensionality of the encoded representation')
    parser.add_argument('--session', type=str, default='AD_HF01_1', help='Select session name')
    parser.add_argument('--plot', action='store_true', help='Plot results for accuracy by dimension and reduction method')
    args = parser.parse_args()
    model, dimension, session_name, plot_acc = args.model, args.dimension, args.session, args.plot
    run_all(model=model, dimension=4, train_session=session_name)

    # with open('AD_HF01_1_results.json', 'r') as file:
    #     accuracy_results = json.load(file)
    # # plot the results
    # plot_results(accuracy_results)