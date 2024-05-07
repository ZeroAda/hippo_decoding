import pandas as pd
import argparse

from data_preprocess import *
from mlp import *
from trans import *
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def run(model_path, reduction, dimension, session_name):
    data_path_json = 'hippo_decoding/data_path.json'
    with open(data_path_json, 'r') as file:
            data_path = json.load(file)
            
    channel_map = read_map(data_path.get('public')['mapping_path'])
    data, label = read_data(data_path, session_name)

    normalized_data = normalize(data)
    print(normalized_data.shape)

    channel_index_label, unique_label = label_data(label, channel_map)
    print(len(channel_index_label))
    print(unique_label)
    if reduction == "umap":
        processed_data = dimension_reduct(normalized_data, method="umap", n_components=dimension)
    elif reduction == "pca":
        processed_data = dimension_reduct(normalized_data, method="pca", n_components=dimension)
    
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
        X.append(processed_data[i])
        y.append(label_index[channel_index_label[i]])
        
    print(len(y))
    print(np.unique(y, return_counts=True))
    
    mlp = MLP(X, y, input_size=dimension)
    mlp.model.load_state_dict(torch.load(model_path))
    # accuracy = mlp.test(X, y)
    accuracy = mlp.evaluate()
            
    return accuracy

def run_all(model_path, reduction, dimension):
    data_path_json = 'hippo_decoding/data_path.json'
    with open(data_path_json, 'r') as file:
            data_path = json.load(file)
            
    channel_map = read_map(data_path.get('public')['mapping_path'])
    all_sessions = ['AD_HF01_1', 'AD_HF02_2', 'AD_HF02_4', 'AD_HF03_1', 'AD_HF03_2', 'NN_syn_01', 'NN_syn_02']
    
    accuracies = []
    for session in all_sessions:
        print(f'Running session: {session}')
        accuracy = run(model_path, reduction, dimension, session)
        print(f'Accuracy of the model on the validation set: {accuracy} %')
        accuracies.append(accuracy)
    
    with open('transfer_accuracies.json', 'w') as file:
        json.dump(accuracies, file)
    
    plt.figure(figsize=(10, 8))
    plt.bar(all_sessions, accuracies)
    plt.xlabel('Session')
    plt.ylabel('Accuracy')
    for index, value in enumerate(accuracies):
        value = round(value, 2)
        plt.text(index, value + 1, str(value))
    plt.title('AD_HF01_1 Model Accuracy by Session (dimesnion = 6)')
    plt.savefig('model_accuracy_by_session.pdf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select model and session name')
    parser.add_argument('--model_path', type=str, default='mlp_model_6.pth', help='Select model path')
    parser.add_argument('--reduction', type=str, default='umap', help='Select dimensionality reduction method')
    parser.add_argument('--dimension', type=int, default=6, help='Select dimensionality of the encoded representation')
    parser.add_argument('--session', type=str, default='AD_HF02_2', help='Select session name')
    parser.add_argument('--plot', action='store_true', help='Plot results by different subjects')
    args = parser.parse_args()
    model_path, reduction, dimension, session_name, plot_acc = args.model_path, args.reduction, args.dimension, args.session, args.plot
    if not plot_acc:
        run(model_path, reduction, dimension, session_name)
    else:
        print("Running all sessions...")
        run_all(model_path, reduction, dimension)
    
    with open('transfer_accuracies.json', 'r') as file:
        accuracies = json.load(file)
    sessions = ['AD_HF01_1', 'AD_HF02_2', 'AD_HF02_4', 'NN_syn_01', 'NN_syn_02']
    plt.figure(figsize=(10, 8))
    plt.bar(sessions, accuracies)
    plt.xlabel('Session')
    plt.ylabel('Accuracy')
    for index, value in enumerate(accuracies):
        value = round(value, 2)
        plt.text(index, value + 1, str(value))
    plt.title('AD_HF01_1 Model Accuracy by Session (dimesnion = 6)')
    plt.savefig('model_accuracy_by_session.pdf')
    