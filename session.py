import pandas as pd
import argparse

from data_preprocess import *
from mlp import *
from trans import *
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def run(session_name):
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

    processed_data = dimension_reduct(normalized_data, method="umap", n_components=10)
    
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
    
    mlp = MLP(X, y, input_size=10)
    mlp.train()
    mlp_accuracy = mlp.evaluate()
    
    X = torch.tensor(X)
    y = torch.tensor(y)

    batch_size = 32
    learning_rate = 0.001
    num_epochs = 100

    transformer_framework = TransformerFramework(X=X, y=y, input_dim=10, num_classes=5,
                                                batch_size=batch_size, learning_rate=learning_rate,
                                                num_epochs=num_epochs)

    print("Starting training...")
    transformer_framework.train()
    print("Evaluating model...")
    transformer_accuracy = transformer_framework.evaluate()
            
    return mlp_accuracy, transformer_accuracy

def run_all():
    all_sessions = ['AD_HF01_1', 'AD_HF02_2', 'AD_HF02_4', 'NN_syn_01', 'NN_syn_02']
    
    mlp_accuracies = []
    trans_accuracies = []
    for session in all_sessions:
        print(f'Running session: {session}')
        mlp_accuracy, trans_accuracy = run(session)
        mlp_accuracies.append(mlp_accuracy)
        trans_accuracies.append(trans_accuracy)
    
    with open('mlp_accuracies_10.json', 'w') as file:
        json.dump(mlp_accuracies, file)
    with open('trans_accuracies_10.json', 'w') as file:
        json.dump(trans_accuracies, file)
    
    # [88.78504672897196, 87.85046728971963, 88.78504672897196, 88.78504672897196, 86.91588785046729]
    # [93.28358208955224, 78.50467289719626, 89.7196261682243, 78.50467289719626, 78.50467289719626]
        
    # x = np.arange(len(all_sessions))  # the label locations
    # width = 0.35  # the width of the bars

    # fig, ax = plt.subplots()
    # rects1 = ax.bar(x - width/2, mlp_accuracies, width, label='MLP')
    # rects2 = ax.bar(x + width/2, trans_accuracies, width, label='Transformer')

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Accuracy (%)')
    # ax.set_title('Accuracy by model and session')
    # ax.set_xticks(x)
    # ax.set_xticklabels(all_sessions)
    # ax.legend()

    # # Function to attach a text label above each bar, displaying its height.
    # def autolabel(rects):
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.annotate('{}'.format(height),
    #                     xy=(rect.get_x() + rect.get_width() / 2, height),
    #                     xytext=(0, 3),  # 3 points vertical offset
    #                     textcoords="offset points",
    #                     ha='center', va='bottom')

    # autolabel(rects1)
    # autolabel(rects2)

    # fig.tight_layout()

    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select model and session name')
    parser.add_argument('--model', type=str, default='mlp_model_6.pth', help='Select model path')
    parser.add_argument('--reduction', type=str, default='umap', help='Select dimensionality reduction method')
    parser.add_argument('--dimension', type=int, default=10, help='Select dimensionality of the encoded representation')
    parser.add_argument('--session', type=str, default='AD_HF02_2', help='Select session name')
    parser.add_argument('--plot', action='store_true', help='Plot results by different subjects')
    args = parser.parse_args()
    model, reduction, dimension, session_name, plot_acc = args.model, args.reduction, args.dimension, args.session, args.plot
    print("Running all sessions...")
    # run_all()
    
    all_sessions = ['AD_HF01_1', 'AD_HF02_2', 'AD_HF02_4', 'NN_syn_01', 'NN_syn_02']
    mlp_accuracies = [87.31343283582089, 86.91588785046729, 81.44329896907216, 95.49549549549549, 90.990990990991]
    trans_accuracies = [91.7910447761194, 78.50467289719626, 69.0721649484536, 97.29729729729729, 94.5945945945946]
    
    x = np.arange(len(all_sessions))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    # round to 2 decimal places
    mlp_accuracies = [round(elem, 2) for elem in mlp_accuracies ]
    trans_accuracies = [round(elem, 2) for elem in trans_accuracies ]
    rects1 = ax.bar(x - width/2, mlp_accuracies, width, label='MLP')
    rects2 = ax.bar(x + width/2, trans_accuracies, width, label='Transformer')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by model and session')
    ax.set_xticks(x)
    ax.set_xticklabels(all_sessions)
    ax.legend()

    # Function to attach a text label above each bar, displaying its height.
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.savefig('10.pdf')