import pandas as pd
import argparse

from data_preprocess import *
from mlp import *
from trans import *


def run():
    # add parser for model selection and session name
    parser = argparse.ArgumentParser(description='Select model and session name')
    parser.add_argument('--model', type=str, default='mlp', help='Select model')
    parser.add_argument('--session', type=str, default='AD_HF01_1', help='Select session name')
    args = parser.parse_args()
    model, session_name = args.model, args.session
    
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

    if model == "mlp":
        processed_data = reduction(normalized_data, method="umap")
        
        # # store the processed data
        # np.save('processed_data.npy', processed_data)
        # np.save('channel_index_label.npy', channel_index_label)
        
        # read the processed data
        # processed_data = np.load('processed_data.npy')
        # channel_index_label = np.load('channel_index_label.npy')
        # print(processed_data.shape)
        # print(len(channel_index_label))
        
        X = []
        y = []
        label_index = {'CA1':0,
                        'CA2':1,
                        'CA3':2,
                        'DG':3,
                        'cortex':4}
        for i in range(len(channel_index_label)):
            # First, check if the value is NaN
            if pd.isna(channel_index_label[i]):
                continue  # Skip this iteration if the value is NaN
            label = str(channel_index_label[i]).strip().lower()  # Convert to string first, then normalize
            if label == "unk" or label == "nan":
                continue
            X.append(processed_data[i])
            y.append(label_index[channel_index_label[i]])
            
        print(len(y))
        print(np.unique(y, return_counts=True))
        
        mlp = MLP(X, y)
        mlp.train()
        mlp.evaluate()

    elif model == "transformer":
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
            # First, check if the value is NaN
            if pd.isna(channel_index_label[i]):
                continue  # Skip this iteration if the value is NaN
            label = str(channel_index_label[i]).strip().lower()  # Convert to string first, then normalize
            if label == "unk" or label == "nan":
                continue
            X.append(normalized_data[i])
            y.append(label_index[channel_index_label[i]])
            
        print(len(y))
        print(np.unique(y, return_counts=True))

        # Parameters for the CombinedFramework
        encoding_dim = 1024  # Dimensionality of the encoded representation
        batch_size = 4  # Adjust based on your system's memory capacity
        learning_rate = 0.001
        num_epochs = 10  # For demonstration, use a small number of epochs

        # Instantiate the combined framework
        combined_framework = CombinedFramework(X, y, input_dim, encoding_dim, num_classes,
                                            batch_size=batch_size, learning_rate=learning_rate, num_epochs=num_epochs)
        
        # Train the model
        print("Starting training...")
        combined_framework.train()

        # Evaluate the model
        print("Evaluating model...")
        combined_framework.evaluate()

if __name__ == "__main__":
    run()