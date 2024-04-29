from blind_localization.pipeline import *
from blind_localization.contrastive import *
from blind_localization.losses import SupConLoss
from sklearn.model_selection import train_test_split
from mlp import MLP
import matplotlib
import pickle

def run_pipeline(session_names, config):
    # load train, val, and test session data and labels
    print("Loading source and target session data...")

    X_train, X_val, y_train, y_val = None, None, None, None
    # X_train, X_val, y_train, y_val = [], [], [], []
    
    if config['train']:
        for source_session in session_names[:-1]:
            print("Loading source session data for: ", source_session)
            raw_signal, D, channel_ridx_map, train_test_indices = load_data_labels(source_session, config)
            print("Raw signal shape: ", raw_signal.shape)
            print("D shape: ", D.shape)
            print("channel_ridx_map shape: ", channel_ridx_map.shape)
            raw_signal, D, channel_ridx_map = transform_data(raw_signal, D, channel_ridx_map)
            print("transformed raw signal shape: ", raw_signal.shape)
            print("transformed D shape: ", D.shape)
            print("transformed channel_ridx_map shape: ", channel_ridx_map.shape)
            
            channel_features = compute_neural_representation(raw_signal, source_session, config)
            X_train_source, X_val_source, X_test_source, y_train_source, y_val_source = load_train_test_data(channel_features, D, channel_ridx_map, random_state=66, train_size=config["source_train_size"], custom_split=config["custom_split"])

            X_train = np.vstack([X_train, X_train_source[0]]) if X_train is not None else X_train_source[0]
            X_val = np.vstack([X_val, X_val_source[0]]) if X_val is not None else X_val_source[0]
            y_train = np.hstack([y_train, y_train_source]) if y_train is not None else y_train_source
            y_val = np.hstack([y_val, y_val_source]) if y_val is not None else y_val_source
            
        raw_signal, D, channel_ridx_map, train_test_indices = load_data_labels(session_names[-1], config)
        print("Raw signal shape: ", raw_signal.shape)
        print("D shape: ", D.shape)
        print("channel_ridx_map shape: ", channel_ridx_map.shape)
        raw_signal, D, channel_ridx_map = transform_data(raw_signal, D, channel_ridx_map)
        print("transformed raw signal shape: ", raw_signal.shape)
        print("transformed D shape: ", D.shape)
        print("transformed channel_ridx_map shape: ", channel_ridx_map.shape)
        channel_features = compute_neural_representation(raw_signal, source_session, config)
        X_train_source, X_val_source, X_test_source, y_train_source, y_val_source = load_train_test_data(channel_features, D, channel_ridx_map, random_state=66, train_size=config["source_train_size"], custom_split=config["custom_split"])
        X_test = np.vstack([X_train_source[0], X_val_source[0]])
        y_test = np.hstack([y_train_source, y_val_source])
        
        print("Training data shape: ", X_train.shape, y_train.shape)
        print("Validation data shape: ", X_val.shape, y_val.shape)
        print("Test data shape: ", X_test.shape, y_test.shape)
        
        # load train and validation dataset
        train_dataset = RawSignalDataset(X_train, y_train)
        val_dataset = RawSignalDataset(X_val, y_val)
        test_dataset = RawSignalDataset(X_test, y_test)

        train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)
        
        # train and evaluate the model
        model = ContrastiveEncoder(input_size=3, output_size=3)
        criterion = SupConLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

        training_losses = []
        validation_losses = []

        for epoch in range(config["epochs"]):
            train_loss = train(model, train_dataloader, optimizer, criterion)
            val_loss = validation(model, val_dataloader, criterion)

            print(f'Epoch [{epoch+1}/{config["epochs"]}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            training_losses.append(train_loss)
            validation_losses.append(val_loss)

        plt.plot(training_losses, label="train")
        plt.plot(validation_losses, label="validation")
        plt.legend()
        # plt.show()
        plt.savefig("contrastive_loss_ft.png")
        
        # save the model
        torch.save(model.state_dict(), 'contrastive_encoder.pth')
        
    elif config['test']:
        # ----------------------- Use Original Data Points -----------------------
        # load the model
        model = ContrastiveEncoder(input_size=3, output_size=3)
        model.load_state_dict(torch.load('contrastive_encoder.pth'))
        
        print("Creating original data points...")
        for source_session in session_names[:-1]:
            print("Loading source session data for: ", source_session)
            raw_signal, D, channel_ridx_map, train_test_indices = load_data_labels(source_session, config)
            channel_features = compute_neural_representation(raw_signal, source_session, config)
            X_train_source, X_val_source, X_test_source, y_train_source, y_val_source = load_train_test_data(channel_features, D, channel_ridx_map, random_state=66, train_size=config["source_train_size"], custom_split=config["custom_split"])

            X_train = np.vstack([X_train, X_train_source[0]]) if X_train is not None else X_train_source[0]
            X_val = np.vstack([X_val, X_val_source[0]]) if X_val is not None else X_val_source[0]
            y_train = np.hstack([y_train, y_train_source]) if y_train is not None else y_train_source
            y_val = np.hstack([y_val, y_val_source]) if y_val is not None else y_val_source
            
        raw_signal, D, channel_ridx_map, train_test_indices = load_data_labels(session_names[-1], config)
        channel_features = compute_neural_representation(raw_signal, source_session, config)
        X_train_source, X_val_source, X_test_source, y_train_source, y_val_source = load_train_test_data(channel_features, D, channel_ridx_map, random_state=66, train_size=config["source_train_size"], custom_split=config["custom_split"])
        X_test = np.vstack([X_train_source[0], X_val_source[0]])
        y_test = np.hstack([y_train_source, y_val_source])
        
        print("Training data shape: ", X_train.shape, y_train.shape)
        print("Validation data shape: ", X_val.shape, y_val.shape)
        print("Test data shape: ", X_test.shape, y_test.shape)
        
        train_dataset = RawSignalDataset(X_train, y_train)
        val_dataset = RawSignalDataset(X_val, y_val)
        test_dataset = RawSignalDataset(X_test, y_test)

        train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)
        
        X_train_embed, y_train_embed = extract_features(model, train_dataloader)
        X_val_embed, y_val_embed = extract_features(model, val_dataloader)
        X_test_embed, y_test_embed = extract_features(model, test_dataloader)
        # X_train_embed, X_val_embed = X_train_embed[:, 0, :],  X_val_embed[:, 0, :]
        
        # # Save embeddings
        save_embeddings(X_train_embed, y_train_embed, './contrastive_embeddings_ft/train_embeddings.pkl')
        save_embeddings(X_val_embed, y_val_embed, './contrastive_embeddings_ft/train_embeddings.pkl')
        save_embeddings(X_test_embed, y_test_embed, './contrastive_embeddings_ft/train_embeddings.pkl')

    else:
        # Load embeddings
        X_train_embed, y_train_embed = load_embeddings('./contrastive_embeddings_ft/train_embeddings.pkl')
        X_val_embed, y_val_embed = load_embeddings('./contrastive_embeddings_ft/val_embeddings.pkl')
        X_test_embed, y_test_embed = load_embeddings('./contrastive_embeddings_ft/test_embeddings.pkl')

    fig = plt.figure(figsize=(18, 6)) 
    ax = fig.add_subplot(131, projection='3d')
    ax.set_title('Raw Training Data')
    scatter = ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap='viridis')
    ax = fig.add_subplot(132, projection='3d')
    ax.set_title('Encoded Training Data')
    scatter = ax.scatter(X_train_embed[:, 0], X_train_embed[:, 1], X_train_embed[:, 2], c=y_train_embed, cmap='viridis')
    ax = fig.add_subplot(133, projection='3d')
    ax.set_title('Encoded Test Data')
    scatter = ax.scatter(X_test_embed[:, 0], X_test_embed[:, 1], X_test_embed[:, 2], c=y_test_embed, cmap='viridis')
    plt.savefig("contrastive_embedding_ft.png", dpi=300)

    exit()

    # classifier = ProbablisticClassifier()
    # classifier.fit(X_train_embed, y_train_embed, distance=config['distance'])

    # _, y_pred_train = classifier.predict(X_train_embed[:100], None, distance=config["distance"])
    # _, y_pred_val = classifier.predict(X_val_embed[:100], None, distance=config["distance"])

    # print("Training accuracy: ", accuracy_score(y_train_embed[:100], y_pred_train))
    # print("Validation accuracy: ", accuracy_score(y_val_embed[:100], y_pred_val))

    # # test the encoder on unseen session
    # _, y_pred_test = classifier.predict(X_test_embed, None, distance=config["distance"])
    # print("Validation accuracy: ", accuracy_score(y_test_embed, y_pred_test))

    # if config["visualize"]:
    #     visualize_accuracy(y_train_embed, y_val_embed, y_pred_train, y_pred_val, alignment=False)
    #     visualize_confusion_matrix(y_val_embed, y_pred_val)
    #     visualize_confusion_matrix(y_test_embed, y_pred_test)
    
    input_size = embeddings.shape[1]
    num_classes = len(np.unique(labels))

    mlp = MLP(X=X_train_val, y=y_train_val, input_size=input_size, hidden_size=64, num_classes=num_classes, batch_size=32, learning_rate=0.001, num_epochs=100, model_save_path='mlp_model.pth')
    mlp.train()
    
    test_accuracy = mlp.test(X_test, y_test)
    print(f'Test accuracy: {test_accuracy}%')


def transform_data(raw_signal, D, channel_ridx_map):
    transformed_raw_signal = []
    transformed_channel_ridx_map = []
    
    good_idx = channel_ridx_map[:, 0]
    idx = 0

    for i in range(len(good_idx)):
        parts = [raw_signal[i][j:j + 20000] for j in range(0, 60000, 20000)]
        while idx < good_idx[i]:
            # print("Skipping channel: ", idx)
            for k in range(3):
                part = parts[k]
                transformed_raw_signal.append(part)
            idx += 1
        label = channel_ridx_map[i][1]
        for k in range(3):
            part = parts[k]
            transformed_raw_signal.append(part)
            transformed_channel_ridx_map.append([len(transformed_raw_signal)-1, label])
        idx += 1
    
    original_size = D.shape[0]
    transformed_D = np.zeros((original_size * 3, original_size * 3))
    
    for i in range(original_size):
        for j in range(original_size):
            transformed_D[i*3:(i+1)*3, j*3:(j+1)*3] = D[i, j]

    return np.array(transformed_raw_signal), np.array(transformed_D), np.array(transformed_channel_ridx_map)

def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0

    for data, label in dataloader:
        signal, pos_signal, neg_signal = data
        optimizer.zero_grad()

        z_i = model(signal).unsqueeze(1)
        # don't need to use z_j and z_k
        # z_j = model(pos_signal)
        # z_k = model(neg_signal)

        loss = criterion(z_i, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validation(model, dataloader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, label in dataloader:
            signal, pos_signal, neg_signal = data

            z_i = model(signal).unsqueeze(1)
            # z_j = model(pos_signal)
            # z_k = model(neg_signal)

            loss = criterion(z_i, label)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for data, label in dataloader:
            feature = model(data[0])
            features.append(feature)
            labels.append(label)

    features = torch.cat(features, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()
    return features, labels


def plot_comparison(data_pts, embeddings, labels):
    fig = plt.figure(figsize=(16, 8))

    # Plotting the original data points
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(data_pts[:, 0], data_pts[:, 1], data_pts[:, 2], c=labels, cmap='viridis')
    ax1.set_title('Original Data Points')

    # Plotting the embeddings
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=labels, cmap='viridis')
    ax2.set_title('Embeddings')

    fig.colorbar(scatter2, ax=[ax1, ax2], shrink=0.5, location='right', label='Class Labels')

    plt.savefig('comparison.png')
    
    
def save_embeddings(embeddings, labels, filename):
    with open(filename, 'wb') as handle:
        pickle.dump({'embeddings': embeddings, 'labels': labels}, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_embeddings(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        return data['embeddings'], data['labels']
    
if __name__ == "__main__":

    config = {
        "custom_split": True,
        "source_train_size":0.8,
        "target_train_size":0.8,
        "distance": False,
        "pre_train": False,
        "visualize": True,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 300,
        "train": False,
        "test": True,
    }

    # run pipeline for a given source and target session
    session_names = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1", "AD_HF03_2"]
    # session_names = ["AD_HF02_2", "AD_HF02_4", "AD_HF03_1", "AD_HF03_2", "AD_HF01_1"]
    run_pipeline(session_names, config)
