import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from blind_localization.data_loading import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold
from blind_localization.alignment import *

def compute_similarity(raw_signal, channel_idx_train, model_path, pre_train=False):
    """
    Here we compute cross correlation between each channel to each training channels
    Output: n_channels * n_train_channels
    """
    sorted_channel_idx_train = np.sort(channel_idx_train)

    if pre_train:
        corr_table = np.load(model_path)
        return corr_table

    norms = np.linalg.norm(raw_signal, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1, norms)
    normalized_signal = raw_signal / safe_norms
    normalized_signal[norms[:, 0] == 0] = 0

    corr_table = np.dot(normalized_signal, normalized_signal.T)
    corr_table = corr_table[:, sorted_channel_idx_train]
    np.save(model_path, corr_table)

    return corr_table


def compute_channel_features(corr_table, channel_region_map, channel_idx_train):
    """
    Here we compute the features for each channel, which represents
    the max correlation score to training channels in cortex, CA1, CA2, CA3, DG
    """

    # find all channels in cortex, CA1, CA2, CA3, DG
    channel_dict = {"cortex":[], "CA1":[], "CA2":[], "CA3":[], "DG":[], "UNK":[]}
    sorted_channel_idx_train = np.sort(channel_idx_train)

    for i in sorted_channel_idx_train:
        row = channel_region_map[channel_region_map["channels"] == i]
        region = row.iloc[0]["regions"] if len(row) > 0 else "UNK"
        channel_dict[region].append(i)
        
    index_map = dict(zip(sorted_channel_idx_train, np.arange(len(sorted_channel_idx_train))))

    # max correlation score to channels in cortex, CA1, CA2, CA3, DG
    channel_features = np.zeros((len(corr_table), 6))

    for i in range(len(corr_table)):
        for r, train_channels in enumerate(channel_dict.values()):
            for c in train_channels:
                if c != i:
                    channel_features[i][r] = max(corr_table[i][index_map[c]], channel_features[i][r])
    
    return channel_features


def get_channel_ridx_map(channel_region_map):
    """
    Here we convert channel region map to channel region idx map 
    """
    mapping = {"cortex":0, "CA1":1, "CA2":2, "CA3":3, "DG":4, "UNK":5}
    channel_ridx_map = channel_region_map.replace(mapping)
    channel_ridx_map = channel_ridx_map.to_numpy().astype(int)

    sorted_indices = channel_ridx_map[:, 0].argsort()
    channel_ridx_map = channel_ridx_map[sorted_indices]

    return channel_ridx_map


def get_train_test_data(channel_features, channel_ridx_map, random_state=66, val_size=0.2):
    idx_test = np.arange(len(channel_ridx_map))[channel_ridx_map[:, 1] == 5]

    mask = np.isin(np.arange(len(channel_ridx_map)), idx_test)
    masked_channel_region_map = channel_ridx_map[~mask]
    inv_masked_channel_region_map = channel_ridx_map[mask]
    
    X = channel_features[masked_channel_region_map[:, 0]][:, :5]
    X_test = channel_features[inv_masked_channel_region_map[:, 0]][:, :5]
    y = masked_channel_region_map[:, 1]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val


def knn_train(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
    knn.fit(X_train, y_train)
    return knn


def knn_classify(knn, X_test, y_test):
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) if y_test is not None else None
    return y_pred, accuracy


def knn_predict_train(knn, X_train, y_train):
    y_pred_train = []

    # Temporarily replace each point with a large value, predict, then reset
    for i in range(len(X_train)):
        original_value = X_train[i].copy()
        X_train[i] = np.full(X_train.shape[1], 1e5)

        knn.fit(X_train, y_train)

        prediction = knn.predict([original_value])
        y_pred_train.append(prediction)

        X_train[i] = original_value

    # Reset the model with the original dataset
    knn.fit(X_train, y_train)
    y_pred_train = np.array(y_pred_train).flatten()
    accuracy = accuracy_score(y_train, y_pred_train)

    return y_pred_train, accuracy


def find_optimal_number_of_labels(raw_signal, channel_ridx_map, channel_region_map, D):
    train_sizes = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    n_training_labels = []
    train_acc = []
    val_acc = []

    for train_size in train_sizes:
        channel_idx_train, channel_idx_val, channel_idx_test, _ = get_train_test_indices(channel_ridx_map, random_state=66, train_size=train_size, custom_split=True)
        print(f"Training labels n = {len(channel_idx_train)}")

        model_path = f"C:/VSCode/blind_localization/model/xcorr_AD_HF01_val_size/xcorr_AD_HF01ts{int(train_size*10)}.npy"
        corr_table = compute_similarity(raw_signal, channel_idx_train, model_path, pre_train=False)
        channel_features = compute_channel_features(corr_table, channel_region_map, channel_idx_train)

        n_training_labels.append(len(channel_idx_train))
        X_train, X_val, X_test, y_train, y_val = load_train_test_data(channel_features, D, channel_ridx_map, random_state=66, train_size=train_size, custom_split=True)
        model = ProbablisticClassifier()
        model.fit(X_train, y_train)

        _, y_pred_train = model.predict(X_train[0], X_train[1])
        _, y_pred_val = model.predict(X_val[0], X_val[1])

        train_acc.append(accuracy_score(y_train, y_pred_train))
        val_acc.append(accuracy_score(y_val, y_pred_val))

    plt.plot(n_training_labels, train_acc, label="train")
    plt.scatter(n_training_labels, train_acc)
    plt.plot(n_training_labels, val_acc, label="val")
    plt.scatter(n_training_labels, val_acc)
    plt.legend()
    plt.xlabel("n training labels")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. n training labels")
    plt.savefig("C:/VSCode/blind_localization/results/accuracy_curve")
    plt.show()


def cross_validate_bayes(model, channel_features, D, channel_ridx_map, transfer=False, random_state=66, custom_split=True, distance=True, transferred_channel_idx_train=None):
    if not transfer:
        X_train, X_val, X_test, y_train, y_val = load_train_test_data(channel_features, D, channel_ridx_map, random_state=66, train_size=0.8, custom_split=custom_split)
    else:
        X_train, X_val, X_test, y_train, y_val = load_train_test_data(channel_features, D, channel_ridx_map, random_state=66, train_size=0.8, transferred_channel_idx_train=transferred_channel_idx_train, custom_split=custom_split)
        
    X_features = np.concatenate([X_train[0], X_val[0]])
    X_dist = np.concatenate([X_train[1], X_val[1]])

    X = np.concatenate([X_features, X_dist], axis=1)
    y = np.concatenate([y_train, y_val])

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    indices = np.arange(len(X))
    train_scores, val_scores = [], []

    for train_index, test_index in kf.split(indices):
        X_train = X[train_index][:, :5], X[train_index][:, 5:]
        X_val = X[test_index][:, :5], X[test_index][:, 5:]
        y_train, y_val = y[train_index], y[test_index]
        
        if not transfer:
            model.fit(X_train, y_train)

        _, y_pred_train = model.predict(X_train[0], X_train[1], distance=distance)
        _, y_pred_val = model.predict(X_val[0], X_val[1], distance=distance)

        train_scores.append(accuracy_score(y_train, y_pred_train))
        val_scores.append(accuracy_score(y_val, y_pred_val))
    
    print("Training accuracy: ", sum(train_scores)/len(train_scores))
    print("Validation accuracy: ", sum(val_scores)/len(val_scores))

    return train_scores, val_scores


def cross_validate_knn(model, channel_features, D, channel_ridx_map, transfer=False, custom_split=True, random_state=66, distance=True, transferred_channel_idx_train=None):
    if not transfer:
        X_train, X_val, X_test, y_train, y_val = load_train_test_data(channel_features, D, channel_ridx_map, random_state=66, train_size=0.8, custom_split=custom_split)
    else:
        X_train, X_val, X_test, y_train, y_val = load_train_test_data(channel_features, D, channel_ridx_map, random_state=66, train_size=0.8, transferred_channel_idx_train=transferred_channel_idx_train, custom_split=custom_split)
        
    X_features = np.concatenate([X_train[0], X_val[0]])
    X_dist = np.concatenate([X_train[1], X_val[1]])

    X = np.concatenate([X_features, X_dist], axis=1) if distance else X_features
    y = np.concatenate([y_train, y_val])

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    indices = np.arange(len(X))
    train_scores, val_scores = [], []

    for train_index, test_index in kf.split(indices):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

        if not transfer:
            model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        train_scores.append(accuracy_score(y_train, y_pred_train))
        val_scores.append(accuracy_score(y_val, y_pred_val))
    
    print("Training accuracy: ", sum(train_scores)/len(train_scores))
    print("Validation accuracy: ", sum(val_scores)/len(val_scores))

    return train_scores, val_scores


def simulate_transfer_training_size(raw_signal, channel_ridx_map, channel_region_map, D, raw_signal_new, channel_region_map_new, custom_split=True):
    train_sizes = [0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    n_training_labels = []
    train_acc = []
    val_acc = []

    random_seeds = [11, 13, 45, 35, 67]
    for random_state in random_seeds:
        # train model using source session
        channel_idx_train, channel_idx_val, channel_idx_test, _ = get_train_test_indices(channel_ridx_map, random_state=random_state, train_size=0.8, custom_split=custom_split)

        model_path = "C:/VSCode/blind_localization/model/xcorr_AD_HF01_seed.npy"
        corr_table = compute_similarity(raw_signal, channel_idx_train, model_path, pre_train=False)
        channel_features = compute_channel_features(corr_table, channel_region_map, channel_idx_train)
        
        X_train_source, X_val_source, X_test_source, y_train_source, y_val_source = load_train_test_data(channel_features, D, channel_ridx_map, random_state=random_state, train_size=0.8, custom_split=custom_split)
        model = ProbablisticClassifier()
        model.fit(X_train_source, y_train_source)

        # transfer model using training data in target session
        for train_size in train_sizes:
            channel_ridx_map_new = get_channel_ridx_map(channel_region_map_new)
            channel_idx_train_new, channel_idx_val_new, channel_idx_test_new, _ = get_train_test_indices(channel_ridx_map_new, train_size=train_size, custom_split=custom_split)

            model_path_new = "C:/VSCode/blind_localization/model/xcorr_AD_HF02_2_seed.npy"
            corr_table_new = compute_similarity(raw_signal_new, channel_idx_train_new, model_path_new, pre_train=False)
            channel_features_new = compute_channel_features(corr_table_new, channel_region_map_new, channel_idx_train_new)

            X_train_target, X_val_target, X_test_target, y_train_target, y_val_target = load_train_test_data(channel_features_new, D, channel_ridx_map_new, custom_split=custom_split,
                                                            random_state=random_state, train_size=train_size, transferred_channel_idx_train = channel_idx_train)

            tform = supervised_align(channel_features, channel_ridx_map, X_train_target, y_train_target)

            X_train_transformed = transform(X_train_target[0], tform), X_train_target[1]
            X_val_transformed = transform(X_val_target[0], tform), X_val_target[1]

            _, y_pred_train = model.predict(X_train_transformed[0], X_train_transformed[1])
            _, y_pred_val = model.predict(X_val_transformed[0], X_val_transformed[1])

            train_acc.append(accuracy_score(y_train_target, y_pred_train))
            val_acc.append(accuracy_score(y_val_target, y_pred_val))
            n_training_labels.append(len(X_train_target[0]))
    
    n_training_labels = n_training_labels[:len(train_sizes)]
    train_acc = np.array(train_acc).reshape(len(random_seeds), len(train_sizes)).mean(axis=0)
    val_acc = np.array(val_acc).reshape(len(random_seeds), len(train_sizes)).mean(axis=0)

    plt.plot(n_training_labels, train_acc, label="train")
    plt.scatter(n_training_labels, train_acc)
    plt.plot(n_training_labels, val_acc, label="val")
    plt.scatter(n_training_labels, val_acc)
    plt.legend()
    plt.xlabel("n alignment labels")
    plt.ylabel("Accuracy")
    plt.title("transfer_learning_accuracy")
    plt.savefig("C:/VSCode/blind_localization/results/transfer_learning_accuracy")
    plt.show()