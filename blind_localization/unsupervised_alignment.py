from blind_localization.alignment import *
from blind_localization.model import *
from blind_localization.visualize import *

def generate_channel_features(corr_table, channel_labels):
    """
    Here we compute the features for each channel, which represents
    the max correlation score to training channels in cortex, CA1, CA2, CA3, DG
    """
    # max correlation score to channels in cortex, CA1, CA2, CA3, DG
    channel_features = np.zeros((len(corr_table), 5))

    for i in range(len(corr_table)):
        for j in range(len(corr_table)):
            r = int(channel_labels[j])
            channel_features[i][r] = max(corr_table[i][j], channel_features[i][r])

    return channel_features

def unsupervised_alignment(channel_features_source, channel_ridx_map_source, model, corr_table_target, D, channel_ridx_map_new, channel_features_new, T=20, threshold=0.95):
    
    # randomly initialize channel labels by priors
    y_diffs, accuracys = [], []
    values, counts = np.unique(channel_ridx_map_source[:, 1], return_counts=True)
    counts = np.delete(counts, np.where(values==5))
    values = np.delete(values, np.where(values==5))

    y_target_gt = channel_ridx_map_new[:, 1][channel_ridx_map_new[:, 1] != 5]
    channel_indices = channel_ridx_map_new[:, 0][channel_ridx_map_new[:, 1] != 5]

    n_labels = len(channel_indices)
    y_targets = np.zeros((T, n_labels))

    # y_targets[0] = np.random.choice(values, size=n_labels, p=counts/sum(counts))
    y_targets[0] = y_target_gt
    y_diffs.append(0)
    accuracys.append(accuracy_score(y_targets[0], y_target_gt))
    
    y_diff = 0
    t = 1

    while t < T and y_diff <= threshold:
        # generate 5-dim similarity features for each channel
        X_target = generate_channel_features(corr_table_target[channel_indices], y_targets[t-1]), D

        # supervised align with training session
        tform = supervised_align(channel_features_source, channel_ridx_map_source, X_target, y_targets[t-1])
        X_target_transformed = transform(X_target[0], tform), X_target[1]

        # only for visualizing distributions (not supposed to know these information)
        channel_features_transformed = np.insert(transform(channel_features_new[:, :-1], tform), len(channel_features_new[0])-1, 0, axis=1)
        visualize_alignment(channel_features_source, channel_ridx_map_source, channel_features_new, channel_ridx_map_new, channel_features_transformed)
        
        # infer new labels from aligned channel features using Bayesian classifier
        _, y_target = model.predict(X_target_transformed[0], X_target_transformed[1], distance=False)
        y_targets[t] = y_target

        # calculate the difference from last iteration
        y_diff = np.sum(y_targets[t] == y_targets[t-1]) / n_labels
        y_diffs.append(y_diff)
        accuracy = accuracy_score(y_targets[t], y_target_gt)
        accuracys.append(accuracy)

        print(f"Epoch {t}/{T}, y_similarity: {y_diff}, accuracy: {accuracy}")
        t += 1

    plt.plot(np.arange(t), y_diffs, label="y_similarity")
    plt.scatter(np.arange(t), y_diffs)
    plt.plot(np.arange(t), accuracys, label="accuracy")
    plt.scatter(np.arange(t), accuracys)
    plt.title("Learning curve")
    plt.legend()
    plt.show()

    return y_targets[t-1]