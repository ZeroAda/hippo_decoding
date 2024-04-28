import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from sklearn.decomposition import PCA
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from functools import partial

def visualize_raw_signals(raw_signal, channel_region_map, skipped_channels, prediction=None, indices=None, T = 1000, sr = 20000):
    """
    Visualize raw signals colored by region (skipping bad channels)
    if prediction is None, color represents ground truth labels
    otherwise, color represents the train/val/test prediction
    """
    offset = 5
    sr_converted = int(sr/1000) #samples/ms
    t_offset = T + 100

    time = np.arange(0, T, 1/sr_converted)

    colors = ["red", "orange", "green", "blue", "magenta", "black"]
    regions = ["cortex", "CA1", "CA2", "CA3", "DG", "UNK"]
    regions_to_colors = dict(zip(regions, colors))
    
    skip_arr = np.zeros(len(raw_signal))
    skip_arr[skipped_channels] = 1

    if prediction:
        y_pred_train, y_pred_val, y_pred_test = prediction
        channel_idx_train, channel_idx_val, channel_idx_test = indices

    plt.figure(figsize=(18, 16))
    for shank in range(8):
        for i in range(128): 
            channel = shank*128+i

            row = channel_region_map[channel_region_map["channels"] == channel]
            region = row.iloc[0]["regions"] if len(row) > 0 else "UNK"

            if not skip_arr[channel]:
                if not prediction:
                    color = regions_to_colors[region]

                else:
                    if channel in channel_idx_val: 
                        r = int(y_pred_val[channel_idx_val == channel][0])
                        color = regions_to_colors[regions[r]]
                    elif channel in channel_idx_train:
                        r = int(y_pred_train[channel_idx_train == channel][0])
                        color = regions_to_colors[regions[r]]
                    elif channel in channel_idx_test:
                        r = int(y_pred_test[channel_idx_test == channel][0])
                        color = regions_to_colors[regions[r]]
                    else:
                        print(channel)
                        color = "black"
            
                plt.plot(time+t_offset*shank, raw_signal[channel, :T*sr_converted]+offset*i, c=color)

    legend_handles = [Line2D([0], [0], color=color, lw=2, label=list(regions_to_colors.keys())[i]) for i, color in enumerate(colors)]

    # plt.legend(handles=legend_handles)

    plt.xlabel("time(ms)")
    plt.ylabel("channel")
    plt.title("Raw signal detected by 8 shanks in 1000 ms")
    plt.show()


def visualize_channel_features(channel_features):
    fig, axes = plt.subplots(8, 1, figsize=(21, 9.6))

    for shank, ax in enumerate(axes.flat):
        im = ax.imshow(channel_features[shank*128:shank*128+128, :-1].T)
        ax.set_yticks(np.arange(5), ["cortex", "CA1", "CA2", "CA3" ,"DG"])
        ax.set_xticks(np.arange(0, 128, 10), np.arange(shank*128, shank*128+128, 10))

    title_obj = plt.suptitle("Similarity between channel to each brain region", fontsize=15)
    title_pos = title_obj.get_position()
    title_obj.set_position((title_pos[0] - 0.05, title_pos[1]))
    plt.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()


def visualize_accuracy(y_train, y_val, y_pred_train, y_pred, alignment=False, ax=None):
    accuracy_per_region_train = []
    accuracy_per_region_val= []
    regions = ["cortex", "CA1", "CA2", "CA3", "DG"]

    for r in range(5):
        if len(y_train[y_train == r]) > 0:
            train_acc = accuracy_score(y_train[y_train == r], y_pred_train[y_train == r])
            val_acc = accuracy_score(y_val[y_val == r], y_pred[y_val == r])
        else:
            train_acc, val_acc = 0, 0

        accuracy_per_region_train.append(train_acc)
        accuracy_per_region_val.append(val_acc)

    bar_width=0.2
    index = np.arange(len(regions))
    ax_flag = False

    if ax is None:
        fig, ax = plt.subplots()
        ax_flag = True

    label = 'Train' if not alignment else "Alignment"
    ax.bar(index, accuracy_per_region_train, bar_width, label=label)
    ax.bar(index + bar_width, accuracy_per_region_val, bar_width, label='Validation')

    for r in range(5):
        count_train = len(y_train[y_train == r])
        count_val = len(y_val[y_val == r])
        ax.text(r, accuracy_per_region_train[r], str(count_train), ha='center', va='bottom')
        ax.text(r+bar_width, accuracy_per_region_val[r], str(count_val), ha='center', va='bottom')

    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(regions)
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.set_ylabel("Accuracy")
    ax.set_title("Region classification accuracy")
    ax.legend(loc='lower left')
    if ax_flag:
        plt.show()
    return ax

def visualize_confusion_matrix(y_true, y_pred):
    c_matrix = confusion_matrix(y_true, y_pred, labels=range(5))
    regions = ["cortex", "CA1", "CA2", "CA3", "DG"]

    sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=regions, yticklabels=regions)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Validation Confusion Plot')
    plt.show()


def animate_features(channel_feature_across_time, frames):
    fig, axes = plt.subplots(8, 1, figsize=(21, 9.6))
    images = []

    # Initialize subplots with zeros (or any initial image data)
    for i in range(8):
        image = axes[i].imshow(np.zeros((100, 100)), animated=True)
        images.append(image)

    def init():
        # Initialize each subplot with zeros (or any initial image data)
        for image in images:
            image.set_array(np.zeros((100, 100)))
        return images

    def update(frame):
        channel_features = channel_feature_across_time[frame]
        
        for shank, ax in enumerate(axes.flat):
            im = ax.imshow(channel_features[shank*128:shank*128+128, :-1].T)
            ax.set_yticks(np.arange(5), ["cortex", "CA1", "CA2", "CA3" ,"DG"])
            ax.set_xticks(np.arange(0, 128, 10), np.arange(shank*128, shank*128+128, 10))
            images[shank].set_array(im.get_array())

        return images

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
    plt.close()
    return HTML(ani.to_jshtml())


def animate_tensor(custom_update, tensor, frames, n_rows, n_cols, titles=None, width=20, height=4, vmin=0, vmax=1):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height))
    images = []

    if n_rows == 1 or n_cols == 1:
        axes = np.array(axes).reshape(n_rows, n_cols)

    for i in range(n_rows):
        for j in range(n_cols):
            if titles:
                axes[i][j].set_title(titles[i][j])
            image = axes[i][j].imshow(np.zeros((100, 100)), vmin=vmin, vmax=vmax, animated=True)
            images.append(image)

    def init():
        for image in images:
            image.set_array(np.zeros((100, 100)))
        return images

    update = partial(custom_update, axes=axes, images=images, tensor=tensor)

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
    plt.close()
    return HTML(ani.to_jshtml())


def animate(data, custom_plot_func, frames):
    fig, ax = plt.subplots(figsize=(8, 6))

    def init():
        return ax,

    def update(frame):
        frame_data = data[frame]
        custom_plot_func(ax, frame_data)
        return ax,

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False)
    
    plt.close(fig)  # Prevents duplicate display of static image
    return HTML(ani.to_jshtml())


def visualize_alignment(channel_features, channel_ridx_map, channel_features_new, channel_ridx_map_new, channel_features_transformed):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    xcorr = channel_features[channel_ridx_map[:, 0]][:, :-1]
    
    regions = ["cortex", "CA1", "CA2", "CA3", "DG"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    pca = PCA(n_components=2)
    pca.fit(xcorr)

    visualize_aligned_features(channel_features, channel_ridx_map, channel_features_new, 
                               channel_ridx_map_new, regions, colors, pca, title="Before Alignment", ax=axes[0])
    visualize_aligned_features(channel_features, channel_ridx_map, channel_features_transformed,
                               channel_ridx_map_new, regions, colors, pca, title="After Alignment",ax=axes[1])
    plt.show()


def visualize_aligned_features(channel_features, channel_ridx_map, channel_features_new, channel_ridx_map_new,
                            regions, colors, pca, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    for r in range(5):
        xcorr_new = channel_features_new[channel_ridx_map_new[:, 0]][channel_ridx_map_new[:, 1] == r][:, :-1] 
        if len(xcorr_new) > 0:
            data_pca = pca.transform(xcorr_new)
            ax.scatter(data_pca[:, 0], data_pca[:, 1], marker='x', color=colors[r], zorder=1, s=20)

        xcorr = channel_features[channel_ridx_map[:, 0]][channel_ridx_map[:, 1] == r][:, :-1]
        if len(xcorr) > 0:
            data_pca = pca.transform(xcorr)
            ax.scatter(data_pca[:, 0], data_pca[:, 1], marker='o', label=regions[r], color=colors[r],zorder=1, s=20)
    
    ax.set_title(title)
    ax.legend()

    return ax


def visualize_features_by_regions(ax, channel_features, r, element_width = 10, element_height = 0.5):

    concat_features = np.concatenate([channel_features[s*128:(s+1)*128, r:r+1] for s in range(8)], axis=1)

    if concat_features.max() != 0 and concat_features.max() != 0:
        normalized_features = (concat_features - concat_features.min()) / (concat_features.max() - concat_features.min())
    else:
        normalized_features = np.zeros_like(concat_features)

    rgba_color = plt.cm.viridis(normalized_features)

    for i in range(concat_features.shape[0]):
        for j in range(concat_features.shape[1]):
            rect = patches.Rectangle((j*element_width, i*element_height), element_width, element_height,
                                    linewidth=1, edgecolor='none', facecolor=rgba_color[i][j])
            ax.add_patch(rect)

    ax.set_xlim([0, element_width * concat_features.shape[1]])
    ax.set_ylim([0, element_height * concat_features.shape[0]])

    ax.set_xticks(np.arange(8)*element_width, np.arange(8), fontsize=12)
    ax.set_yticks(np.arange(0, 128, 10)*element_height, np.arange(0, 128, 10), fontsize=12)

    return ax


def visualize_features_dataset(channel_features_ls):
    n_figures = len(channel_features_ls)
    n_regions = len(channel_features_ls[0][0])

    session_names = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1", "AD_HF03_2", "NN_syn_01", "NN_syn_02"]
    regions = ["cortex", "CA1", "CA2", "CA3", "DG"]
    
    fig, axs = plt.subplots(n_regions, n_figures, figsize=(4*n_figures, 3*n_regions))

    for fig_num in range(n_figures):
        channel_features = channel_features_ls[fig_num]

        for region_index in range(n_regions):
            ax = axs[region_index, fig_num]
            visualize_features_by_regions(ax, channel_features, region_index)

            if region_index == 0:
                axs[region_index, fig_num].set_title(session_names[fig_num], fontsize=20)
            
            if fig_num == 0:
                axs[region_index, fig_num].set_ylabel(regions[region_index], fontsize=20)
        
    plt.tight_layout()
    plt.show()


def visualize_features_3d(session_names, reduced_signal_ls, channel_labels_ls, height=120, width=120, marker_size=2):
    n_rows = 1
    n_cols = len(session_names)
    df_ls = []
    
    for i, session_name in enumerate(session_names):
        channel_labels = channel_labels_ls[i]
        labels = np.argmax(channel_labels, axis=1) + 1
        channel_labels = channel_labels * labels[:, np.newaxis]

        class_labels = np.sum(channel_labels, axis=1).astype(int)
        colors = ["lightgray", "red", "orange", "green", "blue", "purple"]
        channel_colors = [colors[j] for j in class_labels]

        df = pd.DataFrame(reduced_signal_ls[i], columns=['x', 'y', 'z'])
        df['color'] = channel_colors
        df['session_name'] = session_name
        df_ls.append(df)

    fig = make_subplots(rows=n_rows, cols=n_cols,subplot_titles=tuple(session_names),
                        specs=[[{'type':'scatter3d'} for c in range(n_cols)] for r in range(n_rows)])

    for r in range(n_rows):
        for c in range(n_cols):
            df = df_ls[c]
            scatter = px.scatter_3d(df, x='x', y='y', z='z', color='color', color_discrete_sequence=df["color"].unique())
            for trace in scatter.data:
                fig.add_trace(trace, row=r+1, col=c+1)

    fig.update_traces(marker=dict(size=marker_size), showlegend=False, selector=dict(type='scatter3d'))

    fig.update_layout(height=height, width=width*len(session_names),margin=dict(l=0, r=0, b=0, t=0))
                    
    for i in range(1, n_cols + 1):
        fig.update_layout(**{
            f'scene{i}': dict(
                xaxis=dict(showgrid=True, gridcolor='lightgray', backgroundcolor='white',showticklabels=False),
                yaxis=dict(showgrid=True, gridcolor='lightgray', backgroundcolor='white',showticklabels=False),
                zaxis=dict(showgrid=True, gridcolor='lightgray', backgroundcolor='white',showticklabels=False),
                xaxis_title="", yaxis_title="", zaxis_title="",aspectmode='cube'
            )
        })

    fig.show()


def visualize_confusion_matrix_all(y_target_ls, y_pred_ls):
    n_session = int(np.sqrt(len(y_pred_ls)))
    fig, axes = plt.subplots(n_session, n_session, figsize=(3*n_session, 3*n_session))

    for i in range(n_session):
        for j in range(n_session):
            idx = i*n_session+j
            y_true = y_target_ls[idx]
            y_pred = y_pred_ls[idx]

            c_matrix = confusion_matrix(y_true, y_pred, labels=range(5))
            sns.heatmap(c_matrix, ax=axes[i][j], annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=False, yticklabels=False)

    plt.show()

def visualize_accuracy_all(accuracy_ls):
    n_sessions = int(np.sqrt(len(accuracy_ls)))
    sns.heatmap(np.array(accuracy_ls).reshape(n_sessions, n_sessions), cmap='Blues', annot=True)
    plt.show()