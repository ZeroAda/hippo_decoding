import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def plot_loss_values(loss_values, model_name="mlp", dimension=3):
    plt.plot(loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    title = 'Training Loss'
    if model_name != None:
        title += '_' + model_name
    if dimension != None:
        title += '_d' + str(dimension)
    plt.title(title)
    title = title + '.pdf'
    plt.savefig(title)


def plot_accuracy_list(dimension_list, accuracy_list, model_name="mlp"):
    plt.plot(dimension_list, accuracy_list)
    plt.xlabel('Dimension')
    plt.ylabel('Accuracy')
    title = 'Accuracy'
    if model_name != None:
        title += '_' + model_name
    plt.title(title)
    title = title + '.pdf'
    plt.savefig(title)



def cm2str(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None, format='.2f'):
    """Pretty print for confusion matrixes. Taken from https://gist.github.com/zachguo/10296432."""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    out = "    " + empty_cell + " "
    for label in labels:
        out += "%{0}s".format(columnwidth) % label
    out += "\n"
    for i, label1 in enumerate(labels):
        out += "    %{0}s ".format(columnwidth) % label1
        for j in range(len(labels)):
            cell = "%{0}{1}".format(columnwidth, format) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            out += cell + " "
        out += "\n"
    return out


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_data(data, channel_index_label, color_map, label_map, session_name):
    # Define the sampling rate and the window of interest in milliseconds
    sampling_rate = 20000  # Hz: 1s how many samples?
    window_length_ms = 500

    # Calculate the number of samples to cover the window of interest
    window_length_samples = int(window_length_ms * sampling_rate / 1000)  # Convert ms to samples
    print(window_length_samples)

    # Assuming we want to plot the first 500 ms of data
    start_sample = 0
    end_sample = start_sample + window_length_samples

    # Extract the window of data for plotting
    data_to_plot = data[:, start_sample:end_sample]

    # Separate the data into 8 shanks of 128 channels each
    shanks = [data_to_plot[i*128:(i+1)*128, :] for i in range(8)]

    # Plotting
    fig, axes = plt.subplots(1, 8, figsize=(20,15), sharex=True)

    scaled_x = np.arange(window_length_samples) * (window_length_ms / window_length_samples)

    for shank_index, shank_data in enumerate(shanks):
        ax = axes[shank_index]
        for channel_index in range(shank_data.shape[0]):
            # Offset each channel for visual clarity
            offsets = channel_index * np.max(np.abs(shank_data))
            # offsets = channel_index * 5
            channel_index_i = channel_index + shank_index * 128
            if channel_index_label[channel_index_i] in color_map:
                ax.plot(shank_data[channel_index] + offsets, color=color_map[channel_index_label[channel_index_i]])
        # set no border
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.set_title(f'Shank-{shank_index + 1}')
        ax.set_yticks([])
        ax.set_xlim([0, window_length_samples])

    # Set the xlabel for the last subplot
    axes[-1].set_xlabel('Time (ms)')

    # Show the plot
    plt.tight_layout()
    # put legend right upper corner
    plt.legend(handles=[plt.Line2D([0], [0], color=color, label=label_map[label]) for label, color in color_map.items()], loc='upper right')
    title_name = session_name + '_data_plot.pdf'
    plt.savefig(title_name)