import numpy as np
    
class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def softmax(a, T=1):
    a = np.array(a) / T
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

import matplotlib.pyplot as plt
from IPython.display import clear_output


def exponential_smoothing(data, alpha=0.1):
    """Compute exponential smoothing."""
    smoothed = [data[0]]  # Initialize with the first data point
    for i in range(1, len(data)):
        st = alpha * data[i] + (1 - alpha) * smoothed[-1]
        smoothed.append(st)
    return smoothed


def live_plot(data_dict):
    """Plot the live graph with multiple subplots."""

    plt.style.use('ggplot')
    n_plots = len(data_dict)
    fig, axes = plt.subplots(nrows=n_plots, figsize=(7, 4 * n_plots), squeeze=False)
    plt.subplots_adjust(hspace=0.5)
    plt.ion()
    clear_output(wait=True)

    for ax, (label, data) in zip(axes.flatten(), data_dict.items()):
        ax.clear()
        ax.plot(data, label=label, color="yellow", linestyle='--')
        # Compute and plot moving average for total reward
        if len(data) > 0:
            ma = exponential_smoothing(data)
            ma_idx_start = len(data) - len(ma)
            ax.plot(range(ma_idx_start, len(data)), ma, label="Smoothed Value",
                    linestyle="-", color="purple", linewidth=2)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(loc='upper left')

    plt.show()
