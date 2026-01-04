import matplotlib.pyplot as plt
import numpy as np


#######FOR TRAINING GRAPH#########
def compute_dev_plot(n_epochs, data, name, legend=None, title=""):
    """
    Plots mean accuracy and standard deviation across multiple runs over epochs.
    
    Parameters:
    - n_epochs (int): Number of epochs.
    - data (numpy.ndarray): 2D array (runs x epochs) of accuracy values in percentage.
    - name (str): Output file name for saving the plot.
    - legend (list, optional): List of labels for the plot. Defaults to ["Mean Accuracy", "Standard Deviation"].
    - title (str, optional): Title of the plot.
    """
       
    data = data * 100
    epochs = np.arange(5, n_epochs + 1, 5)

    # Compute mean and standard deviation across runs
    mean_acc = np.mean(data, axis=0)
    std_acc = np.std(data, axis=0)

    # Plot mean accuracy curve
    mean_label = legend[0] if legend and len(legend) > 0 else "Mean Accuracy"
    plt.plot(epochs, mean_acc, color="red", label=mean_label)
    
    # Plot standard deviation fill with label
    std_label = legend[2] if legend and len(legend) > 2 else "Standard Deviation"
    plt.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, color="blue", alpha=0.2, label=std_label)

    # Customize the plot
    plt.xlabel("Epoch", fontsize = 15)
    plt.ylabel("Accuracy (%)",fontsize = 15)
    min_acc = np.floor(np.min([min(run) for run in data]) / 3) * 3
    max_acc = np.ceil(np.max([max(run) for run in data]) / 3) * 3
    plt.yticks(np.arange(min_acc, max_acc + 1, 1))
    plt.title(title, fontsize = 15)
    #plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(name)
    plt.close()
