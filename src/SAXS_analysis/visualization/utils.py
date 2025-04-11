import numpy as np
import matplotlib.pyplot as plt
import SAXS_analysis
ROOT_DIR = SAXS_analysis.ROOT_DIR
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_random_data(X, y, num_plots=3, save_path=ROOT_DIR / 'plots' / 'random_data.png'):
    # Define the x-axis
    x_axis = np.linspace(0.001, 1.5, 1000)

    # Plot some random data to ensure that the data is correct
    for i in range(num_plots):
        fig, axs = plt.subplots(4, 2, figsize=(15, 20))
        for j in range(4):
            for k in range(2):
                index = np.random.choice(len(X), 1, replace=False)
                index = index[0]
                axs[j, k].scatter(x_axis, X[index], label=f'y = {y[index]}')
                axs[j, k].plot(x_axis, X[index], 'k--', linewidth=0.5)  # Add thin dashed lines
                axs[j, k].legend()
                axs[j, k].set_xlabel('x')
                axs[j, k].set_ylabel('y')
                axs[j, k].set_title(f'Plot {index}')
                axs[j, k].set_xscale('log')
                axs[j, k].set_yscale('log')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

def plot_log_loss(train_loss, val_loss):
    epochs = len(train_loss)
    x_axis = range(epochs)

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_loss, label='Train')
    ax.plot(x_axis, val_loss, label='Validation')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    colorbar = ax.collections[0].colorbar
    colorbar.set_label('Number of predictions')
    plt.show()