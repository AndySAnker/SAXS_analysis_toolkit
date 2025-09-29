import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import SAXS_analysis
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path
import torch
import pandas as pd
import os
import corner
import joblib
ROOT_DIR = SAXS_analysis.ROOT_DIR

def plot_random_data(X, y, num_plots=3, save_path='random_data.png'):
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
        plt.savefig(ROOT_DIR / 'plots' / save_path)
        plt.close()

def plot_log_loss(train_loss, val_loss, save_path='log_loss.png'):
    epochs = len(train_loss)
    x_axis = range(epochs)

    fig, ax = plt.subplots()
    ax.plot(x_axis, train_loss, label='Train')
    ax.plot(x_axis, val_loss, label='Validation')
    ax.legend(fontsize=8)
    plt.ylabel('Log loss', fontsize=8)
    plt.xlabel('Epochs', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.savefig(ROOT_DIR / 'plots' / save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted', fontsize=8)
    plt.ylabel('Truth', fontsize=8)
    colorbar = ax.collections[0].colorbar
    colorbar.set_label('Number of predictions', fontsize=8, rotation=270, labelpad=10)
    colorbar.ax.tick_params(labelsize=6)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.savefig(ROOT_DIR / 'plots' / save_path)
    plt.close()

def plot_ann_loss(train_losses, val_losses, formfactor):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training vs Validation Loss ({formfactor})")
    plt.legend()
    plt.grid(True)
    plt.savefig(ROOT_DIR / 'plots' / f"loss_curve_{formfactor}.png")
    plt.close()

def plot_prediction_error_histograms(model, test_loader, scaler, formfactor, root_dir):
    model.eval()
    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            y_pred_list.append(outputs)
            y_true_list.append(y_batch)

    y_pred_scaled = torch.cat(y_pred_list, dim=0).cpu().numpy()
    y_true_scaled = torch.cat(y_true_list, dim=0).cpu().numpy()

    # Inverse scale predictions and true values
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_true_scaled)
    errors = y_pred - y_true

    # Define parameter names by formfactor
    if formfactor == "sphere":
        parameter_names = ["Radius (A)", "Polydispersity"]
    elif formfactor == "ellipsoid":
        parameter_names = ["Radius_Polar (A)", "Radius_Equatorial (nm)"]
    elif formfactor == "cylinder":
        parameter_names = ["Radius (A)", "Radius Pd", "Length (A)", "Length Pd"]
    else:
        parameter_names = [f"Parameter {i}" for i in range(y_pred.shape[1])]

    # Create plots directory
    plot_dir = Path(root_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot individual error histograms
    for i in range(y_pred.shape[1]):
        plt.figure()
        plt.hist(errors[:, i], bins=50, alpha=0.75, color='darkorange')
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.title(f"{parameter_names[i]} Prediction Error ({formfactor})")
        plt.grid(True)
        plt.savefig(plot_dir / f"error_hist_{formfactor}_{parameter_names[i].replace(' ', '_')}.png")
        plt.close()

    print(f"Error histograms saved to {plot_dir}")

def plot_rmse_histograms(formfactors, rmse_values, save_dir=None):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame({
        'FormFactor': formfactors,
        'RMSE': rmse_values
    })

    for formfactor in df['FormFactor'].unique():
        plt.figure(figsize=(8, 6))
        subset = df[df['FormFactor'] == formfactor]['RMSE']
        plt.hist(subset, bins=20, edgecolor='black')
        plt.title(f'RMSE Distribution for {formfactor}')
        plt.xlabel('RMSE')
        plt.ylabel('Frequency')
        plt.grid(True)

        if save_dir:
            plot_path = os.path.join(save_dir, f'{formfactor}_rmse_histogram.png')
            plt.savefig(plot_path) 
        else:
            plt.show()

        plt.close()

def save_corner_plot(flat_chain, map_estimate, row_index=None, true_input=None,
                     scaler_path=None, save_dir="plots", parameter_names=None):
    """
    Generate and save a corner plot of MCMC samples using unscaled values.

    Parameters:
    - flat_chain: Flattened MCMC chain of sampled parameters (scaled)
    - map_estimate: MAP estimate (scaled)
    - row_index: Optional row index for filename
    - true_input: True parameter values (unscaled)
    - scaler_path: Path to scaler for inverse-transforming MCMC samples and MAP
    - save_dir: Directory to save the plot
    - parameter_names: Optional list of parameter names for axis labels
    """

    os.makedirs(save_dir, exist_ok=True)
    n_dim = flat_chain.shape[1]

    if scaler_path is None:
        raise ValueError("scaler_path is required for unscaling the MCMC samples and MAP estimate.")

    # Load scaler and inverse transform MCMC samples and MAP estimate
    scaler = joblib.load(scaler_path)
    flat_chain_unscaled = scaler.inverse_transform(flat_chain)
    map_estimate_unscaled = scaler.inverse_transform([map_estimate])[0]

    labels = parameter_names if parameter_names is not None else [f"Param {i}" for i in range(n_dim)]

    # Create corner plot with unscaled values
    fig = corner.corner(
        flat_chain_unscaled,
        labels=labels,
        show_titles=True,
        title_fmt=".3f",
        plot_datapoints=True,
        fill_contours=True,
        plot_density=True,
        truths=None 
    )

    # Overlay lines
    axes = np.array(fig.axes).reshape((n_dim, n_dim))
    for i in range(n_dim):
        ax = axes[i, i] 
        ax.axvline(map_estimate_unscaled[i], color='blue', linestyle='--', label='MAP estimate')
        if true_input is not None:
            ax.axvline(true_input[i], color='red', linestyle=':', label='True value')

    # Add plot legend 
    handles = [
        mlines.Line2D([], [], color='blue', linestyle='--', label='MAP estimate'),
        mlines.Line2D([], [], color='red', linestyle=':', label='True value')
    ]
    axes[0, 0].legend(handles=handles)

    # Save figure
    filename = f"corner_plot_{row_index}.png" if row_index is not None else "corner_plot.png"
    fig_path = os.path.join(save_dir, filename)
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"[Plot] Corner plot saved to {fig_path}")

def plot_experimental_data(q_raw, Iq_raw, std_raw,
                           q_interp, Iq_interp, std_interp,
                           save_path, data_stem):
    """
    Plot raw and interpolated SAXS data.
    """
    plt.figure(figsize=(6, 4))
    plt.errorbar(q_raw, Iq_raw, yerr=std_raw, fmt='x', markersize=3, alpha=0.5,
                 color='blue', markerfacecolor='none', label='Raw data')      
    plt.errorbar(q_interp, Iq_interp, yerr=std_interp, fmt='o', markersize=4, alpha=0.9,
                 color='orange', markerfacecolor='none', label='Interpolated data')  
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('q [1/nm]')
    plt.ylabel('I(q)')
    plt.title(f"Raw and Interpolated SAXS Data: ({data_stem})") 
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_qt_data(q_mid_padded, qt_padded, qt_std_padded, save_path, data_stem):
    """
    Plot padded quotient-transformed SAXS data.
    """
    plt.figure(figsize=(6, 4))
    plt.errorbar(q_mid_padded, qt_padded, yerr=qt_std_padded, fmt='o', markersize=4, alpha=0.9,
                 color='orange', markerfacecolor='none', label='QT Padded')
    plt.xscale('log')
    plt.yscale('linear')
    plt.xlabel('q [1/nm]')
    plt.ylabel('QT-I(q)')
    plt.title(f'Quotient-Transformed SAXS Data ({data_stem})')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()