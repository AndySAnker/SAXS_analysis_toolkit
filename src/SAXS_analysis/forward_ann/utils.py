"""Training utilities for forward ANNs (parameters → curve features)."""

import os
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from SAXS_analysis.data_processing.utils import load_and_preprocess_data, split_data
import SAXS_analysis
from SAXS_analysis.forward_ann.forward_ann import ForwardNN
import joblib
import torch.optim as optim
ROOT_DIR = SAXS_analysis.ROOT_DIR

def process_data_regression(file_name: str, formfactor: str, num_data_points: int = int(9e15)) -> tuple:
    """
    Process data for regression task and return PyTorch tensors for train, val, test with MinMax scaling on parameters.
    Removes the background parameter from y before splitting and scaling.
    Saves the output scaler as '<file_name>_<formfactor>_scaler.joblib'.
    """
    X, y = load_and_preprocess_data(ROOT_DIR / file_name, num_data_points, target_formfactor=formfactor)

    # Remove background parameter
    background_col_idx = 0
    y_no_bg = np.delete(y, background_col_idx, axis=1)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_no_bg)

    print(f"Number of data points: {len(X)}")
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    # Scale parameters
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)

    # Save scalers for subsequent use
    scaler_dir = ROOT_DIR / "scalers"
    scaler_dir.mkdir(parents=True, exist_ok=True)
    scaler_filename = f"{formfactor}_scaler.joblib"
    joblib.dump(y_scaler, scaler_dir / scaler_filename)

    # Convert numpy arrays to PyTorch tensors for model training
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    return (X_train_tensor, y_train_tensor), (X_val_tensor, y_val_tensor), (X_test_tensor, y_test_tensor), y_scaler

def lr_scheduler(optimizer, config_model):
    """Optional ``ReduceLROnPlateau`` from training YAML ``reduce_lr_on_plateau`` block."""
    reduce_lr_cfg = config_model.get('reduce_lr_on_plateau', {})
    reduce_lr_enabled = reduce_lr_cfg.get('enabled', False)

    if not reduce_lr_enabled:
        return None

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=reduce_lr_cfg.get('mode', 'min'),
        factor=float(reduce_lr_cfg.get('factor', 0.5)),
        patience=int(reduce_lr_cfg.get('patience', 5)),
        min_lr=float(reduce_lr_cfg.get('min_lr', 1e-6)),
        verbose=bool(reduce_lr_cfg.get('verbose', True))
    )
    return scheduler

def calculate_test_mae(model, test_loader, scaler, formfactor):
    """Print per-parameter MAE on the test loader after inverse-scaling predictions."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            all_preds.append(outputs.cpu())
            all_targets.append(y_batch.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    all_preds = scaler.inverse_transform(all_preds)
    all_targets = scaler.inverse_transform(all_targets)

    # Define parameter names based on formfactor
    formfactor = formfactor.lower()
    if formfactor == "sphere":
        param_names = ["Radius", "Radius Pd"]
    elif formfactor == "ellipsoid":
        param_names = ["Radius Polar", "Radius Equatorial"]
    elif formfactor == "cylinder":
        param_names = ["Radius", "Radius Pd", "Length", "Length Pd"]
    else:
        n_params = all_preds.shape[1] if all_preds.ndim > 1 else 1
        param_names = [f"param_{i}" for i in range(n_params)]

    print(f"Test MAE per parameter for formfactor '{formfactor}':")
    for i, param_name in enumerate(param_names):
        pred_col = all_preds[:, i] if all_preds.ndim > 1 else all_preds
        target_col = all_targets[:, i] if all_targets.ndim > 1 else all_targets
        mae_val = mean_absolute_error(target_col, pred_col)
        print(f"  {param_name}: {mae_val:.6f}")

def load_forward_regression_model(model_path: str, formfactor: str, X, y):
    """
    Load a pre-trained model.

    Args:
        model_path (str): Path to the pre-trained model folder.
        formfactor (str): Form factor of the model.
        X: Input dataset.
        y: Output dataset.

    Returns:
        torch.nn.Module: Loaded model.
    """

    input_size = X.shape[1]   
    output_size = y.shape[1]  

    model_file = ROOT_DIR / model_path / f"{formfactor}_forward.pth"
    
    model = ForwardNN(input_size=input_size, output_size=output_size)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    return model

