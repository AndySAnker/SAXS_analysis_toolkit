"""Training utilities for inverse ANNs (curve features → parameters)."""

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from SAXS_analysis.data_processing.utils import load_and_preprocess_data, split_data
import SAXS_analysis
from SAXS_analysis.inverse_ann.inverse_ann import InverseNN
import joblib
import torch.optim as optim
ROOT_DIR = SAXS_analysis.ROOT_DIR

def process_data_regression_inverse(file_name: str, formfactor: str, num_data_points: int = int(9e15)):
    """
    Process data for regression task and return PyTorch tensors for train, val, test with MinMax scaling on parameters.
    For inverse model, treat original y as inputs (X) and original x as outputs (y)
    Removes the background parameter from y before splitting and scaling.
    Saves the output scaler as '<file_name>_<formfactor>_scaler.joblib'.
    """
    # Load original data (y as inputs, x as outputs)
    x, y = load_and_preprocess_data(ROOT_DIR / file_name, num_data_points, target_formfactor=formfactor)

    # Treat original y as inputs (X) and original x as outputs (y)
    x, y = y, x

    # Remove background column (assumed index 0) from inputs
    background_col_idx = 0
    x_no_bg = np.delete(x, background_col_idx, axis=1)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(x_no_bg, y)

    # Scale inputs
    x_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)
    X_test_scaled = x_scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Save scalers
    scaler_dir = ROOT_DIR / "scalers"
    scaler_dir.mkdir(parents=True, exist_ok=True)
    scaler_filename = f"{formfactor}_inverse_scaler.joblib"
    joblib.dump(x_scaler, scaler_dir / scaler_filename)

    return (X_train_tensor, y_train_tensor), (X_val_tensor, y_val_tensor), (X_test_tensor, y_test_tensor), x_scaler

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

def calculate_rmse(model, test_loader):
    """RMSE per sample (axis 1) between predictions and targets on the test loader."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            all_preds.append(outputs)
            all_targets.append(y_batch)

    # Load model and true values for RMSE calculation
    preds = torch.cat(all_preds, dim=0).cpu().numpy()
    targets = torch.cat(all_targets, dim=0).cpu().numpy()

    # RMSE calculation
    rmse = np.sqrt(np.mean((preds - targets) ** 2, axis=1))
    return rmse

def load_regression_model(model_path: str, formfactor: str, X, y):
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

    model_file = ROOT_DIR / model_path / f"{formfactor}.pth"
    
    model = InverseNN(input_size=input_size, output_size=output_size)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    return model