import xgboost as xgb
from SAXS_analysis.data_processing.utils import load_and_preprocess_data, split_data
import numpy as np
import SAXS_analysis
ROOT_DIR = SAXS_analysis.ROOT_DIR

def process_data_regression(file_name: str, formfactor: str, num_data_points: int = int(9e15)) -> tuple:
    """Process data for regression task."""
    X, y_decoded = load_and_preprocess_data(ROOT_DIR / file_name, num_data_points, target_formfactor=formfactor)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_decoded)
    
    print(f"Number of data points: {len(X)}")
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    return (
        xgb.DMatrix(X_train, label=y_train),
        xgb.DMatrix(X_val, label=y_val),
        xgb.DMatrix(X_test, label=y_test)
    )

def load_regression_model(model_path: str, formfactor: str):
    """Load a pre-trained model.
    Args:
        model_path (str): Path to the pre-trained model.
        formfactor (str): Form factor of the model.
    Returns:
        xgb.Booster: The loaded model.
    """
    model_path = f"{ROOT_DIR}/{model_path}/{formfactor}.model"
    return xgb.Booster(model_file=model_path)