"""Training data pipelines for form-factor classification (XGBoost)."""

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from SAXS_analysis.data_processing.utils import load_and_preprocess_data, split_data
import SAXS_analysis
ROOT_DIR = SAXS_analysis.ROOT_DIR

def process_data_classification(file_name: str, num_data_points: int = int(9e15), 
                              normalize: bool = False, qmin: float = 0.001, 
                              qmax: float = 1.5) -> tuple:
    """Build XGBoost ``DMatrix`` objects for train/val/test classification.

    Loads HDF5 via :func:`~SAXS_analysis.data_processing.utils.load_and_preprocess_data`,
    splits data, label-encodes form factor names, and returns booster-ready matrices.

    Args:
        file_name: HDF5 path relative to ``ROOT_DIR``.
        num_data_points, qmin, qmax: Passed through to preprocessing.
        normalize: Reserved (splitter currently does not scale X for classification).

    Returns:
        ``(dtrain, dval, dtest, class_names)`` with string class names in label order.
    """
    # Load and preprocess the data
    X, y = load_and_preprocess_data(ROOT_DIR / file_name, num_data_points, qmin, qmax)

    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)#, normalize)

    # Convert labels to integers
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)
    class_names = le.classes_
    
    # Convert data to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    return (
        dtrain,
        dval,
        dtest,
        class_names
    )

def load_classification_model(model_path: str):
    """Load a saved XGBoost booster from disk (path relative to ``ROOT_DIR``)."""
    return xgb.Booster(model_file=ROOT_DIR / model_path)