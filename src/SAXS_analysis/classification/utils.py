import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from SAXS_analysis.data_processing.utils import load_and_preprocess_data, split_data
    
def process_data_classification(file_name: str, num_data_points: int = int(9e15), 
                              normalize: bool = True, qmin: float = 0.001, 
                              qmax: float = 1.5) -> tuple:
    """Process data for classification task."""
    X, y = load_and_preprocess_data(file_name, num_data_points, qmin, qmax)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, normalize)
    
    # Convert labels to integers
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)
    class_names = le.classes_
    
    return (
        xgb.DMatrix(X_train, label=y_train),
        xgb.DMatrix(X_val, label=y_val),
        xgb.DMatrix(X_test, label=y_test),
        class_names
    )

def load_classification_model(model_path: str):
    """Load a pre-trained model.
    Args:
        model_path (str): Path to the pre-trained model.
    Returns:
        xgb.Booster: The loaded model.
    """
    return xgb.Booster(model_file=model_path)