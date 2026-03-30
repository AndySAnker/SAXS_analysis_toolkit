# https://github.com/google-deepmind/uncertain_ground_truth/tree/main

# https://github.com/njszym/XRD-AutoAnalyzer

"""XGBoost training and inference for form-factor classification."""

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization
from sklearn.metrics import log_loss
from SAXS_analysis.visualization.utils import plot_log_loss, plot_confusion_matrix
import SAXS_analysis
ROOT_DIR = SAXS_analysis.ROOT_DIR

def bo_tune_xgb(X_train, y_train, X_val, y_val, max_depth, gamma, n_estimators, learning_rate, early_stopping_rounds):
    """Objective for Bayesian hyperparameter search: negative validation log-loss."""
    params = {
        'max_depth': int(max_depth),
        'gamma': gamma, 
        'n_estimators': int(n_estimators),
        'learning_rate': learning_rate,
        'subsample': 0.8,
        'eta': 0.1,
        'eval_metric': 'mlogloss'
    }
    
    model = xgb.XGBClassifier(**params)
    
    model.fit(
        X_train, 
        y_train, 
        early_stopping_rounds=early_stopping_rounds, 
        eval_set=[(X_val, y_val)], 
        verbose=False
    )
    
    y_val_pred = model.predict_proba(X_val)
    
    return -log_loss(y_val, y_val_pred)

def evaluate_model(model, evals_result, dtrain, dval, dtest, class_names, plot_results=True, save_basename=''):
    """Compute train/val/test accuracy and optionally plot confusion matrix and log-loss curves."""
    # Retrieve performance metrics
    train_loss = evals_result['train']['mlogloss']
    val_loss = evals_result['eval']['mlogloss']

    if plot_results:
        plot_log_loss(train_loss, val_loss, save_basename + '_log_loss.png')

    # Make predictions on the training set
    y_train_pred = model.predict(dtrain)
    
    # Calculate the training accuracy
    train_accuracy = accuracy_score(dtrain.get_label(), np.argmax(y_train_pred, axis=1))

    # Make predictions on the validation set
    y_val_pred = model.predict(dval)

    # Calculate the validation accuracy
    val_accuracy = accuracy_score(dval.get_label(), np.argmax(y_val_pred, axis=1))

    # Make predictions on the test set
    y_test_pred = model.predict(dtest)
    # Calculate the test accuracy
    test_accuracy = accuracy_score(dtest.get_label(), np.argmax(y_test_pred, axis=1))

    # Calculate the baseline accuracy
    baseline_accuracy = 1/len(np.unique(dtrain.get_label()))

    if plot_results:
        plot_confusion_matrix(dtest.get_label(), np.argmax(y_test_pred, axis=1), class_names, save_basename + '_confusion_matrix.png')

    return train_accuracy, val_accuracy, test_accuracy, baseline_accuracy

def train_model(dtrain, dval, early_stopping_rounds=25, hyperparameter_optimisation=False, use_gpu=False):
    """Train XGBoost multi-class classifier with ``multi:softprob``; optional ``bayes_opt`` search."""
    device = 'cuda' if use_gpu else 'cpu'
    tree_method = 'hist'

    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(dtrain.get_label())),
        'reg_alpha': 0,
        'reg_lambda': 0,
        'tree_method': tree_method,
        'device': device
    }

    if hyperparameter_optimisation:
        # Define the bounds of the hyperparameters to be optimized
        hyperparameter_space = {'max_depth': (3, 10),
                                'gamma': (0, 1),
                                'learning_rate':(0, 1),
                                'n_estimators':(100,120)}

        # Initialize the optimizer
        optimizer = BayesianOptimization(f=bo_tune_xgb, pbounds=hyperparameter_space, verbose=2, random_state=1)

        # Optimize
        optimizer.maximize(init_points=5, n_iter=15)

        # Get the best parameters
        best_params = optimizer.max['params']

        # Convert the max_depth and n_estimators to integer because Bayesian Optimization gives float
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['n_estimators'] = int(best_params['n_estimators'])

        # Update the parameters with the best parameters
        params.update(best_params)

    # Train the model with early stopping
    eval_set = [(dtrain, 'train'), (dval, 'eval')]
    evals_result = {}
    model = xgb.train(params, dtrain, num_boost_round=1000, evals=eval_set, early_stopping_rounds=early_stopping_rounds, evals_result=evals_result, verbose_eval=True)

    return model, evals_result

def classify_formfactor(saxs_profile, model_path, class_names_path):
    """Return the single highest-probability class name for a 1D intensity profile."""
    # Load classification model
    model = xgb.Booster()
    model.load_model(str(model_path))

    class_names = np.load(class_names_path)
    input_data = saxs_profile.reshape(1, -1)
    dmatrix = xgb.DMatrix(input_data)
    y_prob = model.predict(dmatrix)

    top1_index = np.argmax(y_prob)
    predicted_class = class_names[top1_index]
    return predicted_class.lower()

def predict_formfactor(data, MLName, class_names):
    """
    Predict the form factor for SAXS data using a pre-trained XGBoost model.

    Loads the model, scores the intensity column, and returns the top three class names.

    Args:
        data: 2D array; column index 1 (intensity) is used for prediction.
        MLName: Path to the trained XGBoost model file (relative to the package root).
        class_names: 1D array of class label strings.

    Returns:
        Top three class names by predicted probability (descending).

    Raises:
        FileNotFoundError: If the model file is missing.
        xgboost.core.XGBoostError: On XGBoost load or prediction errors.
    """
    # Load the model from a file
    model = xgb.Booster()
    model.load_model(ROOT_DIR / MLName)

    # Create the DMatrix
    dataset = xgb.DMatrix(data[:,1].reshape(1, -1))
    
    # Get the probabilities of each class
    y_test_prob = model.predict(dataset)
    # Get the top 3 predictions
    top3_preds = np.argsort(y_test_prob, axis=1)[:,-3:]
    top3_preds_names = class_names[top3_preds]

    return top3_preds_names[0]

