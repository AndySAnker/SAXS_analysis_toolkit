# https://github.com/google-deepmind/uncertain_ground_truth/tree/main

# https://github.com/njszym/XRD-AutoAnalyzer

import h5py, time
import pdb
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
#from bayes_opt import BayesianOptimization
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
import ast 
import random
from formfactors import formfactors, formfactors_original

formfactor_params = {
    'sphere': ['background', 'radius', 'radius_pd'],
    'cylinder': ['background', 'radius', 'radius_pd', 'length', 'length_pd'],
    'ellipsoid': ['background', 'radius_polar', 'radius_equatorial'],
    'elliptical_cylinder': ['background', 'radius_minor', 'axis_ratio', 'length', 'length_pd'],
    'flexible_cylinder': ['background', 'radius', 'radius_pd', 'length', 'length_pd', 'kuhn_length'],
    'flexible_cylinder_elliptical': ['background', 'radius', 'radius_pd', 'axis_ratio', 'length', 'length_pd', 'kuhn_length'],
    'stacked_disks': ['background', 'radius', 'radius_pd', 'thick_layer', 'thick_core', 'n_stacking'],
    'core_shell_ellipsoid': ['background', 'radius_equat_core', 'x_core', 'thick_shell', 'x_polar_shell', 'sld_core', 'sld_shell'],
    'binary_hard_sphere': ['background', 'radius_lg', 'radius_sm', 'volfraction_lg', 'volfraction_sm'],
    'vesicle': ['background', 'radius', 'radius_pd', 'thickness', 'volfraction'],
    'core_shell_sphere': ['background', 'radius', 'radius_pd', 'thickness'],
    'triaxial_ellipsoid': ['background', 'radius_equat_minor', 'radius_equat_major', 'radius_polar'],
    'superball': ['background', 'length_a', 'exponent_p'],
    'fuzzy_sphere': ['background', 'radius', 'radius_pd', 'fuzziness'],
    'hollow_cylinder': ['background', 'radius', 'radius_pd', 'length', 'length_pd', 'thickness'],
    'lamellar': ['background', 'thickness'],
                    }

def bo_tune_xgb(max_depth, gamma, n_estimators ,learning_rate):
    params = {'max_depth': int(max_depth),
              'gamma': gamma, 
              'n_estimators': int(n_estimators),
              'learning_rate':learning_rate,
              'subsample': 0.8,
              'eta': 0.1,
              'eval_metric': 'rmse'}
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, early_stopping_rounds=25, eval_set=[(X_val, y_val)], verbose=False)
    y_val_pred = model.predict_proba(X_val)
    return -log_loss(y_val, y_val_pred)

def process_data(file_name, formfactor, num_data_points=9e15):
    f = h5py.File(file_name, 'r')
    X = f['SAXS_dataset'][()]
    y = f['parameters_formfactor'][()][:len(X)]
    formfactors = f['formfactor'][()][:len(X)]
    formfactors = np.array([x[0].decode() for x in formfactors])
    # Remove rows with 'inf' values
    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]
    formfactors = formfactors[mask]
    
    f.close()
    # Apply a mask for the formfactor
    X = X[formfactors == formfactor]
    y = y[formfactors == formfactor]

    # Load only the specified number of datapoints
    X = X[:int(num_data_points)]
    y = y[:int(num_data_points)]
    print ("Number of data points: ", len(X))
    #y = np.unique(y, axis=1)

    # Decode the byte strings to regular strings and parse them as dictionaries
    y_dicts = [ast.literal_eval(item[0].decode()) for item in y]
    print ('formfactor', formfactor)
    print ('y_dicts[0].keys()', y_dicts[0].keys())
    y_new = np.zeros((len(y_dicts), len(formfactor_params[formfactor])))
    for formfactor_param in formfactor_params[formfactor]:
        y_new[:, formfactor_params[formfactor].index(formfactor_param)] = [item[formfactor_param] for item in y_dicts]
    y = y_new

    # Split the data into 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split the train data into further 75% train and 25% validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    print ("Number of data points for training: ", len(X_train))
    print ("Number of data points for validation: ", len(X_val))
    print ("Number of data points for testing: ", len(X_test))

    # Convert the data to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    return dtrain, dval, dtest

def evaluate_model(model, evals_result, dtrain, dval, dtest, formfactor, plot_results=True):
    # Retrieve performance metrics
    results = evals_result
    epochs = len(results['train']['rmse'])
    x_axis = range(0, epochs)

    if plot_results:
        # Plot log loss
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['train']['rmse'], label='Train')
        ax.plot(x_axis, results['eval']['rmse'], label='Validation')
        ax.legend()
        plt.ylabel('Log Loss')
        plt.title('XGBoost Log Loss')
        plt.show()

    # Reshape the labels
    labels_train = dtrain.get_label().reshape(dtrain.num_row(), -1)
    labels_val = dval.get_label().reshape(dval.num_row(), -1)
    labels_test = dtest.get_label().reshape(dtest.num_row(), -1)

    # Make predictions
    y_train_pred = model.predict(dtrain)
    y_val_pred = model.predict(dval)
    y_test_pred = model.predict(dtest)

    # Calculate the metrics for each row for the trainingtestidation, and test sets
    for i, param in enumerate(formfactor_params[formfactor]):
        train_mae = mean_absolute_error(labels_train[:, i], y_train_pred[:, i])
        train_mse = mean_squared_error(labels_train[:, i], y_train_pred[:, i])
        train_r2 = r2_score(labels_train[:, i], y_train_pred[:, i])
        print(f'Training {param} - MAE: {train_mae:.2f}, MSE: {train_mse:.2f}, R2: {train_r2:.2f}')

        val_mae = mean_absolute_error(labels_val[:, i], y_val_pred[:, i])
        val_mse = mean_squared_error(labels_val[:, i], y_val_pred[:, i])
        val_r2 = r2_score(labels_val[:, i], y_val_pred[:, i])
        print(f'Validation {param} - MAE: {val_mae:.2f}, MSE: {val_mse:.2f}, R2: {val_r2:.2f}')

        test_mae = mean_absolute_error(labels_test[:, i], y_test_pred[:, i])
        test_mse = mean_squared_error(labels_test[:, i], y_test_pred[:, i])
        test_r2 = r2_score(labels_test[:, i], y_test_pred[:, i])
        print(f'Test {param} - MAE: {test_mae:.2f}, MSE: {test_mse:.2f}, R2: {test_r2:.2f}')

        # Calculate the baseline prediction as the mean of the test variables
        baseline_pred = np.full(labels_test[:, i].shape, np.mean(labels_test[:, i]))

        # Calculate the baseline metrics
        baseline_mae = mean_absolute_error(labels_test[:, i], baseline_pred)
        baseline_mse = mean_squared_error(labels_test[:, i], baseline_pred)
        baseline_r2 = r2_score(labels_test[:, i], baseline_pred)
        print(f'Baseline {param} - MAE: {baseline_mae:.2f}, MSE: {baseline_mse:.2f}, R2: {baseline_r2:.2f}')
        print()

    return None

def train_model(dtrain, dval, use_bayesian_optimization=False, use_gpu=False):
    device = 'cuda' if use_gpu else 'cpu'
    tree_method = 'hist'

    params = {
        #'objective': 'multi:softprob',
        #'num_class': len(np.unique(dtrain.get_label())),
        'reg_alpha': 0,
        'reg_lambda': 0,
        'tree_method': tree_method,
        'device': device
    }

    if use_bayesian_optimization:
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
    model = xgb.train(params, dtrain, num_boost_round=1000, evals=eval_set, early_stopping_rounds=25, evals_result=evals_result, verbose_eval=True)

    return model, evals_result

def predict_parameters(data, MLName, formfactor, param_ranges):
    """
    Predicts parameters for a given form factor using a pre-trained XGBoost model and updates param_ranges.

    This function loads a pre-trained XGBoost model specific to the given form factor,
    uses it to predict parameters based on the input SAXS data, and updates the
    param_ranges dictionary with the predicted values if they fall within the
    specified range.

    Parameters:
    data (numpy.ndarray): A 2D array containing the SAXS data. The second column
                          (index 1) is used for prediction.
    MLName (str): The name or identifier of the machine learning model.
    formfactor (str): The form factor for which parameters are to be predicted.
                      This should be one of the keys in the `formfactor_params` dictionary.
    param_ranges (dict): A dictionary containing the current parameter ranges.
                         This dictionary will be modified in-place with predicted values.

    Returns:
    None: The function modifies the `param_ranges` dictionary in-place.

    Raises:
    KeyError: If the provided form factor is not found in the `formfactor_params` dictionary.
    FileNotFoundError: If the model file for the given form factor and ML name is not found.
    xgboost.core.XGBoostError: If there's an error in loading or using the XGBoost model.

    Example usage:
    >>> data = np.array([[q1, I1], [q2, I2], ..., [qn, In]])
    >>> MLName = 'regression_model'
    >>> formfactor = 'sphere'
    >>> param_ranges = {'radius': [None, 1, 100], 'sld': [None, 1e-6, 1e-5], ...}
    >>> predict_parameters(data, MLName, formfactor, param_ranges)
    >>> print(param_ranges)
    {'radius': [50.3, 1, 100], 'sld': [3.2e-6, 1e-6, 1e-5], ...}
    """
    
    # Load the model from a file
    model = xgb.Booster()
    model.load_model(f'XGBoost_models_parameters/XGBoost_model_{formfactor}_{MLName}.json')

    # Create the DMatrix
    dataset = xgb.DMatrix(data[:,1].reshape(1, -1))
    
    # Make predictions
    y_pred = model.predict(dataset)

    # Override the parameters in param_ranges with the predicted parameters
    for param_name in formfactor_params[formfactor]:
        param_index = formfactor_params[formfactor].index(param_name)
        param_values = y_pred[:,param_index]
        if param_values[0] < param_ranges[param_name][1] and param_values[0] > param_ranges[param_name][2]:
            param_ranges[param_name][0] = param_values[0]

    return None

if __name__ == '__main__':
    start_time = time.time()
    
    Num_datapoint = 1.6e6
    for formfactor in formfactors:
        print (f'Formfactor: {formfactor}')
        # Load the data
        dtrain, dval, dtest = process_data(file_name='SAXS_datasets/SAXS_dataset_N100000_float16_0.0percentStructureFactor_noResolution.h5', formfactor=formfactor, num_data_points=Num_datapoint)
        # Train the model
        model, evals_result = train_model(dtrain, dval, use_bayesian_optimization=False, use_gpu=True)
        model.save_model(f'XGBoost_models_parameters/XGBoost_model_{formfactor}_{int(Num_datapoint)}_0.0percentStructureFactor_noResolution.json')
        # Evaluate the model
        evaluate_model(model, evals_result, dtrain, dval, dtest, formfactor=formfactor, plot_results=False)
        print(f'Time taken for {formfactor}: {(time.time() - start_time)/60:.2f} minutes')
