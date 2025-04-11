from SAXS_analysis.utils.formfactors import formfactor_params
from SAXS_analysis.regression.utils import load_regression_model
from SAXS_analysis.classification.utils import load_classification_model, process_data_classification
from SAXS_analysis.data_processing.utils import load_and_process_SAS_data
from SAXS_analysis.fitting.fit_sas import SAS_Fitter
import SAXS_analysis
ROOT_DIR = SAXS_analysis.ROOT_DIR
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

"""Input parameters"""
# Data parameters
data_name = 'data/experimental/dataset_1.dat'
qmin = 0.03
qmax = 1.1
error_weighting = "None" # Options: "None", "intensity", "sqrt", "absolute"
normalization_type = 'None' #Options: 'None' or 'peak'
# Model parameters
formfactor_classification_model = ROOT_DIR / 'models' / 'classification' / 'XGBoost_model_100000_normDataFalse_earlyStopping25_hyperparameterFalse.model'
class_names = np.load(ROOT_DIR / 'models' / 'classification' / 'class_names_100000_normDataFalse_earlyStopping25_hyperparameterFalse.npy')
parameter_regression_model_basename = f"models/regression/XGBoost_model_100000_normDataFalse_earlyStopping25_hyperparameterFalse"
# Fitting parameters
solver = 'dream'
smearing = 0.0




"""Load the experimental SAS data and convert to DMatrix"""
experimental_SAS_data = load_and_process_SAS_data(data_source=data_name, qmin=qmin, qmax=qmax, error_weighting=error_weighting, normalization_type=normalization_type)
print("Loaded experimental SAS data with file: ", data_name)
dmatrix = xgb.DMatrix(experimental_SAS_data.y.reshape(1, -1))
print("dmatrix shape: ", dmatrix.num_row(), "x", dmatrix.num_col())





"""Load the classification model and classify the form factor"""
formfactor_classification_model = load_classification_model(formfactor_classification_model)
print("Loading formfactor classification model: ", formfactor_classification_model)
formfactor_probabilities = formfactor_classification_model.predict(dmatrix)  # Shape: (1, n_classes)
formfactor_prediction = np.argmax(formfactor_probabilities, axis=1)  # Get predicted class index

print("\n===== CLASSIFICATION RESULTS =====")
for i in range(formfactor_probabilities.shape[0]):
    predicted_class = class_names[formfactor_prediction[i]]    
    print(f"\nSample #{i+1}:")
    print(f"  Predicted class: {predicted_class}")    
    print("  Class probabilities:")
    for j, class_name in enumerate(class_names):
        probability = formfactor_probabilities[i][j]
        print(f"    {class_name}: {probability:.4f} ({probability*100:.1f}%)")
print("================================")





"""Load the regression model and do parameter regression for each formfactor"""
print("\n===== REGRESSION RESULTS =====")

# Get the predicted formfactor (using the first sample's prediction)
predicted_formfactor = class_names[formfactor_prediction[0]]
print(f"Parameter estimates for predicted formfactor: {predicted_formfactor}")

# Show parameter estimates for all formfactors
for formfactor in class_names:
    parameter_regression_model = load_regression_model(parameter_regression_model_basename, formfactor)
    print(f"\n  Formfactor: {formfactor}")
    
    # Regress the parameters
    parameter_regression = parameter_regression_model.predict(dmatrix)
    
    # Get all parameters for the formfactor
    all_parameters = formfactor_params[formfactor]
    
    # Print parameter values
    print("  Parameters:")
    for j, param in enumerate(all_parameters):
        value = parameter_regression[0][j]
        print(f"    {param}: {value:.6f}")

print("================================")





"""Perform fitting with the predicted parameters"""
print("\n===== FITTING RESULTS =====")

# Get the predicted formfactor (using the first sample's prediction)
predicted_formfactor = class_names[formfactor_prediction[0]]
print(f"Fitting using formfactor: {predicted_formfactor}")

# Initialize the SAS fitter
SAS_Fitter = SAS_Fitter()

# Load the data
SAS_Fitter.load_data(
    data_source=None, 
    x=experimental_SAS_data.x, 
    y=experimental_SAS_data.y, 
    z=None, 
    qmin=qmin, 
    qmax=qmax, 
    error_weighting=error_weighting, 
    normalization_type=normalization_type
)

# Perform the fit
Icalc, goodness_of_fit, R_w, fitted_params = SAS_Fitter.fit_sas_data(
    predicted_formfactor, 
    solver, 
    smearing
)

# Print fitting results
print("\n  Fit Quality Metrics:")
print(f"    Goodness of fit: {goodness_of_fit:.6f}")
print(f"    R_w:            {R_w:.6f}")

print("\n  Fitted Parameters:")
for param, value in fitted_params.items():
    print(f"    {param}: {value:.6f}")

# Create and save plot
plt.figure(figsize=(10, 6))
plt.plot(experimental_SAS_data.x, experimental_SAS_data.y, 'o', markersize=3, label='Experimental')
plt.plot(experimental_SAS_data.x, Icalc, '-', linewidth=2, label='Fitted')
plt.loglog()
plt.xlabel("q (Å$^{-1}$)")
plt.ylabel("I(q) (cm$^{-1}$)")
plt.title(f"SAS Fitting - {predicted_formfactor}")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT_DIR / f"plots/experimental_SAS_fitting_{predicted_formfactor}.png")
print("\n  Plot saved to: ", ROOT_DIR / f"plots/experimental_SAS_fitting_{predicted_formfactor}.png")
plt.show()

print("================================")
