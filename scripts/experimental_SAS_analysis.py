from SAXS_analysis.utils.formfactors import formfactor_params
from SAXS_analysis.regression.utils import load_regression_model
from SAXS_analysis.classification.utils import load_classification_model, process_data_classification
from SAXS_analysis.data_processing.utils import load_and_process_SAS_data
from SAXS_analysis.fitting.fit_sas import SAS_Fitter
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

# input parameters
data_name = "data/experimental/dataset_1.dat"
qmin = 0.03
qmax = 1.1
error_weighting = "None" # Options: "None", "intensity", "sqrt", "absolute"
normalization_type = 'None' #Options: 'None' or 'peak'






# Load the experimental SAS data
experimental_SAS_data = load_and_process_SAS_data(data_source=data_name, qmin=qmin, qmax=qmax, error_weighting=error_weighting, normalization_type=normalization_type)
print("Loaded experimental SAS data with file: ", data_name)

# Convert experimental_SAS_data to DMatrix
dmatrix = xgb.DMatrix(np.column_stack((experimental_SAS_data.x, experimental_SAS_data.y)))
print("dmatrix shape: ", dmatrix.num_row(), "x", dmatrix.num_col())






# Load simulated SAS data
filename = "data/simulated/SAXS_dataset_100000_floatfloat16_structureFactor0.0percent_powerlaw0.0percent_qmin0.001_qmax1.5_numPoints1000_resolutionMin0.0_resolutionMax0.0_normalisationNone_noiseFalse.h5"
num_data_points = 100

dtrain, dval, dtest, class_names = process_data_classification(
    file_name=filename,
    num_data_points=num_data_points,
    normalize=normalization_type    
)
print("Loaded simulated SAS data with file: ", filename)
# Get data from XGBoost matrix
dmatrix = dtrain
print("dmatrix shape: ", dmatrix.get_data().shape)





# input parameters
formfactor_classification_model = "models/classification/XGBoost_model_100000_normDataFalse_earlyStopping25_hyperparameterFalse.model"
class_names = np.load("models/classification/class_names_100000_normDataFalse_earlyStopping25_hyperparameterFalse.npy")

# Load the classification model
formfactor_classification_model = load_classification_model(formfactor_classification_model)
print("Loading formfactor classification model: ", formfactor_classification_model)

# Classify the form factor
formfactor_probabilities = formfactor_classification_model.predict(dmatrix)  # Shape: (1, n_classes)
formfactor_prediction = np.argmax(formfactor_probabilities, axis=1)  # Get predicted class index

for i in range(formfactor_probabilities.shape[0]):
    print("Formfactor prediction: ", class_names[formfactor_prediction[i]], " with probability: ", formfactor_probabilities[i])
    print("True formfactor: ", class_names[int(dmatrix.get_label()[i])])







# input parameters
parameter_regression_model_basename = f"models/regression/XGBoost_model_100000_normDataFalse_earlyStopping25_hyperparameterFalse"
formfactor = 'sphere'

# Load the regression model
parameter_regression_model = load_regression_model(parameter_regression_model_basename, formfactor)
print("Loading parameter regression model: ", parameter_regression_model_basename)

# Regress the parameters
parameter_regression = parameter_regression_model.predict(dmatrix)

# Get all parameters for the formfactor
all_parameters = formfactor_params[formfactor]

for i in range(parameter_regression.shape[0]):
    # Get all predicted parameters for the formfactor
    predicted_parameters = {}
    for j, param in enumerate(all_parameters):
        predicted_parameters[param] = parameter_regression[i][j]
    print("Predicted parameters for formfactor: ", class_names[formfactor_prediction[i]], " are: ", predicted_parameters)








# Determine parameters for fitting
solver = 'dream'
smearing = 0.0

# Fitting with start values from the predicted parameters
SAS_Fitter = SAS_Fitter()
for i in range(dmatrix.get_data().shape[0]):
    print("data y: ", dmatrix.get_data()[i].toarray().flatten())  # Convert sparse matrix to dense array
    y_data = dmatrix.get_data()[i].toarray().flatten()
    SAS_Fitter.load_data(data_source=None, x=np.linspace(qmin, qmax, 1000), y=y_data, z=None, qmin=qmin, qmax=qmax, error_weighting=error_weighting, normalization_type=normalization_type)
    Icalc, goodness_of_fit, R_w, fitted_params = SAS_Fitter.fit_sas_data(formfactor, solver, smearing)

    print("Goodness of fit: ", goodness_of_fit)
    print("R_w: ", R_w)
    print("Fitted parameters: ", fitted_params)

    # Plot the experimental and fitted SAS data
    plt.plot(experimental_SAS_data.x, experimental_SAS_data.y, label='Experimental')
    plt.plot(experimental_SAS_data.x, Icalc, label='Fitted')
    plt.loglog()
    plt.xlabel("q (Å$^{-1}$)")
    plt.ylabel("I(q) (cm$^{-1}$)")
    plt.legend()
    plt.savefig(f"plots/experimental_SAS_fitting_{i}.png")
    plt.show()