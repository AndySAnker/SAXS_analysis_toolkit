from bumps.names import Parameter, inf
import bumps, sasmodels
from bumps.mapper import MPMapper
from sasmodels.core import load_model
from sasmodels.bumps_model import Model
import numpy as np
import matplotlib.pyplot as plt
from SAXS_analysis.utils.parameter_ranges import param_ranges
from SAXS_analysis.utils.size_distribution_models import size_distribution_models
from SAXS_analysis.data_processing.utils import load_and_process_SAS_data
import SAXS_analysis
ROOT_DIR = SAXS_analysis.ROOT_DIR

class SAS_Fitter:
    """
    A class used to fit Small Angle Scattering (SAS) data using various form factor models and solvers.

    Attributes
    ----------
    DataName : str
        The name of the data file to be loaded.
    MLName : str
        The name of the machine learning model to be used for prediction.
    ML_formfactorModel : bool
        If True, the machine learning model will be used to predict the form factors.
    ML_parameterModel : bool
        If True, the machine learning model will be used to predict the parameters as initial values for the fit.
    class_names : numpy.ndarray
        An array containing the names of the classes for the machine learning model.
    param_ranges : dict
        A dictionary containing the ranges for the parameters of the form factor models.

    Methods
    -------
    load_data(Datafile, qmin, qmax):
        Loads the SAS data from a file or a numpy array, normalizes it, and filters it based on the provided qmin and qmax values.
    predict_formfactor():
        Uses a pre-trained XGBoost model to predict the form factor for the given data.
    predict_parameters(formfactor):
        Uses a pre-trained XGBoost model to predict the parameters for a given form factor.
    fit_sas_data(formfactor, solver, smearing):
        Fits the SAS data using a specified form factor model and solver.
    """

    def __init__(self):
        self.radius_pd_type = size_distribution_models
        self.param_ranges = param_ranges
        pass

    def load_data(self, data_source=None, qmin=None, qmax=None, error_weighting=None, normalization_type='None', x=None, y=None, z=None):
        self.error_weighting = error_weighting
        self.data = load_and_process_SAS_data(data_source=data_source, x=x, y=y, z=z, qmin=qmin, qmax=qmax, error_weighting=error_weighting, normalization_type=normalization_type)

    def fit_sas_data(self, formfactor, solver, smearing):
        """
        Function to fit Small Angle Scattering (SAS) data using a specified form factor model and solver.

        Parameters
        ----------
        formfactor : str
            Form factor model to use for the fit.
        solver : str
            Solver to use for the fit ('dream' or 'lm').
        smearing : float
            Smearing parameter for the fit.

        Returns
        -------
        goodness_of_fit : float
            The goodness of fit measure (chi-squared) for the fitted model.
        R_w : float
            The weighted residual of the fit.
        fitted_params : dict
            A dictionary containing the fitted parameters.

        Notes
        -----
        The function uses a machine learning model to predict the parameters for the given form factor if `self.ML_parameterModel` is True.
        The parameters for the form factor model are defined in `self.param_ranges`.
        The function fits the data using the Bumps library and returns the goodness of fit, the weighted residual, and the fitted parameters.
        """

        # Load the model and get the parameters
        kernel = load_model(ROOT_DIR / formfactor)
        model_parameters_ph = dict()
        # Get the names of the parameters and update the model_parameters attribute
        model_parameters_ph = [param.name for param in kernel.info.parameters.call_parameters] 
        model_parameters_ph.extend([f'{param}_pd' for param in kernel.info.parameters.pd_1d])
        model_parameters_ph.extend([f'{param}_pd_n' for param in kernel.info.parameters.pd_1d])
        model_parameters_ph.extend([f'{param}_pd_n_sigma' for param in kernel.info.parameters.pd_1d])
        model_parameters_ph.extend([f'{param}_pd_type' for param in kernel.info.parameters.pd_1d])

        # Define the parameters to be fitted
        model_parameters = {}
        model_parameters_fit = {}
        for param_name in model_parameters_ph:
            if param_name in self.param_ranges:
                if param_name.endswith('_pd_type'):
                    param_values = np.random.choice(self.radius_pd_type)
                else:   
                    param_values = (np.random.uniform(*self.param_ranges[param_name]), self.param_ranges[param_name][0]-1e-6, self.param_ranges[param_name][1]+1e-6)
                if isinstance(param_values, (int, float, str)):
                    # If there's only one value, fix the parameter to this value
                    model_parameters[param_name] = param_values
                elif len(param_values) == 3:
                    # If there are three values, set the parameter to the first value and allow it to vary between the second and third values
                    model_parameters_fit[param_name] = Parameter(param_values[0], limits=(0,inf), name=param_name).range(param_values[1], param_values[2])

        # Create the problem
        problem = self.make_problem(kernel, model_parameters, model_parameters_fit, smearing)
        mapper = MPMapper.start_mapper(problem, None, cpus=0) #cpu=0 for all CPUs
        result = bumps.fitters.fit(problem, method=solver, mapper=mapper, burn=10, samples=1e4) # https://bumps.readthedocs.io/en/latest/_modules/bumps/mapper.html and https://readthedocs.org/projects/bumps/downloads/pdf/latest/

        #plt.clf()
        #problem.plot()    
        #plt.savefig(f"{self.DataName}_{formfactor}_{self.error_weigthing}_noResolution.png")                               
        # Get the goodness of fit
        goodness_of_fit = problem.chisq()

        # Calculate the calculated intensity from the model
        I_calc = problem.fitness.theory()
        self.I_calc = I_calc
        # Calculate R_w
        if hasattr(self.data, 'dy'):
            R_w = np.sqrt(np.sum(((self.data.y - I_calc) ** 2) / self.data.dy ** 2) / np.sum((self.data.y ** 2) / self.data.dy ** 2))
        else:
            R_w = np.sqrt(np.sum((self.data.y - I_calc) ** 2) / np.sum(self.data.y ** 2))

        # Get the fitted parameters
        fitted_params = problem.fitness.model.state()

        return I_calc, goodness_of_fit, R_w, fitted_params

    def make_problem(self, kernel, model_parameters, model_parameters_fit, smearing=0.0):
        model = Model(model=kernel, **model_parameters, **model_parameters_fit)
        
        # Fit the data
        experiment = sasmodels.bumps_model.Experiment(data=self.data, model=model)
        smearing = sasmodels.resolution.Slit1D(self.data.x, smearing)
        experiment.resolution = smearing # set the resolution
        problem = bumps.fitproblem.FitProblem(experiment)
        return problem
    
    def plot_sas_data(self):
        plt.clf()
        plt.plot(self.data.x, self.data.y, label='Data')
        plt.plot(self.data.x, self.I_calc, label='Fitted')
        plt.legend()
        plt.show()
    






