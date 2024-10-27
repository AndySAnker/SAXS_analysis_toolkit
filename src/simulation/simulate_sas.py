import numpy as np
import random
from sasmodels.core import load_model
from sasmodels.direct_model import DirectModel
from sasmodels.data import empty_data1D
from src.utils.parameter_ranges import param_ranges
from src.data_processing.utils import normalize_intensity

#Can I stop SasView trying to use my GPUs?
#Yes. Create a system environment variable called SAS_OPENCL and give it the value ‘None’.
# See more: https://www.sasview.org/docs/user/qtgui/Perspectives/Fitting/gpu_setup.html

class SAS_Simulator:
    """
    A class used to simulate Small Angle Scattering (SAS) data.

    Attributes
    ----------
    q : numpy.ndarray
        A numpy array representing the q-range for the simulation.
    resolution : float
        The resolution for the simulation.
    model : sasmodels.Model
        The SAS model to be used for the simulation.
    model_parameters : dict
        A dictionary mapping form factor models to their parameters.
    param_ranges : dict
        A dictionary defining the ranges for each parameter.

    Methods
    -------
    simulate_SAS(parameters)
        Simulates SAS data using the model associated with this object.
    assign_parameters_model()
        Assigns random values to the parameters of the model associated with this object.
    generate_parameters_and_simulate_SAS()
        Generates parameters and simulates SAS for the model associated with this object.
    """

    def __init__(self, model_str='sphere', q=np.linspace(0.001, 1.5, 1000), resolution=np.random.uniform(0.0, 0.0)): 
        """
        Initializes the object with a specified model and default values for other attributes.

        The function sets the model string, creates a linspace for q values, sets a random resolution, 
        loads the model, initializes an empty dictionary for model parameters, and sets parameter ranges.

        Parameters:
        model_str (str): The name of the form factor model. Defaults to 'sphere'.

        Returns:
        None
        """

        self.model_str = model_str
        self.q = q
        self.resolution = resolution
        self.model = load_model(self.model_str)
        self.model_parameters = dict()
        # Get the names of the parameters and update the model_parameters attribute
        self.model_parameters[self.model_str] = [param.name for param in self.model.info.parameters.call_parameters] 
        self.model_parameters[self.model_str].extend([f'{param}_pd' for param in self.model.info.parameters.pd_1d])
        self.model_parameters[self.model_str].extend([f'{param}_pd_type' for param in self.model.info.parameters.pd_1d])
        self.model_parameters[self.model_str].extend([f'{param}_pd_n' for param in self.model.info.parameters.pd_1d])
        self.model_parameters[self.model_str].extend([f'{param}_pd_nsigma' for param in self.model.info.parameters.pd_1d])
        self.param_ranges = param_ranges
        
    def simulate_SAS(self, parameters, add_noise=True, normalization_type='peak'):
        """
        Simulates Small Angle Scattering (SAS) data using the form factor model associated with this object.

        The function first creates an empty 1D data set with the q values and resolution stored in the object. 
        It then creates a DirectModel with this data and the model stored in the object. The model is calculated 
        with the specified parameters, and the resulting SAS data is normalized. If add_noise is True, Poisson noise 
        is added to the SAS data.

        Parameters:
        parameters (dict): The parameters to use when calculating the model.
        add_noise (bool): Whether to add Poisson noise to the SAS data.
        normalization_type (str): The type of normalization to apply. 
                                  Options are 'None' or 'peak'. Default is 'peak'.

        Returns:
        numpy.ndarray: The normalized simulated SAS data, with added Poisson noise if add_noise is True.
        """

        # Create an empty 1D data set with the specified q values and resolution
        data = empty_data1D(self.q, resolution=self.resolution)

        # Create a DirectModel with the data and model
        ph = DirectModel(data, self.model)

        # Calculate the model with the specified parameters
        Iq = ph(**parameters)

        # Add Poisson noise to the SAS data if add_noise is True        
        if add_noise:
            # Simulate photon counts (assume Iq represents mean photon counts)
            photon_counts = np.random.poisson(Iq)
            # Convert back to intensity
            Iq = photon_counts

        # Get uncertainty from the data
        dIq = np.sqrt(Iq)
        
        # Normalize the SAS data
        Iq = normalize_intensity(Iq, normalization_type)
        dIq = normalize_intensity(dIq, normalization_type)

        return Iq, dIq

    def assign_parameters_model(self):
        """
        Assigns random values to the parameters of the form factor model associated with this object.

        The function iterates over the parameters of the model, which are stored in `self.model_parameters`. 
        For each parameter, it assigns a random value within the range specified in `self.param_ranges`.

        Returns:
        dict: A dictionary where the keys are the parameter names and the values are the randomly assigned values.

        Raises:
        Warning: If a parameter is not found in `self.param_ranges`, a warning message is printed.
        """

        # Assign parameters to the model
        parameters = {}
        for param in self.model_parameters[self.model_str]:
            if param in self.param_ranges:
                if '_pd_' in param:
                    parameters[param] = random.choice(self.param_ranges[param])
                else:
                    min_val, max_val = self.param_ranges[param]
                    parameters[param] = random.uniform(min_val, max_val)
            else:
                #print(f"Warning: Parameter not found in param_ranges: {param}")
                continue
        return parameters

    def generate_parameters_and_simulate_SAS(self, add_noise=True, normalization_type='peak'):
        """
        Generates parameters and simulates Small Angle Scattering (SAS) for a given model.

        The model is determined by the current state of the object. The function first calls 
        the `assign_parameters_model` method to generate the parameters for the model, and then 
        simulates the SAS data using these parameters by calling the `simulate_SAS` method.

        Returns:
        tuple: A tuple containing two numpy.ndarrays. The first array represents the q values, 
        and the second array represents the simulated SAS data.
        """

        # Generate parameters for the form factor model
        parameters = self.assign_parameters_model()
        
        # Simulate the SAS data
        Iq, dIq = self.simulate_SAS(parameters, add_noise=add_noise, normalization_type=normalization_type)

        return self.q, Iq, dIq, parameters

"""
def run_simulation(input):
    formfactor_model, structurefactor_model, powerlaw_model, i = input
    #print('Formfactor model: ', formfactor_model, 'Structurefactor model: ', structurefactor_model, 'Powerlaw model: ', powerlaw_model, ' Iteration: ', i+1)

    # Create an instance of the SAS_Simulator class
    simulator = SAS_Simulator(formfactor_model)

    # Simulate SAS data for the form factor model
    q, Iq_formfactor, dIq_formfactor, parameters_formfactor = simulator.generate_parameters_and_simulate_SAS()
    formfactor_scaling = np.random.uniform(0, 1)
    Iq_total = formfactor_scaling * Iq_formfactor
    dIq_total = formfactor_scaling * dIq_formfactor

    if structurefactor_model is not None:
        q, Iq_structurefactor, dIq_structurefactor, parameters_structurefactor = simulator.generate_parameters_and_simulate_SAS()
        structurefactor_scaling = np.random.uniform(0, 1)
        Iq_total += structurefactor_scaling * Iq_structurefactor
        dIq_total += structurefactor_scaling * dIq_structurefactor
    else:
        parameters_structurefactor = None

    if powerlaw_model is not None:
        q, Iq_powerlaw, dIq_powerlaw, parameters_powerlaw = simulator.generate_parameters_and_simulate_SAS()
        powerlaw_scaling = np.random.uniform(0, 1)
        Iq_total += powerlaw_scaling * Iq_powerlaw
        dIq_total += powerlaw_scaling * dIq_powerlaw
    else:
        parameters_powerlaw = None

    # Renormalise the total Iq
    dIq_total = dIq_total / np.max(Iq_total)
    Iq_total /= np.max(Iq_total)

    return Iq_total, dIq_total, formfactor_model, parameters_formfactor, structurefactor_model, parameters_structurefactor, powerlaw_model, parameters_powerlaw, i
"""