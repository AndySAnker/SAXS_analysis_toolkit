import numpy as np
import random
import os
import h5py
import multiprocessing
import SAXS_analysis
ROOT_DIR = SAXS_analysis.ROOT_DIR
from sasmodels.core import load_model
from sasmodels.direct_model import DirectModel
from sasmodels.data import empty_data1D
from SAXS_analysis.utils.parameter_ranges import param_ranges
from SAXS_analysis.data_processing.utils import normalize_intensity

"""Simulate one-dimensional SAS curves using sasmodels form factors and optional structure factors.

Writes chunked HDF5 training data under ``ROOT_DIR`` with datasets ``SAXS_dataset``, ``q``,
``formfactor``, ``parameters_formfactor``, and optionally structure factor / power-law fields.

To disable OpenCL/GPU in SasView/sasmodels, set environment variable ``SAS_OPENCL=None``.
"""

# Can I stop SasView trying to use my GPUs?
# Yes. Create a system environment variable called SAS_OPENCL and give it the value 'None'.
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
    """

    def __init__(self, model_str='sphere', q=np.linspace(0.001, 1.5, 1000), resolution=np.random.uniform(0.0, 0.0)): 
        """
        Initialize the simulator with a form factor model and q grid.

        Args:
            model_str: Form factor model name (default ``'sphere'``).
            q: Scattering vector grid (1/Å).
            resolution: Instrument resolution value passed to sasmodels data.
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
        Simulate SAS intensity using the loaded sasmodels form factor.

        Builds a 1D ``DirectModel``, evaluates ``I(q)``, optionally adds Poisson noise,
        then normalizes.

        Args:
            parameters: Model parameters for sasmodels.
            add_noise: If True, apply Poisson noise to intensities.
            normalization_type: ``'peak'`` or ``'none'`` (see ``normalize_intensity``).

        Returns:
            Tuple ``(Iq, dIq)`` after normalization.
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
        Assign random values for each model parameter using ``param_ranges``.

        Returns:
            Parameter name to value mapping for the current ``model_str``.
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
        Sample parameters, simulate intensities, and return q, ``I(q)``, uncertainty, and parameters.

        Returns:
            Tuple ``(q, Iq, dIq, parameters)``.
        """

        # Generate parameters for the form factor model
        parameters = self.assign_parameters_model()
        
        # Simulate the SAS data
        Iq, dIq = self.simulate_SAS(parameters, add_noise=add_noise, normalization_type=normalization_type)

        return self.q, Iq, dIq, parameters

def simulate_sas_datasets(
    num_datasets,
    output_dir,
    filename,
    dtype,
    structurefactor_include_chance,
    powerlaw_include_chance,
    chunk_size=(1000, 1000),
    form_factors=['sphere', 'cylinder', 'ellipsoid'],
    structure_factors=['hardsphere', 'squarewell', 'stickyhardsphere'],
    powerlaws=['power_law'],
    q_range=(0.001, 1.5, 1000),
    resolution_range=(0.0, 0.1),
    normalization_type='peak',
    add_noise=True,
    use_multiprocessing=True,
    logger=None
):
    """Generate many labeled curves and append them to a single HDF5 file.

    Enumerates combinations of form factor / optional structure factor / optional power law,
    then for each combination draws ``num_datasets`` samples. Uses multiprocessing (spawn) by
    default to avoid HDF5 fork issues. Performs a parent-process sasmodels warmup to reduce
    races on kernel compilation.

    Args:
        num_datasets: Curves per (form factor, structure, powerlaw) combination.
        output_dir: Folder under ``ROOT_DIR`` for the HDF5 file.
        filename: HDF5 filename.
        dtype: Storage dtype for intensities.
        structurefactor_include_chance, powerlaw_include_chance: Probabilities in ``[0, 1]``.
        chunk_size: HDF5 chunk and flush cadence ``(rows, q_points)``.
        form_factors, structure_factors, powerlaws: sasmodels names to combine.
        q_range: ``(q_min, q_max, n_points)`` in 1/Å.
        resolution_range: Uniform range for smearing draws.
        normalization_type, add_noise: Passed to :class:`SAS_Simulator`.
        use_multiprocessing: If False, run ``simulate_single`` in-process.
        logger: Logger with ``info`` / ``debug`` methods.
    """
    logger.info(f"Starting simulation of {num_datasets} datasets")
    logger.debug(f"Output directory: {output_dir}, Filename: {filename}")
    logger.debug(f"Chunk size: {chunk_size}, Data type: {dtype}")
    
    os.makedirs(ROOT_DIR / output_dir, exist_ok=True)
    logger.debug(f"Created output directory: {ROOT_DIR / output_dir}")
    
    with h5py.File(ROOT_DIR / output_dir / filename, 'w') as f:
        logger.debug(f"Created HDF5 file: {ROOT_DIR / output_dir / filename}")
        
        dset = f.create_dataset('SAXS_dataset', chunk_size, maxshape=(None, chunk_size[1]), dtype=dtype)
        dset_uncertainty = f.create_dataset('SAXS_dataset_uncertainty', chunk_size, maxshape=(None, chunk_size[1]), dtype=dtype)
        logger.debug(f"Created datasets: SAXS_dataset and SAXS_dataset_uncertainty")

        q = np.linspace(q_range[0], q_range[1], q_range[2])
        f.create_dataset('q', data=q)
        logger.debug(f"Created q dataset with range: {q_range}")

        unique_combinations = list(set((
            formfactor,
            structurefactor if np.random.uniform(0, 1) < structurefactor_include_chance else None,
            powerlaw_model if np.random.uniform(0, 1) < powerlaw_include_chance else None
        ) for formfactor in form_factors 
            for structurefactor in structure_factors 
            for powerlaw_model in powerlaws))

        inputs = [(combo[0], combo[1], combo[2], i, q, resolution_range, normalization_type, add_noise) 
                  for combo in unique_combinations
                  for i in range(num_datasets)]
        logger.info(f"Generated {len(inputs)} unique input combinations")

        formfactor_dset = f.create_dataset('formfactor', (chunk_size[0], 1), maxshape=(None, 1), dtype=h5py.special_dtype(vlen=str))
        parameters_formfactor_dset = f.create_dataset('parameters_formfactor', (chunk_size[0], 1), maxshape=(None, 1), dtype=h5py.special_dtype(vlen=str))
        if structurefactor_include_chance > 0:
            structurefactor_dset = f.create_dataset('structurefactor', (chunk_size[0], 1), maxshape=(None, 1), dtype=h5py.special_dtype(vlen=str))
            parameters_structurefactor_dset = f.create_dataset('parameters_structurefactor', (chunk_size[0], 1), maxshape=(None, 1), dtype=h5py.special_dtype(vlen=str))
        if powerlaw_include_chance > 0:
            powerlaw_dset = f.create_dataset('powerlaw', (chunk_size[0], 1), maxshape=(None, 1), dtype=h5py.special_dtype(vlen=str))
            parameters_powerlaw_dset = f.create_dataset('parameters_powerlaw', (chunk_size[0], 1), maxshape=(None, 1), dtype=h5py.special_dtype(vlen=str))
        logger.debug("Created datasets for form factors, structure factors, and power laws")

        # sasmodels lazily compiles/loads model kernels on first use and stores them under ~/.sasmodels.
        # In practice this can race in multiprocessing (multiple workers trying to build/load the same kernel).
        # To make local testing robust, we "warm up" compilation in the parent process first.
        try:
            for ff in form_factors:
                sim = SAS_Simulator(ff, q=q, resolution=np.random.uniform(resolution_range[0], resolution_range[1]))
                sim_params = sim.assign_parameters_model()
                sim.simulate_SAS(sim_params, add_noise=False, normalization_type=normalization_type)
        except Exception:
            # If warmup fails, continue and let the pool surface the real error with tracebacks.
            pass

        if not use_multiprocessing:
            logger.info("Running simulations sequentially (multiprocessing disabled)")
            iterator = (simulate_single(t) for t in inputs)
            for index, result in enumerate(iterator):
                q, Iq, dIq, formfactor_model, parameters_formfactor, structurefactor_model, parameters_structurefactor, powerlaw_model, parameters_powerlaw, _ = result

                dset.resize(index + 1, axis=0)
                dset[index, :] = np.array([Iq]).astype(dtype)
                dset_uncertainty.resize(index + 1, axis=0)
                dset_uncertainty[index, :] = np.array([dIq]).astype(dtype)

                formfactor_dset.resize(index + 1, axis=0)
                formfactor_dset[index] = str(formfactor_model)
                parameters_formfactor_dset.resize(index + 1, axis=0)
                parameters_formfactor_dset[index] = str(parameters_formfactor)

                if structurefactor_include_chance > 0 and structurefactor_model is not None:
                    structurefactor_dset.resize(index + 1, axis=0)
                    structurefactor_dset[index] = str(structurefactor_model)
                    parameters_structurefactor_dset.resize(index + 1, axis=0)
                    parameters_structurefactor_dset[index] = str(parameters_structurefactor)

                if powerlaw_include_chance > 0 and powerlaw_model is not None:
                    powerlaw_dset.resize(index + 1, axis=0)
                    powerlaw_dset[index] = str(powerlaw_model)
                    parameters_powerlaw_dset.resize(index + 1, axis=0)
                    parameters_powerlaw_dset[index] = str(parameters_powerlaw)

                if (index + 1) % chunk_size[0] == 0:
                    f.flush()
                    logger.info(f"Saved number {index+1} out of {len(inputs)} chunks to DataFile")

                if (index + 1) % 1000 == 0:
                    logger.debug(f"Processed {index+1} datasets")
        else:
            logger.info("Starting multiprocessing pool for simulations")
            # h5py is not fork-safe; use spawn to avoid invalid HDF5 identifiers.
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool() as pool:
                for index, result in enumerate(pool.imap_unordered(simulate_single, inputs)):
                    q, Iq, dIq, formfactor_model, parameters_formfactor, structurefactor_model, parameters_structurefactor, powerlaw_model, parameters_powerlaw, _ = result

                    dset.resize(index + 1, axis=0)
                    dset[index, :] = np.array([Iq]).astype(dtype)
                    dset_uncertainty.resize(index + 1, axis=0)
                    dset_uncertainty[index, :] = np.array([dIq]).astype(dtype)

                    formfactor_dset.resize(index + 1, axis=0)
                    formfactor_dset[index] = str(formfactor_model)
                    parameters_formfactor_dset.resize(index + 1, axis=0)
                    parameters_formfactor_dset[index] = str(parameters_formfactor)

                    if structurefactor_include_chance > 0 and structurefactor_model is not None:
                        structurefactor_dset.resize(index + 1, axis=0)
                        structurefactor_dset[index] = str(structurefactor_model)
                        parameters_structurefactor_dset.resize(index + 1, axis=0)
                        parameters_structurefactor_dset[index] = str(parameters_structurefactor)

                    if powerlaw_include_chance > 0 and powerlaw_model is not None:
                        powerlaw_dset.resize(index + 1, axis=0)
                        powerlaw_dset[index] = str(powerlaw_model)
                        parameters_powerlaw_dset.resize(index + 1, axis=0)
                        parameters_powerlaw_dset[index] = str(parameters_powerlaw)

                    if (index + 1) % chunk_size[0] == 0:
                        f.flush()
                        logger.info(f"Saved number {index+1} out of {len(inputs)} chunks to DataFile")
                    
                    if (index + 1) % 1000 == 0:
                        logger.debug(f"Processed {index+1} datasets")

        logger.info(f"Finished simulation of {num_datasets} datasets")
        logger.debug(f"Final dataset shape: {dset.shape}")

def simulate_single(input_tuple):
    """Simulate one curve: form factor plus optional structure and power-law contributions.

    Scales contributions with a random Dirichlet vector so the sum of weights is 1.
    """
    formfactor_model, structurefactor_model, powerlaw_model, i, q, resolution_range, normalization_type, add_noise = input_tuple
    resolution = np.random.uniform(resolution_range[0], resolution_range[1])

    scalings = np.random.dirichlet(np.ones(3))
    formfactor_scaling, structurefactor_scaling, powerlaw_scaling = scalings

    simulator = SAS_Simulator(formfactor_model, q=q, resolution=resolution)
    q, Iq_formfactor, dIq_formfactor, parameters_formfactor = simulator.generate_parameters_and_simulate_SAS(add_noise=add_noise, normalization_type=normalization_type)
    Iq_total = formfactor_scaling * Iq_formfactor
    dIq_total = formfactor_scaling * dIq_formfactor

    if structurefactor_model is not None:
        simulator = SAS_Simulator(structurefactor_model, q=q, resolution=resolution)
        q, Iq_structurefactor, dIq_structurefactor, parameters_structurefactor = simulator.generate_parameters_and_simulate_SAS(add_noise=add_noise, normalization_type=normalization_type)
        Iq_total += structurefactor_scaling * Iq_structurefactor
        dIq_total += structurefactor_scaling * dIq_structurefactor
    else:
        parameters_structurefactor = None

    if powerlaw_model is not None:
        simulator = SAS_Simulator(powerlaw_model, q=q, resolution=resolution)
        q, Iq_powerlaw, dIq_powerlaw, parameters_powerlaw = simulator.generate_parameters_and_simulate_SAS(add_noise=add_noise, normalization_type=normalization_type)
        Iq_total += powerlaw_scaling * Iq_powerlaw
        dIq_total += powerlaw_scaling * dIq_powerlaw
    else:
        parameters_powerlaw = None

    return q, Iq_total, dIq_total, formfactor_model, parameters_formfactor, structurefactor_model, parameters_structurefactor, powerlaw_model, parameters_powerlaw, i
