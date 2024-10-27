import argparse
import h5py
import os
import numpy as np
import multiprocessing
import time
import logging
from src.simulation.simulate_sas import SAS_Simulator
from src.utils.configs import load_config
from src.utils.constants import ROOT_DIR
from src.utils.logging import setup_logging

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
    logger=None
):
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

        logger.info("Starting multiprocessing pool for simulations")
        with multiprocessing.Pool() as pool:
            for index, result in enumerate(pool.imap_unordered(simulate_single, inputs)):
                Iq, dIq, formfactor_model, parameters_formfactor, structurefactor_model, parameters_structurefactor, powerlaw_model, parameters_powerlaw, _ = result

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

    return Iq_total, dIq_total, formfactor_model, parameters_formfactor, structurefactor_model, parameters_structurefactor, powerlaw_model, parameters_powerlaw, i

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config)

    logger.info(f"Starting experiment {config['experiment_id']}: {config['experiment_name']}")
    logger.info(f"Configuration: {config}")

    start_time = time.time()

    try:
        simulate_sas_datasets(
            num_datasets=config['simulation']['num_datasets'],
            output_dir=config['simulation']['output_dir'],
            filename=config['simulation']['filename_template'].format(num_datasets=config['simulation']['num_datasets']),
            dtype=np.dtype(config['simulation']['dtype']),
            structurefactor_include_chance=config['simulation']['structure_factor_percentage'],
            powerlaw_include_chance=config['simulation']['powerlaw_include_chance'],
            chunk_size=tuple(config['simulation']['chunk_size']),
            form_factors=config['simulation']['form_factors'],
            structure_factors=config['simulation']['structure_factors'],
            powerlaws=config['simulation']['powerlaws'],
            q_range=(config['simulation']['q_range']['start'],
                     config['simulation']['q_range']['end'],
                     config['simulation']['q_range']['num_points']),
            resolution_range=(config['simulation']['resolution']['min'],
                              config['simulation']['resolution']['max']),
            normalization_type=config['simulation']['normalization_type'],
            add_noise=config['simulation']['add_noise'],
            logger=logger
        )
    except Exception as e:
        logger.error(f"An error occurred during simulation: {str(e)}", exc_info=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    days, rem = divmod(elapsed_time, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    logger.info("Elapsed time: {:0>2}:{:0>2}:{:0>2}:{:05.2f}".format(int(days), int(hours), int(minutes), seconds))
