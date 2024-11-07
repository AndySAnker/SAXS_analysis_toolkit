import argparse
import numpy as np
import time
from src.simulation.simulate_sas import simulate_sas_datasets
from src.utils.configs import load_config
from src.utils.logging import setup_logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config)

    logger.info(f"Starting experiment {config['experiment_id']}: {config['experiment_name']}")
    logger.info(f"Configuration: {config}")

    logger.debug(f"num_datasets: {config['simulation']['num_datasets']}")
    logger.debug(f"output_dir: {config['simulation']['output_dir']}")
    logger.debug(f"filename_template: {config['simulation']['filename_template']}")
    logger.debug(f"dtype: {config['simulation']['dtype']}")
    logger.debug(f"structure_factor_percentage: {config['simulation']['structure_factor_percentage']}")
    logger.debug(f"powerlaw_include_chance: {config['simulation']['powerlaw_include_chance']}")
    logger.debug(f"chunk_size: {config['simulation']['chunk_size']}")
    logger.debug(f"form_factors: {config['simulation']['form_factors']}")
    logger.debug(f"structure_factors: {config['simulation']['structure_factors']}")
    logger.debug(f"powerlaws: {config['simulation']['powerlaws']}")
    logger.debug(f"q_range: {config['simulation']['q_range']}")
    logger.debug(f"resolution_range: {config['simulation']['resolution']['min']}, {config['simulation']['resolution']['max']}")
    logger.debug(f"normalization_type: {config['simulation']['normalization_type']}")
    logger.debug(f"add_noise: {config['simulation']['add_noise']}")

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
