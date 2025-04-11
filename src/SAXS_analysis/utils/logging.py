import os
import logging
import SAXS_analysis
ROOT_DIR = SAXS_analysis.ROOT_DIR

def setup_logging(config):
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(ROOT_DIR / 'logs', f"{config['experiment_name']} - {config['experiment_id']}")
    os.makedirs(log_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(log_dir, f"{config['experiment_id']}.log")
    
    # Set logging level based on verbosity
    if config['verbose'] == 0:
        log_level = logging.WARNING  # Only warnings and errors
    elif config['verbose'] == 1:
        log_level = logging.INFO     # Info and above
    else:
        log_level = logging.DEBUG    # All messages
        
    handlers = [logging.FileHandler(log_file)]
    if config['verbose'] > 0:  # Only add stream handler if verbose > 0
        handlers.append(logging.StreamHandler())
        
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)