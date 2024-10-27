import os
import logging

def setup_logging(config):
    # Create logs directory if it doesn't exist
    log_dir = os.path.join('logs', f"{config['experiment_name']} - {config['experiment_id']}")
    os.makedirs(log_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(log_dir, f"{config['experiment_id']}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)