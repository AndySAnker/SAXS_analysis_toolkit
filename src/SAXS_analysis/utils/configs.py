import yaml
import SAXS_analysis    
ROOT_DIR = SAXS_analysis.ROOT_DIR

def load_config(config_path):
    with open(ROOT_DIR / config_path, 'r') as file:
        return yaml.safe_load(file)