import argparse
from src.data_processing.utils import process_data
from src.regression.XGBoost_regression import train_model, evaluate_model
from src.utils.configs import load_config
import time
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    start_time = time.time()

    # Load configuration
    config = load_config(args.config)

    # Load the data
    dtrain, dval, dtest = process_data(
        file_name=config['data']['file_name'],
        formfactor=config['data']['formfactor'],
        num_data_points=config['data']['num_data_points']
    )

    # Train the model
    model, evals_result = train_model(
        dtrain, dval,
        use_bayesian_optimization=config['model']['use_bayesian_optimization'],
        use_gpu=config['model']['use_gpu']
    )

    # Save the model
    model_save_path = config['model']['model_save_path'].format(num_data_points=config['data']['num_data_points'])
    model.save_model(model_save_path)

    # Evaluate the model
    evaluate_model(
        model, evals_result, dtrain, dval, dtest, config['data']['formfactor'],
        plot_results=config['evaluation']['plot_results']
    )

    print(f'Time taken: {(time.time() - start_time)/60:.2f} minutes')
