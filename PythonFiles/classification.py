import argparse
from src.data_processing.utils import process_data
from src.classification.XGBoost_classification import train_model, evaluate_model
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
    dtrain, dval, dtest, class_names = process_data(
        file_name=config['data']['file_name'],
        num_data_points=config['data']['num_data_points'],
        normalise_data=config['data']['normalise_data'],
        plot_data=config['data']['plot_data']
    )

    # Train the model
    model, evals_result = train_model(
        dtrain, dval,
        use_bayesian_optimization=config['model']['use_bayesian_optimization'],
        use_gpu=config['model']['use_gpu']
    )

    # Save the model and class names
    model_save_path = config['model']['model_save_path'].format(num_data_points=config['data']['num_data_points'])
    model.save_model(model_save_path)
    np.save(config['model']['class_names_save_path'], class_names)

    # Evaluate the model
    train_accuracy, val_accuracy, test_accuracy, baseline_accuracy = evaluate_model(
        model, evals_result, dtrain, dval, dtest, class_names,
        plot_results=config['evaluation']['plot_results'],
        plot_confusion_matrix=config['evaluation']['plot_confusion_matrix']
    )

    print(f'Time taken: {(time.time() - start_time)/60:.2f} minutes')
