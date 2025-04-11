import argparse
import time
import numpy as np
from SAXS_analysis.classification.XGBoost_classification import train_model, evaluate_model
from SAXS_analysis.utils.configs import load_config
from SAXS_analysis.utils.logging import setup_logging
from SAXS_analysis.classification.utils import process_data_classification
import SAXS_analysis
ROOT_DIR = SAXS_analysis.ROOT_DIR

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    start_time = time.time()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Starting experiment {config['experiment_id']}: {config['experiment_name']}")
    logger.info(f'Configuration: {config}')

    logger.debug(f"filename: {config['data']['file_name']}")
    logger.debug(f"num_data_points: {config['data']['num_data_points']}")
    logger.debug(f"normalise_data: {config['data']['normalise_data']}")
    logger.debug(f"hyperparameter_optimisation: {config['model']['hyperparameter_optimisation']}")
    logger.debug(f"use_gpu: {config['model']['use_gpu']}")
    logger.debug(f"model_save_path: {config['model']['model_save_path']}")
    logger.debug(f"class_names_save_path: {config['model']['class_names_save_path']}")
    logger.debug(f"plot_results: {config['evaluation']['plot_results']}")
    logger.debug(f"early_stopping_rounds: {config['model']['early_stopping_rounds']}")

    # Load the data
    logger.info("Loading and processing data...")
    dtrain, dval, dtest, class_names = process_data_classification(
        file_name=ROOT_DIR / 'data' / 'simulated' / config['data']['file_name'],
        num_data_points=config['data']['num_data_points'],
        normalize=config['data']['normalise_data']    
    )
    logger.info(f"Number of data points: {dtrain.num_row() + dval.num_row() + dtest.num_row()}")
    logger.info(f"Number of data points for training: {dtrain.num_row()}")
    logger.info(f"dtrain shape: {dtrain.num_row()} x {dtrain.num_col()}")
    logger.info(f"Number of data points for validation: {dval.num_row()}")
    logger.info(f"dval shape: {dval.num_row()} x {dval.num_col()}")
    logger.info(f"Number of data points for testing: {dtest.num_row()}")
    logger.info(f"dtest shape: {dtest.num_row()} x {dtest.num_col()}")
    logger.info(f"Number of unique classes: {len(class_names)}")
    
    # Log number of entries per class for the total dataset
    total_labels = np.concatenate([dtrain.get_label(), dval.get_label(), dtest.get_label()])
    unique_labels, counts = np.unique(total_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        logger.info(f"Number of entries for class '{class_names[int(label)]}': {count}")
    logger.info("Data loading and processing complete.")

    # Train the model
    logger.info("Starting model training...")
    model, evals_result = train_model(
        dtrain, dval,
        hyperparameter_optimisation=config['model']['hyperparameter_optimisation'],
        use_gpu=config['model']['use_gpu'],
        early_stopping_rounds=config['model']['early_stopping_rounds']
    )
    logger.info("Model training complete.")
    logger.debug(f"evals_result: {evals_result}")

    # Save the model and class names
    logger.info("Saving model and class names...")
    model_save_path = ROOT_DIR / 'models' / 'classification' / config['model']['model_save_path'].format(num_data_points=config['data']['num_data_points'], normalise_data=config['data']['normalise_data'], early_stopping_rounds=config['model']['early_stopping_rounds'], hyperparameter_optimisation=config['model']['hyperparameter_optimisation'])
    model.save_model(model_save_path)
    np.save(ROOT_DIR / 'models' / 'classification' / config['model']['class_names_save_path'].format(num_data_points=config['data']['num_data_points'], normalise_data=config['data']['normalise_data'], early_stopping_rounds=config['model']['early_stopping_rounds'], hyperparameter_optimisation=config['model']['hyperparameter_optimisation']), class_names)
    logger.info(f"Model saved to {model_save_path}")
    logger.info(f"Class names saved to {ROOT_DIR / 'models' / 'classification' / config['model']['class_names_save_path']}")

    # Evaluate the model
    logger.info("Starting model evaluation...")

    train_accuracy, val_accuracy, test_accuracy, baseline_accuracy = evaluate_model(
        model, evals_result, dtrain, dval, dtest, class_names,
        plot_results=config['evaluation']['plot_results'],
        save_basename=config['model']['model_save_path'].format(num_data_points=config['data']['num_data_points'], normalise_data=config['data']['normalise_data'], early_stopping_rounds=config['model']['early_stopping_rounds'], hyperparameter_optimisation=config['model']['hyperparameter_optimisation']).replace('.model', '')
    )
    logger.info(f"Model evaluation complete. Train accuracy: {train_accuracy*100:.4f}%, Validation accuracy: {val_accuracy*100:.4f}%, Test accuracy: {test_accuracy*100:.4f}%, Baseline accuracy: {baseline_accuracy*100:.4f}%")

    elapsed_time = (time.time() - start_time) / 60
    logger.info(f'Total time taken: {elapsed_time:.2f} minutes')
