import argparse
import time
import numpy as np
from src.classification.XGBoost_classification import train_model, evaluate_model
from src.utils.configs import load_config
from src.utils.logging import setup_logging
from src.classification.utils import process_data_classification

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
    logger.debug(f"plot_data: {config['data']['plot_data']}")
    logger.debug(f"use_bayesian_optimization: {config['model']['use_bayesian_optimization']}")
    logger.debug(f"use_gpu: {config['model']['use_gpu']}")
    logger.debug(f"model_save_path: {config['model']['model_save_path']}")
    logger.debug(f"class_names_save_path: {config['model']['class_names_save_path']}")
    logger.debug(f"plot_results: {config['evaluation']['plot_results']}")
    logger.debug(f"plot_confusion_matrix: {config['evaluation']['plot_confusion_matrix']}")
    logger.debug(f"early_stopping_rounds: {config['model']['early_stopping_rounds']}")

    # Load the data
    logger.info("Loading and processing data...")
    dtrain, dval, dtest, class_names = process_data_classification(
        file_name=config['data']['file_name'],
        num_data_points=config['data']['num_data_points'],
        normalize=config['data']['normalise_data'],
        plot_data=config['data']['plot_data']
    )
    logger.info(f"Number of data points: {len(dtrain) + len(dval) + len(dtest)}")
    logger.info(f"Number of data points for training: {len(dtrain)}")
    logger.info(f"Number of data points for validation: {len(dval)}")
    logger.info(f"Number of data points for testing: {len(dtest)}")
    logger.info(f"Number of unique classes: {len(np.unique(class_names))}")
    logger.info("Data loading and processing complete.")

    # Train the model
    logger.info("Starting model training...")
    model, evals_result = train_model(
        dtrain, dval,
        use_bayesian_optimization=config['model']['use_bayesian_optimization'],
        use_gpu=config['model']['use_gpu'],
        early_stopping_rounds=config['model']['early_stopping_rounds']
    )
    logger.info("Model training complete.")
    logger.debug(f"evals_result: {evals_result}")

    # Save the model and class names
    logger.info("Saving model and class names...")
    model_save_path = config['model']['model_save_path'].format(num_data_points=config['data']['num_data_points'])
    model.save_model(model_save_path)
    np.save(config['model']['class_names_save_path'], class_names)
    logger.info(f"Model saved to {model_save_path}")
    logger.info(f"Class names saved to {config['model']['class_names_save_path']}")

    # Evaluate the model
    logger.info("Starting model evaluation...")
    train_accuracy, val_accuracy, test_accuracy, baseline_accuracy = evaluate_model(
        model, evals_result, dtrain, dval, dtest, class_names,
        plot_results=config['evaluation']['plot_results'],
        plot_confusion_matrix=config['evaluation']['plot_confusion_matrix']
    )
    logger.info(f"Model evaluation complete. Train accuracy: {train_accuracy:.4f}, Validation accuracy: {val_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}, Baseline accuracy: {baseline_accuracy:.4f}")

    elapsed_time = (time.time() - start_time) / 60
    logger.info(f'Total time taken: {elapsed_time:.2f} minutes')
