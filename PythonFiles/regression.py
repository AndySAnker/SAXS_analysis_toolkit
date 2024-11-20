import os
import time
import argparse
from pathlib import Path
from src.regression.XGBoost_regression import train_model, evaluate_model
from src.utils.configs import load_config
from src.utils.logging import setup_logging
from src.regression.utils import process_data_regression

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)
        logger = setup_logging(config)
        start_time = time.time()

        logger.info(f"Starting experiment {config['experiment_id']}: {config['experiment_name']}")
        logger.info(f'Configuration: {config}')

        # Debug configuration settings
        logger.debug("Configuration settings:")
        logger.debug(f"Data file: {config['data']['file_name']}")
        logger.debug(f"Number of data points: {config['data']['num_data_points']}")
        logger.debug(f"Form factor: {config['data']['formfactors']}")
        logger.debug(f"Model settings:")
        logger.debug(f"- Use GPU: {config['model']['use_gpu']}")
        logger.debug(f"- Use hyperparameter optimisation: {config['model']['hyperparameter_optimisation']}")
        logger.debug(f"- Early stopping rounds: {config['model']['early_stopping_rounds']}")
        logger.debug(f"- Model save path: {config['model']['model_save_path'].format(num_data_points=config['data']['num_data_points'], normalise_data=config['data']['normalise_data'], early_stopping_rounds=config['model']['early_stopping_rounds'], hyperparameter_optimisation=config['model']['hyperparameter_optimisation'])}")

        for formfactor in config['data']['formfactors']:
            logger.info(f"Processing data for formfactor: {formfactor}")
            # Load and process data
            logger.info("Loading and processing data...")
            dtrain, dval, dtest = process_data_regression(
                file_name=config['data']['file_name'],
                formfactor=formfactor,
                num_data_points=config['data']['num_data_points']
            )
            logger.info(f"Data loaded - Train size: {dtrain.num_row()}, Val size: {dval.num_row()}, Test size: {dtest.num_row()}")

            # Train model
            logger.info("Training model...")
            model, evals_result = train_model(
                dtrain=dtrain,
                dval=dval,
                early_stopping_rounds=config['model']['early_stopping_rounds'],
                hyperparameter_optimisation=config['model']['hyperparameter_optimisation'],
                use_gpu=config['model']['use_gpu']
            )

            # Save model
            logger.info("Saving model...")
            model_path = Path(config['model']['model_save_path'].format(num_data_points=config['data']['num_data_points'], normalise_data=config['data']['normalise_data'], early_stopping_rounds=config['model']['early_stopping_rounds'], hyperparameter_optimisation=config['model']['hyperparameter_optimisation']))
            os.makedirs(model_path, exist_ok=True)
            model.save_model(f"{model_path}/{formfactor}.model")
            logger.info(f"Model saved to {model_path}")

            # Evaluate model
            logger.info("Evaluating model...")
            evaluate_model(
                model=model,
                evals_result=evals_result,
                dtrain=dtrain,
                dval=dval,
                dtest=dtest,
                formfactor=formfactor,
                plot_results=config['evaluation']['plot_results']
            )

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise

    finally:
        logger.info(f'Total time: {(time.time() - start_time)/60:.2f} minutes')

if __name__ == '__main__':
    main()
