import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from SAXS_analysis.forward_ann.utils import process_data_regression, calculate_test_mae
from SAXS_analysis.forward_ann.forward_ann import ForwardNN
from SAXS_analysis.visualization.utils import plot_ann_loss, plot_prediction_error_histograms
from SAXS_analysis.utils.configs import load_config
from SAXS_analysis.utils.logging import setup_logging
from SAXS_analysis.forward_ann.utils import lr_scheduler
import SAXS_analysis
ROOT_DIR = SAXS_analysis.ROOT_DIR

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
        logger.debug(f"Configuration: {config}")

        # Prepare common paths and hyperparameters
        file_path = ROOT_DIR / config['data']['file_name']
        formfactors = config['data'].get('formfactors', None)
        output_dir = ROOT_DIR / config['model']['model_save_path']
        num_epochs = config['model']['num_epochs']
        learning_rate = config['model']['learning_rate']
        batch_size = config['model']['batch_size']

        # Debug configuration settings
        for formfactor in formfactors:
            logger.info(f"Processing data for formfactor: {formfactor}")
            logger.info(f"Data file: {file_path}")
            logger.info(f"Num epochs: {num_epochs}, Learning rate: {learning_rate}, Batch size: {batch_size}")
            logger.info(f"Model output dir: {output_dir}")

            # Load and process data
            logger.info("Loading and processing data...")
            (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = process_data_regression(file_path, formfactor)
            logger.info(f"Data loaded: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

            # Model setup
            input_size = X_train.shape[1]
            output_size = y_train.shape[1] if y_train.ndim > 1 else 1
            model = ForwardNN(input_size=input_size, output_size=output_size)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = lr_scheduler(optimizer, config['model'])

            # Load data
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
            test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

            train_losses = []
            val_losses = []

            # Train model
            logger.info("Starting training...")
            train_start_time = time.time()

            early_stopping_cfg = config['model'].get('early_stopping', {})
            early_stopping_enabled = early_stopping_cfg.get('enabled', False)
            patience = early_stopping_cfg.get('patience', 10)
            min_delta = early_stopping_cfg.get('min_delta', 0.0)

            best_val_loss = float('inf')
            epochs_no_improve = 0

            for epoch in range(num_epochs):
                model.train()
                total_loss = 0.0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_train_loss = total_loss / len(train_loader)
                train_losses.append(avg_train_loss)

                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                if scheduler is not None:
                    prev_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(avg_val_loss)
                    new_lr = optimizer.param_groups[0]['lr']
                    if new_lr < prev_lr:
                        print(f"[lr_scheduler] Learning rate reduced from {prev_lr:.6e} to {new_lr:.6e}")

                logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

                if early_stopping_enabled:
                    if best_val_loss - avg_val_loss > min_delta:
                        best_val_loss = avg_val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break

            total_train_time = time.time() - train_start_time
            logger.info(f"Training completed in {total_train_time:.2f} seconds.")

            calculate_test_mae(model, test_loader, scaler, formfactor)

            # Save model
            relative_path = Path(config["model"]["model_save_path"])
            model_dir = ROOT_DIR / relative_path
            model_dir.mkdir(parents=True, exist_ok=True)
            model_save_path = model_dir / f"{formfactor}_forward.pth"
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Model saved to: {model_save_path}")

            # Evaluate model
            logger.info("Plotting results...")
            plot_ann_loss(train_losses, val_losses, formfactor)
            plot_prediction_error_histograms(model, test_loader, scaler, formfactor, ROOT_DIR)

    except Exception as e:
        if logger:
            logger.error(f"An error occurred: {e}", exc_info=True)
        else:
            print(f"An error occurred: {e}")
        raise

    finally:
        if logger:
            logger.info(f'Total script time: {(time.time() - start_time)/60:.2f} minutes')
        else:
            print(f'Total script time: {(time.time() - start_time)/60:.2f} minutes')

if __name__ == '__main__':
    main()
