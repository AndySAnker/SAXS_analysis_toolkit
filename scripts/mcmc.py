import argparse
import time
import numpy as np
import torch
import joblib
from pathlib import Path
import h5py
import ast
from SAXS_analysis.classification.XGBoost_classification import classify_formfactor
from SAXS_analysis.utils.configs import load_config
from SAXS_analysis.utils.logging import setup_logging
from SAXS_analysis.mcmc.utils import run_mcmc
from SAXS_analysis.forward_ann.forward_ann import ForwardNN
from SAXS_analysis.inverse_ann.inverse_ann import InverseNN
from SAXS_analysis.visualization.utils import save_corner_plot
import SAXS_analysis
ROOT_DIR = Path(SAXS_analysis.ROOT_DIR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger = setup_logging(config)

    row_index = config['row_index']
    logger.info(f"Starting inference with row_index {row_index}, file: {config['data']['h5_file']}")

    # Load classification model and class names
    clf_model_path = ROOT_DIR / config['data']['classifier_model']
    clf_class_names_path = ROOT_DIR / config['data']['classifier_class_names']

    # Load dataset
    h5_path = ROOT_DIR / config['data']['h5_file']
    with h5py.File(h5_path, 'r') as f:
        y = f['SAXS_dataset'][row_index]
        param_raw = f['parameters_formfactor'][row_index]
        formfactor_raw = f['formfactor'][row_index]

    # Prepare actual form factor for comparison with predicted form factor
    if isinstance(formfactor_raw, np.ndarray):
        actual_formfactor_bytes = formfactor_raw[0]
        if isinstance(actual_formfactor_bytes, bytes):
            actual_formfactor = actual_formfactor_bytes.decode('utf-8').lower()
        else:
            actual_formfactor = str(actual_formfactor_bytes).lower()
    elif isinstance(formfactor_raw, bytes):
        actual_formfactor = formfactor_raw.decode('utf-8').lower()
    else:
        actual_formfactor = str(formfactor_raw).lower()

    y_std = config.get('y_std', 0.01)

    # Use classification model on data and compare with actual form factor
    predicted_formfactor = classify_formfactor(y, clf_model_path, clf_class_names_path)
    logger.info(f"Predicted formfactor: {predicted_formfactor}")
    logger.info(f"Actual formfactor: {actual_formfactor}")

    if predicted_formfactor != actual_formfactor:
        logger.info(f"Predicted formfactor '{predicted_formfactor}' does NOT match actual formfactor '{actual_formfactor}'. Skipping inference for row {row_index}.")
        return  # Skip inference if form factors do not match
    else:
        logger.info(f"Predicted formfactor matches actual formfactor '{actual_formfactor}'. Proceeding with inference.")

    # Prepare actual parameters for comparison with predicted parameters
    param_bytes = param_raw[0]
    if isinstance(param_bytes, bytes):
        param_str_decoded = param_bytes.decode('utf-8')

    # Parse the dict string to a Python dictionary
    param_dict = ast.literal_eval(param_str_decoded)

    if predicted_formfactor == "sphere":
        parameter_names = ["radius", "radius_pd"]
        forward_model_path = ROOT_DIR / config['data']['forward_model_paths']['sphere']
        inverse_model_path = ROOT_DIR / config['data']['inverse_model_paths']['sphere']
        forward_scaler_path = ROOT_DIR / 'scalers' / 'sphere_scaler.joblib'
        inverse_scaler_path = ROOT_DIR / 'scalers' / 'sphere_inverse_scaler.joblib'

    elif predicted_formfactor == "ellipsoid":
        parameter_names = ["radius_polar", "radius_equatorial"]
        forward_model_path = ROOT_DIR / config['data']['forward_model_paths']['ellipsoid']
        inverse_model_path = ROOT_DIR / config['data']['inverse_model_paths']['ellipsoid']
        forward_scaler_path = ROOT_DIR / 'scalers' / 'ellipsoid_scaler.joblib'
        inverse_scaler_path = ROOT_DIR / 'scalers' / 'ellipsoid_inverse_scaler.joblib'

    elif predicted_formfactor == "cylinder":
        parameter_names = ["radius", "radius_pd", "length", "length_pd"]
        forward_model_path = ROOT_DIR / config['data']['forward_model_paths']['cylinder']
        inverse_model_path = ROOT_DIR / config['data']['inverse_model_paths']['cylinder']
        forward_scaler_path = ROOT_DIR / 'scalers' / 'cylinder_scaler.joblib'
        inverse_scaler_path = ROOT_DIR / 'scalers' / 'cylinder_inverse_scaler.joblib'

    else:
        raise ValueError(f"Unsupported predicted formfactor: {predicted_formfactor}")

    true_input = np.array([param_dict[name] for name in parameter_names])

    # Load scalers
    forward_scaler = joblib.load(forward_scaler_path)
    inverse_scaler = joblib.load(inverse_scaler_path)

    # Load forward model for priors
    forward_model = ForwardNN(input_size=len(y), output_size=len(parameter_names))
    forward_model.load_state_dict(torch.load(forward_model_path, map_location='cpu'))
    forward_model.eval()

    # Use forward model to obtain priors 
    input_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        priors_scaled_tensor = forward_model(input_tensor)
    priors_scaled = priors_scaled_tensor.numpy().flatten()
    logger.info(f"Priors: {priors_scaled}")

    # Inverse scaling to get priors in original scale and scale for inverse model; not exactly necessary
    priors_unscaled = forward_scaler.inverse_transform(priors_scaled.reshape(1, -1)).flatten()
    priors_for_inverse_ann = inverse_scaler.transform(priors_unscaled.reshape(1, -1)).flatten()

    # Load inverse model as surrogate for MCMC sampling 
    s_ml_model = InverseNN(input_size=len(parameter_names), output_size=len(y))
    s_ml_model.load_state_dict(torch.load(inverse_model_path, map_location='cpu'))
    s_ml_model.eval()

    # Load MCMC parameters from config with defaults
    num_params = config.get('num_params', len(parameter_names))
    nwalkers = config.get('nwalkers', 25)
    burn_in = config.get('burn_in', 5000)
    niter = config.get('niter', 5000)

    # Initialize walkers randomly in (0,1)
    p0 = [np.random.rand(num_params) for _ in range(nwalkers)]
    
    # Run MCMC inference with surrogate model
    start_time = time.time()
    chain, log_probs, map_estimate = run_mcmc(p0, nwalkers, niter, y, y_std, row_index, s_ml_model, priors_for_inverse_ann, burn_in)

    theta_max_scaled = chain[np.argmax(log_probs.flatten())]
    MAP_unscaled = inverse_scaler.inverse_transform(theta_max_scaled.reshape(1, -1)).flatten()

    logger.info(f"MAP estimate: {MAP_unscaled}")
    logger.info(f"True values: {true_input}")
    logger.info(f"Inference completed in {time.time() - start_time:.2f} seconds")

    flat_chain = chain.reshape(-1, chain.shape[-1])
    plot_save_dir = ROOT_DIR / "plots"

    save_corner_plot(
        flat_chain=flat_chain,
        map_estimate=map_estimate,
        row_index=row_index,
        true_input=true_input,
        scaler_path=inverse_scaler_path,
        save_dir=plot_save_dir,
        parameter_names=parameter_names
    )

if __name__ == "__main__":
    main()