import numpy as np
import argparse
import time
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import torch
import SAXS_analysis
from pathlib import Path
from SAXS_analysis.utils.configs import load_config
from SAXS_analysis.utils.logging import setup_logging
from SAXS_analysis.data_processing.utils import load_and_process_SAS_data, adaptive_downsample, interpolate_to_n_points, quotient_transform
from SAXS_analysis.classification.utils import load_classification_model
from SAXS_analysis.classification.XGBoost_classification import classify_formfactor
from SAXS_analysis.forward_ann.forward_ann import ForwardNN
from SAXS_analysis.inverse_ann.inverse_ann import InverseNN
from SAXS_analysis.visualization.utils import plot_experimental_data, plot_qt_data, save_corner_plot
from SAXS_analysis.mcmc.utils import run_mcmc
ROOT_DIR = Path(SAXS_analysis.ROOT_DIR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger = setup_logging(config)

    data_name = ROOT_DIR / config['data']['data_file']
    data_stem = Path(data_name).stem 
    
    plot_dir = ROOT_DIR / 'plots' / data_stem
    plot_dir.mkdir(exist_ok=True, parents=True) 

    logger.info(f"Processing data file: {data_name}...")

    # Load experimental data
    q_raw, Iq_raw, std_raw = load_and_process_SAS_data(data_name)
    logger.info(f"Loaded raw data: {len(q_raw)} points, data q-range: {q_raw.min():.5f} - {q_raw.max():.5f} nm^-1")

    # Data preprocessing (adaptive downsampling and interpolation)
    dq = 0.01499
    q_split = config['preprocessing']['q_split']
    n_low = config['preprocessing']['n_low']
    n_high = config['preprocessing']['n_high']

    # Adaptive downsampling for n_low < q_split and n_high > q_split
    logger.info(f"Adaptive downsampling parameters -> q_split: {q_split} nm^-1,  n_low: {n_low}, n_high: {n_high}")
    q_ds, Iq_ds, std_ds = adaptive_downsample(q_raw, Iq_raw, std_raw, q_split=q_split, n_low=n_low, n_high=n_high)
    logger.info(f"Downsampled data: {len(q_ds)} points (from {len(q_raw)} raw points)")
    
    # Interpolation
    num_points = int(np.ceil((q_raw.max() - q_raw.min()) / dq))
    logger.info(f"Interpolation step -> dq: {dq}, target points: {num_points}")
    q_interp, Iq_interp, std_interp = interpolate_to_n_points(q_ds, Iq_ds, std_ds, num_points=num_points)
    logger.info(f"Interpolated data: {len(q_interp)} points corresponding to data q-range: {q_raw.min():.5f} - {q_raw.max():.5f} nm^-1")
        
    # Plot experimental data for data visualisation    
    plot_experimental_data(q_raw, Iq_raw, std_raw,
                           q_interp, Iq_interp, std_interp,
                           save_path=plot_dir / f"{data_stem}_SAXS_Data.png",
                           data_stem=data_name.stem)
    
    logger.info(f"Experimental & Interpolated Plot saved to: {plot_dir}")

    # Quotient transform I(q) and std dev
    qt, qt_std = quotient_transform(Iq_interp, std_interp)
    q_mid = 0.5 * (q_interp[1:] + q_interp[:-1])
    logger.info(f"Quotient transform completed: {len(qt)} points")

    # Padding qt data and std dev for ML 
    q_min_target = 0.01
    total_points_target = 999
    if q_mid[0] > q_min_target:
        n_front = int(np.ceil((q_mid[0] - q_min_target) / dq))
    else:
        n_front = 0
    n_back = max(total_points_target - n_front - len(q_mid), 0)

    # Pad qt and qt_std for ML
    qt_padded = np.concatenate([np.zeros(n_front), qt, np.zeros(n_back)])
    qt_std_padded = np.concatenate([0.05 * np.ones(n_front), qt_std, 0.05 * np.ones(n_back)])
    q_mid_padded = np.concatenate([
        np.linspace(q_min_target - n_front*dq, q_mid[0]-dq, n_front) if n_front > 0 else np.array([]),
        q_mid,
        np.linspace(q_mid[-1]+dq, q_mid[-1]+n_back*dq, n_back) if n_back > 0 else np.array([])
    ])

    logger.info(f"Padded front points: {n_front}, Padded back points: {n_back}")
    logger.info(f"Total points after padding: {len(qt_padded)}")

    # Plot qt data for data visualisation
    plot_qt_data(
        q_mid_padded, qt_padded, qt_std_padded,
        save_path=plot_dir / f"{data_stem}_quotient_transformed.png",
        data_stem=data_name.stem
    )

    logger.info(f"QT Plot saved: {plot_dir}")

    # Use classification model on data
    clf_model_path = ROOT_DIR / config['models']['classification']['model_path']
    clf_class_names_path = ROOT_DIR / config['models']['classification']['class_names_path']

    formfactor_classification_model = load_classification_model(clf_model_path)
    class_names = np.load(clf_class_names_path, allow_pickle=True)

    dmatrix = xgb.DMatrix(qt_padded.reshape(1, -1))
    formfactor_probabilities = formfactor_classification_model.predict(dmatrix)
    formfactor_prediction = np.argmax(formfactor_probabilities, axis=1)
    predicted_formfactor = class_names[formfactor_prediction[0]]

    logger.info("Class probabilities:")
    for name, prob in zip(class_names, formfactor_probabilities[0]):
        logger.info(f"  {name}: {prob:.4f} ({prob*100:.1f}%)")
    
    logger.info(f"Predicted form factor class: {predicted_formfactor}")

    # Load forward, inverse models and scalers
    parameter_names = config['parameters'][predicted_formfactor]
    forward_model_path = ROOT_DIR / config['models']['forward'][predicted_formfactor]
    inverse_model_path = ROOT_DIR / config['models']['inverse'][predicted_formfactor]
    forward_scaler_path = ROOT_DIR / config['scalers']['forward'][predicted_formfactor]
    inverse_scaler_path = ROOT_DIR / config['scalers']['inverse'][predicted_formfactor]

    forward_scaler = joblib.load(forward_scaler_path)
    inverse_scaler = joblib.load(inverse_scaler_path)

    # Use forward model to obtain priors 
    forward_model = ForwardNN(input_size=len(qt_padded), output_size=len(parameter_names))
    forward_model.load_state_dict(torch.load(forward_model_path, map_location='cpu'))
    forward_model.eval()

    input_tensor = torch.tensor(qt_padded, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        priors_scaled_tensor = forward_model(input_tensor)

    priors_scaled = priors_scaled_tensor.numpy().flatten()
    logger.info(f"Forward model output (scaled): {priors_scaled}")

    priors_unscaled = forward_scaler.inverse_transform(priors_scaled.reshape(1, -1)).flatten()
    logger.info(f"Forward model output (unscaled): {priors_unscaled}")

    # Load inverse model as surrogate for MCMC sampling 
    s_ml_model = InverseNN(input_size=len(parameter_names), output_size=len(qt_padded))
    s_ml_model.load_state_dict(torch.load(inverse_model_path, map_location='cpu'))
    s_ml_model.eval()

    # Load MCMC parameters from config 
    num_params = len(parameter_names)
    nwalkers = config['mcmc']['nwalkers']
    burn_in = config['mcmc']['burn_in']
    niter = config['mcmc']['niter']

    p0 = [np.random.rand(num_params) for _ in range(nwalkers)]

    # MCMC sampling
    start_time = time.time()
    chain, log_probs, map_estimate = run_mcmc(
        p0, nwalkers, niter, qt_padded, qt_std_padded, 0, s_ml_model, priors_scaled, burn_in
    )

    theta_max_scaled = chain[np.argmax(log_probs.flatten())]
    MAP_unscaled = inverse_scaler.inverse_transform(theta_max_scaled.reshape(1, -1)).flatten()
    logger.info(f"MAP estimate: {MAP_unscaled}")

    # Save corner plot
    flat_chain = chain.reshape(-1, chain.shape[-1])
    save_corner_plot(
        flat_chain=flat_chain,
        map_estimate=map_estimate,
        row_index=data_stem,
        true_input=None,
        scaler_path=inverse_scaler_path,
        save_dir=plot_dir,
        parameter_names=parameter_names
    )

    logger.info(f"Inference completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

