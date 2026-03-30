"""Load, filter, and transform SAS curves for ML and fitting."""

import h5py
import numpy as np
import sasmodels.data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ast
from SAXS_analysis.utils.formfactors import formfactor_params
import SAXS_analysis
import joblib
from scipy.interpolate import interp1d
ROOT_DIR = SAXS_analysis.ROOT_DIR

def load_hdf5_data(filename, num_files=None, qmin=None, qmax=None):
    """Load simulated SAXS HDF5 data and stack q with intensities for training.

    Reads ``SAXS_dataset``, ``formfactor``, and optional ``q``. Rows are shuffled.
    Intensities may be quotient-normalized (length ``len(q)-1``); q is trimmed to match.

    Args:
        filename: Path relative to ``ROOT_DIR`` to an HDF5 file.
        num_files: Max number of curves to return (default: all).
        qmin, qmax: Fallback q range if the file has no ``q`` dataset.

    Returns:
        ``(Datafiles, y_labels)`` where ``Datafiles`` has shape
        ``(n, 2, n_q)`` — channel 0 is q, channel 1 is intensity — and ``y_labels``
        are decoded form factor names per row.
    """
    with h5py.File(ROOT_DIR / filename, 'r') as f:
        X = f['SAXS_dataset'][()]
        y = f['formfactor'][()][:len(X)]
        q_file = f['q'][()] if 'q' in f else None
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    y = np.array(y).astype(str)
    y_decoded = np.unique(y, axis=1)
    
    if num_files is None:
        num_files = X.shape[0]

    # Build a q-grid matching the stored intensity length.
    # For quotient-normalization, intensities have length len(q)-1.
    if q_file is not None:
        q_arr = np.asarray(q_file).ravel()
        if q_arr.size == X.shape[1] + 1:
            q_arr = q_arr[1:]
        if q_arr.size != X.shape[1]:
            q_arr = np.linspace(q_arr.min(), q_arr.max(), X.shape[1])
    else:
        if qmin is None:
            qmin = 0.001
        if qmax is None:
            qmax = 1.5
        q_arr = np.linspace(qmin, qmax, X.shape[1])

    q_repeated = np.repeat(q_arr[np.newaxis, :], num_files, axis=0)

    #print("q_repeated shape:", q_repeated.shape)
    #print("X[:num_files] shape:", X[:num_files].shape)

    Datafiles = np.stack((q_repeated, X[:num_files]), axis=1)
    
    return Datafiles, y_decoded[:num_files]

# --- Functions ---
def load_and_process_SAS_data(data_source, qmin=None, qmax=None):
    """Load columns ``q``, ``I(q)``, optional ``sigma`` from a whitespace-separated text file.

    Args:
        data_source: Path relative to ``ROOT_DIR``.
        qmin, qmax: Clip range; defaults to data extent.

    Returns:
        ``(q, I, sigma)`` with positive-intensity points only.
    """
    data = np.loadtxt(ROOT_DIR / data_source, dtype=float)
    q_raw, Iq_raw = data[:,0], data[:,1]
    std_raw = np.abs(data[:,2]) if data.shape[1] > 2 else np.zeros_like(Iq_raw)

    # Set qmin/qmax defaults
    if qmin is None: qmin = q_raw.min()
    if qmax is None: qmax = q_raw.max()

    # Filter
    mask = (q_raw >= qmin) & (q_raw <= qmax) & (Iq_raw > 0)
    return q_raw[mask], Iq_raw[mask], std_raw[mask]

def adaptive_downsample(q, Iq, std, q_split=1, n_low=1, n_high=20):
    """Reduce points below ``q_split`` every ``n_low`` steps and above every ``n_high`` steps.

    Keeps more detail at low q by default when ``n_low < n_high``. Result is sorted by q.
    """
    low_mask = q < q_split
    high_mask = q >= q_split
    q_ds = np.concatenate([q[low_mask][::n_low], q[high_mask][::n_high]])
    Iq_ds = np.concatenate([Iq[low_mask][::n_low], Iq[high_mask][::n_high]])
    std_ds = np.concatenate([std[low_mask][::n_low], std[high_mask][::n_high]])
    # Sort by q
    sort_idx = np.argsort(q_ds)
    return q_ds[sort_idx], Iq_ds[sort_idx], std_ds[sort_idx]

def interpolate_to_n_points(q, Iq, std, num_points):
    """Linearly interpolate ``I`` and ``sigma`` onto ``num_points`` evenly spaced in q."""
    q_interp = np.linspace(q.min(), q.max(), num_points)
    Iq_interp = np.interp(q_interp, q, Iq)
    std_interp = np.interp(q_interp, q, std)
    return q_interp, Iq_interp, std_interp

def quotient_transform(Iq, std):
    """Compute ``2 * log(I(q_i)/I(q_{i-1}))`` and first-order error propagation on ``std``."""
    ratio = Iq[1:] / Iq[:-1]
    qt = 2 * np.log(ratio)
    qt_std = 2 * np.sqrt((std[1:]/Iq[1:])**2 + (std[:-1]/Iq[:-1])**2)
    return qt, qt_std

def normalize_intensity(intensity, normalization_type='peak'):
    """
    Normalize the scattering intensity based on the specified type.

    Args:
        intensity: Scattering intensity array.
        normalization_type: ``'none'``, ``'peak'``, or ``'quotient'``.

    Returns:
        Normalized intensity (or log-ratio for ``quotient``).
    """
    if normalization_type.lower() == 'none':
        return intensity
    elif normalization_type.lower() == 'peak':
        return intensity / np.max(intensity)
    elif normalization_type.lower() == 'quotient':
        return 2 * np.log(intensity[..., 1:] / intensity[..., :-1])
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")

def apply_error_weighting(data, error_weighting='sqrt'):
    """
    Apply error weighting to the data.

    Args:
        data: Array with at least two columns (q and intensity).
        error_weighting: ``'None'``, ``'intensity'``, ``'sqrt'``, or ``'absolute'``.

    Returns:
        Per-point error weights.

    Raises:
        ValueError: If ``error_weighting`` is not recognized.
    """
    if error_weighting == 'None':
        return np.ones(len(data)) * 1e-9  # Impossible to predict noise when not reported in data file
    elif error_weighting == 'intensity':
        return data[:, 1]
    elif error_weighting == 'sqrt':
        return np.abs(np.sqrt(data[:, 1]))
    elif error_weighting == 'absolute':
        return np.abs(data[:, 1])
    else:
        raise ValueError(f"Invalid value for error_weighting: {error_weighting}")


def load_and_preprocess_data(file_name: str, num_data_points: int = int(9e15), 
                           qmin: float = 0.001, qmax: float = 1.5,
                           target_formfactor: str = None) -> tuple:
    """
    Data loading and masking of qmin, qmax, num_data_points, and target_formfactor.
    
    Args:
        file_name: Path to HDF5 file
        num_data_points: Maximum number of data points to process
        qmin: Minimum q value
        qmax: Maximum q value
        target_formfactor: If specified, filter data for this formfactor only
    
    Returns:
        tuple: (X, y) or (X, y, formfactors) depending on target_formfactor parameter
    """
    # First load the data
    with h5py.File(ROOT_DIR / file_name, 'r') as f:
        X = f['SAXS_dataset'][()]
        if target_formfactor:
            y = f['parameters_formfactor'][()][:len(X)]
            all_formfactors = np.array([x[0].decode() for x in f['formfactor'][()][:len(X)]])
        else:
            Datafiles, y = load_hdf5_data(ROOT_DIR / file_name, num_data_points, qmin, qmax)
            X = Datafiles[:, :, :]  # Extract intensity data

            # Only use the intensity data
            X = X[:,1]  
            return X, y
    

    # Remove infinities
    mask_finite = np.all(np.isfinite(X), axis=1)
    X = X[mask_finite]
    y = y[mask_finite] if 'y' in locals() else None
    all_formfactors = all_formfactors[mask_finite] if 'all_formfactors' in locals() else None

    if target_formfactor:
        # Filter for specific formfactor
        mask_formfactor = all_formfactors == target_formfactor
        X = X[mask_formfactor]
        y = y[mask_formfactor] if y is not None else None
        
        # Process parameters for regression
        y_dicts = [ast.literal_eval(item[0].decode()) for item in y]
        y_new = np.zeros((len(y_dicts), len(formfactor_params[target_formfactor])))
        
        for param in formfactor_params[target_formfactor]:
            param_idx = formfactor_params[target_formfactor].index(param)
            y_new[:, param_idx] = [d[param] for d in y_dicts]
        y = y_new

    # Limit data points
    X = X[:int(num_data_points)]
    y = y[:int(num_data_points)] if y is not None else None

    return X, y

def split_data(X: np.ndarray, y: np.ndarray, normalize: bool = False) -> tuple:
    """Split ``X, y`` into train / validation / test (60/20/20) with fixed random seed.

    Args:
        X: Feature matrix (e.g. intensities or q–I stacks).
        y: Targets (class indices or parameter vectors).
        normalize: If True, apply ``StandardScaler`` to X (fit on train only) and save
            ``scalers/classification_standard_scaler.joblib``.

    Returns:
        ``X_train, X_val, X_test, y_train, y_val, y_test``.
    """
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Save the scaler
        scaler_dir = ROOT_DIR / "scalers"
        joblib.dump(scaler, scaler_dir / "classification_standard_scaler.joblib")

    return X_train, X_val, X_test, y_train, y_val, y_test

