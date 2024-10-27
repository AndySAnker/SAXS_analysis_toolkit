import h5py
import numpy as np
import sasmodels.data

def load_hdf5_data(filename, num_files, qmin, qmax):
    """
    Load data from an HDF5 file, shuffle it, and prepare it for machine learning purposes.
    """
    with h5py.File(filename, 'r') as f:
        X = f['SAXS_dataset'][()]
        y = f['formfactor'][()][:len(X)]
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    y = np.array(y).astype(str)
    y_decoded = np.unique(y, axis=1)
    
    q = np.linspace(qmin, qmax, 1000)
    q_repeated = np.repeat(q[np.newaxis, :], num_files, axis=0)
    Datafiles = np.stack((q_repeated, X[:num_files]), axis=2)
    
    return Datafiles, y_decoded[:num_files]

def load_and_process_data(data_source, qmin, qmax, error_weighting, normalization_type='None'):
    """
    Load data from a file or numpy array, normalize it, and filter it based on qmin and qmax.
    """
    if isinstance(data_source, str):
        try:
            data = np.loadtxt(data_source, delimiter=',')
        except ValueError:
            data = np.loadtxt(data_source, delimiter=' ')
    elif isinstance(data_source, np.ndarray):
        data = data_source
    else:
        raise TypeError("data_source must be a string or a numpy array")

    mask = (data[:,0] >= qmin) & (data[:,0] <= qmax) & (data[:,1] > 0)

    if data.shape[1] > 2:
        data[:,2] = normalize_intensity(data[:,2], normalization_type)
        data[:,1] = normalize_intensity(data[:,1], normalization_type)
        data = sasmodels.data.Data1D(x=data[mask,0], y=data[mask,1], dy=np.abs(data[mask,2]))
    else:
        data[:,1] = normalize_intensity(data[:,1], normalization_type)
        noise = apply_error_weighting(data[mask], error_weighting)
        data = sasmodels.data.Data1D(x=data[mask,0], y=data[mask,1], dy=noise)

    return data, data

def normalize_intensity(intensity, normalization_type='peak'):
    """
    Normalize the scattering intensity based on the specified type.
    
    Parameters:
    intensity (numpy.ndarray): The scattering intensity to normalize.
    normalization_type (str): The type of normalization to apply. 
                              Options are 'None' or 'peak'. Default is 'peak'.
    
    Returns:
    numpy.ndarray: The normalized intensity.
    """
    if normalization_type.lower() == 'none':
        return intensity
    elif normalization_type.lower() == 'peak':
        return intensity / np.max(intensity)
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")

def apply_error_weighting(data, error_weighting='sqrt'):
    """
    Apply error weighting to the data.

    Parameters:
    data (numpy.ndarray): The input data array. It should have at least 2 columns: q and intensity.
    error_weighting (str): The type of error weighting to apply. 
                           Options are 'None', 'intensity', 'sqrt', or 'absolute'. Default is 'sqrt'.

    Returns:
    numpy.ndarray: The error weights for the data.

    Raises:
    ValueError: If an invalid error_weighting option is provided.
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