"""HDF5 dataset creation and row appends for simulated SAS training data."""

import h5py
import numpy as np


def create_datasets(f, Num_datasets, dtype):
    """Create empty extensible datasets for intensity, uncertainty, and labels.

    Args:
        f: Open HDF5 file handle (write mode).
        Num_datasets: Unused legacy name (kept for call-site compatibility).
        dtype: NumPy dtype for intensity arrays (e.g. ``float16``).

    Returns:
        Tuple of dataset handles:
        ``SAS_data``, ``SAS_data_uncertainty``, ``formfactor``, ``structurefactor``,
        ``powerlaw``, ``parameters_formfactor``, ``parameters_structurefactor``,
        ``parameters_powerlaw``.
    """
    dset = f.create_dataset("SAS_data", (0, 100), maxshape=(None, 100), dtype=dtype)
    dset_uncertainty = f.create_dataset(
        "SAS_data_uncertainty", (0, 100), maxshape=(None, 100), dtype=dtype
    )
    dset_formfactor = f.create_dataset(
        "formfactor", (0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str)
    )
    dset_structurefactor = f.create_dataset(
        "structurefactor", (0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str)
    )
    dset_powerlaw = f.create_dataset(
        "powerlaw", (0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str)
    )
    dset_parameters_formfactor = f.create_dataset(
        "parameters_formfactor", (0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str)
    )
    dset_parameters_structurefactor = f.create_dataset(
        "parameters_structurefactor", (0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str)
    )
    dset_parameters_powerlaw = f.create_dataset(
        "parameters_powerlaw", (0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str)
    )
    return (
        dset,
        dset_uncertainty,
        dset_formfactor,
        dset_structurefactor,
        dset_powerlaw,
        dset_parameters_formfactor,
        dset_parameters_structurefactor,
        dset_parameters_powerlaw,
    )


def save_data(
    datasets,
    index,
    Iq,
    dIq,
    formfactor_model,
    parameters_formfactor,
    structurefactor_model,
    parameters_structurefactor,
    powerlaw_model,
    parameters_powerlaw,
    dtype,
):
    """Append one simulated row to all relevant HDF5 datasets (resize + assign).

    Args:
        datasets: Output of :func:`create_datasets`.
        index: Row index (0-based).
        Iq, dIq: 1D intensity and uncertainty arrays.
        formfactor_model: Form factor name string.
        parameters_formfactor: Object stringified for storage.
        structurefactor_model, parameters_structurefactor: Optional structure factor; ``"None"`` if absent.
        powerlaw_model, parameters_powerlaw: Optional power-law; ``"None"`` if absent.
        dtype: Storage dtype for intensities.
    """
    (
        dset,
        dset_uncertainty,
        dset_formfactor,
        dset_structurefactor,
        dset_powerlaw,
        dset_parameters_formfactor,
        dset_parameters_structurefactor,
        dset_parameters_powerlaw,
    ) = datasets

    new_data = np.array([Iq]).astype(dtype)
    dset.resize(index + 1, axis=0)
    dset[index, :] = new_data

    dnew_data_uncertainty = np.array([dIq]).astype(dtype)
    dset_uncertainty.resize(index + 1, axis=0)
    dset_uncertainty[index, :] = dnew_data_uncertainty

    dset_formfactor.resize(index + 1, axis=0)
    dset_formfactor[index] = formfactor_model

    dset_structurefactor.resize(index + 1, axis=0)
    dset_structurefactor[index] = structurefactor_model if structurefactor_model else "None"

    dset_powerlaw.resize(index + 1, axis=0)
    dset_powerlaw[index] = powerlaw_model if powerlaw_model else "None"

    dset_parameters_formfactor.resize(index + 1, axis=0)
    dset_parameters_formfactor[index] = str(parameters_formfactor)

    dset_parameters_structurefactor.resize(index + 1, axis=0)
    dset_parameters_structurefactor[index] = (
        str(parameters_structurefactor) if parameters_structurefactor else "None"
    )

    dset_parameters_powerlaw.resize(index + 1, axis=0)
    dset_parameters_powerlaw[index] = str(parameters_powerlaw) if parameters_powerlaw else "None"
