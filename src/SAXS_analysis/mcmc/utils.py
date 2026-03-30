"""MCMC sampling (emcee) using a neural forward model as likelihood surrogate."""

import numpy as np
import torch
import emcee


def lnlike(theta, y, y_std, model):
    """Gaussian log-likelihood between data ``y`` and forward-model prediction ``model(theta)``.

    Args:
        theta: Parameter vector in the scaled space expected by ``model``.
        y, y_std: Observed quotient or intensity features and uncertainties.
        model: Callable ``torch.nn.Module`` mapping (1, n_param) tensor to prediction.
    """
    theta_tensor = torch.tensor(theta, dtype=torch.float32).unsqueeze(0)
    y_model = model(theta_tensor).detach().numpy().flatten()
    return -0.5 * np.sum(((y - y_model) / y_std) ** 2)


def lnprior(theta, priors, std_dev=0.15):
    """Uniform box on (0, 1) for each parameter plus Gaussian pulls toward ``priors`` (ML anchors).

    Args:
        theta: Same length as ``priors``.
        priors: Center values (e.g. inverse-ANN predictions) per dimension.
        std_dev: Width of the Gaussian prior term.
    """
    if all(0 < x < 1 for x in theta):
        return np.sum([-0.5 * ((x - priors[i]) / std_dev) ** 2 for i, x in enumerate(theta)])
    return -np.inf


def lnprob(theta, y, y_std, model, priors):
    """Log posterior up to constant: prior + likelihood."""
    lp = lnprior(theta, priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, y, y_std, model)


def run_mcmc(p0, nwalkers, niter, y, y_std, row_index, s_ml_model, priors_for_inverse_ann, burn_in):
    """Run emcee ensemble MCMC with :func:`lnprob`.

    Args:
        p0: Initial walker positions, shape ``(nwalkers, ndim)``.
        nwalkers, niter: Ensemble size and production steps (after ``burn_in`` discarded).
        y, y_std: Data vector and uncertainties.
        row_index: Identifier for logging only.
        s_ml_model: Torch forward model (surrogate for intensity vs parameters).
        priors_for_inverse_ann: Prior centers passed to :func:`lnprior`.
        burn_in: Discarded steps before production.

    Returns:
        ``(flat_chain, flat_log_prob, map_estimate)`` where ``map_estimate`` is the sample
        with highest log probability along the flattened chain.
    """
    ndim = len(p0[0])
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob, args=(y, y_std, s_ml_model, priors_for_inverse_ann)
    )

    print("[Main] Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, burn_in, progress=True)
    sampler.reset()

    print("[Main] Running production for...")
    sampler.run_mcmc(p0, niter, progress=True)

    flat_chain = sampler.get_chain(flat=True)
    flat_log_prob = sampler.get_log_prob(flat=True)

    max_idx = np.argmax(flat_log_prob)
    map_estimate = flat_chain[max_idx]

    print(f"[Main] MCMC complete for row {row_index}")

    return flat_chain, flat_log_prob, map_estimate
