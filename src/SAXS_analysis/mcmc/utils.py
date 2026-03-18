import numpy as np 
import torch
import emcee

# === Likelihood & Prior ===
def lnlike(theta, y, y_std, model):
    theta_tensor = torch.tensor(theta, dtype=torch.float32).unsqueeze(0)
    y_model = model(theta_tensor).detach().numpy().flatten()
    return -0.5 * np.sum(((y - y_model) / y_std) ** 2)

def lnprior(theta, priors, std_dev=0.15):
    if all(0 < x < 1 for x in theta):
        return np.sum([-0.5 * ((x - priors[i]) / std_dev) ** 2 for i, x in enumerate(theta)])
    else:
        return -np.inf

def lnprob(theta, y, y_std, model, priors):
    lp = lnprior(theta, priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, y, y_std, model)

def run_mcmc(p0, nwalkers, niter, y, y_std, row_index, s_ml_model, priors_for_inverse_ann, burn_in):
    ndim = len(p0[0])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(y, y_std, s_ml_model, priors_for_inverse_ann))

    print(f"[Main] Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, burn_in, progress=True)
    sampler.reset()

    print(f"[Main] Running production for...")
    sampler.run_mcmc(p0, niter, progress=True)

    flat_chain = sampler.get_chain(flat=True)
    flat_log_prob = sampler.get_log_prob(flat=True)

    max_idx = np.argmax(flat_log_prob)
    map_estimate = flat_chain[max_idx]

    print(f"[Main] MCMC complete for row {row_index}")

    return flat_chain, flat_log_prob, map_estimate
