### Small utility script to compare results with TurboPE

import seaborn as sns
import copy
import pickle
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt 
import corner
import pandas as pd

from ripple import get_chi_eff, Mc_eta_to_ms, lambda_tildes_to_lambdas, lambdas_to_lambda_tildes

### Utilities

default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=24),
                        title_kwargs=dict(fontsize=24), 
                        color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        save=False
)

params = {
    "axes.labelsize": 30,
    "axes.titlesize": 30,
    "text.usetex": True,
    "font.family": "serif",
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
}
plt.rcParams.update(params)

    
### Postprocessing analysis tools

def weight_function(x):
    return x**2

def reweigh_distance(chains, d_idx = 6):
    """
    Get weights based on distance to mimic cosmological distance prior.
    """
    d_samples = chains[:, d_idx]
    print(d_samples)
    weights = weight_function(d_samples)
    weights = weights / np.sum(weights)
    
    return weights

def powerlaw_transform(d_L_quantile, max_distance:float=75, alpha:float=2):
    return (1.0 ** (1 + alpha) + d_L_quantile * (max_distance ** (1 + alpha) - 1.0 ** (1 + alpha))) ** (1. / (1 + alpha))


### Plotting hyperparameters

param_names = ["Mc", "q", "chi1", "chi2", "lambda_tilde", "delta_lambda_tilde", "d_L", "t_c", "phi_c", "iota", "psi", "alpha", "delta"]
idx_list = [i for i in range(len(param_names))]
n_dim = len(param_names)

def read_samples(filename: str,
                 use_weights: bool = False,
                 use_d_L_quantile: bool = False):
    
    print(f"Reading samples from {filename}... ")
    
    if "production" in filename:
        which = "production"
    else:
        which = "NF"
    
    print(f"Loading {which} samples... ")
    copy_idx_list = copy.deepcopy(idx_list)
    
    # My samples
    if which == "production":
        data = np.load(filename)
        chains = data['chains'][:,:,copy_idx_list].reshape(-1, n_dim)
        
        # Fetch only d_L and lambda parameters:
        d_L_index = param_names.index("d_L")
        d_samples = chains[:, d_L_index]
        weights = weight_function(d_samples)
        weights = weights / np.sum(weights)
        
        if use_d_L_quantile:
            chains[:,d_L_index] = powerlaw_transform(chains[:,d_L_index])
        chains = np.asarray(chains)
    
    else:
        data = np.load(filename)
        chains = data["chains"][:, copy_idx_list]
        if use_d_L_quantile:
            chains[:,d_L_index] = powerlaw_transform(chains[:,d_L_index])
        chains = np.asarray(chains)

    # Reweight based on distance
    print("Loading complete")
    if use_weights:
        weights = reweigh_distance(chains)
    else:
        weights = None
        
    # Get the masses, convert them to component masses
    Mc_idx = param_names.index("Mc")
    q_idx = param_names.index("q")
    # From q to eta:
    mc = chains[:, Mc_idx]
    q = chains[:, q_idx]
    eta = q / ((1 + q) ** 2)
    # From Mc and eta to m1 and m2:
    m1, m2 = Mc_eta_to_ms(jnp.array([mc, eta]))
    # From m1, m2 and lambda_tilde to lambda1 and lambda2:
    lambda_tilde_idx = param_names.index("lambda_tilde")
    delta_lambda_tilde_idx = param_names.index("delta_lambda_tilde")
    lambdaT, delta_lambdaT = chains[:, lambda_tilde_idx], chains[:, delta_lambda_tilde_idx]
    lambda1, lambda2 = lambda_tildes_to_lambdas(jnp.array([lambdaT, delta_lambdaT, m1, m2]))
    
    samples = np.vstack([lambda1, lambda2]).T
        
    return samples, weights, (m1, m2)


####################################
#### Main plotting comes here ######
####################################

### Fetch data

which_list = ["production", "NF"]
outdir_list = ["../GW190425_TaylorF2/outdir/"]
colors = ["blue", "red"]
corner_kwargs = default_corner_kwargs

print(f"Reading data from {outdir_list}")

### Plotting

use_weights = True
labels = [r"$\Lambda_1$", r"$\Lambda_2$"]

for which in which_list:
    
    if use_weights:
        save_name = f"./outdir/lambdas_{which}_reweighted.png"
    else:
        save_name = f"./outdir/lambdas_{which}.png"
    
    # Iterate over directories in outdir_list, read the samples, and plot in single corner plot
    for i, (dir, col) in enumerate(zip(outdir_list, colors)):
        # Set color and read in samples
        corner_kwargs["color"] = col
        filename = f"{dir}results_{which}.npz"
        samples, weights, (m1, m2) = read_samples(filename, use_weights=use_weights)
        
        # Limit the samples and masses to only include the ones with lambda1 and lambda2 below 2000
        print(jnp.shape(samples))
        idx = np.where((samples[:,0] < 2000) & (samples[:,1] < 2000))[0]
        print(len(idx))
        samples = samples[idx]
        m1 = m1[idx]
        m2 = m2[idx]
        if use_weights:
            weights = weights[idx]
        
        if i == 0:
            fig = corner.corner(samples, labels = labels, weights=weights, hist_kwargs={'density': True}, **default_corner_kwargs)
        else:
            corner.corner(samples, fig=fig, labels = labels, weights=weights, hist_kwargs={'density': True}, **default_corner_kwargs)
        fig.savefig(save_name, bbox_inches='tight')  
        
        # Now plot the lambda and lambda tilde for the restricted samples
        lambda1 = samples[:,0]
        lambda2 = samples[:,1]
        
        lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(jnp.array([lambda1, lambda2, m1, m2]))
        lambda_tilde = np.asarray(lambda_tilde)
        delta_lambda_tilde = np.asarray(delta_lambda_tilde)
        
        # Stack together
        samples = np.vstack([lambda_tilde, delta_lambda_tilde]).T
        
        # Plot lambda_tilde and delta_lambda_tilde
        save_name = f"./outdir/lambda_tildes_{which}.png"
        
        labels = [r"$\tilde{\Lambda}$", r"$\delta\tilde{\Lambda}$"]
        
        if i == 0:
            fig = corner.corner(samples, labels = labels, weights=weights, hist_kwargs={'density': True}, **default_corner_kwargs)
        else:
            corner.corner(samples, fig=fig, labels = labels, weights=weights, hist_kwargs={'density': True}, **default_corner_kwargs)
        
        fig.savefig(save_name, bbox_inches='tight')  
    
    print("Done")