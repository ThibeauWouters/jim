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

from ripple import get_chi_eff, Mc_eta_to_ms

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

labels_no_chi_eff = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda$', r'$\delta\Lambda$' ,r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$t_c$', r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

labels_chi_eff = [r'$M_c/M_\odot$', r'$q$', r'$\chi_{\rm eff}$', r'$\Lambda$', r'$\delta\Lambda$' ,r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$t_c$', r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

use_weights = True
use_d_L_quantile = False
use_chi_eff = True
remove_tc = False
    
# Put the above booleans in a dictionary kwargs:
kwargs = dict(use_weights=use_weights, 
              use_d_L_quantile=use_d_L_quantile, 
              use_chi_eff=use_chi_eff, 
              remove_tc=remove_tc)
    
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
                 use_weights: bool = True,
                 use_d_L_quantile: bool = False,
                 use_chi_eff: bool = True,
                 remove_tc: bool = True):
    
    print(f"Reading samples from {filename}... ")
    
    print(f"Plot hyperparameters:")
    print(f"use_weights: {use_weights}")
    print(f"use_d_L_quantile: {use_d_L_quantile}")
    print(f"use_chi_eff: {use_chi_eff}")
    print(f"remove_tc: {remove_tc}")
    
    if "production" in filename:
        which = "production"
    else:
        which = "NF"
    
    print(f"Loading {which} samples... ")
    
    copy_idx_list = copy.deepcopy(idx_list)
    if remove_tc:
        tc_index = param_names.index("t_c")
        # Delete tc_index from copy_idx_list
        copy_idx_list.pop(tc_index)
    
    new_ndim = len(copy_idx_list)    
    iota_idx = param_names.index("iota")
    dec_idx = param_names.index("delta")
    d_L_idx = param_names.index("d_L")
    
    # My samples
    if which == "production":
        data = np.load(filename)
        chains = data['chains'][:,:,copy_idx_list].reshape(-1, new_ndim)
        
        chains[:,iota_idx] = np.arccos(chains[:,iota_idx])
        chains[:,dec_idx] = np.arcsin(chains[:,dec_idx])
        if use_d_L_quantile:
            chains[:,d_L_idx] = powerlaw_transform(chains[:,d_L_idx])
        chains = np.asarray(chains)
    
    else:
        data = np.load(filename)
        chains = data["chains"][:, copy_idx_list]
        if use_d_L_quantile:
            chains[:,d_L_idx] = powerlaw_transform(chains[:,d_L_idx])
        chains = np.asarray(chains)

    # Reweight based on distance
    print("Loading complete")
    if use_weights:
        weights = reweigh_distance(chains)
    else:
        weights = None
    
    # Conversion to chi_eff
    if use_chi_eff:
            
        mc, q, chi1, chi2 = chains[:,0], chains[:,1], chains[:,2], chains[:,3]
        eta = q/(1+q)**2
        
        m1, m2 = Mc_eta_to_ms(jnp.array([mc, eta]))
        
        params = jnp.array([m1, m2, chi1, chi2])
        chi_eff = get_chi_eff(params)
        
        # Now, we remove the second and third column from chains, and add chi_eff as new column at index 2
        chains = np.delete(chains, [2,3], axis=1)
        chains = np.insert(chains, 2, chi_eff, axis=1)
        
    return chains, weights


####################################
#### Main plotting comes here ######
####################################

### Fetch data

which_list = ["production", "NF"]
outdir_list = ["../GW170817_TaylorF2_FIM/outdir/", "../GW170817_TaylorF2/outdir/"]
colors = ["blue", "red"]
corner_kwargs = default_corner_kwargs

print(f"Reading data from {outdir_list}")

### Plotting

for which in which_list:
    
    if use_weights:
        save_name = f"./outdir/compare_corners_{which}_reweighted.png"
    else:
        save_name = f"./outdir/compare_corners_{which}.png"
    
    # Iterate over directories in outdir_list, read the samples, and plot in single corner plot
    for i, (dir, col) in enumerate(zip(outdir_list, colors)):
        # Set color and read in samples
        corner_kwargs["color"] = col
        filename = f"{dir}results_{which}.npz"
        chains, weights = read_samples(filename, **kwargs)
        if use_chi_eff:
            labels = labels_chi_eff
        else:
            labels = labels_no_chi_eff
            
        if remove_tc:
            t_c_index = labels.index(r"$t_c$")
            labels.pop(t_c_index)
        
        if i == 0:
            fig = corner.corner(chains, labels = labels, weights=weights, hist_kwargs={'density': True}, **default_corner_kwargs)
        else:
            corner.corner(chains, fig=fig, labels = labels, weights=weights, hist_kwargs={'density': True}, **default_corner_kwargs)
            
    fig.savefig(save_name, bbox_inches='tight')  
    print("Done")