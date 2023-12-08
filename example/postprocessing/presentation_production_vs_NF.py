### Small utility script to compare results with TurboPE

import seaborn as sns
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
                        color="red",
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

use_masses = False

if use_masses:
    labels = [r'$M_c/M_\odot$', r'$q$', r'$\tilde{\Lambda}$', r'$\delta\tilde{\Lambda}$']
    idx_list = [0,1,4,5]
    bbox_to_anchor = (2.25, 3)
else:
    labels = [r'$\tilde{\Lambda}$', r'$\delta\tilde{\Lambda}$']
    idx_list = [4,5]
    bbox_to_anchor = (2.25, 2)


### Postprocessing analysis tools

def weight_function(x):
    return x**2

# TODO improve getting the index right
def reweigh_distance(chains, d_idx = 6):
    """
    Get weights based on distance to mimic cosmological distance prior.
    """
    d_samples = chains[:, d_idx]
    print(d_samples)
    weights = weight_function(d_samples)
    weights = weights / np.sum(weights)
    
    return weights

### Fetch data

which_list = ["production", "NF"]
outdir = "../GW170817_TaylorF2/outdir/"
corner_kwargs = default_corner_kwargs

print(f"Reading data from {outdir}")

### Plotting hyperparameters

use_weights = True
use_d_L_quantile = False
use_chi_eff = True

print(f"Creating plots with use_weights={use_weights}, use_d_L_quantile={use_d_L_quantile}, use_chi_eff={use_chi_eff}")

### Production chains
filename = f"{outdir}results_production.npz"
data = np.load(filename)
chains = data['chains'].reshape(-1,13)
production_chains = np.asarray(chains)
# Reweight based on distance
production_weights = reweigh_distance(production_chains)
# Only use Mc, q, lambda_tilde and delta_lambda_tilde
production_chains = production_chains[:, idx_list]

### NF chains
filename = f"{outdir}results_NF.npz"
data = np.load(filename)
chains = data["chains"]
NF_chains = np.asarray(chains)
NF_weights = reweigh_distance(NF_chains)
# Only use Mc, q, lambda_tilde and delta_lambda_tilde
NF_chains = NF_chains[:, idx_list]

print("Loading complete")

### Plotting

# name = "./outdir/presentation_GW170817_production_vs_NF.png"
name = "./outdir/presentation_GW170817_production_vs_NF"
    
print(f"Creating plot")

# New corner kwargs:
alpha = 0.5
lw = 2
hist_kwargs = {'density': True,
               'alpha': alpha,
               'linewidth': lw,
               'color': 'red'}

# update_corner_kwargs = {'alpha': alpha,}
# corner_kwargs.update(update_corner_kwargs)

### Finally, make the plot
fig = corner.corner(production_chains, labels = labels, weights=production_weights, hist_kwargs=hist_kwargs, **corner_kwargs)
corner_kwargs["color"] = "blue"
hist_kwargs["color"] = "blue"
corner.corner(NF_chains, labels = labels, weights=NF_weights, fig=fig, hist_kwargs=hist_kwargs, **corner_kwargs)
# Add legend here
chain_names = ["Production", "NF"]
colors = ["red", "blue"]
# top = 0.005
# left = 0.05
# vstep = 0.0005
# hstep = 0.05
# fontsize = 24

# for i, (chains_name, col) in enumerate(zip(chain_names, colors)):
#     # Put names vertically below each other
#     plt.text(left, top - i * vstep, chains_name, color=col, fontsize=fontsize)

from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=40, label='Production'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=40, label='NF')
]

legend_fontsize = 36
plt.legend(handles=legend_elements, loc='upper right', fontsize = legend_fontsize, bbox_to_anchor = bbox_to_anchor)
    
for ext in ["png", "pdf"]:
    if use_masses:
        plot_name = name + "_masses." + ext
    else:
        plot_name = name + "." + ext
    print(f"Creating plot {plot_name}")
    fig.savefig(plot_name, bbox_inches='tight')
print("Done")