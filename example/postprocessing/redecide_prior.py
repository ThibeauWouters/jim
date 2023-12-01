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

from ripple import get_chi_eff, Mc_eta_to_ms, lambda_tildes_to_lambdas, lambdas_to_lambda_tildes, lambda_tildes_to_lambdas_from_q, lambdas_to_lambda_tildes_from_q

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

    

####################################
#### Main plotting comes here ######
####################################

### Fetch data

corner_kwargs = default_corner_kwargs

### Plotting

labels = [r"$\tilde{\Lambda}$", r"$\delta \tilde{\Lambda}$"]
save_name = f"./outdir/lambdas_redecide_prior.png"

# Prior bounds
q_low, q_high = 0.34, 1.0
lambda_low, lambda_high = 0.0, 2000.0

# Generate a synthetic dataset of lambda1 and lambda2
N = 10000
lambda1 = np.random.uniform(low=lambda_low, high=lambda_high, size=N)
lambda2 = np.random.uniform(low=lambda_low, high=lambda_high, size=N)
q = np.random.uniform(low=q_low, high=q_high, size=N)

# Get lambda_tilde and delta_lambda_tilde
lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes_from_q(jnp.array([lambda1, lambda2, q]))
# Stack together into samples
samples = np.vstack([lambda_tilde, delta_lambda_tilde]).T
fig = corner.corner(samples, labels = labels, weights=None, hist_kwargs={'density': True}, **default_corner_kwargs)
fig.savefig(save_name, bbox_inches='tight')  
plt.close()
print(f"Saved {save_name}")