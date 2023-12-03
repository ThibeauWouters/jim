"""
Here we compare our posterior samples with those obtained from GWOSC.

More information:
- GWOSC page for GW190425 results: https://gwosc.org/eventapi/html/GWTC-2.1-confident/GW190425/v3/
- Posterior samples can be found here: https://dcc.ligo.org/public/0165/P2000026/002/posterior_samples.h5
"""

import numpy as np
import matplotlib.pyplot as plt 
import corner
import h5py
import jax.numpy as jnp

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

labels = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda$', r'$\delta\Lambda$' ,r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

labels_chi_eff = [r'$M_c/M_\odot$', r'$q$', r'$\chi_{\rm eff}$', r'$\tilde{\Lambda}$', r'$\delta\tilde{\Lambda}$' ,r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

gwosc_names = ['chirp_mass', 'mass_ratio', 'chi_eff', 'lambda_tilde', 'delta_lambda', 'luminosity_distance', 'phase', 'iota', 'psi', 'ra', 'dec']

use_lambdas = False
if use_lambdas:
    lambda_tilde_idx = gwosc_names.index('lambda_tilde')
    delta_lambda_idx = gwosc_names.index('delta_lambda')
    gwosc_names[lambda_tilde_idx] = 'lambda_1'
    gwosc_names[delta_lambda_idx] = 'lambda_2'

def get_chains_GWOSC(filename: str = "posterior_samples.h5", key: str = "TaylorF2"):
    """
    Retrieve posterior samples of an event
    """
    
    # key should be: PhenomPNRT or TaylorF2
    key += "-LS"
    
    # Load the posterior samples from the HDF5 file
    with h5py.File(filename, 'r') as file:
        print('Top-level data structures:',file.keys())

        # Fetch indices of the names of parameters that we are interested in
        posterior = file[key]['posterior_samples']#[()]
        pnames = posterior.dtype.names
        gwosc_indices = [pnames.index(name) for name in gwosc_names]
        
        # print("parameter names:") 
        # for name in pnames:
        #     print(name)

        # Fetch the posterior samples for the parameters that we are interested in
        samples = []
        for ind in gwosc_indices:
            samples.append([samp[ind] for samp in posterior[()]])

        samples = np.asarray(samples).T
        
        print("samples shape:", np.shape(samples))
        
        print("example sample")
        print(samples[0])

    return samples

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

which_list = ["production"]
outdir = "../GW190425_TaylorF2_v4/outdir/"
corner_kwargs = default_corner_kwargs

print(f"Reading data from {outdir}")
# we remove tc:
idx_list = [0,1,2,3,4,5,6,8,9,10,11,12]

### Plotting hyperparameters

use_weights = True
use_d_L_quantile = False
use_chi_eff = True

print(f"Creating plots with use_weights={use_weights}, use_d_L_quantile={use_d_L_quantile}, use_chi_eff={use_chi_eff}")

gwosc_samples = get_chains_GWOSC()

### Plotting
for which in which_list:
    corner_kwargs["color"] = "blue"
    filename = f"{outdir}results_{which}.npz"

    print(f"Loading my {which} samples... ")
    # My samples
    if which == "production":
        data = np.load(filename)
        chains = data['chains'][:,:,idx_list].reshape(-1,12)
        chains[:,8] = np.arccos(chains[:,8])
        chains[:,11] = np.arcsin(chains[:,11])
        chains = np.asarray(chains)
    else:
        data = np.load(filename)
        chains = data["chains"][:, idx_list]
        chains = np.asarray(chains)
    
    # Reweight based on distance
    weights = reweigh_distance(chains)
    
    print("Loading complete")

    ### Plot
    
    if use_weights:
        name = f"postprocessing_{which}_reweighted.png"
    
    if not use_weights:
        name = f"postprocessing_{which}.png"
        weights = None
        
    print(f"Saving plot of chains to {name}")
    
    # Do the conversion to chi eff
    
    if use_chi_eff:
            
        mc, q, chi1, chi2 = chains[:,0], chains[:,1], chains[:,2], chains[:,3]
        eta = q/(1+q)**2
        
        m1, m2 = Mc_eta_to_ms(jnp.array([mc, eta]))
        
        params = jnp.array([m1, m2, chi1, chi2])
        chi_eff = get_chi_eff(params)
        
        # Now, we remove the second and third column from chains, and add chi_eff as new column at index 2
        chains = np.delete(chains, [2,3], axis=1)
        chains = np.insert(chains, 2, chi_eff, axis=1)
        labels = labels_chi_eff
    
    fig = corner.corner(chains, labels = labels, weights=weights, hist_kwargs={'density': True}, **default_corner_kwargs)
    corner_kwargs["color"] = "red"
    # Now plot the posterior samples from GWOSC
    corner.corner(gwosc_samples, labels = labels, fig=fig, hist_kwargs={'density': True}, **corner_kwargs)
    fig.savefig(outdir + name, bbox_inches='tight')  
    print("Done")