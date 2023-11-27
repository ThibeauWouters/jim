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

labels_chi_eff = [r'$M_c/M_\odot$', r'$q$', r'$\chi_{\rm eff}$', r'$\Lambda$', r'$\delta\Lambda$' ,r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

def get_chains(event, data_path = "../../data/"):
    """
    Retrieve chains for each event.
    
    Note: flowMC and Bilby refer to the ones obtained for the TurboPE repository (jim paper)
    """
    if event.upper() == 'GW150914':
        flowMC_data = np.load(data_path + 'GW150914_flowMC.npz')
        bilby_data = np.genfromtxt(data_path + 'GW150914_Bilby.dat')
        flowMC_chains = flowMC_data['chains'][:,:,[0,1,2,3,4,6,7,8,9,10]].reshape(-1,10)
        bilby_chains = bilby_data[1:,[1,0,2,3,6,11,9,10,8,7]]
        flowMC_chains[:,6] = np.arccos(flowMC_chains[:,6])
        flowMC_chains[:,9] = np.arcsin(flowMC_chains[:,9])
    elif event.upper() == 'GW170817':
        flowMC_data = np.load(data_path + 'GW170817_flowMC_1800.npz')
        bilby_data = np.genfromtxt(data_path + 'GW170817_Bilby_flat.dat')
        flowMC_chains = flowMC_data['chains'][:,:,[0,1,2,3,4,6,7,8,9,10]].reshape(-1,10)
        bilby_chains = bilby_data[1:,[0,1,2,3,4,8,10,7,6,5]]
        # the bilby run had an extended spin prior to avoid boundary issues,
        # so reject-sample down to our spin bounds of -0.05 < chi < 0.05
        bilby_chains = bilby_chains[(np.abs(bilby_chains[:,2]) < 0.05) & (np.abs(bilby_chains[:,3]) < 0.05)]
        flowMC_chains[:,6] = np.arccos(flowMC_chains[:,6])
        flowMC_chains[:,9] = np.arcsin(flowMC_chains[:,9])
    return flowMC_chains, bilby_chains

### Fetch data

def weight_function(x):
    return x**2

def reweigh_distance(chains, d_idx = 4):
    """
    Get weights based on distance to mimic cosmological distance prior.
    """
    d_samples = chains[:, d_idx]
    weights = weight_function(d_samples)
    weights = weights / np.sum(weights)
    
    return weights


which_list = ["NF", "production"]
outdir = "../GW170817_TaylorF2/outdir/"
corner_kwargs = default_corner_kwargs

idx_list = [0,1,2,3,4,5,6,8,9,10,11,12]

### Plotting hyperparameters

use_weights = True
use_chi_eff = True

### Plotting
for which in which_list:
    corner_kwargs["color"] = "blue"
    filename = f"{outdir}results_{which}.npz"

    print(f"Loading {which} samples... ")
    # My samples
    if which == "production":
        data = np.load(filename)
        chains = data['chains'][:,:,idx_list].reshape(-1,12)
        chains[:,8] = np.arccos(chains[:,8])
        chains[:,11] = np.arcsin(chains[:,11])
        chains = np.asarray(chains)
        print(np.shape(chains))
    else:
        data = np.load(filename)
        chains = data["chains"][:, idx_list]
        chains = np.asarray(chains)
        print(np.shape(chains))
    
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
        
        print("chi_eff")
        print(chi_eff)
        print("np.shape(chi_eff)")
        print(np.shape(chi_eff))
        
        # Now, we remove the second and third column from chains, and add chi_eff as new column at index 2
        chains = np.delete(chains, [2,3], axis=1)
        chains = np.insert(chains, 2, chi_eff, axis=1)
        labels = labels_chi_eff
    
    fig = corner.corner(chains, labels = labels, weights=weights, hist_kwargs={'density': True}, **default_corner_kwargs)
    corner_kwargs["color"] = "red"
    # corner.corner(flowMC_chains, labels = labels, fig=fig, hist_kwargs={'density': True}, **corner_kwargs)
    fig.savefig(outdir + name, bbox_inches='tight')  
    print("Done")