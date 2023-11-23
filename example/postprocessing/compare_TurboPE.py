### Small utility script to compare results with TurboPE

import seaborn as sns
import pickle
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt 
import corner
import pandas as pd

### Utilities

default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        save=False)

params = {
    "axes.labelsize": 30,
    "axes.titlesize": 30,
    "text.usetex": True,
    "font.family": "serif",
    # "font.serif": "cmr10",
    # "mathtext.fontset": "cm",
    # "axes.formatter.use_mathtext": True,  # needed when using cm=cmr10 for normal text
}
plt.rcParams.update(params)

labels = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$d_{\rm{L}}/{\rm Mpc}$',
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

which_list = ["production", "NF"]
outdir = "../GW170817_reproduction/outdir/"
corner_kwargs = default_corner_kwargs

for which in which_list:
    corner_kwargs["color"] = "blue"
    filename = f"{outdir}results_{which}.npz"

    print(f"Loading {which} samples...")
    # My samples
    if which == "production":
        data = np.load(filename)
        chains = data['chains'][:,:,[0,1,2,3,4,6,7,8,9,10]].reshape(-1,10)
        chains[:,6] = np.arccos(chains[:,6])
        chains[:,9] = np.arcsin(chains[:,9])
        chains = np.asarray(chains)
        print(np.shape(chains))
    else:
        data = np.load(filename)
        chains = data["chains"][:, [0,1,2,3,4,6,7,8,9,10]]
        chains = np.asarray(chains)
        print(np.shape(chains))
    # TurboPE samples
    flowMC_chains, _ = get_chains('GW170817')
    print("Loading complete")

    ### Plot
    name = f"comparison_TurboPE_{which}.png"
    print(f"Saving plot of chains to {name}")
    
    fig = corner.corner(chains, labels = labels, hist_kwargs={'density': True}, **default_corner_kwargs)
    corner_kwargs["color"] = "red"
    corner.corner(flowMC_chains, labels = labels, fig=fig, hist_kwargs={'density': True}, **corner_kwargs)
    fig.savefig(outdir + name, bbox_inches='tight')  
    print("Done")