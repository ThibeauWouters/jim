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

### Fetch and check data (choose training or production run)

which = "production" 
filename = f"../GW170817_reproduction/outdir/samples_{which}_GW170817_IMRPhenomD.pickle"
print("Loading samples...")
with open(filename, 'rb') as handle:
    loaded_samples = pickle.load(handle) 
print("Loading complete")
    
print("Printing samples")
keys = list(loaded_samples.keys())
# We will ignore t_c in the plots
n_dim = len(keys) - 1 

samples = []
for key in keys:
    if key == "t_c":
        continue
    else:
        myarray = loaded_samples[key].flatten()
        samples.append(myarray)
        
samples = jnp.array(samples)
samples = jnp.swapaxes(samples, 0, 1)

### Plotting my chains

samples = np.asarray(samples)
fig = corner.corner(samples, labels = labels, hist_kwargs={'density': True}, **default_corner_kwargs)
name = f"my_samples_{which}.png"
print(f"Saving plot of chains to {name}")
fig.savefig(name, bbox_inches='tight')

### Plotting other chains

corner_kwargs = default_corner_kwargs

print("Getting TurboPE chains")
flowMC_chains, bilby_chains = get_chains('GW170817')
print("Plotting TurboPE chains")
fig = corner.corner(flowMC_chains, labels = labels, hist_kwargs={'density': True}, **corner_kwargs)
corner_kwargs["color"] = "red"
corner.corner(bilby_chains, fig=fig, hist_kwargs={'density': True}, **corner_kwargs)
fig.savefig("turboPE_test.png", bbox_inches='tight')
print("Done")