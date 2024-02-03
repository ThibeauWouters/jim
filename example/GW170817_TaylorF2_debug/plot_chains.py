# jim
from jimgw.jim import Jim
from jimgw.detector import H1, L1, V1
from jimgw.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD, RippleTaylorF2
from jimgw.prior import Uniform, Powerlaw, Composite 
from jimgw.fisher_information_matrix import FisherInformationMatrix
# ripple
# flowmc
from flowMC.utils.PRNG_keys import initialize_rng_keys
# jax
import jax.numpy as jnp
import jax
jax.config.update("jax_platform_name", "cpu")
# others
import numpy as np
jax.config.update("jax_enable_x64", True)
from astropy.time import Time

# import urllib.request
import os
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
import corner

do_postprocessing = True
from jax.tree_util import tree_structure

# TODO move!!!
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
}
plt.rcParams.update(params)
labels = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda$', r'$\delta\Lambda$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$t_c$', r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']


### Plot chains and samples

# file = "./outdir/results_training.npz"
file = "/home/thibeau.wouters/projects/jim_runs/injections/tidal/outdir_TaylorF2/injection_144/results_production.npz"

data = np.load(file)
# TODO improve the following: ignore t_c, and reshape with n_dims, and do conversions
chains = data['chains']
chains = np.asarray(chains)
print("np.shape(chains)")
print(np.shape(chains))
chains[:, :, 9] = np.arccos(chains[:, :, 9])
chains[:, :, 12] = np.arcsin(chains[:, :, 12])

n_steps = 2 * 200
mc_data = chains[:, -n_steps-1 : -1, 0]
x = [i+1 for i in range(n_steps)]
mc_data = np.asarray(mc_data)
plt.figure(figsize = (10, 6))
for chain in mc_data:
    plt.plot(x, chain, color = 'blue', alpha = 0.2)
plt.xlabel('Step')
plt.ylabel(r'$M_c/M_\odot$')
plt.axvline(x = n_steps // 2, color = 'black', linestyle = '--')
plt.tight_layout()
plt.savefig('./postprocessing/mc_chain.png', bbox_inches='tight')    

### Extra stuff here
mc_data_MALA = chains[:, 0 : (n_steps // 2), 0]
diffs = mc_data_MALA[:, 0] - mc_data_MALA[:, -1]
print("np.mean(diffs)")
print(np.mean(diffs))
