# jim
from jimgw.jim import Jim
from jimgw.detector import H1, L1, V1
from jimgw.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD, RippleTaylorF2
from jimgw.prior import Uniform
# ripple
# flowmc
from flowMC.utils.PRNG_keys import initialize_rng_keys
# jax
import jax.numpy as jnp
import jax
# others
import time
import numpy as np
from lal import GreenwichMeanSiderealTime
jax.config.update("jax_enable_x64", True)
from astropy.time import Time

# import urllib.request
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import corner

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
               r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

chosen_device = jax.devices()[3] # e.g. device with index 2
jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_default_device", chosen_device)

### Data definitions

total_time_start = time.time()
gps = 1187008882.43
trigger_time = gps
fmin = 20
fmax = 2048
minimum_frequency = fmin
maximum_frequency = fmax
T = 128
duration = T
post_trigger_duration = 2
epoch = duration - post_trigger_duration
f_ref = fmin 
# gmst = GreenwichMeanSiderealTime(trigger_time)
# gsmt = Time(trigger_time, format="gps").sidereal_time("apparent", "greenwich").rad

### Getting detector data

H1_frequency, H1_data_re, H1_data_im = np.genfromtxt('../../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_H1_fd_strain.txt').T
H1_data = H1_data_re + 1j*H1_data_im
H1_psd_frequency, H1_psd = np.genfromtxt('../../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_H1_psd.txt').T

H1_data = H1_data[(H1_frequency>minimum_frequency)*(H1_frequency<maximum_frequency)]
H1_psd = H1_psd[(H1_frequency>minimum_frequency)*(H1_frequency<maximum_frequency)]
H1_frequency = H1_frequency[(H1_frequency>minimum_frequency)*(H1_frequency<maximum_frequency)]

L1_frequency, L1_data_re, L1_data_im = np.genfromtxt('../../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_L1_fd_strain.txt').T
L1_data = L1_data_re + 1j*L1_data_im
L1_psd_frequency, L1_psd = np.genfromtxt('../../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_L1_psd.txt').T

L1_data = L1_data[(L1_frequency>minimum_frequency)*(L1_frequency<maximum_frequency)]
L1_psd = L1_psd[(L1_frequency>minimum_frequency)*(L1_frequency<maximum_frequency)]
L1_frequency = L1_frequency[(L1_frequency>minimum_frequency)*(L1_frequency<maximum_frequency)]

V1_frequency, V1_data_re, V1_data_im = np.genfromtxt('../../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_V1_fd_strain.txt').T
V1_data = V1_data_re + 1j*V1_data_im
V1_psd_frequency, V1_psd = np.genfromtxt('../../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_V1_psd.txt').T

V1_data = V1_data[(V1_frequency>minimum_frequency)*(V1_frequency<maximum_frequency)]
V1_psd = V1_psd[(V1_frequency>minimum_frequency)*(V1_frequency<maximum_frequency)]
V1_frequency = V1_frequency[(V1_frequency>minimum_frequency)*(V1_frequency<maximum_frequency)]

### Getting ifos and overwriting with above data

H1.load_data(gps, post_trigger_duration, post_trigger_duration, fmin, fmax, psd_pad=16, tukey_alpha=0.2)
L1.load_data(gps, post_trigger_duration, post_trigger_duration, fmin, fmax, psd_pad=16, tukey_alpha=0.2)
V1.load_data(gps, post_trigger_duration, post_trigger_duration, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

# Overwrite results
H1.frequencies = H1_frequency
H1.data = H1_data
H1.psd = H1_psd 

L1.frequencies = L1_frequency
L1.data = L1_data
L1.psd = L1_psd 

V1.frequencies = V1_frequency
V1.data = V1_data
V1.psd = V1_psd 


# TODO double-check whether these are params before or after transformation

# prior_range = jnp.array([[1.18,1.21],[0.125,1],[-0.05,0.05],[-0.05,0.05],[1,75],[-0.01,0.02],[0,2*np.pi],[-1,1],[0,np.pi],[0,2*np.pi],[-1,1]])
# rng_key_set = initialize_rng_keys(n_chains, seed=42)
# initial_position = jax.random.uniform(rng_key_set[0], shape=(int(n_chains), n_dim)) * 1
# for i in range(n_dim):
#     initial_position = initial_position.at[:,i].set(initial_position[:,i]*(prior_range[i,1]-prior_range[i,0])+prior_range[i,0])
    
# Prior
prior = Uniform(
    xmin=[1.18, 0.125, -0.05, -0.05,    0.0, -500.0,  1.0, -0.1,        0.0, -1.0,    0.0,        0.0, -1],
    xmax=[1.21,   1.0,  0.05,  0.05, 3000.0,  500.0, 75.0,  0.1, 2 * jnp.pi,  1.0, jnp.pi, 2 * jnp.pi,  1],
    naming=[
        "M_c",
        "q",
        "s1_z", 
        "s2_z", 
        "lambda_tilde",
        "delta_lambda_tilde",
        "d_L",
        "t_c",
        "phase_c",
        "cos_iota",
        "psi",
        "ra",
        "sin_dec",
    ],
    transforms = {"q": ("eta", lambda params: params['q']/(1+params['q'])**2),
                 "cos_iota": ("iota",lambda params: jnp.arccos(jnp.arcsin(jnp.sin(params['cos_iota']/2*jnp.pi))*2/jnp.pi)),
                 "sin_dec": ("dec",lambda params: jnp.arcsin(jnp.arcsin(jnp.sin(params['sin_dec']/2*jnp.pi))*2/jnp.pi))}
)

### Create likelihood object


# TODO get the parameters here!

# ref_params = {'M_c': 1.19754835, 
#               'eta': 0.24211905, 
#               's1_z': 0.04992184, 
#               's2_z': -0.0375549, 
#               'lambda_tilde': 236.19042388, 
#               'delta_lambda_tilde': 95.33493973, 
#               'd_L': 19.27281561, 
#               't_c': 0.0326196, 
#               'phase_c': 4.43696823, 
#               'iota': 1.73586993, 
#               'psi': 2.04194889, 
#               'ra': 1.72313012, 
#               'dec': 0.72667927
# }

# The following line will (by default) automatically print the parameters of the reference waveform
n_walkers = 200
n_loops = 200
which_ES = "CMA_ES"
print(f"Running with {n_walkers} walkers and {n_loops} loops and {which_ES} ES.")
likelihood = HeterodynedTransientLikelihoodFD([H1, L1, V1], prior=prior, bounds=[prior.xmin, prior.xmax], waveform=RippleTaylorF2(), trigger_time=gps, duration=T, n_bins=500, n_walkers = n_walkers, n_loops = n_loops)


