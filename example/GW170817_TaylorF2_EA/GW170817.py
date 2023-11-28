# jim
from jimgw.jim import Jim
from jimgw.detector import H1, L1, V1
from jimgw.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD, RippleTaylorF2
from jimgw.prior import Uniform
# ripple
# jax
import jax.numpy as jnp
import jax
# others
import time
import numpy as np
jax.config.update("jax_enable_x64", True)

import numpy as np
import matplotlib.pyplot as plt
import csv

from evosax import Strategies

# Choose our GPU device
chosen_device = jax.devices()[3]
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

# All knwon ES names
ES_names = Strategies.keys()

# The following line will (by default) automatically print the parameters of the reference waveform
n_walkers = 200
n_loops = 200
# ES_list = ["CMA_ES", "OpenES"]
ES_list = ES_names

# Initialize a new CSV file
CSV_filename = "GW170817_ES.csv"
with open(CSV_filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["ES_name", "best_fitness", "best_member", "elapsed_time"])

# The following for loop will iterate over a given list of keys of ES strategies to be used in likelihood. 
# Then, it will run the likelihood with the given ES strategy and print the elapsed time.
# Finally, we save everything to one single big CSV file.

for which_ES in ES_list:
    start = time.time()
    print(f"Running with {n_walkers} walkers and {n_loops} loops and {which_ES} ES.")
    likelihood = HeterodynedTransientLikelihoodFD([H1, L1, V1], 
                                                  prior=prior, 
                                                  bounds=[prior.xmin, prior.xmax], 
                                                  waveform=RippleTaylorF2(), 
                                                  trigger_time=gps, 
                                                  duration=T, 
                                                  n_bins=500, 
                                                  n_walkers = n_walkers, # TODO rename to popsize
                                                  n_loops = n_loops, 
                                                  which_ES=which_ES
                                                  )
    
    best_member, best_result = likelihood.best_member, likelihood.best_result
    end = time.time()
    elapsed = end - start
    
    # Save results to CSV
    with open(CSV_filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([which_ES, best_result, best_member, elapsed])
    
    print(f"Elapsed time: {elapsed} seconds.")


