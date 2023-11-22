import time
from jimgw.jim import Jim
from jimgw.detector import H1, L1
from jimgw.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD
from jimgw.prior import Uniform
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from gwpy.timeseries import TimeSeries

import pickle
import urllib.request

# CUDA stuff
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import jaxlib
print(jax.devices())
gpu_index = 1
jax.config.update('jax_platform_name', f'gpu:{gpu_index}')
jax.default_device = jax.devices("gpu")[1]

waveform = RippleIMRPhenomD

###########################################
########## First we grab data #############
###########################################

total_time_start = time.time()

gps = 1187008882.43

T = 128
start = gps - T/2
end   = gps + T/2
fmin = 20.0
fmax = 2048.0
psd_pad = 16

ifos = ["H1", "L1"]
H1.load_data(gps, T/2, T/2, fmin, fmax, psd_pad=psd_pad, tukey_alpha=0.2)
L1.load_data(gps, T/2, T/2, fmin, fmax, psd_pad=psd_pad, tukey_alpha=0.2)

base_url = "https://raw.githubusercontent.com/ThibeauWouters/gw-datasets/main/"

def get_psd(filename):
    print(f"Fetching PSD for {filename}")
    psd_file = base_url + filename
    
    with urllib.request.urlopen(psd_file) as response:
        data = response.read().decode('utf-8')
        
        lines = data.split('\n')
        f, psd = [], []

        for line in lines:
            if line.strip():  # Check if the line is not empty
                columns = line.split()
                f.append(float(columns[0]))
                psd.append(float(columns[1]))
        print(f"Shapes: {np.shape(f)}, {np.shape(psd)}")
    
    return f, psd

override_psd = True
if override_psd:
    psd_pad = 0
    print("Overriding PSD")
    ### Override the PSD
    # Taking PSD values from Peter:
    ## H1

    # We fetch the data separately in order to build the same frequency array, in order to get the same mask
    data_td = TimeSeries.fetch_open_data("H1", gps - T/2, gps + T/2, cache=True)
    segment_length = data_td.duration.value
    n = len(data_td)
    delta_t = data_td.dt.value
    freq = jnp.fft.rfftfreq(n, delta_t)
    freq = freq[(freq>fmin)&(freq<fmax)]
    
    f, psd = get_psd("h1_psd.txt")
    
    # Get at specific frequencies for jim
    psd = np.interp(freq, f, psd)
    H1.psd = psd

    ## L1
    data_td = TimeSeries.fetch_open_data("L1", gps - T/2, gps + T/2, cache=True)
    segment_length = data_td.duration.value
    n = len(data_td)
    delta_t = data_td.dt.value
    freq = jnp.fft.rfftfreq(n, delta_t)
    freq = freq[(freq>fmin)&(freq<fmax)]
            
    f, psd = get_psd("l1_psd.txt")
    
    # Get at specific frequencies for jim
    psd = np.interp(freq, f, psd)
    L1.psd = psd


# Priors and likelihood
eps = 1e-3

prior = Uniform(
    xmin=[1.18, 0.125, -0.05, -0.05,     0.0, -1000.0,  1.0, -0.1,        0.0, -1.0,    0.0,        0.0, -1],
    xmax=[1.21,   1.0,  0.05,  0.05, 10000.0,  1000.0, 75.0,  0.1, 2 * jnp.pi,  1.0, jnp.pi, 2 * jnp.pi,  1],
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
                 "sin_dec": ("dec",lambda params: jnp.arcsin(jnp.arcsin(jnp.sin(params['sin_dec']/2*jnp.pi))*2/jnp.pi))} # sin and arcsin are periodize cos_iota and sin_dec
)
likelihood = HeterodynedTransientLikelihoodFD([H1, L1], prior=prior, bounds=[prior.xmin, prior.xmax], waveform=waveform(), trigger_time=gps, duration=T, post_trigger_duration=T/2)

eps = 3e-6
mass_matrix = jnp.eye(13) 
mass_matrix = mass_matrix.at[1, 1].set(1e-3)
mass_matrix = mass_matrix.at[7, 7].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix * eps}

outdir_name = "./outdir_GW170817_NRTidalv2/"

jim = Jim(
    likelihood,
    prior,
    n_loop_pretraining=10,
    n_loop_training=100,
    n_loop_production=10,
    n_local_steps=150,
    n_global_steps=150,
    n_chains=500,
    n_epochs=50,
    learning_rate=0.001,
    max_samples=45000,
    momentum=0.9,
    batch_size=50000,
    use_global=True,
    keep_quantile=0.0,
    train_thinning=1,
    output_thinning=10,
    seed = 42,
    local_sampler_arg=local_sampler_arg,
    outdir_name=outdir_name
)


### Sampling begins here
jim.sample(jax.random.PRNGKey(24))
jim.print_summary()

jim.Sampler.plot_summary("pretraining")
jim.Sampler.plot_summary("training")
jim.Sampler.plot_summary("production")

# # Show results
# result = jim.get_samples()
# print(result)


# # Save the results to external output files
# print("Saving jim object (Normalizing flow)")
# jim.Sampler.save_flow("my_nf_NRTidalv2")

# # name ='samples_NRTidalv2.pickle'
# # print(f"Saving samples to {name}")
# # with open(name, 'wb') as handle:
# #     pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# # name ='likelihood_NRTidalv2.pickle'
# # print(f"Saving likelihood to {name}")
# # with open(name, 'wb') as handle:
# #     pickle.dump(likelihood, handle, protocol=pickle.HIGHEST_PROTOCOL)
