import os
import shutil
import time
from jimgw.jim import Jim
from jimgw.detector import H1, L1, V1
from jimgw.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD
from jimgw.prior import Uniform
import jax.numpy as jnp
import jax

from gwpy.timeseries import TimeSeries

jax.config.update("jax_enable_x64", True)

import pickle
import urllib.request

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

labels = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

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
V1.load_data(gps, T/2, T/2, fmin, fmax, psd_pad=psd_pad, tukey_alpha=0.2)

base_url = "https://raw.githubusercontent.com/ThibeauWouters/gw-datasets/main/"

outdir_name = "./outdir/"

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
    
    ## V1
    data_td = TimeSeries.fetch_open_data("V1", gps - T/2, gps + T/2, cache=True)
    segment_length = data_td.duration.value
    n = len(data_td)
    delta_t = data_td.dt.value
    freq = jnp.fft.rfftfreq(n, delta_t)
    freq = freq[(freq>fmin)&(freq<fmax)]
            
    f, psd = get_psd("v1_psd.txt")
    
    # Get at specific frequencies for jim
    psd = np.interp(freq, f, psd)
    V1.psd = psd


# Priors and likelihood
prior = Uniform(
    xmin=[1.18, 0.125, -0.05, -0.05,  1.0, -0.1,        0.0, -1.0,    0.0,        0.0, -1],
    xmax=[1.21,   1.0,  0.05,  0.05, 75.0,  0.1, 2 * jnp.pi,  1.0, jnp.pi, 2 * jnp.pi,  1],
    naming=[
        "M_c",
        "q",
        "s1_z",
        "s2_z", 
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

# TODO are these parameters any good?
# Found after 2000 epochs of max likelihood
ref_params = {'M_c': 1.20094279, 
              'eta': 0.24869089, 
              's1_z': 0.04999872,
              's2_z': 0.03370802,  
              'd_L': 53.04809216, 
              't_c': 0.08423194, 
              'phase_c': 0.99098395,  
              'iota': 1.6507561, 
              'psi': 1.40839314, 
              'ra': 2.24920225, 
              'dec': 0.05244613
}

likelihood = HeterodynedTransientLikelihoodFD([H1, L1, V1], prior=prior, bounds=[prior.xmin, prior.xmax], waveform=RippleIMRPhenomD(), trigger_time=gps, duration=T, n_bins=500, ref_params=ref_params)

eps = 1e-5
mass_matrix = jnp.eye(11) 
mass_matrix = mass_matrix.at[1, 1].set(1e-3)
mass_matrix = mass_matrix.at[5, 5].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix * eps}

jim = Jim(
    likelihood,
    prior,
    n_loop_pretraining=0,
    n_loop_training=20,
    n_loop_production=20,
    n_local_steps=200,
    n_global_steps=200,
    n_chains=1000,
    n_epochs=60,
    max_samples=45000,
    batch_size=50000,
    train_thinning=40,
    output_thinning=10,
    n_loops_maximize_likelihood = 2000,
    local_sampler_arg=local_sampler_arg,
    outdir_name=outdir_name
)

### Heavy computation
# jim.maximize_likelihood([prior.xmin, prior.xmax])
jim.sample(jax.random.PRNGKey(42))

# Cleaning outdir
for filename in os.listdir(outdir_name):
    file_path = os.path.join(outdir_name, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

# === Show results, save output ===

### Summary
jim.print_summary()

### Diagnosis plots of summaries
print("Creating plots")
jim.Sampler.plot_summary("pretraining")
jim.Sampler.plot_summary("training")
jim.Sampler.plot_summary("production")

# TODO - save the NF object to sample from later on
# print("Saving jim object (Normalizing flow)")
# jim.Sampler.save_flow("my_nf_IMRPhenomD")

### Write to 
which_list = ["training", "production"]
for which in which_list:
    name = outdir_name + f'results_{which}.npz'
    print(f"Saving {which} samples in npz format to {name}")
    state = jim.Sampler.get_sampler_state(which)
    chains, log_prob, local_accs, global_accs = state["chains"], state["log_prob"], state["local_accs"], state["global_accs"]
    np.savez(name, chains=chains, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs)

print("Sampling from the flow")
chains = jim.Sampler.sample_flow(10000)
name = outdir_name + 'results_NF.npz'
print(f"Saving flow samples to {name}")
np.savez(name, chains=chains)

### Plot chains and samples

# files = [outdir_name + s for s in ["results_production.npz", "results_NF.npz"]]
# names = [outdir_name + s for s in ["production.png", "NF.png"]]

# Production samples:
file = outdir_name + "results_production.npz"
name = outdir_name + "results_production.png"

data = np.load(file)
# TODO improve the following: ignore t_c, and reshape with n_dims, and do conversions
chains = data['chains'][:,:,[0,1,2,3,4,6,7,8,9,10]].reshape(-1,10)
chains[:,6] = np.arccos(chains[:,6])
chains[:,9] = np.arcsin(chains[:,9])
chains = np.asarray(chains)
corner_kwargs = default_corner_kwargs
fig = corner.corner(chains, labels = labels, hist_kwargs={'density': True}, **default_corner_kwargs)
fig.savefig(name, bbox_inches='tight')  

# Production samples:
file = outdir_name + "results_NF.npz"
name = outdir_name + "results_NF.png"

data = np.load(file)["chains"]
print("np.shape(data)")
print(np.shape(data))

# TODO improve the following: ignore t_c, and reshape with n_dims, and do conversions
chains = data[:, [0,1,2,3,4,6,7,8,9,10]]
# chains[:,6] = np.arccos(chains[:,6])
# chains[:,9] = np.arcsin(chains[:,9]) # TODO not sure if this is still necessary?
chains = np.asarray(chains)
corner_kwargs = default_corner_kwargs
fig = corner.corner(chains, labels = labels, hist_kwargs={'density': True}, **default_corner_kwargs)
fig.savefig(name, bbox_inches='tight')  
    
    
print("Saving the hyperparameters")
jim.save_hyperparameters()