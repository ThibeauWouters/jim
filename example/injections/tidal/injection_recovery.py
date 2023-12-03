import os
import shutil
import sys
import yaml
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

### Read in the filename location given in to this script as argument
outdir = sys.argv[1]
print("Reading in the filename from the command line argument")
print(outdir)

# Read the config file:
with open(outdir + "config.json") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
outdir = config["outdir"]
    
# Read the prior file:
with open("prior.json") as f:
    prior_dict = yaml.load(f, Loader=yaml.FullLoader)
    
print("Prior:")
print(prior_dict)

print(prior_dict.items())

naming = list(prior_dict.keys())
bounds = []
for key, value in prior_dict.items():
    bounds.append(value)

bounds = np.asarray(bounds)
xmin = bounds[:, 0]
xmax = bounds[:, 1]

print("results from prior dict")
print(naming)
print(xmin)
print(xmax)

# Read the hyperparameters file:
with open("hyperparams.json") as f:
    hyperparameters_dict = yaml.load(f, Loader=yaml.FullLoader)
    
# Fetch the flowMC and jim hyperparameters, and put together into one dict:
flowmc_hyperparameters = hyperparameters_dict["flowmc"]
jim_hyperparameters = hyperparameters_dict["jim"]
hyperparameters = {**flowmc_hyperparameters, **jim_hyperparameters}

print("Hyperparameters:")
print(hyperparameters)

print("Injection:")
print(config)

### Data definitions

total_time_start = time.time()
gps = 1187008882.43
trigger_time = gps
fmin = 20
fmax = 2048
f_sampling = 2 * fmax
minimum_frequency = fmin
maximum_frequency = fmax
T = 128
duration = T
post_trigger_duration = 2
epoch = duration - post_trigger_duration
f_ref = fmin 
epoch = duration - post_trigger_duration
gmst = Time(trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad

freqs = jnp.linspace(fmin, fmax, duration * f_sampling)

# Define prior
prior = Uniform(
    xmin = xmin,
    xmax = xmax,
    naming = naming,
    transforms = {"q": ("eta", lambda params: params['q']/(1+params['q'])**2),
                 "cos_iota": ("iota",lambda params: jnp.arccos(jnp.arcsin(jnp.sin(params['cos_iota']/2*jnp.pi))*2/jnp.pi)),
                 "sin_dec": ("dec",lambda params: jnp.arcsin(jnp.arcsin(jnp.sin(params['sin_dec']/2*jnp.pi))*2/jnp.pi))} # sin and arcsin are periodize cos_iota and sin_dec
)

### Getting ifos and overwriting with above data

# TODO why +1234?
key, subkey = jax.random.split(jax.random.PRNGKey(config["seed"] + 1234))
true_params = jnp.array([config[name] for name in naming])
true_params = prior.add_name(true_params, transform_name = True, transform_value = True)

print("True params:")
print(true_params)


detector_param = {"ra": true_params["ra"], 
                  "dec": true_params["dec"], 
                  "gmst": gmst, 
                  "psi": true_params["psi"], 
                  "epoch": epoch, 
                  "t_c": true_params["t_c"]}

print("detector_param")
print(detector_param)

waveform = RippleTaylorF2(f_ref=f_ref)
h_sky = waveform(freqs, true_params)

print("Injecting signal")
H1.inject_signal(subkey, freqs, h_sky, detector_param)
key, subkey = jax.random.split(key)
L1.inject_signal(subkey, freqs, h_sky, detector_param)
key, subkey = jax.random.split(key)
V1.inject_signal(subkey, freqs, h_sky, detector_param)

### Fetch PSD and overwrite

print("Loading PSDs")
H1.load_psd_from_file("../H1.txt")
L1.load_psd_from_file("../L1.txt")
L1.load_psd_from_file("../V1.txt")
print("Loading PSDs: done")

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

likelihood = HeterodynedTransientLikelihoodFD([H1, L1, V1], prior=prior, bounds=[prior.xmin, prior.xmax], waveform=RippleTaylorF2(), trigger_time=gps, duration=T, n_bins=500)

### Create sampler and jim objects

# Mass matrix (this is copy pasted from the TurboPE set up)
# TODO get automated mass matrix, or make sure this doesn't break things
eps = 1e-3
n_chains = 1000
n_dim = 13
mass_matrix = jnp.eye(n_dim)
mass_matrix = mass_matrix.at[0,0].set(1e-5)
mass_matrix = mass_matrix.at[1,1].set(1e-4)
mass_matrix = mass_matrix.at[2,2].set(1e-3)
mass_matrix = mass_matrix.at[3,3].set(1e-3)
mass_matrix = mass_matrix.at[7,7].set(1e-5)
mass_matrix = mass_matrix.at[11,11].set(1e-2)
mass_matrix = mass_matrix.at[12,12].set(1e-2)
local_sampler_arg = {"step_size": mass_matrix * eps}

hyperparameters["outdir_name"] = outdir
hyperparameters["local_sampler_arg"] = local_sampler_arg
hyperparameters["n_loops_training"] = 5
hyperparameters["n_loops_production"] = 5

jim = Jim(
    likelihood,
    prior,
    **hyperparameters
)

### Heavy computation begins
jim.sample(jax.random.PRNGKey(42))
### Heavy computation ends

# === Show results, save output ===

# Cleaning outdir
for filename in os.listdir(outdir):
    file_path = os.path.join(outdir, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

### Summary
jim.print_summary()

### Diagnosis plots of summaries
print("Creating plots")
jim.Sampler.plot_summary("training")
jim.Sampler.plot_summary("production")

### Write to 
which_list = ["training", "production"]
for which in which_list:
    name = outdir + f'results_{which}.npz'
    print(f"Saving {which} samples in npz format to {name}")
    state = jim.Sampler.get_sampler_state(which)
    chains, log_prob, local_accs, global_accs = state["chains"], state["log_prob"], state["local_accs"], state["global_accs"]
    np.savez(name, chains=chains, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs)

print("Sampling from the flow")
chains = jim.Sampler.sample_flow(10000)
name = outdir + 'results_NF.npz'
print(f"Saving flow samples to {name}")
np.savez(name, chains=chains)

### Plot chains and samples

# Production samples:
file = outdir + "results_production.npz"
name = outdir + "results_production.png"

data = np.load(file)
# TODO improve the following: ignore t_c, and reshape with n_dims, and do conversions
idx_list = [0,1,2,3,4,5,6,8,9,10,11,12]
chains = data['chains'][:,:,idx_list].reshape(-1,12)
chains[:,8] = np.arccos(chains[:,8])
chains[:,11] = np.arcsin(chains[:,11])
chains = np.asarray(chains)
corner_kwargs = default_corner_kwargs
fig = corner.corner(chains, labels = labels, hist_kwargs={'density': True}, **default_corner_kwargs)
fig.savefig(name, bbox_inches='tight')  

# Production samples:
file = outdir + "results_NF.npz"
name = outdir + "results_NF.png"

data = np.load(file)["chains"]
print("np.shape(data)")
print(np.shape(data))

# TODO improve the following: ignore t_c, and reshape with n_dims, and do conversions
chains = data[:, idx_list]
# chains[:,6] = np.arccos(chains[:,6])
# chains[:,9] = np.arcsin(chains[:,9]) # TODO not sure if this is still necessary?
chains = np.asarray(chains)
corner_kwargs = default_corner_kwargs
fig = corner.corner(chains, labels = labels, hist_kwargs={'density': True}, **default_corner_kwargs)
fig.savefig(name, bbox_inches='tight')  
    
    
print("Saving the hyperparameters")
jim.save_hyperparameters()