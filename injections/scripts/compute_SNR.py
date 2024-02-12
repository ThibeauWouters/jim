"""
This is a deprecated script that was used to compute the SNR of the injected signals.

Now, the code to compute the SNR is in the main postprocessing script.
"""

import os
import shutil
import sys
import json
# jim
from jimgw.jim import Jim
from jimgw.detector import H1, L1, V1
from jimgw.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.waveform import RippleTaylorF2
from jimgw.prior import Uniform, Powerlaw, Composite
# jax
import jax.numpy as jnp
import jax
chosen_device = jax.devices()[2]
jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_default_device", chosen_device)
# others
import numpy as np
jax.config.update("jax_enable_x64", True)
from astropy.time import Time

import numpy as np
import corner
import matplotlib.pyplot as plt
import pandas as pd

import gc

outdir = "./outdir/"

####################
### Script setup ###
####################

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

matplotlib_params = {
    "axes.labelsize": 30,
    "axes.titlesize": 30,
    "text.usetex": True,
    "font.family": "serif",
}
plt.rcParams.update(matplotlib_params)

labels = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda$', r'$\delta\Lambda$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

### Shared injection constants

HYPERPARAMETERS = {
    "flowmc": 
        {
            "n_loop_training": 200, 
            "n_loop_production": 50, 
            "n_local_steps": 200,
            "n_global_steps": 200, 
            "n_chains": 1000, 
            "n_epochs": 100, 
            "learning_rate": 0.001, 
            "max_samples": 50000, 
            "momentum": 0.9, 
            "batch_size": 50000, 
            "use_global": True, 
            "logging": True, 
            "keep_quantile": 0.0, 
            "local_autotune": None, 
            "train_thinning": 10, 
            "output_thinning": 30, 
            "n_sample_max": 10000, 
            "precompile": False, 
            "verbose": False, 
            "outdir_name": "./outdir/"
        }, 
    "jim": 
        {
            "seed": 0, 
            "n_chains": 1000, 
            "num_layers": 10, 
            "hidden_size": [128, 128], 
            "num_bins": 8, 
            # "local_sampler_arg": {"step_size": 1e-2 * jnp.array([[1e-08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1e-07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-08, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-05, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-05]])}
            }
}

PRIOR = {
    # "M_c": [0.435275, 2.61165],
    "M_c": [1.2, 2.61165], 
    "q": [0.125, 1.0], 
    "s1_z": [-0.05, 0.05], 
    "s2_z": [-0.05, 0.05], 
    "lambda_tilde": [0.0, 9000.0], 
    "delta_lambda_tilde": [-1000.0, 1000.0], 
    "d_L": [1.0, 100.0], 
    "t_c": [-0.1, 0.1], 
    "phase_c": [0.0, 6.283185307179586], 
    "cos_iota": [-1.0, 1.0], 
    "psi": [0.0, 3.141592653589793], 
    "ra": [0.0, 6.283185307179586], 
    "sin_dec": [-1, 1]
}

NAMING = ['M_c', 'q', 's1_z', 's2_z', 'lambda_tilde', 'delta_lambda_tilde', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']

#########################
### Utility functions ###
#########################

def compute_snr(detector, h_sky, detector_params):
    """Compute the SNR for a single detector

    Args:
        detector (_type_): _description_
        h_sky (_type_): _description_
        detector_params (_type_): _description_
    """
    frequencies = detector.frequencies
    df = frequencies[1] - frequencies[0]
    align_time = jnp.exp(
        -1j * 2 * jnp.pi * frequencies * (detector_params["epoch"] + detector_params["t_c"])
    )
    
    waveform_dec = (
                detector.fd_response(detector.frequencies, h_sky, detector_params) * align_time
            )
    
    snr = 4 * jnp.sum(jnp.conj(waveform_dec) * waveform_dec / detector.psd * df).real
    snr = jnp.sqrt(snr)
    return snr


def generate_config(prior_low: np.array, 
                    prior_high: np.array, 
                    params_names: "list[str]", 
                    N_config: int = 1,
                    no_noise: bool = True) -> str:
    """
    From a given prior range and parameter names, generate the config files.
    
    Args:
        prior_low: lower bound of the prior range
        prior_high: upper bound of the prior range
        params_names: list of parameter names
        N_config: identification number of this config file.
    
    Returns:
        outdir (str): the directory where the config files are saved
    """
    
    # First, generate the GW parameters:
    params_dict = {}
    for low, high, param in zip(prior_low, prior_high, params_names):
        params_dict[param] = np.random.uniform(low, high)
    
    print("params_dict")
    print(params_dict)
        
    # Create new injection file
    output_path = f'./outdir/injection_{str(N_config)}/'
    filename = output_path + f"config.json"
    
    # This injection dictionary will store all needed information for the injection
    injection_dict = {
        'outdir': output_path,
        'seed': np.random.randint(low=0, high=10000),
        'f_sampling': 2*2048,
        'duration': 128,
        'fmin': 20,
        'ifos': ['H1', 'L1', 'V1'],
        'no_noise': no_noise
    }
    
    injection_dict.update(params_dict)
    
    print("injection_dict")
    print(injection_dict)
    
    # Save the injection file to the output directory as JSON
    with open(filename, 'w') as f:
        json.dump(injection_dict, f)
        
    # Show the user that the file has been generated
    print(f"Generated config file {filename}")
    
    return injection_dict

def get_N():
    """
    Check outdir, get the subdirectories and return the length of subdirectories list
    """
    
    # Get the subdirectories
    subdirs = [x[0] for x in os.walk(outdir)]
    
    # Remove the first element, which is the outdir itself
    subdirs.pop(0)
    
    print("subdirs")
    print(subdirs)
    
    # Return the length of the list
    return len(subdirs)


###############################
### Main body of the script ###
###############################

def body(N):

    # NOTE we no longer read in the prior.json and hyperparameters.json files, but instead use the dictionaries above in this script directly

    # Get naming and prior bounds
    naming = list(PRIOR.keys())
    prior_ranges = jnp.array([PRIOR[name] for name in naming])
    prior_low, prior_high = prior_ranges[:, 0], prior_ranges[:, 1]

    # Check result:
    print("Prior:")
    print("low: ", prior_low)
    print("high:", prior_high)
    print("params_names:", naming)

    # # Generate the config files
    # print("Generating injection config files...")
    # # TODO make sure the correct N is passed
    # print(f"Running generate config with {N}")
    # config = generate_config(prior_low, prior_high, naming, N)
    # outdir = config["outdir"]
    # no_noise = config["no_noise"]
    # print(f"The outdir is {outdir}")
    # print("Done!")
    
    # Read the config file that is in the outdir with N
    config_file = f"./outdir/injection_{N}/config.json"
    print(f"Reading config file {config_file}")
    with open(config_file) as f:
        config = json.load(f)
    print("Done!")
    outdir = config["outdir"]
    no_noise = config["no_noise"]
    no_noise = False

    print("Prior:")
    print(PRIOR)

    print(PRIOR.items())

    naming = list(PRIOR.keys())
    bounds = []
    for key, value in PRIOR.items():
        bounds.append(value)

    bounds = np.asarray(bounds)
    xmin = bounds[:, 0]
    xmax = bounds[:, 1]

    print("Results from prior dict")
    print("naming")
    print(naming)
    print("xmin")
    print(xmin)
    print("xmax")
    print(xmax)

    # Fetch the flowMC and jim hyperparameters, and put together into one dict:
    flowmc_hyperparameters = HYPERPARAMETERS["flowmc"]
    jim_hyperparameters = HYPERPARAMETERS["jim"]
    hyperparameters = {**flowmc_hyperparameters, **jim_hyperparameters}

    print("Hyperparameters:")
    print(hyperparameters)

    print("Injection:")
    print(config)

    ### Data definitions

    gps = 1187008882.43
    trigger_time = gps
    gmst = Time(trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad
    fmin = 20
    f_ref = fmin 
    fmax = 2048
    f_sampling = 2 * fmax
    T = 128
    duration = T
    post_trigger_duration = 2
    epoch = duration - post_trigger_duration
    freqs = jnp.linspace(fmin, fmax, duration * f_sampling)

    ### Define priors
    
    # Internal parameters
    Mc_prior = Uniform(xmin[0], xmax[0], naming=["M_c"])
    q_prior = Uniform(
        xmin[1], 
        xmax[1],
        naming=["q"],
        transforms={"q": ("eta", lambda params: params["q"] / (1 + params["q"]) ** 2)},
    )
    s1z_prior                = Uniform(xmin[2], xmax[2], naming=["s1_z"])
    s2z_prior                = Uniform(xmin[3], xmax[3], naming=["s2_z"])
    lambda_tilde_prior       = Uniform(xmin[4], xmax[4], naming=["lambda_tilde"])
    delta_lambda_tilde_prior = Uniform(xmin[5], xmax[5], naming=["delta_lambda_tilde"])

    # External parameters
    dL_prior       = Powerlaw(xmin[6], xmax[6], 2.0, naming=["d_L"])
    t_c_prior      = Uniform(xmin[7], xmax[7], naming=["t_c"])

    # These priors below are always the same, no xmin and xmax needed
    phase_c_prior  = Uniform(0.0, 2 * jnp.pi, naming=["phase_c"])
    cos_iota_prior = Uniform(
        -1.0,
        1.0,
        naming=["cos_iota"],
        transforms={
            "cos_iota": (
                "iota",
                lambda params: jnp.arccos(
                    jnp.arcsin(jnp.sin(params["cos_iota"] / 2 * jnp.pi)) * 2 / jnp.pi
                ),
            )
        },
    )
    psi_prior     = Uniform(0.0, jnp.pi, naming=["psi"])
    ra_prior      = Uniform(0.0, 2 * jnp.pi, naming=["ra"])
    sin_dec_prior = Uniform(
        -1.0,
        1.0,
        naming=["sin_dec"],
        transforms={
            "sin_dec": (
                "dec",
                lambda params: jnp.arcsin(
                    jnp.arcsin(jnp.sin(params["sin_dec"] / 2 * jnp.pi)) * 2 / jnp.pi
                ),
            )
        },
    )

    prior_list = [
            Mc_prior,
            q_prior,
            s1z_prior,
            s2z_prior,
            lambda_tilde_prior,
            delta_lambda_tilde_prior,
            dL_prior,
            t_c_prior,
            phase_c_prior,
            cos_iota_prior,
            psi_prior,
            ra_prior,
            sin_dec_prior,
    ]

    # Compose prior
    prior = Composite(prior_list)
    bounds = jnp.array([[p.xmin, p.xmax] for p in prior.priors]).T
    naming = NAMING
    
    ### Get the injected parameters, but apply the transforms first
    true_params_values = jnp.array([config[name] for name in naming])

    true_params = dict(zip(naming, true_params_values))

    # Apply prior transforms to this list:
    true_params = prior.transform(true_params)

    # Convert values from single float arrays to just float
    true_params = {key: value.item() for key, value in true_params.items()}
    print("true_params")
    print(true_params)

    ### Getting ifos and overwriting with above data

    detector_param = {"ra": true_params["ra"], 
                    "dec": true_params["dec"], 
                    "gmst": gmst, 
                    "psi": true_params["psi"], 
                    "epoch": epoch, 
                    "t_c": true_params["t_c"]}

    print("detector_param")
    print(detector_param)
    
    ### Fetch PSD and overwrite

    waveform = RippleTaylorF2(f_ref=f_ref)
    h_sky = waveform(freqs, true_params)
    
    H1.frequencies = freqs
    L1.frequencies = freqs
    V1.frequencies = freqs
    
    print("Loading PSDs")
    H1.load_psd_from_file("../H1.txt")
    L1.load_psd_from_file("../L1.txt")
    L1.load_psd_from_file("../V1.txt")
    print("Loading PSDs: done")

    print("Injecting signal. No noise is set to: ", no_noise)
    key, subkey = jax.random.split(jax.random.PRNGKey(config["seed"] + 1234))
    H1.inject_signal(subkey, freqs, h_sky, detector_param, no_noise=no_noise)
    key, subkey = jax.random.split(key)
    L1.inject_signal(subkey, freqs, h_sky, detector_param, no_noise=no_noise)
    key, subkey = jax.random.split(key)
    V1.inject_signal(subkey, freqs, h_sky, detector_param, no_noise=no_noise)
    
    # Compute the SNR
    h1_snr = compute_snr(H1, h_sky, detector_param)
    l1_snr = compute_snr(L1, h_sky, detector_param)
    v1_snr = compute_snr(V1, h_sky, detector_param)
    
    network_snr = np.sqrt(h1_snr**2 + l1_snr**2 + v1_snr**2)
    
    print("SNRs:")
    print("H1:", h1_snr)
    print("L1:", l1_snr)
    print("V1:", v1_snr)
    
    print("Network:", network_snr)
    
    # Now, in the given outdir, write these to a file
    snr_file = outdir + "snr.csv"
    snr_dict = {"detector": ["H1", "L1", "V1", "network"], 
                "snr": [h1_snr, l1_snr, v1_snr, network_snr]}
    
    df = pd.DataFrame(snr_dict)
    df.to_csv(snr_file)
    
def main():
    # N = get_N()
    ### Iterate over the directories
    for N in range(30):
        my_string = "================================================================================================================================================================================================================================================"
        print(my_string)
        print(f"Running compute_SNR script for  N = {N}")
        print(my_string)
        body(N)
    
if __name__ == "__main__":
    main()
    
    