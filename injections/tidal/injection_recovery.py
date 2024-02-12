import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.50"
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import psutil
p = psutil.Process()
p.cpu_affinity([0])
import shutil
import sys
import json
from scipy.interpolate import interp1d
# jim
from jimgw.jim import Jim
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.single_event.waveform import RippleTaylorF2
from jimgw.prior import Uniform, PowerLaw, AlignedSpin, Composite
# jax
import jax.numpy as jnp
import jax
import jax.profiler
# ripple
from ripple import Mc_eta_to_ms
# get available jax devices
nb_devices = len(jax.devices())
print(f"GPU available? {nb_devices} devices available, using final device from this list")
print(jax.devices())    
chosen_device = jax.devices()[-1]
print(f"chosen_device: {chosen_device}")
jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_default_device", chosen_device)
# others
import numpy as np
jax.config.update("jax_enable_x64", True)
from astropy.time import Time
import numpy as np
import corner
import matplotlib.pyplot as plt
import time
import gc

import plotting_utils as utils

####################
### Script setup ###
####################

### Script constants
SNR_THRESHOLD = 1e-40 # skip injections with SNR below this threshold
waveform_approximant = "TaylorF2" # which waveform approximant to use, either TaylorF2 or IMRPhenomD_NRTidalv2
OUTDIR = f"./outdir_{waveform_approximant}/"

### Script hyperparameters


HYPERPARAMETERS = {
    "flowmc": 
        {
            "n_loop_training": 20, # 130 
            "n_loop_production": 20, # 50
            "n_local_steps": 20, # 200
            "n_global_steps": 200, # 200
            "n_epochs": 50, # 100
            "n_chains": 1000, 
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
            "outdir_name": OUTDIR
        }, 
    "jim": 
        {
            "seed": 0, 
            "n_chains": 1000, 
            "num_layers": 10, 
            "hidden_size": [128, 128], 
            "num_bins": 8, 
        }
}

NAMING = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']
PRIOR = {
        "M_c": [0.8759659737275101, 2.6060030916165484],
        "q": [0.5, 1.0], 
        "s1_z": [-0.05, 0.05], 
        "s2_z": [-0.05, 0.05], 
        "lambda_1": [0.0, 5000.0], 
        "lambda_2": [0.0, 5000.0], 
        "d_L": [30.0, 300.0], 
        "t_c": [-0.1, 0.1], 
        "phase_c": [0.0, 6.283185307179586], 
        "cos_iota": [-1.0, 1.0], 
        "psi": [0.0, 3.141592653589793], 
        "ra": [0.0, 6.283185307179586], 
        "sin_dec": [-1, 1]
}



#########################
### Utility functions ###
#########################

def compute_snr(detector, h_sky, detector_params):
    """Compute the SNR of an event for a single detector, given the waveform generated in the sky.

    Args:
        detector (Detector): Detector object from jim.
        h_sky (Array): Jax numpy array containing the waveform strain as a function of frequency in the sky frame
        detector_params (dict): Dictionary containing parameters of the event relevant for the detector.
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

def generate_params_dict(prior_low, prior_high, params_names):
    params_dict = {}
    for low, high, param in zip(prior_low, prior_high, params_names):
        params_dict[param] = np.random.uniform(low, high)
    return params_dict

def generate_config(prior_low: np.array, 
                    prior_high: np.array, 
                    params_names: "list[str]", 
                    N_config: int = 1,
                    ) -> str:
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
    
    params_dict = generate_params_dict(prior_low, prior_high, params_names)
        
    # Create new injection file
    output_path = f'{OUTDIR}injection_{str(N_config)}/'
    filename = output_path + f"config.json"
    
    # Check if directory exists, if not, create it. Otherwise, delete it and create it again
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    print("Made injection directory")

    # This injection dictionary will store all needed information for the injection
    seed = np.random.randint(low=0, high=10000)
    injection_dict = {
        'seed': seed,
        'f_sampling': 2*2048,
        'duration': 256,
        'fmin': 20,
        'fref': 20,
        'post_trigger_duration': 2,
        'ifos': ['H1', 'L1', 'V1'],
        'outdir' : output_path
    }
    
    injection_dict.update(params_dict)
    
    # Save the injection file to the output directory as JSON
    with open(filename, 'w') as f:
        json.dump(injection_dict, f)
    
    return injection_dict

def get_N(outdir):
    """
    Check outdir, get the subdirectories and return the length of subdirectories list
    """
    subdirs = [x[0] for x in os.walk(outdir)]
    return len(subdirs)


###############################
### Main body of the script ###
###############################

def body(N, outdir, load_existing_config = False):

    # Preamble
    naming = NAMING
    network_snr = 0.0
    print(f"The SNR threshold parameter is set to {SNR_THRESHOLD}")

    while network_snr < SNR_THRESHOLD:
        
        # Fetch the flowMC and jim hyperparameters, and put together into one dict:
        flowmc_hyperparameters = HYPERPARAMETERS["flowmc"]
        jim_hyperparameters = HYPERPARAMETERS["jim"]
        hyperparameters = {**flowmc_hyperparameters, **jim_hyperparameters}
        
        # Generate the parameters or load them from an existing file
        prior_ranges = jnp.array([PRIOR[name] for name in naming])
        prior_low, prior_high = prior_ranges[:, 0], prior_ranges[:, 1]
        if load_existing_config:
            print("Loading existing config, path:")
            config_path = f"{outdir}injection_{N}/config.json"
            print(config_path)
            config = json.load(open(config_path))
        else:
            config = generate_config(prior_low, prior_high, naming, N)
        outdir = config["outdir"]

        bounds = []
        for key, value in PRIOR.items():
            bounds.append(value)
        bounds = np.asarray(bounds)
        
        # # TODO I just move on anyway
        network_snr = 14
        # if network_snr < SNR_THRESHOLD:
        #     print(f"Network SNR is less than {SNR_THRESHOLD}, generating new parameters")
            
        

        
    print("Injecting signals")
    waveform = RippleTaylorF2(f_ref=config["fmin"])
    key = jax.random.PRNGKey(config["seed"])
    # creating frequency grid
    freqs = jnp.arange(
        config["fmin"],
        config["f_sampling"] / 2,  # maximum frequency being halved of sampling frequency
        1. / config["duration"]
        )
    # convert injected mass ratio to eta
    eta = config["q"] / (1 + config["q"]) ** 2
    iota = float(jnp.arccos(config["cos_iota"]))
    dec = float(jnp.arcsin(config["sin_dec"]))
    # setup the timing setting for the injection
    print("config[trigger_time]")
    print(config["trigger_time"])
    epoch = config["duration"] - config["post_trigger_duration"]
    gmst = Time(config["trigger_time"], format='gps').sidereal_time('apparent', 'greenwich').rad
    print("gmst")
    print(gmst)
    # array of injection parameters
    true_param = {
        'M_c':       config["M_c"],       # chirp mass
        'eta':       eta,            # symmetric mass ratio 0 < eta <= 0.25
        's1_z':      config["s1_z"],      # aligned spin of priminary component s1_z.
        's2_z':      config["s2_z"],      # aligned spin of secondary component s2_z.
        'lambda_1':  config["lambda_1"],  # tidal deformability of priminary component lambda_1.
        'lambda_2':  config["lambda_2"],  # tidal deformability of secondary component lambda_2.
        'd_L':       config["d_L"],  # luminosity distance
        't_c':       config["t_c"],        # timeshift w.r.t. trigger time
        'phase_c':   config["phase_c"],      # merging phase
        'iota':      iota,
        'psi':       config["psi"],
        'ra':        config["ra"],
        'dec':       dec
        }
    detector_param = {
        'ra':     config["ra"],
        'dec':    dec,
        'gmst':   gmst,
        'psi':    config["psi"],
        'epoch':  epoch,
        't_c':    config["t_c"],
        }
    print(f"The injected parameters are {true_param}")
    # generating the geocenter waveform
    h_sky = waveform(freqs, true_param)
    # setup ifo list
    ifos = [H1, L1, V1]
    psd_files = ["./psd.txt", "./psd.txt", "./psd_virgo.txt"]
    # inject signal into ifos
    for idx, ifo in enumerate(ifos):
        key, subkey = jax.random.split(key)
        ifo.inject_signal(
            subkey,
            freqs,
            h_sky,
            detector_param,
            psd_file=psd_files[idx]  # the function load_psd actaully load asd
        )
    print("Signal injected")
    
    # Compute the SNR
    h1_snr = compute_snr(H1, h_sky, detector_param)
    l1_snr = compute_snr(L1, h_sky, detector_param)
    v1_snr = compute_snr(V1, h_sky, detector_param)
    
    network_snr = np.sqrt(h1_snr**2 + l1_snr**2 + v1_snr**2)
    
    print("H1 SNR:", h1_snr)
    print("L1 SNR:", l1_snr)
    print("V1 SNR:", v1_snr)
    print("Network SNR:", network_snr)

    print("Start prior setup")
    # priors without transformation 
    Mc_prior    = Uniform(0.8759659737275101, 2.6060030916165484, naming=['M_c'])
    s1z_prior   = Uniform(-0.05, 0.05, naming=['s1_z'])
    s2z_prior   = Uniform(-0.05, 0.05, naming=['s2_z'])
    lambda_1_prior = Uniform(0., 5000., naming=['lambda_1'])
    lambda_2_prior = Uniform(0., 5000., naming=['lambda_2'])
    dL_prior    = Uniform(30, 300, naming=['d_L'])
    tc_prior    = Uniform(-0.1, 0.1, naming=['t_c'])
    phic_prior  = Uniform(0., 2. * jnp.pi, naming=['phase_c'])
    psi_prior   = Uniform(0., jnp.pi, naming=["psi"])
    ra_prior    = Uniform(0., 2 * jnp.pi, naming=["ra"])
    # priors with transformations
    q_prior = Uniform(
        0.5,
        1,
        naming=['q'],
        transforms={
            'q': (
                'eta',
                lambda params: params['q'] / (1 + params['q']) ** 2
                )
            }
        )
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
    # compose the prior
    prior_list = [
            Mc_prior,
            q_prior,
            s1z_prior,
            s2z_prior,
            lambda_1_prior,
            lambda_2_prior,
            dL_prior,
            tc_prior,
            phic_prior,
            cos_iota_prior,
            psi_prior,
            ra_prior,
            sin_dec_prior,
    ]
    complete_prior = Composite(prior_list)
    bounds = jnp.array([[p.xmin, p.xmax] for p in complete_prior.priors])
    print("Finished prior setup")

    print("Initializing likelihood")
    likelihood = HeterodynedTransientLikelihoodFD(
        ifos,
        prior=complete_prior,
        bounds=bounds,
        n_bins=100,
        waveform=waveform,
        trigger_time=config["trigger_time"],
        duration=config["duration"],
        post_trigger_duration=config["post_trigger_duration"],
        ref_params=true_param
        )

    mass_matrix = jnp.eye(len(prior_list))
    for idx, prior in enumerate(prior_list):
        mass_matrix = mass_matrix.at[idx, idx].set(prior.xmax - prior.xmin) # fetch the prior range
    local_sampler_arg = {'step_size': mass_matrix * 1e-4} # set the step size to be 0.3% of the prior range

    hyperparameters["local_sampler_arg"] = local_sampler_arg
    
    print("outdir")
    print(outdir)

    jim = Jim(
        likelihood, 
        complete_prior,
        **hyperparameters
    )
    key = jax.random.PRNGKey(24)
    jim.sample(key)

    # === Show results, save output ===

    ### Summary to screen:
    jim.print_summary()

    # jim.Sampler.plot_summary("training")
    # jim.Sampler.plot_summary("production")
    name = outdir + f'results_production.npz'
    state = jim.Sampler.get_sampler_state(training = False)
    chains, log_prob, local_accs, global_accs = state["chains"], state["log_prob"], state["local_accs"], state["global_accs"]
    np.savez(name, chains=chains, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs)
    chains = jim.Sampler.sample_flow(10_000)
    name = outdir + 'results_NF.npz'
    np.savez(name, chains=chains)

    ### Plot chains and samples
    print("Creating plots")
    
    # Training plots
    state = jim.Sampler.get_sampler_state(training = True)
    local_accs, global_accs = state["local_accs"], state["global_accs"]
    local_accs = jnp.mean(local_accs, axis=0)
    global_accs = jnp.mean(global_accs, axis=0)
    utils.plot_accs(local_accs, "training", "local_accs_training", outdir)
    utils.plot_accs(global_accs, "training", "global_accs_training", outdir)
    
    # Production plots
    state = jim.Sampler.get_sampler_state(training = False)
    chains, log_prob, local_accs, global_accs = state["chains"], state["log_prob"], state["local_accs"], state["global_accs"]
    local_accs = jnp.mean(local_accs, axis=0)
    global_accs = jnp.mean(global_accs, axis=0)
    utils.plot_accs(local_accs, "production", "local_accs_production", outdir)
    utils.plot_accs(global_accs, "production", "global_accs_production", outdir)
    truths = true_param.values()
    utils.plot_chains(chains, truths, "chains", outdir)

    # TODO implement this again
    # print("Saving the jim hyperparameters")
    # jim.save_hyperparameters()
    
    print("Saving the NF")
    jim.Sampler.save_flow(outdir + "nf_model")

    # Remove in order to hopefully reduce memory?
    jim = None
    gc.collect()
    print("INJECTION RECOVERY FINISHED SUCCESSFULLY")
    
def main():
    
    ## Normal, new injection:
    # N = get_N(OUTDIR)
    # my_string = "================================================================================================================================================================================================================================================"
    # print(my_string)
    # print(f"Running injection script for  N = {N}")
    # print(my_string)
    # body(N, outdir=OUTDIR) # regular, computing on the fly
    
    # ### Rerun a specific injection
    body(144, outdir = OUTDIR, load_existing_config = True) 
    
    # jax.profiler.save_device_memory_profile("memory.prof")
    
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Time taken: {end-start} seconds ({(end-start)/60} minutes)")
    
