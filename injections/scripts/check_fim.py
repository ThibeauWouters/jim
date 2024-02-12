import os
import shutil
import sys
import json
from scipy.interpolate import interp1d
# jim
from jimgw.jim import Jim
from jimgw.detector import H1, L1, V1 
from jimgw.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.waveform import RippleTaylorF2, RippleIMRPhenomD_NRTidalv2
from jimgw.fisher_information_matrix import FisherInformationMatrix, plot_ratio_comparison, ORIGINAL_MASS_MATRIX
from jimgw.prior import Uniform, Powerlaw, Composite
# jax
import jax.numpy as jnp
import jax
# ripple
from ripple import Mc_eta_to_ms
# get available jax devices
nb_devices = len(jax.devices())
print(f"GPU available? {nb_devices} devices available, using final device from this list")
print(jax.devices())
chosen_device = jax.devices()[1]
print(f"chosen_device: {chosen_device}")
jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_default_device", chosen_device)
# others
import numpy as np
jax.config.update("jax_enable_x64", True)
from astropy.time import Time

import numpy as np
import pandas as pd
import corner
import matplotlib.pyplot as plt

import time

# import lalsimulation as lalsim

import gc

####################
### Script setup ###
####################

### Script constants
no_noise = False # whether to use noise in the injection
SNR_THRESHOLD = 0.001 # skip injections with SNR below this threshold
override_PSD = False # whether to load another PSD file -- unused now
use_lambda_tildes = False # whether to use lambda tildes instead of individual component lambdas or not
duration_with_lalsim = False # TODO check with Peter whether this is OK/useful?
waveform_approximant = "TaylorF2" # which waveform approximant to use, either TaylorF2 or IMRPhenomD_NRTidalv2
print(f"Waveform approximant: {waveform_approximant}")
OUTDIR = f"./outdir_{waveform_approximant}/"
# load_existing_config = True # whether to load an existing config file or generate a new one on the fly

### Script hyperparameters

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

# These are the labels that we use when plotting right after an injection has finished
if use_lambda_tildes:
    labels_results_plot = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\tilde{\Lambda}$', r'$\delta\tilde{\Lambda}$', r'$d_{\rm{L}}/{\rm Mpc}$',
                r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']
else:
    labels_results_plot = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda_1$', r'$\Lambda_2$', r'$d_{\rm{L}}/{\rm Mpc}$',
                r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

### Shared injection constants


HYPERPARAMETERS = {
    "flowmc": 
        {
            "n_loop_training": 150, 
            "n_loop_production": 50, 
            "n_local_steps": 200,
            "n_global_steps": 200, 
            "n_chains": 1000, 
            "n_epochs": 50, 
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


# For Mc prior: 0.870551 if lower bound is 1, 0.435275 if lower bound is 0.5
MC_PRIOR_1 = [0.8759659737275101, 2.6060030916165484] # lowest individual mass is 1
# MC_LOW_PRIOR = [0.435275, 0.870551] # NOTE only for testing low Mc values, arbitrarily chose to be between above 2 lower bounds

if use_lambda_tildes:
    NAMING = ['M_c', 'q', 's1_z', 's2_z', 'lambda_tilde', 'delta_lambda_tilde', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']
    PRIOR = {
        "M_c": MC_PRIOR_1,
        "q": [0.5, 1.0], 
        "s1_z": [-0.05, 0.05], 
        "s2_z": [-0.05, 0.05], 
        "lambda_tilde": [0.0, 9000.0], 
        "delta_lambda_tilde": [-1000.0, 1000.0], 
        "d_L": [30.0, 300.0], 
        "t_c": [-0.1, 0.1], 
        "phase_c": [0.0, 6.283185307179586], 
        "cos_iota": [-1.0, 1.0], 
        "psi": [0.0, 3.141592653589793], 
        "ra": [0.0, 6.283185307179586], 
        "sin_dec": [-1, 1]
    }
else:
        NAMING = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']
        PRIOR = {
        "M_c": MC_PRIOR_1,
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


###############################
### Main body of the script ###
###############################

def interp_psd(freqs, f_psd, psd_vals):
    psd = interp1d(f_psd, psd_vals, fill_value=(psd_vals[0], psd_vals[-1]))(freqs)
    return psd

def compute_fim(N, outdir, plot = False, save_matrix = True):

    naming = list(PRIOR.keys())
    prior_ranges = jnp.array([PRIOR[name] for name in naming])
    prior_low, prior_high = prior_ranges[:, 0], prior_ranges[:, 1]

    print("Loading existing config, path:")
    config_path = f"{outdir}injection_{str(N)}/config.json"
    config = json.load(open(config_path))
    outdir = config["outdir"]
    print(f"Outdir is set to {outdir}")
    no_noise = config["no_noise"]

    print(PRIOR.items())

    naming = list(PRIOR.keys())
    bounds = []
    for key, value in PRIOR.items():
        bounds.append(value)

    bounds = np.asarray(bounds)
    xmin = bounds[:, 0]
    xmax = bounds[:, 1]

    # Fetch the flowMC and jim hyperparameters, and put together into one dict:
    flowmc_hyperparameters = HYPERPARAMETERS["flowmc"]
    jim_hyperparameters = HYPERPARAMETERS["jim"]
    hyperparameters = {**flowmc_hyperparameters, **jim_hyperparameters}

    ### Data definitions

    gps = 1187008882.43
    trigger_time = gps
    gmst = Time(trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad
    fmin = 20
    f_ref = fmin 
    fmax = 2048
    f_sampling = 2 * fmax
    T = 256
    duration = T
    post_trigger_duration = 2

    ### Define priors
    print(f"Sampling from prior (for tidal parameters: use_lambda_tildes = {use_lambda_tildes})")
    
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
    
    if use_lambda_tildes:
        first_lambda_prior  = Uniform(xmin[4], xmax[4], naming=["lambda_tilde"])
        second_lambda_prior = Uniform(xmin[5], xmax[5], naming=["delta_lambda_tilde"])
    else:
        first_lambda_prior  = Uniform(xmin[4], xmax[4], naming=["lambda_1"])
        second_lambda_prior = Uniform(xmin[5], xmax[5], naming=["lambda_2"])
        

    # External parameters
    # dL_prior       = Powerlaw(xmin[6], xmax[6], 2.0, naming=["d_L"])
    dL_prior       = Uniform(xmin[6], xmax[6], naming=["d_L"])
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

    ### Compose the prior
    prior_list = [
            Mc_prior,
            q_prior,
            s1z_prior,
            s2z_prior,
            first_lambda_prior,
            second_lambda_prior,
            dL_prior,
            t_c_prior,
            phase_c_prior,
            cos_iota_prior,
            psi_prior,
            ra_prior,
            sin_dec_prior,
    ]
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
    
    T = 256
    duration = T
    epoch = duration - post_trigger_duration
    freqs = jnp.linspace(fmin, fmax, duration * f_sampling)

    ### Getting ifos and overwriting with above data

    detector_param = {"ra": true_params["ra"], 
                    "dec": true_params["dec"], 
                    "gmst": gmst, 
                    "psi": true_params["psi"], 
                    "epoch": epoch, 
                    "t_c": true_params["t_c"]}

    ### Inject signal, fetch PSD and overwrite

    waveform = RippleTaylorF2(f_ref=f_ref)
    h_sky = waveform(freqs, true_params)
    
    print("Injecting signal. No noise is set to: ", no_noise)
    key, subkey = jax.random.split(jax.random.PRNGKey(config["seed"] + 1234))
    H1.inject_signal(subkey, freqs, h_sky, detector_param, psd_file = "psd.txt", no_noise=no_noise)
    key, subkey = jax.random.split(key)
    L1.inject_signal(subkey, freqs, h_sky, detector_param, psd_file = "psd.txt", no_noise=no_noise)
    key, subkey = jax.random.split(key)
    V1.inject_signal(subkey, freqs, h_sky, detector_param, psd_file = "psd_virgo.txt", no_noise=no_noise)
    
    ### Create likelihood object

    
    if waveform_approximant == "IMRPhenomD_NRTidalv2":
        waveform = RippleIMRPhenomD_NRTidalv2(use_lambda_tildes=use_lambda_tildes)
    
    elif waveform_approximant == "TaylorF2":
        waveform = RippleTaylorF2(use_lambda_tildes=use_lambda_tildes)
        
    
    # print("Creating likelihood object")    
    # n_bins = 100
    # likelihood = HeterodynedTransientLikelihoodFD([H1, L1, V1], prior=prior, bounds=bounds, waveform=waveform, trigger_time=gps, duration=T, n_bins=n_bins, ref_params=true_params)
    # print("Creating likelihood object: done")

    ### Create sampler and jim objects
    eps = 1e-2
    
    n_dim = 13
    mass_matrix = jnp.eye(n_dim)
    mass_matrix = mass_matrix.at[0,0].set(1e-5)
    mass_matrix = mass_matrix.at[1,1].set(1e-4)
    mass_matrix = mass_matrix.at[2,2].set(1e-3)
    mass_matrix = mass_matrix.at[3,3].set(1e-3)
    mass_matrix = mass_matrix.at[7,7].set(1e-5)
    mass_matrix = mass_matrix.at[11,11].set(1e-2)
    mass_matrix = mass_matrix.at[12,12].set(1e-2)
    
    original_mass_matrix = mass_matrix * eps

    ### FIM calculations

    fim = FisherInformationMatrix([H1, L1, V1], 
                                  waveform = waveform, 
                                  prior = prior,
                                  trigger_time = gps, 
                                  duration = T,
                                  verbose=False)

    print("Calculating FIM")
    fim.compute_fim(true_params, H1.frequencies)
    print("Calculating FIM: DONE")

    # print("fim.fisher_information_matrix")
    # print(fim.fisher_information_matrix)

    print("Calculating inverted FIM")
    fim.invert()
    print("Calculating inverted FIM: DONE")

    # print("fim.inverse")
    # print(fim.inverse)

    print("Calculating inverted FIM")
    tuned_mass_matrix = fim.get_tuned_mass_matrix()

    naming = list(true_params.keys())

    print("original_mass_matrix")
    print(jnp.diag(original_mass_matrix))
    # plot_mass_matrix(original_mass_matrix, "original_mass_matrix")
    # plot_ratio(original_mass_matrix, "original_mass_matrix", naming)
    # plot_ratio(original_mass_matrix, "original_mass_matrix", naming, use_ratio = False)

    print("tuned_mass_matrix")
    print(jnp.diag(tuned_mass_matrix))
    # plot_mass_matrix(tuned_mass_matrix, "tuned_mass_matrix")
    # plot_ratio(tuned_mass_matrix, "tuned_mass_matrix", naming)
    # plot_ratio(tuned_mass_matrix, "tuned_mass_matrix", naming, use_ratio = False)
    
    # Save tuned mass matrix values to a txt file:
    if save_matrix:
        print("INFO: Saving tuned mass matrix to a txt file")
        fname = f"{outdir}/tuned_mass_matrix.txt"
        tuned_mass_matrix_diag = np.diag(tuned_mass_matrix)
        np.savetxt(fname, tuned_mass_matrix_diag)
        # with open(f"{outdir}/tuned_mass_matrix.txt", "w") as f:
        #     tuned_mass_matrix_diag = np.diag(tuned_mass_matrix_diag)
        #     f.write(str(list(tuned_mass_matrix_diag)))
        
    if plot:
        print("INFO: Plotting mass matrices")
        plot_ratio_comparison(original_mass_matrix, tuned_mass_matrix, "Original", "Tuned", naming, use_ratio=True, outdir=outdir)
        plot_ratio_comparison(original_mass_matrix, tuned_mass_matrix, "Original", "Tuned", naming, use_ratio=False, outdir=outdir)

    print("fim.eps:")
    print(fim.eps)
    
    # Write value of fim.eps to a txt file:
    with open(f"{outdir}/fim_eps.txt", "w") as f:
        f.write(str(fim.eps))

    print("DONE")
    
def check_inversion_errors():
    
    # Iterate over all subdirectories of OUTDIR, load eps.txt in each one, and make a histogram
    
    eps_values = []
    
    for subdir in os.listdir(OUTDIR):
        if os.path.isdir(os.path.join(OUTDIR, subdir)):
            # Check if they have fim_eps.txt, otherwise skip
            if not os.path.exists(os.path.join(OUTDIR, subdir, "fim_eps.txt")):
                continue
            with open(os.path.join(OUTDIR, subdir, "fim_eps.txt"), "r") as f:
                eps = float(f.read())
                eps_values.append(eps)
                
    eps_values = np.array(eps_values)
    
    # Make histogram
    nb_bins = 20
    bins = 10.0 ** (np.linspace(-5, 2, nb_bins))
    plt.hist(eps_values, bins=bins, histtype="step", density=True, linewidth=2)
    plt.xscale("log")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel("Density")
    plt.title("Histogram of $\epsilon$ values")
    plt.savefig(f"./postprocessing/fim_eps_histogram.png")
    plt.close()
    
def plot_all_fim(outdir, 
                 naming,
                 plot_original_matrix = True, 
                 use_ratio = False, 
                 alpha = 0.1,
                 plot_histograms = True):
    
    # Iterate over all directories in outdir, check if they have a tuned_mass_matrix.txt file, and save it
    
    fim_matrices = []
    for subdir in os.listdir(outdir):
        if os.path.isdir(os.path.join(outdir, subdir)):
            # Check if they have tuned_mass_matrix.txt, otherwise skip
            if not os.path.exists(os.path.join(outdir, subdir, "tuned_mass_matrix.txt")):
                continue
            fname = os.path.join(outdir, subdir, "tuned_mass_matrix.txt")
            matrix = np.loadtxt(fname)
            fim_matrices.append(matrix)
    
    fim_matrices = np.array(fim_matrices)
    print("np.shape(fim_matrices)")
    print(np.shape(fim_matrices))
    
    original = np.diag(ORIGINAL_MASS_MATRIX)
    
    plt.figure(figsize=(10, 10))
    
    if use_ratio:
        for i, matrix in enumerate(fim_matrices):
            fim_matrices[i] = matrix / matrix[0]
        ylabel = r"Step size ratio $\varepsilon / \varepsilon_{\mathcal{M}_c}$"
    else:
        ylabel = r"Step size $\varepsilon$"
    
    for i, matrix in enumerate(fim_matrices):
        if i == 0:
            plt.plot(matrix, "-o", alpha=alpha, color = "red", label = "Tuned")
        else:
            plt.plot(matrix, "-o", alpha=alpha, color = "red")
    plt.plot(original, "-o", color = "blue")
    plt.xticks(np.arange(len(naming)), naming, rotation=90)
    plt.legend()
    plt.yscale("log")
    plt.ylabel(ylabel)
    save_name = f"./postprocessing/fim_matrices_comparison.png"
    print(f"Saving plot to {save_name}")
    plt.savefig(save_name, bbox_inches = 'tight')
    plt.close()
    
    # if plot_histograms:
    #     for i, param_name in enumerate(naming):

        
def main():
    
    ### Computing FIM for our injections
    
    # # Get number of subdirectories of OUTDIR
    # N = len([name for name in os.listdir(OUTDIR) if os.path.isdir(os.path.join(OUTDIR, name))])
    # print(f"Number of injections: {N}")
    
    # # Iterate over subdirectories (go over N) and run body
    # for i in range(N):
    #     print(f"Running injection script for  N = {i+1}")
    #     compute_fim(i+1, outdir=OUTDIR, plot=False, save_matrix=True)
    #     print(f"Running injection script for  N = {i+1} DONE")
    #     gc.collect()
    
    ### Checking inversion errors
    # check_inversion_errors()
    
    ### Plotting FIM realizations across the injections
    plot_all_fim("./outdir_TaylorF2", NAMING, plot_original_matrix = True, use_ratio = False, alpha = 0.1)
    
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Time taken: {end-start} seconds ({(end-start)/60} minutes)")
    