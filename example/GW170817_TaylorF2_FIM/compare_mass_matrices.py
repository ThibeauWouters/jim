# jim
from jimgw.jim import Jim
from jimgw.detector import H1, L1, V1
from jimgw.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD, RippleTaylorF2, RippleIMRPhenomD_NRTidalv2
from jimgw.prior import Uniform, Powerlaw, Composite 
from jimgw.fisher_information_matrix import FisherInformationMatrix, plot_ratio_comparison
# ripple
# flowmc
from flowMC.utils.PRNG_keys import initialize_rng_keys
# jax
import jax.numpy as jnp
import jax
# chosen_device = jax.devices()[1]
jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_default_device", chosen_device)
print(jax.devices())
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
from matplotlib.colors import LogNorm
import corner

### DIAGNOSES

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


load_online_data = False
labels = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda$', r'$\delta\Lambda$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

# waveform = RippleIMRPhenomD_NRTidalv2()
waveform = RippleTaylorF2()

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

assert np.allclose(H1_frequency, L1_frequency), "Frequencies are not the same for H1 and L1"
assert np.allclose(H1_frequency, V1_frequency), "Frequencies are not the same for H1 and V1"

frequencies = H1_frequency

### Getting ifos and overwriting with above data

if load_online_data:
    tukey_alpha = 2 / (duration / 2)

    H1.load_data(gps, duration, 2, fmin, fmax, psd_pad=16, tukey_alpha=tukey_alpha)
    L1.load_data(gps, duration, 2, fmin, fmax, psd_pad=16, tukey_alpha=tukey_alpha)
    V1.load_data(gps, duration, 2, fmin, fmax, psd_pad=16, tukey_alpha=tukey_alpha)

    H1.load_psd_from_file('../../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_H1_psd.txt')
    L1.load_psd_from_file('../../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_L1_psd.txt')
    V1.load_psd_from_file('../../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_V1_psd.txt')

else:
    H1.frequencies = H1_frequency
    H1.data = H1_data
    H1.psd = H1_psd 

    L1.frequencies = L1_frequency
    L1.data = L1_data
    L1.psd = L1_psd 

    V1.frequencies = V1_frequency
    V1.data = V1_data
    V1.psd = V1_psd 

### Define priors

# Internal parameters
Mc_prior = Uniform(1.18, 1.21, naming=["M_c"])
q_prior = Uniform(
    0.125,
    1.0,
    naming=["q"],
    transforms={"q": ("eta", lambda params: params["q"] / (1 + params["q"]) ** 2)},
)
s1z_prior                = Uniform(-0.05, 0.05, naming=["s1_z"])
s2z_prior                = Uniform(-0.05, 0.05, naming=["s2_z"])
# lambda_tilde_prior       = Uniform(0.0, 3000.0, naming=["lambda_tilde"])
# delta_lambda_tilde_prior = Uniform(-500.0, 500.0, naming=["delta_lambda_tilde"])
lambda1_prior = Uniform(0.0, 5000.0, naming=["lambda_1"])
lambda2_prior = Uniform(0.0, 5000.0, naming=["lambda_2"])

# External parameters
# dL_prior       = Uniform(0.0, 75.0, naming=["d_L"])
dL_prior       = Powerlaw(1.0, 75.0, 2.0, naming=["d_L"])
t_c_prior      = Uniform(-0.1, 0.1, naming=["t_c"])
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

prior = Composite([
        Mc_prior,
        q_prior,
        s1z_prior,
        s2z_prior,
        lambda1_prior,
        lambda2_prior,
        dL_prior,
        t_c_prior,
        phase_c_prior,
        cos_iota_prior,
        psi_prior,
        ra_prior,
        sin_dec_prior,
    ]
)

# The following only works if every prior has xmin and xmax property, which is OK for Uniform and Powerlaw
bounds = jnp.array([[p.xmin, p.xmax] for p in prior.priors]).T

### Create likelihood object

# TODO put the reference parameters here?
ref_params = {
    'M_c': 1.19755196,
    'eta': 0.23889171,
    's1_z': 0.04989052,
    's2_z': -0.02622163,
    'lambda_1': 60.80207005,
    'lambda_2': 316.70276899,
    'd_L': 11.88292643,
    't_c': -0.00154799,
    'phase_c': 3.66348683,
    'iota': 1.81371344,
    'psi': 1.6000364,
    'ra': 3.4179911,
    'dec': -0.40603519
}

n_bins = 100
likelihood = HeterodynedTransientLikelihoodFD([H1, L1, V1], prior=prior, bounds=bounds, waveform=waveform, trigger_time=gps, duration=T, n_bins=n_bins, ref_params=ref_params)

print("Running with n_bins  = ", n_bins)

### Create sampler and jim objects

eps = 1e-2
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

original_mass_matrix = mass_matrix * eps

## FIM calculations


fim = FisherInformationMatrix([H1, L1, V1], 
                              waveform = waveform, 
                              prior = prior,
                              trigger_time = gps, 
                              duration = T)

print("Calculating FIM")
fim.compute_fim(ref_params, frequencies)
print("Calculating FIM: DONE")

print("fim.fisher_information_matrix")
print(fim.fisher_information_matrix)

print("Calculating inverted FIM")
fim.invert()
print("Calculating inverted FIM: DONE")

print("fim.inverse")
print(fim.inverse)

print("Calculating inverted FIM")
tuned_mass_matrix = fim.get_tuned_mass_matrix()

naming = list(ref_params.keys())

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

plot_ratio_comparison(original_mass_matrix, tuned_mass_matrix, "Original", "Tuned", naming, use_ratio=True, outdir="./postprocessing/")
plot_ratio_comparison(original_mass_matrix, tuned_mass_matrix, "Original", "Tuned", naming, use_ratio=False, outdir="./postprocessing/")

print("DONE")