# jim
from jimgw.jim import Jim
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.single_event.waveform import RippleTaylorF2, RippleIMRPhenomD_NRTidalv2
from jimgw.prior import Uniform, PowerLaw, Composite
# ripple
# flowmc
from flowMC.utils.PRNG_keys import initialize_rng_keys
# jax
import jax.numpy as jnp
import jax
chosen_device = jax.devices()[1]
jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_default_device", chosen_device)
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
waveform = RippleIMRPhenomD_NRTidalv2(f_ref=f_ref)

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
    
assert np.allclose(H1_frequency, L1_frequency), "Frequencies are not the same for H1 and L1"
assert np.allclose(H1_frequency, V1_frequency), "Frequencies are not the same for H1 and V1"

frequencies = H1_frequency

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
lambda1_prior = Uniform(0.0, 5000.0, naming=["lambda_1"])
lambda2_prior = Uniform(0.0, 5000.0, naming=["lambda_2"])

# External parameters
dL_prior       = PowerLaw(1.0, 75.0, 2.0, naming=["d_L"])
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
ref_params = None
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
local_sampler_arg = {"step_size": mass_matrix * eps}

outdir_name = "./outdir/"

jim = Jim(
    likelihood,
    prior,
    n_loop_pretraining=0,
    n_loop_training=100,
    n_loop_production=20,
    n_local_steps=200,
    n_global_steps=200,
    n_chains=n_chains,
    n_epochs=100,
    learning_rate=0.001,
    max_samples=50000,
    momentum=0.9,
    batch_size=50000,
    use_global=True,
    keep_quantile=0.0,
    train_thinning=10,
    output_thinning=30,    
    n_loops_maximize_likelihood = 2000,
    local_sampler_arg=local_sampler_arg,
    outdir_name=outdir_name
)

start = time.time()
### Heavy computation begins
jim.sample(jax.random.PRNGKey(42))
### Heavy computation ends
end = time.time()

# Print time in minutes
print("Total time taken = ", (end - start) / 60, " minutes")

# === Show results, save output ===

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

### Summary
jim.print_summary()

### Diagnosis plots of summaries
print("Creating plots")
jim.Sampler.plot_summary("training")
jim.Sampler.plot_summary("production")

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

# Production samples:
file = outdir_name + "results_production.npz"
name = outdir_name + "results_production.png"

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
file = outdir_name + "results_NF.npz"
name = outdir_name + "results_NF.png"

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
