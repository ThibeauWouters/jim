import os 
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import psutil
p = psutil.Process()
p.cpu_affinity([0])
# jim
from jimgw.jim import Jim
from jimgw.single_event.detector import H1, L1, V1, Detector
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomD, RippleTaylorF2
from jimgw.prior import Uniform, PowerLaw, Composite 
# ripple
# flowmc
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.utils.postprocessing import plot_summary
# jax
import jax.numpy as jnp
import jax
print(jax.devices())
# chosen_device = jax.devices()[1]
# jax.config.update("jax_platform_name", "gpu")
# jax.config.update("jax_default_device", chosen_device)
# others
import time
import numpy as np
jax.config.update("jax_enable_x64", True)
from astropy.time import Time

# import urllib.request
import shutil
import numpy as np
import matplotlib.pyplot as plt
import corner

from jaxtyping import Float

smart_initial_guess = True
clean_outdir = False

### 
### UTILITY FUNCTIONALITIES
###

def reflect_sky_location(
    gmst: Float,
    detectors: list[Detector],
    ra: Float,
    dec: Float,
    tc: Float,
    iota: Float
    ) -> tuple[Float, Float, Float]:

    assert len(detectors) == 3, "This reflection only holds for a 3-detector network"

    # convert tc to radian
    tc_rad = tc / (24 * 60 * 60) * 2 * jnp.pi

    # source location in cartesian coordinates
    # with the geocentric frame, thus the shift in ra by gmst
    v = jnp.array([jnp.cos(dec) * jnp.cos(ra - gmst - tc_rad),
                   jnp.cos(dec) * jnp.sin(ra - gmst - tc_rad),
                   jnp.sin(dec)])

    # construct the detector plane
    # fetch the detectors' locations
    x, y, z = detectors[0].vertex, detectors[1].vertex, detectors[2].vertex
    # vector normal to the detector plane
    n = jnp.cross(y - x, z - x)
    # normalize the vector
    nhat = n / jnp.sqrt(jnp.dot(n, n))
    # parametrize v as v = v_n * nhat + v_p, where nhat * v_p = 0
    v_n = jnp.dot(v, nhat)
    v_p = v - v_n * nhat
    # get the plan-reflect location
    v_ref = -v_n * nhat + v_p
    # convert back to ra, dec
    # adjust ra_prime so that it is in [0, 2pi)
    # i.e., negative values are map to [pi, 2pi)
    ra_prime = jnp.arctan2(v_ref[1], v_ref[0])
    ra_prime = ra_prime - (jnp.sign(ra_prime) - 1) * jnp.pi
    ra_prime = ra_prime + gmst + tc_rad  # add back the gps time and tc
    ra_prime = jnp.mod(ra_prime, 2 * jnp.pi)
    dec_prime = jnp.arcsin(v_ref[2])

    # calculate the time delay
    # just pick the first detector
    old_time_delay = detectors[0].delay_from_geocenter(ra, dec, gmst + tc_rad)
    new_time_delay = detectors[0].delay_from_geocenter(ra_prime, dec_prime,
                                                       gmst + tc_rad)
    tc_prime = tc + old_time_delay - new_time_delay
    
    # Also flip iota
    iota_prime = jnp.pi - iota

    return ra_prime, dec_prime, tc_prime, iota_prime

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

labels_with_tc = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda$', r'$\delta\Lambda$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$t_c$', r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

### Data definitions

total_time_start = time.time()
gps = 1187008882.43
trigger_time = gps
gmst = Time(trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad
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
s1z_prior = Uniform(-0.05, 0.05, naming=["s1_z"])
s2z_prior = Uniform(-0.05, 0.05, naming=["s2_z"])
lambda_1_prior = Uniform(0.0, 5000.0, naming=["lambda_1"])
lambda_2_prior = Uniform(0.0, 5000.0, naming=["lambda_2"])

# External parameters
dL_prior       = Uniform(0.0, 75.0, naming=["d_L"])
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
        lambda_1_prior,
        lambda_2_prior,
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
bounds = jnp.array([[p.xmin, p.xmax] for p in prior.priors])
prior_ranges = jnp.array([bound[1] - bound[0] for bound in bounds])

### Create likelihood object

# ref_params = {'M_c': 1.19754357, 
#               'eta': 0.24984541, 
#               's1_z': -0.00429651, 
#               's2_z': 0.00470304, 
#               'lambda_1': 1816.51300368, 
#               'lambda_2': 0.10161503, 
#               'd_L': 10.87770389, 
#               't_c': 0.00864911, 
#               'phase_c': 4.33436689, 
#               'iota': 1.59216065, 
#               'psi': 1.69112445, 
#               'ra': 5.08658471, 
#               'dec': 0.47136332
# }

ref_params = {'M_c': 1.19754357, 
              'eta': 0.24984541, 
              's1_z': -0.00429651, 
              's2_z': 0.00470304, 
              'lambda_1': 1816.51300368, 
              'lambda_2': 0.10161503, 
              'd_L': 10.87770389, 
              't_c': 0.00864911, 
              'phase_c': 4.33436689, 
              'iota': 2.5, # 1.59216065
              'psi': 1.69112445, 
              'ra': 2.2, # 5.08658471
              'dec': -1.25  # 0.47136332
}


# NOTE I am checking whether 100 bins also gives fine results or not
n_bins = 100
likelihood = HeterodynedTransientLikelihoodFD([H1, L1, V1], prior=prior, bounds=bounds, waveform=RippleTaylorF2(f_ref=f_ref), trigger_time=gps, duration=T, n_bins=n_bins, ref_params=ref_params)

print("Running with n_bins  = ", n_bins)

### Create sampler and jim objects

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

outdir_name = "./outdir/"

jim = Jim(
    likelihood,
    prior,
    n_loop_pretraining=0,
    n_loop_training=200,
    n_loop_production=20,
    n_local_steps=10,
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

# TODO Distribute the walkers for the initial guess

if smart_initial_guess:
    print("Going to initialize the walkers")

    # Fix the indices
    tc_index = 7
    iota_index = 9
    ra_index = 11
    dec_index = 12
    special_idx = [0, 6, tc_index, iota_index, ra_index, dec_index]
    other_idx = [i for i in range(n_dim) if i not in special_idx]
    
    # Start new PRNG key
    my_seed = 1234556
    my_key = jax.random.PRNGKey(my_seed)
    my_key, subkey = jax.random.split(my_key)

    # Mc
    z = jax.random.normal(subkey, shape = (int(n_chains),))
    mc_mean = ref_params["M_c"]
    mc_std = 0.1 
    mc_samples = mc_mean + mc_std * z

    ### Sample for ra, dec, tc, iota
    # TODO make less cumbersome here!
    assert n_chains % 2 == 0, "n_chains must be multiple of two"
    n_chains_half = int(n_chains // 2) 
    my_key, subkey = jax.random.split(my_key)
    z = jax.random.normal(subkey, shape = (int(n_chains_half), 4))
    my_key, subkey = jax.random.split(my_key)
    z_prime = jax.random.normal(subkey, shape = (int(n_chains_half), 4))
    sky_std = 0.1 * jnp.array([prior_ranges[ra_index], prior_ranges[dec_index], prior_ranges[tc_index], prior_ranges[iota_index]])

    # True sky location
    ra, dec, tc, iota = ref_params["ra"], ref_params["dec"], ref_params["t_c"], ref_params["iota"]
    sky_means = jnp.array([ra, dec, tc, iota])
    sky_samples = sky_means + sky_std * z
    ra_samples, dec_samples, tc_samples, iota_samples = sky_samples[:,0], sky_samples[:,1], sky_samples[:,2], sky_samples[:,3]

    # Reflected sky location
    ra_prime, dec_prime, tc_prime, iota_prime = reflect_sky_location(gmst, [H1, L1, V1], ra, dec, tc, iota)
    sky_means_prime = jnp.array([ra_prime, dec_prime, tc_prime, iota_prime])
    sky_samples_prime = sky_means_prime + sky_std * z_prime
    ra_samples_prime, dec_samples_prime, tc_samples_prime, iota_samples_prime = sky_samples_prime[:,0], sky_samples_prime[:,1], sky_samples_prime[:,2], sky_samples_prime[:,3]

    # Merge original and reflected samples
    merged_ra = jnp.concatenate([ra_samples, ra_samples_prime], axis=0)
    merged_dec = jnp.concatenate([dec_samples, dec_samples_prime], axis=0)
    merged_tc = jnp.concatenate([tc_samples, tc_samples_prime], axis=0)
    merged_iota = jnp.concatenate([iota_samples, iota_samples_prime], axis=0)

    # dL samples with powerlaw
    my_key, subkey = jax.random.split(my_key)
    dL_samples = dL_prior.sample(subkey, n_chains)
    # Convert to jnp array
    dL_samples = jnp.array(dL_samples["d_L"])

    # Rest of samples is uniform
    uniform_samples = jax.random.uniform(subkey, shape = (int(n_chains), n_dim - 6))
    for i, idx in enumerate(other_idx):
        # Get the relevant shift for this parameter, param is fetched by idx
        shift = prior_ranges[idx]
        # At this unifor/m samples, set the value using the shifts
        uniform_samples = uniform_samples.at[:,i].set(bounds[idx, 0] + uniform_samples[:,i] * shift)

    # Now build up the initial guess
    initial_guess = jnp.array([mc_samples, # Mc, 0
                            uniform_samples[:,0], # q, 1
                            uniform_samples[:,1], # chi1, 2
                            uniform_samples[:,2], # chi2, 3
                            uniform_samples[:,3], # lambda1, 4
                            uniform_samples[:,4], # lambda2, 5
                            dL_samples, # dL, 6
                            merged_tc, # t_c, 7
                            uniform_samples[:,5], # phase_c, 8
                            jnp.cos(merged_iota), # cos_iota, 9
                            uniform_samples[:,6], # psi, 10
                            merged_ra, # ra, 11
                            jnp.sin(merged_dec), # sin_dec, 12
                            ]).T

    # Make a corner plot
    print("Going to plot the walkers")
    initial_guess_numpy = np.array(initial_guess)
    ref_params["iota"] = jnp.cos(ref_params["iota"])
    ref_params["dec"] = jnp.sin(ref_params["dec"])
    truths = np.array([ref_params[p] for p in ref_params if p != "gmst"])
    print("Plotting the initial guess")
    fig = corner.corner(initial_guess_numpy, labels = labels_with_tc, truths = truths, hist_kwargs={'density': True}, **default_corner_kwargs)
    # Save
    fig.savefig(outdir_name + "initial_guess_corner.png", bbox_inches='tight')
else:
    initial_guess = None

### Heavy computation begins
jim.sample(jax.random.PRNGKey(42), initial_guess=initial_guess)
### Heavy computation ends

# === Show results, save output ===

if clean_outdir:
    print("Cleaning outdir")
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


# TODO - save the NF object to sample from later on
# print("Saving jim object (Normalizing flow)")
# jim.Sampler.save_flow("my_nf_IMRPhenomD")

### Write to 
name = outdir_name + f'results_production.npz'
print(f"Saving production samples in npz format to {name}")
state = jim.Sampler.get_sampler_state(training = False)
chains, log_prob, local_accs, global_accs = state["chains"], state["log_prob"], state["local_accs"], state["global_accs"]
np.savez(name, chains=chains, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs)

print("Sampling from the flow")
chains = jim.Sampler.sample_flow(10000)
name = outdir_name + 'results_NF.npz'
print(f"Saving flow samples to {name}")
np.savez(name, chains=chains)

### Plot chains and samples

# Production samples:
print("Going to plot the chains")
file = outdir_name + "results_production.npz"
name = outdir_name + "results_production.png"

data = np.load(file)
idx_list = [0,1,2,3,4,5,6,7,8,9,10,11,12]
chains = data['chains'][:,:,idx_list].reshape(-1, len(idx_list))
chains[:,9] = np.arccos(chains[:,9])
chains[:,12] = np.arcsin(chains[:,12])
chains = np.asarray(chains)
corner_kwargs = default_corner_kwargs
fig = corner.corner(chains, labels = labels, hist_kwargs={'density': True}, **default_corner_kwargs)
fig.savefig(name, bbox_inches='tight')  

# Production samples:
print("Going to plot the NF chains")
file = outdir_name + "results_NF.npz"
name = outdir_name + "results_NF.png"

data = np.load(file)["chains"]
print("np.shape(data)")
print(np.shape(data))

# # TODO improve the following: ignore t_c, and reshape with n_dims, and do conversions
# chains = data[:, idx_list]
# chains = np.asarray(chains)
# corner_kwargs = default_corner_kwargs
# fig = corner.corner(chains, labels = labels, hist_kwargs={'density': True}, **default_corner_kwargs)
# fig.savefig(name, bbox_inches='tight')  
    
# print("Saving the hyperparameters")
# jim.save_hyperparameters()

print("Creating plots for flowMC")
plot_summary(jim.Sampler, "production")