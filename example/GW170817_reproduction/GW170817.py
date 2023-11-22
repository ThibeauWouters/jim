# jim
from jimgw.jim import Jim
from jimgw.detector import H1, L1, V1
from jimgw.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD
from jimgw.prior import Uniform
# ripple
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD, gen_IMRPhenomD_hphc
from ripple import Mc_eta_to_ms
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
import pickle
from astropy.time import Time

### Data definitions

total_time_start = time.time()
gps = 1187008882.43
trigger_time = 1187008882.43
fmin = 23
fmax = 1792
minimum_frequency = fmin
maximum_frequency = fmax
T = 128
duration = T
post_trigger_duration = 2
epoch = duration - post_trigger_duration
gmst = GreenwichMeanSiderealTime(trigger_time)
f_ref = 20 
# gsmt = Time(trigger_time, format="gps").sidereal_time("apparent", "greenwich").rad

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

### Copy pasted form TurboPE script, but outdated code:
# def genWaveform(theta):
#     # Get the relevant parameters
#     theta_waveform = theta[:8]
#     # We set t_c to zero
#     theta_waveform = theta_waveform.at[5].set(0)
#     # Get extrinsic parameters
#     ra = theta[9]
#     dec = theta[10]
#     hp_test, hc_test = gen_IMRPhenomD_hphc(H1_frequency, theta_waveform, f_ref)
#     h_dict = {"p": hp_test, "c": hc_test}
#     params = {"ra": ra, "dec": dec, "gmst": gmst, "psi": 0}
#     # TODO gmst
#     # align_time = jnp.exp(-1j*2*jnp.pi*H1_frequency*(epoch+theta[5]))
#     h_test_H1 = H1.fd_response(H1_frequency, hp_test, hc_test, params) # * align_time
#     return h_test_H1

# def calculate_match_filter_SNR(theta):
#     # Separate waveformm parameters and set t_c to zero
#     theta_waveform = theta[:8]
#     theta_waveform = theta_waveform.at[5].set(0)
#     ra = theta[9]
#     dec = theta[10]
#     # Generate the sky waveform
#     hp_test, hc_test = gen_IMRPhenomD_hphc(H1_frequency, theta_waveform, f_ref)
#     h_dict = {"p": hp_test, "c": hc_test}
#     params = {"ra": ra, "dec": dec, "gmst": gmst, "psi": theta[8]}
#     # align_time = jnp.exp(-1j*2*jnp.pi*H1_frequency*(epoch+theta[5])) # TODO unused here?
#     h_test_H1 = H1.fd_response(H1_frequency, h_dict, params)# * align_time
#     h_test_L1 = L1.fd_response(L1_frequency, h_dict, params)# * align_time
#     h_test_V1 = V1.fd_response(V1_frequency, h_dict, params)# * align_time
#     # Get the inner product to compute SNR
#     df = H1_frequency[1] - H1_frequency[0]
#     match_filter_SNR_H1 = 4*jnp.sum((jnp.conj(H1_data)*h_test_H1)/H1_psd*df).real
#     match_filter_SNR_L1 = 4*jnp.sum((jnp.conj(L1_data)*h_test_L1)/L1_psd*df).real
#     match_filter_SNR_V1 = 4*jnp.sum((jnp.conj(V1_data)*h_test_V1)/V1_psd*df).real
#     optimal_SNR_H1 = 4*jnp.sum((jnp.conj(h_test_H1)*h_test_H1)/H1_psd*df).real
#     optimal_SNR_L1 = 4*jnp.sum((jnp.conj(h_test_L1)*h_test_L1)/L1_psd*df).real
#     optimal_SNR_V1 = 4*jnp.sum((jnp.conj(h_test_V1)*h_test_V1)/V1_psd*df).real
#     return match_filter_SNR_H1, match_filter_SNR_L1, match_filter_SNR_V1, optimal_SNR_H1, optimal_SNR_L1, optimal_SNR_V1


# def LogLikelihood(theta):
#     # TODO mostly duplicate from above?
#     # Do conversions (q to eta, cos iota to iota, cos dec to dec)
#     theta = theta.at[1].set(theta[1]/(1+theta[1])**2)
#     theta = theta.at[7].set(jnp.arccos(theta[7]))
#     theta = theta.at[10].set(jnp.arcsin(theta[10]))
#     # Separate waveform
#     theta_waveform = theta[:8]
#     theta_waveform = theta_waveform.at[5].set(0)
#     ra = theta[9]
#     dec = theta[10]
#     # Generate the sky waveform
#     hp_test, hc_test = gen_IMRPhenomD_hphc(H1_frequency, theta_waveform, f_ref)
#     h_dict = {"p": hp_test, "c": hc_test}
#     params = {"ra": ra, "dec": dec, "gmst": gmst, "psi": theta[8]}
#     # align_time = jnp.exp(-1j*2*jnp.pi*H1_frequency*(epoch+theta[5])) # TODO unused here?
#     h_test_H1 = H1.fd_response(H1_frequency, h_dict, params)# * align_time
#     h_test_L1 = L1.fd_response(L1_frequency, h_dict, params)# * align_time
#     h_test_V1 = V1.fd_response(V1_frequency, h_dict, params)# * align_time
#     # Get the inner product to compute SNR
#     df = H1_frequency[1] - H1_frequency[0]
#     match_filter_SNR_H1 = 4*jnp.sum((jnp.conj(H1_data)*h_test_H1)/H1_psd*df).real
#     match_filter_SNR_L1 = 4*jnp.sum((jnp.conj(L1_data)*h_test_L1)/L1_psd*df).real
#     match_filter_SNR_V1 = 4*jnp.sum((jnp.conj(V1_data)*h_test_V1)/V1_psd*df).real
#     optimal_SNR_H1 = 4*jnp.sum((jnp.conj(h_test_H1)*h_test_H1)/H1_psd*df).real
#     optimal_SNR_L1 = 4*jnp.sum((jnp.conj(h_test_L1)*h_test_L1)/L1_psd*df).real
#     optimal_SNR_V1 = 4*jnp.sum((jnp.conj(h_test_V1)*h_test_V1)/V1_psd*df).real

#     return (match_filter_SNR_H1-optimal_SNR_H1/2) + (match_filter_SNR_L1-optimal_SNR_L1/2) + (match_filter_SNR_V1-optimal_SNR_V1/2)

### Setting up the initial positions

# TODO double-check whether these are params before or after transformation


ref_params = jnp.array([1.19765181e+00, # Mc
                        2.30871959e-01, # eta (?)
                        2.89696686e-02, # spin1z
                        7.57299436e-02, # spin2z 
                        3.67225424e+01, # distance
                        4.97355831e-04, # t_c
                        4.94017055e+00, # phi_c
                        2.54597984e+00, # iota (?)
                        8.48429439e-01, # psi
                        3.40970408e+00, # ra
                        -3.42097428e-01]) # dec (?)
prior_range = jnp.array([[1.18,1.21],[0.125,1],[-0.05,0.05],[-0.05,0.05],[1,75],[-0.01,0.02],[0,2*np.pi],[-1,1],[0,np.pi],[0,2*np.pi],[-1,1]])
n_chains = 1000
n_dim = len(ref_params)
guess_param = ref_params
guess_param = np.array(jnp.repeat(guess_param[None,:],int(n_chains),axis=0) * np.random.normal(loc=1,scale=0.01,size=(int(n_chains), n_dim)))
guess_param[guess_param[:,1]>0.25,1] = 0.249 # limit eta to avoid singularity

rng_key_set = initialize_rng_keys(n_chains, seed=42)
initial_position = jax.random.uniform(rng_key_set[0], shape=(int(n_chains), n_dim)) * 1
for i in range(n_dim):
    initial_position = initial_position.at[:,i].set(initial_position[:,i]*(prior_range[i,1]-prior_range[i,0])+prior_range[i,0])

# m1,m2 = jax.vmap(Mc_eta_to_ms)(guess_param[:,:2])
# q = m2/m1

### TODO do we have to implement this?
# from astropy.cosmology import Planck18 as cosmo

# z = np.linspace(0.0002,0.03,10000)
# dL = cosmo.luminosity_distance(z).value
# dVdz = cosmo.differential_comoving_volume(z).value

# Prior
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

### Create likelihood object

# likelihood = TransientLikelihoodFD([H1, L1], waveform=RippleIMRPhenomD(), trigger_time=gps, duration=T, post_trigger_duration=T/2)
likelihood = HeterodynedTransientLikelihoodFD([H1, L1, V1], prior=prior, bounds=[prior.xmin, prior.xmax], waveform=RippleIMRPhenomD(), trigger_time=gps, duration=T, post_trigger_duration=T/2, n_bins=500, ref_params=ref_params)

### Create sampler and jim objects

# Mass matrix (this is copy pasted from the TurboPE set up)
eps = 3e-2
mass_matrix = jnp.eye(11)
mass_matrix = mass_matrix.at[0,0].set(1e-5)
mass_matrix = mass_matrix.at[1,1].set(1e-4)
mass_matrix = mass_matrix.at[2,2].set(1e-3)
mass_matrix = mass_matrix.at[3,3].set(1e-3)
mass_matrix = mass_matrix.at[5,5].set(1e-5)
mass_matrix = mass_matrix.at[9,9].set(1e-2)
mass_matrix = mass_matrix.at[10,10].set(1e-2)
local_sampler_arg = {"step_size": mass_matrix * eps}

outdir_name = "./outdir/"

jim = Jim(
    likelihood,
    prior,
    n_loop_pretraining=0,
    n_loop_training=20,
    n_loop_production=20,
    n_local_steps=200,
    n_global_steps=200,
    n_chains=n_chains,
    n_epochs=60,
    learning_rate=0.001,
    max_samples=50000,
    momentum=0.9,
    batch_size=50000,
    use_global=True,
    keep_quantile=0.0,
    train_thinning=40,
    output_thinning=1,
    local_sampler_arg=local_sampler_arg,
    outdir_name=outdir_name
)

### Heavy computation begins
jim.sample(jax.random.PRNGKey(24), initial_guess=initial_position) # start from our initial guess
### Heavy computation ends

### Postprocessing 

# Check the PE results
jim.print_summary()

# Create plots
print("Creating plots")
jim.Sampler.plot_summary("pretraining")
jim.Sampler.plot_summary("training")
jim.Sampler.plot_summary("production")

# Save output to files
samples_training = jim.get_samples("training")
samples_production = jim.get_samples("production")
samples_list = [samples_training, samples_production]
names = [outdir_name + 'samples_training_GW170817_IMRPhenomD.pickle', 
         outdir_name + 'samples_production_GW170817_IMRPhenomD.pickle']

for sample, name in zip(samples_list, names):
    print(f"Saving samples to {name}")
    with open(name, 'wb') as handle:
        pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Testing sampling from the flow")
flow_samples = jim.Sampler.sample_flow(10000)
name = outdir_name + 'flow_samples_GW170817_IMRPhenomD.pickle'
print(f"Saving flow samples to {name}")
with open(name, 'wb') as handle:
    pickle.dump(flow_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)