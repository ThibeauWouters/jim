### This is for running on the LIGO CIT cluster
import psutil
p = psutil.Process()
p.cpu_affinity([0])

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.20"
### --------------------------------------------

import time
import jax
import jax.numpy as jnp

import numpy as np

from jimgw.jim import Jim
from jimgw.jim import Jim
from jimgw.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior,
    UniformSpherePrior,
)
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomD_NRTidalv2
from jimgw.transforms import BoundToUnbound
from jimgw.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    SphereSpinToCartesianSpinTransform,
    MassRatioToSymmetricMassRatioTransform,
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
)
from jimgw.single_event.utils import Mc_q_to_m1_m2
from flowMC.strategy.optimization import optimization_Adam

jax.config.update("jax_enable_x64", True)

###########################################
########## First we grab data #############
###########################################

total_time_start = time.time()

# first, fetch a 4s segment centered on GW150914

gps = 1187008882.43
trigger_time = gps
fmin = 20
fmax = 2048
minimum_frequency = fmin
maximum_frequency = fmax
duration = 128
post_trigger_duration = 2
epoch = duration - post_trigger_duration
f_ref = fmin

ifos = [H1, L1, V1]

### Load the pre-processed data

data_path = "./data/GW170817/"

# This is our preprocessed data obtained from the TXT files at the GWOSC website (the GWF gave me NaNs?)
H1.frequencies = jnp.array(np.genfromtxt(f'{data_path}H1_freq.txt'))
H1_data_re, H1_data_im = np.genfromtxt(f'{data_path}H1_data_re.txt'), np.genfromtxt(f'{data_path}H1_data_im.txt')
H1.data = jnp.array(H1_data_re + 1j * H1_data_im)

L1.frequencies = jnp.array(np.genfromtxt(f'{data_path}L1_freq.txt'))
L1_data_re, L1_data_im = np.genfromtxt(f'{data_path}L1_data_re.txt'), np.genfromtxt(f'{data_path}L1_data_im.txt')
L1.data = jnp.array(L1_data_re + 1j * L1_data_im)

V1.frequencies = jnp.array(np.genfromtxt(f'{data_path}V1_freq.txt'))
V1_data_re, V1_data_im = np.genfromtxt(f'{data_path}V1_data_re.txt'), np.genfromtxt(f'{data_path}V1_data_im.txt')
V1.data = jnp.array(V1_data_re + 1j * V1_data_im)

# Load the PSD

H1.psd = H1.load_psd(H1.frequencies, psd_file = data_path + "H1_psd.txt")
L1.psd = L1.load_psd(L1.frequencies, psd_file = data_path + "L1_psd.txt")
V1.psd = V1.load_psd(V1.frequencies, psd_file = data_path + "V1_psd.txt")


waveform = RippleIMRPhenomD_NRTidalv2(f_ref=f_ref)

###########################################
########## Set up priors ##################
###########################################

prior = []

# Mass prior
M_c_min, M_c_max = 1.18, 1.21
q_min, q_max = 0.125, 1.0
Mc_prior = UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"])
q_prior = UniformPrior(q_min, q_max, parameter_names=["q"])

prior = prior + [Mc_prior, q_prior]

# Spin prior
spin_min, spin_max = -0.05, 0.05
s1_prior = UniformPrior(spin_min, spin_max, parameter_names=["s1_z"])
s2_prior = UniformPrior(spin_min, spin_max, parameter_names=["s2_z"])
iota_prior = SinePrior(parameter_names=["iota"])

# Lambda prior
lambda_min, lambda_max = 0.0, 5000.0
lambda1_prior = UniformPrior(lambda_min, lambda_max, parameter_names=["lambda_1"])
lambda2_prior = UniformPrior(lambda_min, lambda_max, parameter_names=["lambda_2"])

prior = prior + [
    s1_prior,
    s2_prior,
    lambda1_prior,
    lambda2_prior,
    iota_prior,
]

# Extrinsic prior
dL_prior = PowerLawPrior(1.0, 75.0, 2.0, parameter_names=["d_L"])
t_c_prior = UniformPrior(-0.1, 0.1, parameter_names=["t_c"])
phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
dec_prior = CosinePrior(parameter_names=["dec"])

prior = prior + [
    dL_prior,
    t_c_prior,
    phase_c_prior,
    psi_prior,
    ra_prior,
    dec_prior,
]

prior = CombinePrior(prior)

# Defining Transforms

sample_transforms = [
    DistanceToSNRWeightedDistanceTransform(gps_time=gps, ifos=ifos, dL_min=dL_prior.xmin, dL_max=dL_prior.xmax),
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(gps_time=gps, ifo=ifos[0]),
    GeocentricArrivalTimeToDetectorArrivalTimeTransform(tc_min=t_c_prior.xmin, tc_max=t_c_prior.xmax, gps_time=gps, ifo=ifos[0]),
    SkyFrameToDetectorFrameSkyPositionTransform(gps_time=gps, ifos=ifos),
    BoundToUnbound(name_mapping = (["M_c"], ["M_c_unbounded"]), original_lower_bound=M_c_min, original_upper_bound=M_c_max),
    BoundToUnbound(name_mapping = (["q"], ["q_unbounded"]), original_lower_bound=q_min, original_upper_bound=q_max),
    BoundToUnbound(name_mapping = (["iota"], ["iota_unbounded"]) , original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = (["s1_z"], ["s1_z_unbounded"]) , original_lower_bound=spin_min, original_upper_bound=spin_max),
    BoundToUnbound(name_mapping = (["s2_z"], ["s2_z_unbounded"]) , original_lower_bound=spin_min, original_upper_bound=spin_max),
    BoundToUnbound(name_mapping = (["lambda_1"], ["lambda_1_unbounded"]) , original_lower_bound=lambda_min, original_upper_bound=lambda_max),
    BoundToUnbound(name_mapping = (["lambda_2"], ["lambda_2_unbounded"]) , original_lower_bound=lambda_min, original_upper_bound=lambda_max),
    BoundToUnbound(name_mapping = (["phase_det"], ["phase_det_unbounded"]), original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    BoundToUnbound(name_mapping = (["psi"], ["psi_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = (["zenith"], ["zenith_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = (["azimuth"], ["azimuth_unbounded"]), original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
]

likelihood_transforms = [
    MassRatioToSymmetricMassRatioTransform,
]

likelihood = HeterodynedTransientLikelihoodFD(ifos, 
                                              waveform=waveform, 
                                              n_bins = 1_000, 
                                              trigger_time=trigger_time, 
                                              duration=duration, 
                                              post_trigger_duration=post_trigger_duration, 
                                              prior = prior, 
                                              sample_transforms = sample_transforms, 
                                              likelihood_transforms = likelihood_transforms, 
                                              popsize = 10, 
                                              n_steps = 50)

mass_matrix = jnp.eye(prior.n_dim)
# mass_matrix = mass_matrix.at[1, 1].set(1e-3)
# mass_matrix = mass_matrix.at[9, 9].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix * 1e-3}

Adam_optimizer = optimization_Adam(n_steps=3000, learning_rate=0.01, noise_level=1)

import optax

n_epochs = 20
n_loop_training = 100
n_loop_production = 10
total_epochs = n_epochs * n_loop_training
start = total_epochs // 10
learning_rate = optax.polynomial_schedule(
    1e-3, 1e-4, 4.0, total_epochs - start, transition_begin=start
)

jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    n_loop_training=n_loop_training,
    n_loop_production=n_loop_production,
    n_local_steps=10,
    n_global_steps=1000,
    n_chains=1_000,
    n_epochs=n_epochs,
    learning_rate=learning_rate,
    n_max_examples=30000,
    n_flow_sample=100000,
    momentum=0.9,
    batch_size=30000,
    use_global=True,
    keep_quantile=0.0,
    train_thinning=1,
    output_thinning=10,
    local_sampler_arg=local_sampler_arg,
    # strategies=[Adam_optimizer,"default"],
)


jim.sample(jax.random.PRNGKey(42))
jim.print_summary()

output_samples = jim.get_samples()

jnp.savez("../../jim_testing/integration/output_samples.npz", **output_samples)

print("DONE")