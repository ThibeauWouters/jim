import time
from jimgw.jim import Jim
from jimgw.detector import H1, L1
from jimgw.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD
from jimgw.fisher_information_matrix import FisherInformationMatrix
from jimgw.prior import Uniform
import jax.numpy as jnp
import jax

from flowMC.sampler import MMALA, Sampler

jax.config.update("jax_enable_x64", True)

###########################################
########## First we grab data #############
###########################################

total_time_start = time.time()

# first, fetch a 4s segment centered on GW150914
gps = 1126259462.4
duration = 4
start = gps - duration/2
end = gps + duration/2
fmin = 20.0
fmax = 1024.0

ifos = ["H1", "L1"]

H1.load_data(gps, duration/2, duration/2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)
L1.load_data(gps, duration/2, duration/2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

detectors = [H1, L1]
waveform = RippleIMRPhenomD()

prior = Uniform(
    xmin=[10, 0.125, -1.0, -1.0, 0.0, -0.05, 0.0, -1, 0.0, 0.0, -1.0],
    xmax=[80.0, 1.0, 1.0, 1.0, 2000.0, 0.05, 2 * jnp.pi, 1.0, jnp.pi, 2 * jnp.pi, 1.0],
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

likelihood = TransientLikelihoodFD([H1, L1], waveform=waveform, trigger_time=gps, duration=duration, post_trigger_duration=duration/2)
# likelihood = HeterodynedTransientLikelihoodFD([H1, L1], prior=prior, bounds=[prior.xmin, prior.xmax], waveform=waveform, trigger_time=gps, duration=duration, post_trigger_duration=duration/2)

mass_matrix = jnp.eye(11)
mass_matrix = mass_matrix.at[1, 1].set(1e-3)
mass_matrix = mass_matrix.at[5, 5].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix * 3e-3}

jim = Jim(
    likelihood,
    prior,
    n_loop_training=100,
    n_loop_production=10,
    n_local_steps=150,
    n_global_steps=150,
    n_chains=500,
    n_epochs=50,
    learning_rate=0.001,
    max_samples=45000,
    momentum=0.9,
    batch_size=50000,
    use_global=True,
    keep_quantile=0.0,
    train_thinning=1,
    output_thinning=10,
    local_sampler_arg=local_sampler_arg,
)

# We are going to compute the FIM at the point obtained by the Evolutionary optimizer
jim.maximize_likelihood([prior.xmin, prior.xmax], seed=42)
best_params = jim.best_fit_params
naming=prior.get_naming(transform_name=True)
# Get the transformed and named values from the optimizer
named_params = jim.Prior.add_name(best_params, transform_name=True, transform_value=True)

print("Evaluating Fisher matrix at:")
print(named_params)

print("Computing Fisher information matrix")
fisher = FisherInformationMatrix(detectors, waveform, gps, duration, duration/2)
frequencies = H1.frequencies
mass_matrix = fisher.tune_mass_matrix(prior, RippleIMRPhenomD(), named_params, frequencies)

fisher_information_matrix = fisher.fisher_information_matrix
tc_index = naming.index("t_c")
fisher_information_matrix = fisher_information_matrix.at[tc_index, tc_index].set(1e-5)

print("fisher_information_matrix")
print(fisher_information_matrix)

print("--- New object with MMALA")

fisher_information_matrix = fisher_information_matrix.at[tc_index, tc_index].set(1e-5)
matrix = jnp.diag(jnp.diag(fisher_information_matrix))
print(matrix)

mass_matrix = jnp.eye(11)
mass_matrix = mass_matrix.at[1, 1].set(1e-3)
mass_matrix = mass_matrix.at[5, 5].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix * 3e-3}

jim = Jim(
    likelihood,
    prior,
    G=matrix,
    n_loop_training=100,
    n_loop_production=50,
    n_local_steps=150,
    n_global_steps=150,
    n_chains=500,
    n_epochs=50,
    learning_rate=0.001,
    max_samples=45000,
    momentum=0.9,
    batch_size=50000,
    use_global=True,
    keep_quantile=0.0,
    train_thinning=1,
    output_thinning=10,
    local_sampler_arg=local_sampler_arg
)


# Start the sampling
jim.sample(jax.random.PRNGKey(42))
jim.print_summary()