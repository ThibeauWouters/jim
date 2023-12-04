import time
from jimgw.jim import Jim
from jimgw.detector import H1, L1
from jimgw.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD
<<<<<<< HEAD:example/experimental/GW150914_tuned_mass_matrix.py
from jimgw.fisher_information_matrix import FisherInformationMatrix
from jimgw.prior import Uniform
=======
from jimgw.prior import Unconstrained_Uniform, Composite
>>>>>>> main:example/GW150914.py
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

###########################################
########## First we grab data #############
###########################################

total_time_start = time.time()

# first, fetch a 4s segment centered on GW150914
gps = 1126259462.4
duration = 4
<<<<<<< HEAD:example/experimental/GW150914_tuned_mass_matrix.py
start = gps - duration/2
end = gps + duration/2
=======
post_trigger_duration = 2
start_pad = duration - post_trigger_duration
end_pad = post_trigger_duration
>>>>>>> main:example/GW150914.py
fmin = 20.0
fmax = 1024.0

ifos = ["H1", "L1"]

<<<<<<< HEAD:example/experimental/GW150914_tuned_mass_matrix.py
H1.load_data(gps, duration/2, duration/2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)
L1.load_data(gps, duration/2, duration/2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

detectors = [H1, L1]
waveform = RippleIMRPhenomD()
=======
H1.load_data(gps, start_pad, end_pad, fmin, fmax, psd_pad=16, tukey_alpha=0.2)
L1.load_data(gps, start_pad, end_pad, fmin, fmax, psd_pad=16, tukey_alpha=0.2)
>>>>>>> main:example/GW150914.py

Mc_prior = Unconstrained_Uniform(10.0, 80.0, naming=["M_c"])
q_prior = Unconstrained_Uniform(
    0.125,
    1.0,
    naming=["q"],
    transforms={"q": ("eta", lambda params: params["q"] / (1 + params["q"]) ** 2)},
)
s1z_prior = Unconstrained_Uniform(-1.0, 1.0, naming=["s1_z"])
s2z_prior = Unconstrained_Uniform(-1.0, 1.0, naming=["s2_z"])
dL_prior = Unconstrained_Uniform(0.0, 2000.0, naming=["d_L"])
t_c_prior = Unconstrained_Uniform(-0.05, 0.05, naming=["t_c"])
phase_c_prior = Unconstrained_Uniform(0.0, 2 * jnp.pi, naming=["phase_c"])
cos_iota_prior = Unconstrained_Uniform(
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
psi_prior = Unconstrained_Uniform(0.0, jnp.pi, naming=["psi"])
ra_prior = Unconstrained_Uniform(0.0, 2 * jnp.pi, naming=["ra"])
sin_dec_prior = Unconstrained_Uniform(
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

prior = Composite(
    [
        Mc_prior,
        q_prior,
        s1z_prior,
        s2z_prior,
        dL_prior,
        t_c_prior,
        phase_c_prior,
        cos_iota_prior,
        psi_prior,
        ra_prior,
        sin_dec_prior,
    ]
)
likelihood = TransientLikelihoodFD(
    [H1, L1],
    waveform=RippleIMRPhenomD(),
    trigger_time=gps,
    duration=4,
    post_trigger_duration=2,
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

<<<<<<< HEAD:example/experimental/GW150914_tuned_mass_matrix.py
# We are going to compute the FIM at the point obtained by the Evolutionary optimizer
jim.maximize_likelihood([prior.xmin, prior.xmax], seed=42)
best_params = jim.best_fit_params
# naming=prior.get_naming(transform_name=True)
# Get the transformed and named values from the optimizer
named_params = jim.Prior.add_name(best_params, transform_name=True, transform_value=True)

print("Evaluating Fisher matrix at:")
print(named_params)

print("Computing Fisher information matrix")
fisher = FisherInformationMatrix(detectors, waveform, gps, duration, duration/2)
frequencies = H1.frequencies
mass_matrix = fisher.tune_mass_matrix(prior, RippleIMRPhenomD(), named_params, frequencies)

mass_matrix_diag = jnp.diag(mass_matrix)
print("mass_matrix_diag")
print(mass_matrix_diag)
s_vec = jnp.sqrt(mass_matrix_diag)
print("s_vec")
print(s_vec)

# TODO need to find a way to update the existing jim sampler, or to initialize it immediately with the tuned mass matrix 
print("--- Changing jim object to new mass matrix")
local_sampler_arg = {"step_size": mass_matrix * 1}

# TODO check this implementation, and improve jim source code to improve upon this
jim.Sampler.local_sampler.params = local_sampler_arg

# Start the sampling
=======
>>>>>>> main:example/GW150914.py
jim.sample(jax.random.PRNGKey(42))
jim.print_summary()