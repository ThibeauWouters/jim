import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"
# Regular imports
import time
from jimgw.jim import Jim
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomPv2
from jimgw.prior import Uniform, Composite, Sphere
import jax.numpy as jnp
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

###########################################
########## First we grab data #############
###########################################

import optax
n_epochs = 50
n_loop_training = 400
total_epochs = n_epochs * n_loop_training
start = int(total_epochs / 10)
start_lr = 1e-3
end_lr = 1e-5
power = 4.0
schedule_fn = optax.polynomial_schedule(start_lr, end_lr, power, total_epochs-start, transition_begin=start)

total_time_start = time.time()

# first, fetch a 4s segment centered on GW150914
gps = 1126259462.4
start = gps - 2
end = gps + 2
fmin = 20.0
fmax = 1024.0

ifos = ["H1", "L1"]

H1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)
L1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

waveform = RippleIMRPhenomPv2(f_ref=20)

###########################################
########## Set up priors ##################
###########################################

Mc_prior = Uniform(10.0, 80.0, naming=["M_c"])
q_prior = Uniform(
    0.125,
    1.0,
    naming=["q"],
    transforms={"q": ("eta", lambda params: params["q"] / (1 + params["q"]) ** 2)},
)
s1_prior = Sphere(naming="s1")
s2_prior = Sphere(naming="s2")
dL_prior = Uniform(0.0, 2000.0, naming=["d_L"])
t_c_prior = Uniform(-0.05, 0.05, naming=["t_c"])
phase_c_prior = Uniform(0.0, 2 * jnp.pi, naming=["phase_c"])
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
psi_prior = Uniform(0.0, jnp.pi, naming=["psi"])
ra_prior = Uniform(0.0, 2 * jnp.pi, naming=["ra"])
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

prior = Composite(
    [
        Mc_prior,
        q_prior,
        s1_prior,
        s2_prior,
        dL_prior,
        t_c_prior,
        phase_c_prior,
        cos_iota_prior,
        psi_prior,
        ra_prior,
        sin_dec_prior,
    ],
)

bounds = jnp.array(
    [
        [10.0, 80.0],
        [0.125, 1.0],
        [0, jnp.pi],
        [0, 2*jnp.pi],
        [0.0, 1.0],
        [0, jnp.pi],
        [0, 2*jnp.pi],
        [0.0, 1.0],
        [0.0, 2000.0],
        [-0.05, 0.05],
        [0.0, 2 * jnp.pi],
        [-1.0, 1.0],
        [0.0, jnp.pi],
        [0.0, 2 * jnp.pi],
        [-1.0, 1.0],
    ]
)
# likelihood = TransientLikelihoodFD([H1, L1], waveform=waveform, trigger_time=gps, duration=4, post_trigger_duration=2)
likelihood = HeterodynedTransientLikelihoodFD([H1, L1], prior=prior, bounds=bounds, waveform=waveform, trigger_time=gps, duration=4, post_trigger_duration=2)


mass_matrix = jnp.eye(prior.n_dim)
mass_matrix = mass_matrix.at[1, 1].set(1e-3)
mass_matrix = mass_matrix.at[9, 9].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix * 1e-4}

jim = Jim(
    likelihood,
    prior,
    n_loop_training=n_loop_training,
    n_loop_production=50,
    n_local_steps=5,
    n_global_steps=400,
    n_chains=1000,
    n_epochs=n_epochs,
    learning_rate=0.001,
    max_samples = 10000,
    momentum=0.9,
    batch_size=10000,
    use_global=True,
    keep_quantile=0.,
    train_thinning=1,
    output_thinning=30,
    num_layers=6,
    hidden_size=[64, 64],
    num_bins=8,
    local_sampler_arg=local_sampler_arg,
)

jim.sample(jax.random.PRNGKey(42))

jim.print_summary()

name = f'./outdir_PV2/results_production.npz'
state = jim.Sampler.get_sampler_state(training=False)
chains, log_prob, local_accs, global_accs = state["chains"], state[
        "log_prob"], state["local_accs"], state["global_accs"]
local_accs = jnp.mean(local_accs, axis=0)
global_accs = jnp.mean(global_accs, axis=0)
np.savez(name, chains=chains, log_prob=log_prob,
             local_accs=local_accs, global_accs=global_accs)