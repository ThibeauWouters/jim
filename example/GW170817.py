import time
from jimgw.jim import Jim
from jimgw.detector import H1, L1
from jimgw.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD
from jimgw.prior import Uniform
import jax.numpy as jnp
import jax

from gwpy.timeseries import TimeSeries

jax.config.update("jax_enable_x64", True)

import pickle

# Quickly check presence of CUDA driver as sanity check

import jaxlib
print(jax.devices())

import numpy as np

###########################################
########## First we grab data #############
###########################################

total_time_start = time.time()

# GPS time for GW170817
gps = 1187008882.43

T = 128
psd_pad = 16
start = gps - T/2
end   = gps + T/2
fmin = 20.0
fmax = 2048.0

ifos = ["H1", "L1"]

H1.load_data(gps, T/2, T/2, fmin, fmax, psd_pad=psd_pad, tukey_alpha=0.2)
L1.load_data(gps, T/2, T/2, fmin, fmax, psd_pad=psd_pad, tukey_alpha=0.2)

# Priors and likelihood

eps = 1e-3

prior = Uniform(
    xmin=[1.18, 0.125, -1, -1,  0.0, -0.05,        0.0, -1.0,    0.0, 3.44616 - eps, np.sin(-0.408084) - eps],
    xmax=[1.21,   1.0,  1,  1, 70.0,  0.05, 2 * jnp.pi,  1.0, jnp.pi, 3.44616 + eps, np.sin(-0.408084) + eps],
    # xmin=[1.18, 0.125, -1, -1,  0.0, -0.05,        0.0, -1.0,    0.0, 0.0, -1.0],
    # xmax=[1.21,   1.0,  1,  1, 70.0,  0.05, 2 * jnp.pi,  1.0, jnp.pi, 2 * jnp.pi, 1.0],
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
    transforms = {
        "q": ("eta", lambda params: params['q']/(1+params['q'])**2),
        "cos_iota": ("iota",lambda params: jnp.arccos(jnp.arcsin(jnp.sin(params['cos_iota']/2*jnp.pi))*2/jnp.pi)),
        "sin_dec": ("dec",lambda params: jnp.arcsin(jnp.arcsin(jnp.sin(params['sin_dec']/2*jnp.pi))*2/jnp.pi)) # sin and arcsin are periodize cos_iota and sin_dec
        }
)
# likelihood = TransientLikelihoodFD([H1, L1], waveform=RippleIMRPhenomD(), trigger_time=gps, duration=T, post_trigger_duration=T/2)
likelihood = HeterodynedTransientLikelihoodFD([H1, L1], prior=prior, bounds=[prior.xmin, prior.xmax], waveform=RippleIMRPhenomD(), trigger_time=gps, duration=T, post_trigger_duration=T/2)

mass_matrix = jnp.eye(11)
mass_matrix = mass_matrix.at[1, 1].set(1e-3)
mass_matrix = mass_matrix.at[5, 5].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix * 3e-3}

jim = Jim(
    likelihood,
    prior,
    n_loop_training=50,
    n_loop_production=10,
    n_local_steps=150,
    n_global_steps=150,
    n_chains=100,
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

jim.maximize_likelihood([prior.xmin, prior.xmax])
jim.sample(jax.random.PRNGKey(42))

# Show results
result = jim.get_samples()
print(result)
jim.print_summary()

# ### Optional: save results externally
# print("Saving jim object (Normalizing flow)")
# jim.Sampler.save_flow("my_nf_TaylorF2_new")
# name ='samples.pickle'
# print(f"Saving samples to {name}")
# with open(name, 'wb') as handle:
#     pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
