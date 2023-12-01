
## OLD jim script


from jimgw.jim import Jim
from jimgw.detector import H1, L1, V1
from jimgw.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomPv2, RippleIMRPhenomD
from jimgw.prior import Uniform
from ripple import ms_to_Mc_eta
import jax.numpy as jnp
import jax
from astropy.time import Time

from tap import Tap
import yaml
from tqdm import tqdm
import numpy as np

jax.config.update("jax_enable_x64", True)


# TODO change param names
class InjectionRecoveryParser(Tap):
    config: str 
    
    # Noise parameters
    seed: int = 0
    f_sampling: int  = 4096
    duration: int = 4
    fmin: float = 20.0
    ifos: list[str]  = ["H1", "L1", "V1"]

    # Injection parameters
    m1: float = 30.0
    m2: float = 25.0
    s1_z: float = 0.04
    s2_z: float = 0.02
    dist_mpc: float = 400.
    tc: float = 0.
    phic: float = 0.1
    inclination: float = 0.5
    polarization_angle: float = 0.7
    ra: float = 1.2
    dec: float = 0.3

    # Sampler parameters
    n_dim: int = 11
    n_chains: int = 500
    n_loop_training: int = 200
    n_loop_production: int = 10
    n_local_steps: int = 200
    n_global_steps: int = 200
    learning_rate: float = 0.001
    max_samples: int = 60000
    momentum: float = 0.9
    num_epochs: int = 200
    batch_size: int = 60000
    stepsize: float = 0.01
    use_global: bool = True
    keep_quantile: float = 0.0
    train_thinning: int = 1
    output_thinning: int = 30
    num_layers: int = 6
    hidden_size: list[int] = [64,64]
    num_bins: int = 8

    # Output parameters
    output_path: str = "./outdir/"
    downsample_factor: int = 10

# Get config and start parse injection recovery

injection_file = "./configs/injection_config_0.yaml"
args = InjectionRecoveryParser().parse_args()

# Fetch noise parameters 

print("Constructing detectors")
print("Making noises")

#Fetch injection parameters and inject signal

print("Injection signals")

# Define frequency grid and time of trigger
# TODO get zero noise or artificial noise?
freqs = jnp.linspace(args.fmin, args.f_sampling/2, args.duration*args.f_sampling)
f_ref = args.fmin
trigger_time = 1126259462.4
post_trigger_duration = 2
epoch = args.duration - post_trigger_duration
gmst = Time(trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad

# Define prior
prior = Uniform(
    xmin = prior_low,
    xmax = prior_high,
    naming = ["M_c", "q", "s1_z", "s2_z", "d_L", "t_c", "phase_c", "cos_iota", "psi", "ra", "sin_dec"],
    transforms = {"q": ("eta", lambda params: params['q']/(1+params['q'])**2),
                 "cos_iota": ("iota",lambda params: jnp.arccos(jnp.arcsin(jnp.sin(params['cos_iota']/2*jnp.pi))*2/jnp.pi)),
                 "sin_dec": ("dec",lambda params: jnp.arcsin(jnp.arcsin(jnp.sin(params['sin_dec']/2*jnp.pi))*2/jnp.pi))} # sin and arcsin are periodize cos_iota and sin_dec
)

# Get true signal and detector params
Mc, _ = ms_to_Mc_eta(jnp.array([args.m1, args.m2]))
true_param = jnp.array([Mc, args.m2/args.m1, args.s1_z, args.s2_z, args.dist_mpc, args.tc, args.phic, args.inclination, args.polarization_angle, args.ra, args.dec])
true_param = prior.add_name(true_param, transform_name = True, transform_value = True)

print("true_param")
print(true_param)

detector_param = {"ra": args.ra, "dec": args.dec, "gmst": gmst, "psi": args.polarization_angle, "epoch": epoch, "t_c": args.tc}
waveform = RippleIMRPhenomD(f_ref=f_ref)
h_sky = waveform(freqs, true_param)

# Inject signal
# TODO grab PSD from somewhere
key, subkey = jax.random.split(jax.random.PRNGKey(args.seed+1234))
H1.inject_signal(subkey, freqs, h_sky, detector_param)
key, subkey = jax.random.split(key)
L1.inject_signal(subkey, freqs, h_sky, detector_param)
key, subkey = jax.random.split(key)
V1.inject_signal(subkey, freqs, h_sky, detector_param)

# Create likelihood object
# TODO add Virgo?
likelihood = TransientLikelihoodFD([H1, L1], waveform, trigger_time, args.duration, post_trigger_duration)
# likelihood = HeterodynedTransientLikelihoodFD([H1, L1, V1], prior=prior, bounds=[prior.xmin, prior.xmax],  waveform = waveform, trigger_time = trigger_time, duration = args.duration, post_trigger_duration = post_trigger_duration)

# Create jim object

mass_matrix = jnp.eye(args.n_dim)
mass_matrix = mass_matrix.at[1,1].set(1e-3)
mass_matrix = mass_matrix.at[9,9].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix*3e-3}

jim = Jim(likelihood, 
        prior,
        n_loop_pretraining=0,
        n_loop_training=args.n_loop_training,
        n_loop_production = args.n_loop_production,
        n_local_steps=args.n_local_steps,
        n_global_steps=args.n_global_steps,
        n_chains=args.n_chains,
        n_epochs=args.num_epochs,
        learning_rate = args.learning_rate,
        max_samples = args.max_samples,
        momentum = args.momentum,
        batch_size = args.batch_size,
        use_global=args.use_global,
        keep_quantile= args.keep_quantile,
        train_thinning = args.train_thinning,
        output_thinning = args.output_thinning,
        local_sampler_arg = local_sampler_arg,
        seed = args.seed,
        num_layers = args.num_layers,
        hidden_size = args.hidden_size,
        num_bins = args.num_bins
        )

key, subkey = jax.random.split(key)
jim.sample(subkey)
samples = jim.get_samples()

jim.print_summary()