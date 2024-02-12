import psutil
p = psutil.Process()
p.cpu_affinity([0])

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import pandas as pd

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from jimgw.jim import Jim
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.single_event.waveform import RippleTaylorF2
from jimgw.prior import Uniform, PowerLaw, AlignedSpin, Composite

from ripple import ms_to_Mc_eta

from astropy.time import Time

from tap import Tap

class InjectionRecoveryParser(Tap):
    # Noise parameters
    noise_seed: int = 8010 + 1234
    sampling_seed: int = 42
    f_sampling: int  = 4096
    duration: int = 256
    fmin: float = 20.0
    fref: float = 20.0
    ifos: list[str]  = ["H1", "L1", "V1"]
    psds: list[str]  = ["./psd.txt",
                        "./psd.txt",
                        "./psd_virgo.txt"]

    # Injection parameters
    M_c: float = 1.6241342228137727
    q: float = 0.668298338249546
    s1_z: float = -0.04322900906979665
    s2_z: float = 0.0018364993350326042
    lambda_1: float = 2705.980957696422
    lambda_2: float = 3284.755794210511
    dist_mpc: float = 195.15224544041467
    tc: float = 0.06399874286352789
    phic: float = 0.7458447627099869
    iota: float = 0.4
    psi: float = 2.7807861411019457
    ra: float = 5.236983876031977
    dec: float = float(jnp.arcsin(-0.6758174992722121))
    trigger_time: float = 1187008882.43
    post_trigger_duration: float = 2.0

    # Sampler parameters
    n_chains: int = 1000
    n_loop_training: int = 400
    n_loop_production: int = 20
    n_local_steps: int = 200
    n_global_steps: int = 200
    learning_rate: float = 0.001
    max_samples: int = 50000
    momentum: float = 0.9
    num_epochs: int = 50
    batch_size: int = 50000
    stepsize: float = 0.01
    use_global: bool = True
    keep_quantile: float = 0.0
    train_thinning: int = 10
    output_thinning: int = 30
    num_layers: int = 10
    hidden_size: list[int] = [128,128]
    num_bins: int = 8

    # Output parameters
    output_path: str = "./output/"
    downsample_factor: int = 10


args = InjectionRecoveryParser().parse_args()

print("Injecting signals")
# setup waveform
waveform = RippleTaylorF2(f_ref=args.fref)
# key for noise generation
key = jax.random.PRNGKey(args.noise_seed)
# creating frequency grid
freqs = jnp.arange(
    args.fmin,
    args.f_sampling / 2,  # maximum frequency being halved of sampling frequency
    1. / args.duration
    )
# convert injected mass ratio to eta
eta = args.q / (1 + args.q) ** 2
# setup the timing setting for the injection
epoch = args.duration - args.post_trigger_duration
print("args.trigger_time")
print(args.trigger_time)
gmst = Time(args.trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad
print("gmst")
print(gmst)
# array of injection parameters
true_param = {
    'M_c':       args.M_c,       # chirp mass
    'eta':       eta,            # symmetric mass ratio 0 < eta <= 0.25
    's1_z':      args.s1_z,      # aligned spin of priminary component s1_z.
    's2_z':      args.s2_z,      # aligned spin of secondary component s2_z.
    'lambda_1':  args.lambda_1,  # tidal deformability of priminary component lambda_1.
    'lambda_2':  args.lambda_2,  # tidal deformability of secondary component lambda_2.
    'd_L':       args.dist_mpc,  # luminosity distance
    't_c':       args.tc,        # timeshift w.r.t. trigger time
    'phase_c':   args.phic,      # merging phase
    'iota':      args.iota,
    'psi':       args.psi,
    'ra':        args.ra,
    'dec':       args.dec
    }
detector_param = {
    'ra':     args.ra,
    'dec':    args.dec,
    'gmst':   gmst,
    'psi':    args.psi,
    'epoch':  epoch,
    't_c':    args.tc
    }
print(f"The injected parameters are {true_param}")
# generating the geocenter waveform
h_sky = waveform(freqs, true_param)
# setup ifo list
ifos = []
for ifo in args.ifos:
    eval(f'ifos.append({ifo})')
# inject signal into ifos
for idx, ifo in enumerate(ifos):
    key, subkey = jax.random.split(key)
    ifo.inject_signal(
        subkey,
        freqs,
        h_sky,
        detector_param,
        psd_file=args.psds[idx]  # the function load_psd actaully load asd
    )
print("Signal injected")

print("Start prior setup")
# priors without transformation 
Mc_prior    = Uniform(0.8759659737275101, 2.6060030916165484, naming=['M_c'])
s1z_prior   = Uniform(-0.05, 0.05, naming=['s1_z'])
s2z_prior   = Uniform(-0.05, 0.05, naming=['s2_z'])
lambda_1_prior = Uniform(0., 5000., naming=['lambda_1'])
lambda_2_prior = Uniform(0., 5000., naming=['lambda_2'])
dL_prior    = Uniform(30, 300, naming=['d_L'])
tc_prior    = Uniform(-0.1, 0.1, naming=['t_c'])
phic_prior  = Uniform(0., 2. * jnp.pi, naming=['phase_c'])
psi_prior   = Uniform(0., jnp.pi, naming=["psi"])
ra_prior    = Uniform(0., 2 * jnp.pi, naming=["ra"])
# priors with transformations
q_prior = Uniform(
    0.5,
    1,
    naming=['q'],
    transforms={
        'q': (
            'eta',
            lambda params: params['q'] / (1 + params['q']) ** 2
            )
        }
    )
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
# compose the prior
prior_list = [
        Mc_prior,
        q_prior,
        s1z_prior,
        s2z_prior,
        lambda_1_prior,
        lambda_2_prior,
        dL_prior,
        tc_prior,
        phic_prior,
        cos_iota_prior,
        psi_prior,
        ra_prior,
        sin_dec_prior,
]
complete_prior = Composite(prior_list)
bounds = jnp.array([[p.xmin, p.xmax] for p in complete_prior.priors])
print("Finished prior setup")

print("Initializing likelihood")
likelihood = HeterodynedTransientLikelihoodFD(
    ifos,
    prior=complete_prior,
    bounds=bounds,
    n_bins=100,
    waveform=waveform,
    trigger_time=args.trigger_time,
    duration=args.duration,
    post_trigger_duration=args.post_trigger_duration,
    ref_params=true_param
    )

mass_matrix = jnp.eye(len(prior_list))
for idx, prior in enumerate(prior_list):
    mass_matrix = mass_matrix.at[idx, idx].set(prior.xmax - prior.xmin) # fetch the prior range
local_sampler_arg = {'step_size': mass_matrix * 3e-3} # set the step size to be 0.3% of the prior range

jim = Jim(
    likelihood, 
    complete_prior,
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
    seed = args.sampling_seed,
    num_layers = args.num_layers,
    hidden_size = args.hidden_size,
    num_bins = args.num_bins
)

print("Start sampling")
key = jax.random.PRNGKey(args.sampling_seed)
jim.sample(key)
jim.print_summary()
samples = jim.get_samples()

# make output directory
import pandas as pd
os.makedirs(args.output_path)
for key in samples.keys():
    samples[key] = jnp.ravel(samples[key])
df = pd.DataFrame.from_dict(samples)
df.to_csv(f'./{args.output_path}/posterior_samples.dat', sep=' ', index=False)

# make corner plot
import utils
utils.corner_plot(samples, true_param, args.output_path)
