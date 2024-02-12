### Check waveform: check how the waveform looks like in time domain

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import jaxlib
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import ripple
from ripple.waveforms import TaylorF2, X_NRTidalv2
params = {"axes.grid": True,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(params)
# This is from: https://ldas-jobs.ligo.caltech.edu/~thibeau.wouters/jim_injections/tests_TaylorF2_19_01_2024/injection_144/config.json
config = {"seed": 8010, 
          "f_sampling": 4096, 
          "duration": 256, 
          "fmin": 20, 
          "ifos": ["H1", "L1", "V1"], 
          "no_noise": False, 
          "outdir": "./outdir_TaylorF2/injection_144/", 
          "M_c": 1.6241342228137727, 
          "q": 0.668298338249546, 
          "s1_z": -0.04322900906979665, 
          "s2_z": 0.0018364993350326042, 
          "lambda_1": 2705.980957696422, 
          "lambda_2": 3284.755794210511, 
          "d_L": 195.15224544041467, 
          "t_c": 0.06399874286352789, 
          "phase_c": 0.7458447627099869, 
          "cos_iota": -0.04453032073740104, 
          "psi": 2.7807861411019457, 
          "ra": 5.236983876031977, 
          "sin_dec": -0.6758174992722121}

def main(which: str = "TaylorF2"):
    
    T = 256
    f_l = 20.0
    f_sampling = 2 * 2084
    f_u = f_sampling // 2
    f_ref = f_l
    
    delta_t = 1 / f_sampling
    tlen = int(round(T / delta_t))
    freqs = np.fft.rfftfreq(tlen, delta_t)
    df = freqs[1] - freqs[0]
    fs = freqs[(freqs > f_l) & (freqs < f_u)]
    
    # m1_msun = m1
    # m2_msun = m2
    mc = config["M_c"]
    q = config["q"]
    eta = q / (1 + q)**2
    chi1 = config["s1_z"]
    chi2 = config["s2_z"]
    lambda_1 = config["lambda_1"]
    lambda_2 = config["lambda_2"]
    tc = config["t_c"]
    phic = config["phase_c"]
    dist_mpc = config["d_L"]
    cos_iota = config["cos_iota"]
    inclination = np.arccos(cos_iota)

    theta_ripple = jnp.array([mc, eta, chi1, chi2, lambda_1, lambda_2, dist_mpc, tc, phic, inclination])
    fs_ripple = jnp.arange(f_l, f_u, df)[1:]

    # And finally lets generate the waveform!
    if which == "TaylorF2":
        print("Using TaylorF2")
        hp_ripple, hc_ripple = TaylorF2.gen_TaylorF2_hphc(fs_ripple, theta_ripple, f_ref)
    else:
        print("Using NRTv2")
        hp_ripple, hc_ripple = X_NRTidalv2.gen_NRTidalv2_hphc(fs_ripple, theta_ripple, f_ref)
    
    # Convert to the time domain
    hp_ripple_td = jnp.fft.irfft(hp_ripple, tlen)
    hc_ripple_td = jnp.fft.irfft(hc_ripple, tlen)
    t = np.linspace(0, T, tlen)
    
    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(fs_ripple, hp_ripple, label="hp")
    ax[1].plot(fs_ripple, hc_ripple, label="hc")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[0].set_ylabel(r"$\tilde{h}_+(f)$")
    ax[1].set_ylabel(r"$\tilde{h}_\times(f)$")
    # ax[0].legend()
    # ax[1].legend()
    save_name = "./postprocessing/ripple_waveform_FD.png"
    print("Saving to: ", save_name)
    plt.savefig(save_name)
    plt.legend()
    plt.close()
    
    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    # Limit t and strain to after 100 seconds
    cut_idx = np.where(t > 100)[0][0]
    
    ax[0].plot(t[cut_idx:], hp_ripple_td[cut_idx:], label="hp")
    ax[1].plot(t[cut_idx:], hc_ripple_td[cut_idx:], label="hc")
    ax[1].set_xlabel("Time (s)")
    ax[0].set_ylabel(r"$h_+(t)$")
    ax[1].set_ylabel(r"$h_\times(t)$")
    # ax[0].legend()
    # ax[1].legend()
    save_name = "./postprocessing/ripple_waveform_TD.png"
    print("Saving to: ", save_name)
    plt.savefig(save_name)
    plt.legend()
    plt.close()

if __name__ == "__main__":
    main()
    print("Done!")