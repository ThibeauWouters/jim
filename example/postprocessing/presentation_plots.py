import psutil
p = psutil.Process()
p.cpu_affinity([0])
import matplotlib.pyplot as plt
import json
import numpy as np
import corner
import h5py
import arviz

NAMING = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']

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

matplotlib_params = {"axes.grid": True,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 32,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(matplotlib_params)

labels = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda_1$', r'$\Lambda_2$', r'$d_{\rm{L}}/{\rm Mpc}$',
            r'$t_c$', r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

labels_gwosc = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda_1$', r'$\Lambda_2$', r'$d_{\rm{L}}/{\rm Mpc}$',
            r'$\iota$', r'$\alpha$', r'$\delta$']

labels_no_lambdas = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$d_{\rm{L}}/{\rm Mpc}$',
            r'$t_c$', r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

def plot_accs(accs, label, name, outdir):
    
    eps = 1e-3
    plt.figure(figsize=(10, 6))
    plt.plot(accs, label=label)
    plt.ylim(0 - eps, 1 + eps)
    
    plt.ylabel(label)
    plt.xlabel("Iteration")
    plt.savefig(f"{outdir}{name}.png", bbox_inches='tight')  
    plt.close()

    
def plot_single_chains(chains, labels, savename, subplots_adjust = 3, fs_ticks = 62, fs_labels = 72, linewidth_1d = 4):
    
    hist_kwargs = {'density': True,
                   "linewidth": linewidth_1d}
    
    fig = corner.corner(chains, labels = labels, hist_kwargs=hist_kwargs, **default_corner_kwargs)
    
    # Clean up
    fig.subplots_adjust(right=subplots_adjust,top=subplots_adjust)
    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=fs_ticks)
        # also increase labels fontsize
        ax.set_xlabel(ax.get_xlabel(), fontsize=fs_labels)
        ax.set_ylabel(ax.get_ylabel(), fontsize=fs_labels)

    fig.savefig(f"{savename}.png", bbox_inches='tight', dpi = 300)  
    
def get_chains_GWOSC():
    """Compare to the HDF5 file as well"""
    
    # Read the HDF5 file
    with h5py.File("./data/GW170817_posterior.hdf5", 'r') as f:
        print(f.keys())
        data = f['IMRPhenomPv2NRT_lowSpin_posterior']
        print(data.dtype)
        m1 = data['m1_detector_frame_Msun']
        m2 = data['m2_detector_frame_Msun']
        chi1 = data['spin1']
        chi2 = data['spin2']
        lambda1 = data['lambda1']
        lambda2 = data['lambda2']
        
        dL = data['costheta_jn']
        ra = data['right_ascension']
        dec = data['declination']
        iota = data['costheta_jn']
        
        # Convert masses
        mc = ms_to_chirp_mass(m1, m2)
        q = m2 / m1
        
        # samples = np.array([mc, q, chi1, chi2, lambda1, lambda2, dL, iota, ra, dec]).T
        # print("np.shape(samples)")
        # print(np.shape(samples))
        samples = np.array([mc, q, chi1, chi2, lambda1, lambda2, dL]).T
        
    return samples

def plot_chains(chains_1, chains_2, labels, ls = 22):
    
    fig = corner.corner(chains_1, labels = labels, hist_kwargs={'density': True}, **default_corner_kwargs)
    default_corner_kwargs["color"] = "red"
    corner.corner(chains_2, fig = fig, labels = labels, hist_kwargs={'density': True}, **default_corner_kwargs)
    
    # Clean up the plot
    fig.subplots_adjust(right=1.5,top=1.5)
    fig.savefig(f"./outdir/chains_jim_gwosc.png", bbox_inches='tight')  
    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=ls)
    plt.close()


#### UTILITIES

def ms_to_chirp_mass(m1, m2):
    """Convert to chirp mass
    """
    
    return (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
    
    
    
def get_jim_chains_from_file(filename, remove_non_gwosc = False, remove_non_bilby = False, drop_lambdas = False):
    
    data = np.load(filename)
    chains = data['chains']
    my_chains = []
    n_dim = np.shape(chains)[-1]
    for i in range(n_dim):
        values = chains[:, :, i].flatten()
        my_chains.append(values)
    my_chains = np.array(my_chains).T
    chains = chains.reshape(-1, 13)
    cos_iota_index = labels.index(r'$\iota$')
    sin_dec_index = labels.index(r'$\delta$')
    chains[:,cos_iota_index] = np.arccos(chains[:,cos_iota_index])
    chains[:,sin_dec_index] = np.arcsin(chains[:,sin_dec_index])
    cos_iota_index = labels.index(r'$\iota$')
    sin_dec_index = labels.index(r'$\delta$')
    chains = np.asarray(chains)
    
    # Remove the t_c and phase_c and psi
    if remove_non_gwosc:
        t_c_index = labels.index(r'$t_c$')
        phase_c_index = labels.index(r'$\phi_c$')
        psi_index = labels.index(r'$\psi$')
        
        # Extra
        iota_index = labels.index(r'$\iota$')
        sin_dec_index = labels.index(r'$\delta$')
        ra_index = labels.index(r'$\alpha$')
        
        # chains = np.delete(chains, [t_c_index, phase_c_index, psi_index], 1)
        chains = np.delete(chains, [t_c_index, phase_c_index, psi_index, iota_index, sin_dec_index, ra_index], 1)
        
    if remove_non_bilby:
        t_c_index = labels.index(r'$t_c$')
        phase_c_index = labels.index(r'$\phi_c$')
        psi_index = labels.index(r'$\psi$')
        
        # Extra
        iota_index = labels.index(r'$\iota$')
        sin_dec_index = labels.index(r'$\delta$')
        ra_index = labels.index(r'$\alpha$')
        
        # chains = np.delete(chains, [t_c_index, phase_c_index, psi_index], 1)
        chains = np.delete(chains, [t_c_index, phase_c_index, psi_index, iota_index, sin_dec_index, ra_index], 1)
        
    if drop_lambdas:
        lambda_1_index = labels.index(r'$\Lambda_1$')
        lambda_2_index = labels.index(r'$\Lambda_2$')
        
        chains = np.delete(chains, [lambda_1_index, lambda_2_index], 1)
    
    return chains

def get_chains_bilby(fake_lambdas = False):
    filename = "/home/thibeau.wouters/jim/example/GW170817_TaylorF2/data/GW170817_Bilby.dat"
    
    # Load it, but skip the first row since this is a header
    data = np.loadtxt(filename, skiprows=1)
    original_header = ["chirp_mass", "mass_ratio", "chi_1", "chi_2", "luminosity_distance", "dec", "ra", "psi", "phase", "geocent_time", "theta_jn"]
    header = ["chirp_mass", "mass_ratio", "chi_1", "chi_2", "luminosity_distance", "geocent_time", "phase", "theta_jn", "psi", "ra", "dec"]
    
    result = []
    for name in header:
        index = original_header.index(name)
        result.append(data[:, index])
    
    # Insert empty arrays for lambda_1 and lambda_2
    if fake_lambdas:
        result.insert(4, np.zeros(len(data)))
        result.insert(5, np.zeros(len(data)))
    
    result = np.array(result).T
    
    return result
    
    
def main():
    jim_chains = get_jim_chains_from_file("/home/thibeau.wouters/public_html/jim_runs/GW170817_TaylorF2/GW170817_TaylorF2/outdir/results_production.npz")
    
    # sanity check
    d_L = jim_chains[:, 6]
    med = np.median(d_L)
    low, high = arviz.hdi(d_L, hdi_prob=0.9)
    low = med - low
    high = high - med
    
    nb_round = 2
    low, med, high = round(low, nb_round), round(med, nb_round), round(high, nb_round)
    
    print(f"Median: {med}- {low} + {high}")
    
    idx_list = [0, 1, 4, 5, 6] + [9, 10, 11, 12]
    jim_chains = jim_chains[:, idx_list]
    my_labels = [labels[i] for i in idx_list]

    plot_single_chains(jim_chains, my_labels, savename="./figures/posterior")
    plot_single_chains(jim_chains, my_labels, savename="/home/thibeau.wouters/public_html/jim_plots/GW170817_TaylorF2/GW170817_posterior")
    
    ### TODO plot other chains as well from trusted sources e.g. bilby etc
    # bilby_result = get_chains_bilby(fake_lambdas = False)
    # gwosc_chains = get_chains_GWOSC()
    # plot_chains(jim_chains, gwosc_chains)
    
    
if __name__ == "__main__":
    main()
    print("DONE")