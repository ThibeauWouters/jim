import matplotlib.pyplot as plt
import json
import numpy as np
import corner

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
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(matplotlib_params)

labels_results_plot = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda_1$', r'$\Lambda_2$', r'$d_{\rm{L}}/{\rm Mpc}$',
            r'$t_c$', r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

labels_with_tc = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda$', r'$\delta\Lambda$', r'$d_{\rm{L}}/{\rm Mpc}$',
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

    
def plot_chains(chains, truths, name, outdir, labels = labels_results_plot):
    
    chains = np.array(chains)
    
    # Check if 3D, then reshape
    if len(np.shape(chains)) == 3:
        chains = chains.reshape(-1, 13)
    
    print("np.shape(chains)")
    print(np.shape(chains))
    # Find index of cos iota and sin dec
    cos_iota_index = labels.index(r'$\iota$')
    sin_dec_index = labels.index(r'$\delta$')
    # Convert cos iota and sin dec to cos and sin
    chains[:,cos_iota_index] = np.arccos(chains[:,cos_iota_index])
    chains[:,sin_dec_index] = np.arcsin(chains[:,sin_dec_index])
    chains = np.asarray(chains)
    fig = corner.corner(chains, labels = labels, truths = truths, hist_kwargs={'density': True}, **default_corner_kwargs)
    fig.savefig(f"{outdir}{name}.png", bbox_inches='tight')  
    
def plot_chains_from_file(outdir, load_true_params: bool = False):
    
    filename = outdir + 'results_production.npz'
    data = np.load(filename)
    chains = data['chains']
    my_chains = []
    n_dim = np.shape(chains)[-1]
    for i in range(n_dim):
        values = chains[:, :, i].flatten()
        my_chains.append(values)
    my_chains = np.array(my_chains).T
    chains = chains.reshape(-1, 13)
    if load_true_params:
        truths = load_true_params_from_config(outdir)
    else:
        truths = None
    
    plot_chains(chains, truths, 'results', outdir)
    
def plot_accs_from_file(outdir):
    
    filename = outdir + 'results_production.npz'
    data = np.load(filename)
    local_accs = data['local_accs']
    global_accs = data['global_accs']
    
    local_accs = np.mean(local_accs, axis = 0)
    global_accs = np.mean(global_accs, axis = 0)
    
    plot_accs(local_accs, 'local_accs', 'local_accs_production', outdir)
    plot_accs(global_accs, 'global_accs', 'global_accs_production', outdir)
    
def load_true_params_from_config(outdir):
    
    config = outdir + 'config.json'
    # Load the config   
    with open(config) as f:
        config = json.load(f)
    true_params = np.array([config[key] for key in NAMING])
    
    # Convert cos_iota and sin_dec to iota and dec
    cos_iota_index = NAMING.index('cos_iota')
    sin_dec_index = NAMING.index('sin_dec')
    true_params[cos_iota_index] = np.arccos(true_params[cos_iota_index])
    true_params[sin_dec_index] = np.arcsin(true_params[sin_dec_index])
    
    return true_params

def main():
    # plot_chains_from_file("./outdir_TaylorF2/injection_144/")
    plot_accs_from_file("./outdir_TaylorF2/injection_144/")
    plot_chains_from_file("./outdir_TaylorF2/injection_144/", load_true_params=True)
    
    plot_accs_from_file("./outdir_TaylorF2/injection_144_original/")
    plot_chains_from_file("./outdir_TaylorF2/injection_144_original/", load_true_params=True)
    
if __name__ == "__main__":
    main()