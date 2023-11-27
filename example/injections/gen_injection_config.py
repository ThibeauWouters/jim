import numpy as np

def Mc_eta_to_ms(m):
    Mchirp, eta = m
    M = Mchirp / (eta ** (3 / 5))
    m2 = (M - np.sqrt(M ** 2 - 4 * M ** 2 * eta)) / 2
    m1 = M - m2
    return m1, m2

def generate_params_dict(prior_low: np.array, prior_high: np.array, params_names: list[str], N_config: int = 3) -> dict:
    params_dict = {}
    for low, high, param in zip(prior_low, prior_high, params_names):
        params_dict[param] = np.random.uniform(low, high, N_config)
    
    # Convert M_chirp, q to masses
    q = params_dict['q']
    mc = params_dict['mc']
    eta = q/(1+q)**2
    m1, m2 = Mc_eta_to_ms(np.stack([mc,eta]))
    
    # Convert inclination and dec
    inclination = np.arccos(params_dict["cos_inclination"])
    params_dict["inclination"] = inclination
    dec = np.arcsin(params_dict["sin_dec"])
    params_dict["dec"] = dec

    
    params_dict["q"] = q
    params_dict["eta"] = eta
    params_dict["m1"] = m1
    params_dict["m2"] = m2
    
    return params_dict

def generate_configs(prior_low: np.array, prior_high: np.array, params_names: list[str], config_dir: str = './configs/', N_config: int = 3) -> None:
    
    params_dict = generate_params_dict(prior_low, prior_high, params_names, N_config)

    for i in range(N_config):
        
        # Create new injection file
        filename = f"{config_dir}injection_config_{str(i)}.yaml"
        output_path = './outdir/ppPlots/injection_'+str(i)+'\n'
        
        with open(filename, 'w') as f:
            f.write(f'output_path: {output_path}')
            f.write('downsample_factor: 10\n')
            f.write('seed: '+str(np.random.randint(low=0,high=10000))+'\n')
            f.write('f_sampling: 2048\n')
            f.write('duration: 16\n')
            f.write('fmin: 30\n')
            f.write('ifos:\n')
            f.write('  - H1\n')
            f.write('  - L1\n')
            f.write('  - V1\n')

            f.write("m1: "+str(params_dict["m1"][i])+"\n")
            f.write("m2: "+str(params_dict["m2"][i])+"\n")
            f.write("chi1: "+str(params_dict["chi1"][i])+"\n")
            f.write("chi2: "+str(params_dict["chi2"][i])+"\n")
            f.write("dist_mpc: "+str(params_dict["dist_mpc"][i])+"\n")
            f.write("tc: "+str(params_dict["tc"][i])+"\n")
            f.write("phic: "+str(params_dict["phic"][i])+"\n")
            f.write("inclination: "+str(params_dict["inclination"][i])+"\n")
            f.write("polarization_angle: "+str(params_dict["polarization_angle"][i])+"\n")
            f.write("ra: "+str(params_dict["ra"][i])+"\n")
            f.write("dec: "+str(params_dict["dec"][i])+"\n")
            f.write("heterodyne_bins: 501\n")

            f.write("n_dim: 11\n")
            f.write("n_chains: 500\n")
            f.write("n_loop_training: 20\n")
            f.write("n_loop_production: 20\n")
            f.write("n_local_steps: 200\n")
            f.write("n_global_steps: 200\n")
            f.write("learning_rate: 0.001\n")
            f.write("max_samples: 50000\n")
            f.write("momentum: 0.9\n")
            f.write("num_epochs: 240\n")
            f.write("batch_size: 50000\n")
            # TODO fix local sampler arg in recovery file
            f.write("stepsize: 0.01\n")

# test case

prior_low  = np.array([10.0, 0.5, -0.5, -0.5,  300, -0.5,     0.0, -1,   0.0,     0.0, -1])
prior_high = np.array([50.0, 1.0,  0.5,  0.5, 2000,  0.5, 2*np.pi,  1, np.pi, 2*np.pi,  1])
params_names = ["mc", "q", "chi1", "chi2", "dist_mpc", "tc", "phic", "cos_inclination", "polarization_angle", "ra", "sin_dec"]
    
generate_configs(prior_low, prior_high, params_names)