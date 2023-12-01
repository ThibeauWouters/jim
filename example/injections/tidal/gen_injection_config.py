import numpy as np
import jax.numpy as jnp
import json
import pandas as pd
import os, shutil

from ripple import Mc_eta_to_ms, ms_to_Mc_eta

def generate_params_dict(prior_low: np.array, prior_high: np.array, params_names: list[str], N_config: int = 3) -> dict:
    """
    Generate a dictionary of parameters for injections.
    
    """
    
    # Sample over the prior, as often as desired
    params_dict = {}
    for low, high, param in zip(prior_low, prior_high, params_names):
        params_dict[param] = np.random.uniform(low, high, N_config)
        
    # # Wherever m1 < m2, swap them and their spins:
    # m1 = params_dict['m1']
    # m2 = params_dict['m2']
    # s1_z = params_dict['s1_z']
    # s2_z = params_dict['s2_z']
    
    # swap = m1 < m2
    # m1[swap], m2[swap] = m2[swap], m1[swap]
    # s1_z[swap], s2_z[swap] = s2_z[swap], s1_z[swap]
    
    # params_dict['m1'] = m1
    # params_dict['m2'] = m2
    # params_dict['s1_z'] = s1_z
    # params_dict['s2_z'] = s2_z
    
    ## TODO unused?
    # Convert M_chirp, q to masses
    # q = params_dict['q']
    # mc = params_dict['mc']
    # eta = q/(1+q)**2
    # m1, m2 = Mc_eta_to_ms(np.stack([mc,eta]))
    
    # mc, eta = ms_to_Mc_eta(jnp.stack([params_dict['m1'], params_dict['m2']]))
    # q = params_dict['m2'] / params_dict['m1']
    
    ## TODO unused?
    # # Convert inclination and dec
    # inclination = np.arccos(params_dict["cos_inclination"])
    # # params_dict["inclination"] = inclination
    # dec = np.arcsin(params_dict["sin_dec"])
    # # params_dict["dec"] = dec

    # params_dict["q"] = q
    # params_dict["eta"] = np.asarray(eta)
    # params_dict["M_c"] = np.asarray(mc)
    
    # Create a Pandas dataframe, then save to a CSV file
    df = pd.DataFrame(params_dict)
    df.to_csv('./injection_params.csv', index=False)
    
    return params_dict

def generate_configs(prior_low: np.array, prior_high: np.array, params_names: list[str], config_dir: str = './configs/', N_config: int = 3) -> None:
    
    # First, generate the parameters:
    params_dict = generate_params_dict(prior_low, prior_high, params_names, N_config)

    # Then, generate the config files from the parameters in JSON format
    for i in range(N_config):
        
        # Create new injection file
        output_path = f'./outdir/injection_{str(i)}/'
        filename = output_path + f"config.json"
        
        # Check if directory exists, if not, create it. Otherwise, delete it and create it again
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

        injection_dict = {
            'outdir': output_path,
            # 'downsample_factor': 10,
            'seed': np.random.randint(low=0, high=10000),
            'f_sampling': 2*2048,
            'duration': 128,
            'fmin': 20,
            'ifos': ['H1', 'L1', 'V1']
        }
        
        for param, value in params_dict.items():
            injection_dict[param] = value[i].item()
        
        with open(filename, 'w') as f:
            json.dump(injection_dict, f)
            

if __name__ == "__main__":
    
    ## Masses: Mc and q
    # prior_low  = [1.18, 0.125, -0.05, -0.05,    0.0, -500.0,  1.0, -0.1,        0.0, -1.0,    0.0,        0.0, -1],
    # prior_high = [1.21,   1.0,  0.05,  0.05, 3000.0,  500.0, 75.0,  0.1, 2 * jnp.pi,  1.0, jnp.pi, 2 * jnp.pi,  1],
    
    ## Masses: component masses
    prior_low  = [1.18, 0.125, -0.05, -0.05,    0.0, -500.0,  1.0, -0.1,        0.0, -1.0,    0.0,        0.0, -1]
    prior_high = [1.21,   1.0,  0.05,  0.05, 3000.0,  500.0, 75.0,  0.1, 2 * jnp.pi,  1.0, jnp.pi, 2 * jnp.pi,  1]
    bounds = zip(prior_low, prior_high)
    naming = ["M_c", "q", "s1_z", "s2_z", "lambda_tilde", "delta_lambda_tilde", "d_L", "t_c", "phase_c", "cos_iota", "psi", "ra", "sin_dec"]
    
    prior_dict  = {name: bound for name, bound in zip(naming, bounds)}
    
    # Now save to json:
    with open('./prior.json', 'w') as f:
        json.dump(prior_dict, f)
    
    print("Prior:")
    print("low: ", prior_low)
    print("high:", prior_high)
    print("params_names:", naming)
    
    print("Generating injection config files...")
    generate_configs(prior_low, prior_high, naming)
    print("Done!")
