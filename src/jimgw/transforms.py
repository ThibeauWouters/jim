import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped
from typing import Callable, Union

# Regularly used transform functions:

def q_to_eta(params: dict) -> dict:
    eta = params["q"] / (1 + params["q"]) ** 2
    return {"eta": eta}

def cos_iota_to_iota(params: dict) -> dict:
    iota = jnp.arccos(params["cos_iota"])
    return {"iota": iota}

def sin_dec_to_dec(params: dict) -> dict:
    dec = jnp.arcsin(params["sin_dec"])
    return {"dec": dec}

def H0_z_to_dL(params: dict):
    H0, z = params["H_0"], params["z"]
    c = 299792458.0 / 1e3 # TODO: improve this, but only testing for now
    d_L = z / H0 * c
    return {"d_L": d_L}

default_functions = {"q_to_eta": q_to_eta,
                     "cos_iota_to_iota": cos_iota_to_iota,
                     "sin_dec_to_dec": sin_dec_to_dec,
                     "H0_z_to_dL": H0_z_to_dL}

class Transform:
    
    transforms: list[Callable]
    
    def __init__(self, 
                 transforms: list[Callable],
                 identity_params: list[str]):
        
        # TODO: it is not optimal that users have to give all parameters that are not transformed...
        
        self.transforms = transforms
        self.identity_params = identity_params
        
    def transform(self,
                  params: dict) -> dict:
        
        result = {}
        
        # First perform the transforms:
        for func in self.transforms:
            result.update(func(params))
            
        # Then simply add the others (identity transform):
        for key in self.identity_params:
            result[key] = params[key]
            
        return result