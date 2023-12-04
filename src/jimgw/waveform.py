from jaxtyping import Array
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc, gen_IMRPhenomD
from ripple.waveforms.IMRPhenomPv2 import gen_IMRPhenomPv2_hphc
# # Tidal waveforms
from ripple.waveforms.TaylorF2 import gen_TaylorF2_hphc
from ripple.waveforms.X_NRTidalv2 import gen_NRTidalv2_hphc
import jax.numpy as jnp
from abc import ABC


class Waveform(ABC):
    def __init__(self):
        return NotImplemented

    def __call__(self, axis: Array, params: Array) -> dict:
        return NotImplemented
    

class RippleIMRPhenomD(Waveform):

    f_ref: float

    def __init__(self, f_ref: float = 20.0):
        self.f_ref = f_ref

    def __call__(self, frequency: Array, params: dict) -> dict:
        output = {}
        ra = params["ra"]
        dec = params["dec"]
        theta = [
            params["M_c"],
            params["eta"],
            params["s1_z"],
            params["s2_z"],
            params["d_L"],
            0,
            params["phase_c"],
            params["iota"],
        ]
        hp, hc = gen_IMRPhenomD_hphc(frequency, theta, self.f_ref)
        output["p"] = hp
        output["c"] = hc
        return output
    
    
    def gen_complex_strain(self, frequency: Array, theta: Array) -> Array:
        
        return gen_IMRPhenomD(frequency, theta, self.f_ref)
    
class RippleIMRPhenomPv2(Waveform):

    f_ref: float

    def __init__(self, f_ref: float = 20.0):
        self.f_ref = f_ref

    def __call__(self, frequency: Array, params: dict) -> dict:
        output = {}
        theta = [
            params["M_c"],
            params["eta"],
            params['s1_x'],
            params['s1_y'],
            params["s1_z"],
            params['s2_x'],
            params['s2_y'],
            params["s2_z"],
            params["d_L"],
            0,
            params["phase_c"],
            params["iota"],
        ]
        hp, hc = gen_IMRPhenomPv2_hphc(frequency, theta, self.f_ref)
        output["p"] = hp
        output["c"] = hc
        return output

class RippleTaylorF2(Waveform):

    f_ref: float

    def __init__(self, f_ref: float = 20.0, use_lambda_tildes: bool = True):
        self.f_ref = f_ref
        self.use_lambda_tildes = use_lambda_tildes

    def __call__(self, frequency: Array, params: dict) -> dict:
        output = {}
        ra = params["ra"]
        dec = params["dec"]
        if self.use_lambda_tildes:
            first_lambda_params = params["lambda_tilde"]
            second_lambda_params = params["delta_lambda_tilde"]
        else:
            first_lambda_params = params["lambda1"]
            second_lambda_params = params["lambda2"]
        
        theta = [
            params["M_c"],
            params["eta"],
            params["s1_z"],
            params["s2_z"],
            first_lambda_params,
            second_lambda_params,
            params["d_L"],
            0,
            params["phase_c"],
            params["iota"],
        ]
        hp, hc = gen_TaylorF2_hphc(frequency, theta, self.f_ref, use_lambda_tildes=self.use_lambda_tildes)
        output["p"] = hp
        output["c"] = hc
        return output
    
class RippleTaylorF2NoTidal(Waveform):

    f_ref: float

    def __init__(self, f_ref: float = 20.0):
        self.f_ref = f_ref

    def __call__(self, frequency: Array, params: dict) -> dict:
        output = {}
        ra = params["ra"]
        dec = params["dec"]
        theta = [
            params["M_c"],
            params["eta"],
            params["s1_z"],
            params["s2_z"],
            0, # no tidal
            0, # no tidal
            params["d_L"],
            0,
            params["phase_c"],
            params["iota"],
        ]
        hp, hc = gen_TaylorF2_hphc(frequency, theta, self.f_ref)
        output["p"] = hp
        output["c"] = hc
        return output
    
class RippleIMRPhenomD_NRTidalv2(Waveform):

    f_ref: float

    def __init__(self, f_ref: float = 20.0):
        self.f_ref = f_ref

    def __call__(self, frequency: Array, params: dict, use_lambda_tildes: bool=True) -> dict:
        output = {}
        ra = params["ra"]
        dec = params["dec"]
        
        if use_lambda_tildes:
            first_lambda_params = params["lambda_tilde"]
            second_lambda_params = params["delta_lambda_tilde"]
        else:
            first_lambda_params = params["lambda1"]
            second_lambda_params = params["lambda2"]
        
        theta = [
            params["M_c"],
            params["eta"],
            params["s1_z"],
            params["s2_z"],
            first_lambda_params,
            second_lambda_params,
            params["d_L"],
            0,
            params["phase_c"],
            params["iota"],
        ]
        hp, hc = gen_NRTidalv2_hphc(frequency, theta, self.f_ref, use_lambda_tildes=use_lambda_tildes)
        output["p"] = hp
        output["c"] = hc
        return output