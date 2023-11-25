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

    def __call__(self, frequency: Array, params: dict) -> Array:
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

    def __call__(self, frequency: Array, params: dict) -> Array:
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
            params["lambda_tilde"],
            params["delta_lambda_tilde"],
            params["d_L"],
            0,
            params["phase_c"],
            params["iota"],
        ]
        hp, hc = gen_TaylorF2_hphc(frequency, theta, self.f_ref)
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

    def __call__(self, frequency: Array, params: dict) -> dict:
        output = {}
        ra = params["ra"]
        dec = params["dec"]
        theta = [
            params["M_c"],
            params["eta"],
            params["s1_z"],
            params["s2_z"],
            params["lambda_tilde"],
            params["delta_lambda_tilde"],
            params["d_L"],
            0,
            params["phase_c"],
            params["iota"],
        ]
        hp, hc = gen_NRTidalv2_hphc(frequency, theta, self.f_ref)
        output["p"] = hp
        output["c"] = hc
        return output