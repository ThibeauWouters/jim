from abc import ABC, abstractmethod
from jaxtyping import Array, Float
from jimgw.waveform import Waveform
from jimgw.detector import Detector
import jax.numpy as jnp
from astropy.time import Time
import numpy as np
from scipy.interpolate import interp1d
import jax
from flowMC.utils.EvolutionaryOptimizer import EvolutionaryOptimizer
from jimgw.prior import Prior
from jimgw.likelihood import LikelihoodBase
from jimgw.waveform import Waveform, RippleIMRPhenomD


# TODO find maybe a better spot to place this?

class FisherInformationMatrix:
    
    def __init__(self,
                detectors: list[Detector],
                waveform: Waveform = RippleIMRPhenomD,
                trigger_time: float = 0,
                duration: float = 4,
                post_trigger_duration: float = 2,) -> None:
        
        self.detectors = detectors 
        self.waveform = waveform 
        self.trigger_time = trigger_time 
        self.duration = duration 
        self.post_trigger_duration = post_trigger_duration
        
        # Convert the trigger time to a Time object
        self.gmst = (
            Time(trigger_time, format="gps").sidereal_time("apparent", "greenwich").rad
        )
        
    @property
    def epoch(self):
        """
        The epoch of the data.
        """
        return self.duration - self.post_trigger_duration

    @property
    def ifos(self):
        """
        The interferometers for the likelihood.
        """
        return [detector.name for detector in self.detectors]
    
    @staticmethod
    def noise_weighted_inner_product(h: Array, g: Array, detectors: list[Detector], df: float):
        result = 0
        for detector in detectors:
            product =  4 * jnp.sum((jnp.conj(h) * g) / detector.psd * df).real
            result += product
        return result


    def fisher_information_matrix(self,
                                prior: Prior, 
                                waveform_generator: Waveform,
                                params: Array,
                                frequencies: Array) -> Array:
        """
        Computes the Fisher information matrix at the params location and uses it to tune the mass matrix.
        """
        
        # TODO improve this implementation
        
        n_dim = prior.n_dim
        # TODO remove this? Or implement some failsafe with it?
        # mass_matrix = jnp.eye(n_dim) * 3e-3
        
        df = frequencies[1] - frequencies[0]
        
        # Now do the Fisher matrix computation
        fn = lambda x: waveform_generator(frequencies, x)
        _, dh_dlambda_func = jax.value_and_grad(fn)
        dh_dlambda = dh_dlambda_func(params)
        
        fisher_information_matrix = jnp.zeros((n_dim, n_dim))
    
        for i in range(n_dim):
            for j in range(n_dim):
                value = self.noise_weighted_inner_product(dh_dlambda[i], dh_dlambda[j], self.detectors, df)
                fisher_information_matrix.at[i, j].set(value)
        
        return fisher_information_matrix