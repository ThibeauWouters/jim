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
                                params: dict,
                                frequencies: Array,
                                verbose=False) -> Array:
        """
        Computes the Fisher information matrix at the params location and uses it to tune the mass matrix.
        """
        
        # TODO improve this implementation
        # TODO check the condition number and override certain entries! Cf paper
        # TODO remove verbose, for debugging
        
        # Get auxiliary quantities
        n_dim = prior.n_dim
        naming = prior.get_naming(transform_name=True)
        fisher_information_matrix = jnp.zeros((n_dim, n_dim))
        params["gmst"] = self.gmst
        
        df = frequencies[1] - frequencies[0]
        
        # Iterate over all detectors in the network
        for detector in self.detectors:
            # Initialize a new Fisher matrix for this detector
            this_fisher_information_matrix = jnp.zeros((n_dim, n_dim))
            # Get gradient of function that gives the waveform in this detector
            fn = lambda x: detector._get_h_detector(frequencies, waveform_generator, x)
            dh_dlambda_func = jax.jacfwd(fn)
            # Evaluate the derivatives at the given point
            dh_dlambda = dh_dlambda_func(params)
            # Sort based on naming, and get as values instead of dict
            dh_dlambda = [dh_dlambda[key] for key in naming]
            
            # TODO debug
            if verbose:
                print("jnp.shape(dh_dlambda)")
                print(jnp.shape(dh_dlambda))
                print("dh_dlambda")
                print(dh_dlambda)
            
            # Fill in the fisher information matrix for this detector
            for i in range(n_dim):
                for j in range(n_dim):
                    value = self.noise_weighted_inner_product(dh_dlambda[i], dh_dlambda[j], [detector], df)
                    this_fisher_information_matrix = this_fisher_information_matrix.at[i, j].set(value)
                    
            # At the end, add it to the overall fisher matrix
            fisher_information_matrix += this_fisher_information_matrix
        
        # DEUBUGGING
        if verbose:
            print(jnp.shape(dh_dlambda))
            print("dh_dlambda")
            for i, value in enumerate(dh_dlambda):
                print(naming[i])
                print(value)
            
            print("fisher_information_matrix")
            print(fisher_information_matrix)
            
            diagonal = jnp.diag(fisher_information_matrix)
            diagonal= jnp.where(diagonal == 0, 1e-3, diagonal) # Get rid of the zeros
            print("fisher_information_matrix diagonal")
            print(diagonal)
        
        self.fisher_information_matrix = fisher_information_matrix 
        
        # Also use it to get a tuned mass matrix. For now: diagonal entries only
        fisher_diagonal = np.diag(fisher_information_matrix)
        self.mass_matrix = np.diag(fisher_diagonal)
        
        print("Parameters:")
        print(naming)
        print("Tuned mass matrix: diagonal is:")
        print(jnp.round(fisher_diagonal, 2))
        
        print("Summary")
        for name, value in zip(naming, jnp.round(fisher_diagonal, 2)):
            print(f"Parameter: {name}, value Fisher matrix diagonal: {jnp.round(value, 2)}")
        
        return fisher_information_matrix
    
    
    ### TODO Can also implement it with the ensemble average definition? Any benefit?
    # def new_fisher_information_matrix(self,
    #                             prior: Prior, 
    #                             likelihood: LikelihoodBase,
    #                             params: Array,
    #                             points: Array,
    #                             diagonal_only: bool=True) -> Array:
    #     """
    #     Computes the Fisher information matrix at the params location, but now using the likelihood function and uses it to tune the mass matrix.
        
    #     Points are used to perform the ensemble average. 
    #     """
        
    #     n_dim = prior.n_dim
        
    #     # Lambda function that gets complex strain h when given parameter dict x
    #     fn = lambda x: - likelihood.evaluate(params, None)
    #     likelihood_hessian = jax.hessian(fn)
        
    #     # Ensemble average
    #     hessian_values = jax.vmap(likelihood_hessian)(points)
        
    #     fisher_information_matrix = jnp.zeros((n_dim, n_dim))
        
    #     weights = None
        
    #     weighted_average = jnp.sum(jnp.dot(weights, hessian_values)) / jnp.sum(weights)
                
    #     # Invert it to get the tuned mass matrix
    #     print("fisher_information_matrix")
    #     print(fisher_information_matrix)
        
    #     print("fisher_information_matrix diagonal")
    #     print(jnp.diag(fisher_information_matrix))
        
    #     inverse = jnp.linalg.inv(fisher_information_matrix)
        
    #     return inverse