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
from scipy.stats import norm


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
    
    def tune_mass_matrix(self,
                        prior: Prior, 
                        waveform_generator: Waveform,
                        params: dict,
                        frequencies: Array,
                        verbose=False) -> Array:
        """
        Computes the Fisher information matrix at the params location and uses it to tune the mass matrix.
        """
        
        # TODO remove verbose if debugging finished
        # TODO save Fisher information, if used later on?
        
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
            
            # TODO debug, check if derivatives make sense now
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
        
        self.fisher_information_matrix = fisher_information_matrix 
        
        # Go from Fisher information matrix to the tuned mass matrix
        fisher_diagonal = jnp.diag(fisher_information_matrix)
        prior_range = prior.xmax - prior.xmin
        mass_matrix_diagonal = jnp.sqrt(1 / fisher_diagonal) / prior_range
        
        # Clip so that it all scales are below 1 
        mass_matrix_diagonal = jnp.clip(mass_matrix_diagonal, 0, 1)
        
        # TODO override values should be done more informed?
        idx = naming.index("t_c") # t_c is uninformed, set to default value instead
        mass_matrix_diagonal = mass_matrix_diagonal.at[idx].set(1e-5)
        
        # Finally, convert from diagonal to matrix
        mass_matrix = jnp.diag(mass_matrix_diagonal)
        self.mass_matrix = mass_matrix
        
        if verbose:
            # for debugging
            print(jnp.shape(dh_dlambda))
            print("dh_dlambda")
            for i, value in enumerate(dh_dlambda):
                print(naming[i])
                print(value)
            
            print("fisher_information_matrix")
            print(fisher_information_matrix)
            
            print("Summary")
            for name, value in zip(naming, fisher_diagonal):
                print(f"Parameter: {name}, value Fisher matrix diagonal: {value}")
        
        return mass_matrix
    
    
    ## TODO Can also implement it with the ensemble average definition? Any benefit?
    def new_fisher_information_matrix(self,
                        prior: Prior, 
                        waveform_generator: Waveform,
                        params: dict,
                        frequencies: Array,
                        n_samples: int = 100,
                        verbose: bool=False) -> Array:
        """
        Computes the Fisher information matrix at the params location, but now using the likelihood function and uses it to tune the mass matrix.
        
        Points are used to perform the ensemble average. 
        """
        
        n_dim = prior.n_dim
        df = frequencies[1] - frequencies[0]
        fisher_information_matrix = jnp.zeros((n_dim, n_dim))
        
        # def get_gamma_estimate(data, params, detector):
        #     # Evaluate a single term in the discrete approximation of the FIM
        #     fn_h = lambda x: detector._get_h_detector(frequencies, waveform_generator, x)
        #     sq_diff = lambda x: jnp.abs(data - fn_h(x)) ** 2
        #     hessian = jax.hessian(sq_diff)
        #     hessian_values = hessian(params)
        #     # Sum over frequencies
        #     term = jnp.sum(2 * df / detector.psd * hessian_values)
        #     return term
        
        def get_hessian_values(data, params, detector):
            h_detector = lambda x: detector._get_h_detector(frequencies, waveform_generator, x)
            sq_diff = lambda x: jnp.abs(data - h_detector(x)) ** 2
            hessian = jax.hessian(sq_diff)
            hessian_values = hessian(params)
            
            return hessian_values
            
        
        for detector in self.detectors:
            # Build the argument of the FIM expectation value (x are waveform parameters below)
            # TODO is the cov right in multivariate normal, or need sqrt?
            
            print(f"Getting hessian for detector {detector.name}")
            h_detector = lambda x: detector._get_h_detector(frequencies, waveform_generator, x)
            h = h_detector(params)
            h_np = np.asarray(h, dtype=np.complex_)
            scale = 2 * np.sqrt(detector.psd) / np.sqrt(df)
            # Single 
            sample_d = [norm.rvs(loc=h_np[i], scale = scale[i], size=1) for i in range(len(h_np))]
            sample_d = jnp.asarray(sample_d)
            print("sample_d shape")
            print(jnp.shape(sample_d))
            fn_get_hessian_values = lambda d: get_hessian_values(d, params, detector)
            get_hessian_values_vmap = jax.vmap(fn_get_hessian_values)
            hessian_values = get_hessian_values_vmap(sample_d)
            term = jnp.mean(jnp.sum(2 * df / detector.psd * hessian_values, axis = 1))
            fisher_information_matrix += term
            
                
        # Invert it to get the tuned mass matrix
        if verbose:
                
            print("fisher_information_matrix")
            print(fisher_information_matrix)
            self.fisher_information_matrix = fisher_information_matrix
            print("fisher_information_matrix diagonal")
            print(jnp.diag(fisher_information_matrix))
        
        return fisher_information_matrix