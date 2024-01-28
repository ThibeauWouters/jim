from abc import ABC, abstractmethod
import mpmath
import copy
from jaxtyping import Array, Float
from jimgw.waveform import Waveform
from jimgw.detector import Detector
import jax.numpy as jnp
from astropy.time import Time
import numpy as np
from scipy.interpolate import interp1d
import jax
from flowMC.utils.EvolutionaryOptimizer import EvolutionaryOptimizer
from jimgw.prior import Prior, Composite
from jimgw.likelihood import LikelihoodBase
from jimgw.waveform import Waveform, RippleIMRPhenomD
from scipy.stats import norm
import matplotlib.pyplot as plt

# TODO find maybe a better spot to place this?

SUPPORTED_INVERSION_METHODS = ["svd"]
DEFAULT_INVERSION_METHOD = "svd"

ORIGINAL_MASS_MATRIX = 1e-2 * jnp.eye(13)
ORIGINAL_MASS_MATRIX = ORIGINAL_MASS_MATRIX.at[0,0].set(1e-5)
ORIGINAL_MASS_MATRIX = ORIGINAL_MASS_MATRIX.at[1,1].set(1e-4)
ORIGINAL_MASS_MATRIX = ORIGINAL_MASS_MATRIX.at[2,2].set(1e-3)
ORIGINAL_MASS_MATRIX = ORIGINAL_MASS_MATRIX.at[3,3].set(1e-3)
ORIGINAL_MASS_MATRIX = ORIGINAL_MASS_MATRIX.at[7,7].set(1e-5)
ORIGINAL_MASS_MATRIX = ORIGINAL_MASS_MATRIX.at[11,11].set(1e-2)
ORIGINAL_MASS_MATRIX = ORIGINAL_MASS_MATRIX.at[12,12].set(1e-2)

def plot_ratio(matrix, name, naming, use_ratio = True):
    # Check if 2D, if so, then use diag
    if len(np.shape(matrix)) == 2:
        matrix = np.diag(matrix)
    
    # Get ratio wrt first entry
    if use_ratio:
        ratio = matrix / matrix[0]
    else:
        ratio = matrix
        name = name + "_absolute"
    
    plt.plot(ratio, "-o")
    plt.xticks(np.arange(len(naming)), naming, rotation=90)
    plt.yscale("log")
    
    plt.savefig(f"./postprocessing/{name}_ratios.png")
    plt.close()
    
def plot_ratio_comparison(mass_matrix_1, mass_matrix_2, name_1, name_2, naming, use_ratio = True, outdir="./outdir/"):
    # Check if 2D, if so, then use diag
    if len(mass_matrix_1.shape) == 2:
        mass_matrix_1 = np.diag(mass_matrix_1)
        
    if len(mass_matrix_2.shape) == 2:
        mass_matrix_2 = np.diag(mass_matrix_2)
    
    name = "comparison_ratios"
    if use_ratio:
        ratio_1 = mass_matrix_1 / mass_matrix_1[0]
        ratio_2 = mass_matrix_2 / mass_matrix_2[0]
        ylabel = r"Step size ratio $\varepsilon / \varepsilon_{\mathcal{M}_c}$"
    else:
        ratio_1 = mass_matrix_1
        ratio_2 = mass_matrix_2
        name = name + "_absolute"
        ylabel = r"Step size $\varepsilon$"
    
    # Skip gmst in the params
    naming = naming[:-1]
    
    plt.plot(ratio_1, "-o", color = "red", label = name_1)
    plt.plot(ratio_2, "-o", color = "blue", label = name_2)
    plt.xticks(np.arange(len(naming)), naming, rotation=90)
    plt.legend()
    plt.yscale("log")
    plt.ylabel(ylabel)
    plt.savefig(f"{outdir}{name}.png", bbox_inches = 'tight', dpi=300)
    plt.close()

class FisherInformationMatrix:
    
    def __init__(self,
                detectors: list[Detector],
                waveform: Waveform,
                prior: Composite,
                trigger_time: float = 0,
                duration: float = 4,
                post_trigger_duration: float = 2,
                inversion_method: str = "svd",
                transform_names: bool = True,
                verbose = False) -> None:
        
        self.detectors = detectors 
        self.waveform = waveform 
        
        # Prior-related stuff comes here
        self.prior = prior
        self.naming = prior.get_naming(transform_names = transform_names)
        self.n_dim = prior.n_dim
        try:
            prior_range = [single_prior.xmax - single_prior.xmin for single_prior in self.prior.priors]
            # prior_range = {name: value for name, value in zip(self.naming, prior_range)}
            prior_range = jnp.array(prior_range)
            self.prior_range = prior_range
            self.divide_by_prior_range = True
        except:
            # TODO more meaningful statement and print the error message as well.
            print("Something went wrong with computing the prior range, perhaps the prior has no xmin and xmax?")
            self.divide_by_prior_range = False
        
        # Remaining setters: 
        self.trigger_time = trigger_time 
        self.duration = duration 
        self.post_trigger_duration = post_trigger_duration
        self.verbose = verbose
        if inversion_method not in SUPPORTED_INVERSION_METHODS:
            print(f"Warning: inversion method {inversion_method} not supported, using default {DEFAULT_INVERSION_METHOD}")
            inversion_method = DEFAULT_INVERSION_METHOD
        self.inversion_method = inversion_method
        self.fisher_information_matrix = None
        self.inverse = None
        
        # Convert the trigger time to a Time object
        self.gmst = (
            Time(trigger_time, format="gps").sidereal_time("apparent", "greenwich").rad
        )
    
    def compute_fim(self,
                    params: dict,
                    frequencies: Array) -> Array:
        """
        Computes the Fisher information matrix at the params location and uses it to tune the mass matrix.
        """
        
        if self.verbose:
            print("naming")
            print(self.naming)
        fisher_information_matrix = jnp.zeros((self.n_dim, self.n_dim))
        params["gmst"] = self.gmst
        
        df = frequencies[1] - frequencies[0]
        
        # Iterate over all detectors in the network, compute FIM and add all together
        for detector in self.detectors:
            # Initialize a new Fisher matrix for this detector
            this_fisher_information_matrix = jnp.zeros((self.n_dim, self.n_dim))
            # Get gradient of function that gives the waveform in this detector
            fn = lambda x: detector._get_h_detector(frequencies, self.waveform, x) * jnp.exp(-1j * 2 * jnp.pi * frequencies * x["t_c"])
            
            ## Original (weird results)
            dh_dlambda = jax.jacfwd(fn)(params)
            
            ### jacrev (bug about holomorphic)
            # dh_dlambda = jax.jacrev(fn)(params)
            
            # ### jacvjp
            # primals, dh_dlambda_fun = jax.vjp(fn, params)
            # dh_dlambda = dh_dlambda_fun(params)

            
            ### Split in real and imag (giving errors)
            # fn_real = lambda x: jnp.real(detector._get_h_detector(frequencies, self.waveform, x) * jnp.exp(-1j * 2 * jnp.pi * frequencies * x["t_c"]))
            # fn_imag = lambda x: jnp.imag(detector._get_h_detector(frequencies, self.waveform, x) * jnp.exp(-1j * 2 * jnp.pi * frequencies * x["t_c"]))
            # dv_real = jax.jacrev(fn_real)(params)
            # dv_imag = jax.jacrev(fn_imag)(params)
            # dh_dlambda = dv_real + 1j * dv_imag
            
            if self.verbose:            
                print("dh_dlambda")
                print(dh_dlambda)
                
                print("type(dh_dlambda)")
                print(type(dh_dlambda))
                
                print("Use the naming scheme and convert:")
            
            # Sort based on naming, and get as values instead of dict
            dh_dlambda = [dh_dlambda[key] for key in self.naming]
                        
            if self.verbose:
                print("dh_dlambda")
                print(dh_dlambda)
                
                print("type(dh_dlambda)")
                print(type(dh_dlambda))
            
            # TODO debug, check if derivatives make sense now
            if self.verbose:
                # for debugging
                print(jnp.shape(dh_dlambda))
                print("dh_dlambda")
                for i, value in enumerate(dh_dlambda):
                    print(self.naming[i])
                    print(value)
                
                print("fisher_information_matrix")
                print(fisher_information_matrix)
                
            # Fill in the fisher information matrix for this detector
            for i in range(self.n_dim):
                for j in range(self.n_dim):
                    value = self.noise_weighted_inner_product(dh_dlambda[i], dh_dlambda[j], detector, df)
                    this_fisher_information_matrix = this_fisher_information_matrix.at[i, j].set(value)
                    
            # At the end, add it to the overall fisher matrix
            fisher_information_matrix += this_fisher_information_matrix
        
        self.fisher_information_matrix = fisher_information_matrix 
        return fisher_information_matrix 
        
        
    def invert(self,
               truncate: bool = True,
               svals_thresh: float = 1e-15,
               max_cond_number: float = 1e50):
        
        # Check if FIM is computed, otherwise, if not, then return zeroes matrix
        if self.fisher_information_matrix is None:
            print("Fisher information matrix not computed yet, returning zeroes matrix")
            return jnp.zeros((self.n_dim, self.n_dim))
        
        # Store a copy of the FIM
        self.fisher_information_matrix_copy = copy.deepcopy(self.fisher_information_matrix)
        
        # Convert to mpmath matrix and normalize it
        fisher_information_matrix_mpmath = mpmath.matrix(self.fisher_information_matrix.astype('float64'))
        normalized_matrix, cond, ws = self.normalize_matrix(fisher_information_matrix_mpmath)
        
        # Invert the matrix
        if self.inversion_method == "svd":
            
            U, Sm, V = mpmath.svd_r(normalized_matrix)
            S = jnp.array(Sm.tolist(), dtype='float64')
            if ((truncate) and (jnp.abs(cond) > max_cond_number)):
                if self.verbose:
                    print('Truncating singular values below %s' %svals_thresh)
                
                maxev = jnp.max(jnp.abs(S))
                Sinv = mpmath.matrix(jnp.array([1/s if jnp.abs(s)/maxev>svals_thresh else 1/(maxev*svals_thresh) for s in S ]).astype('float64'))
                St = mpmath.matrix(jnp.array([s if jnp.abs(s)/maxev>svals_thresh else maxev*svals_thresh for s in S ]).astype('float64'))
                
                # Also copute truncated Fisher to quantify inversion error consistently
                truncFisher = U*mpmath.diag([s for s in St])*V
                truncFisher = (truncFisher+truncFisher.T)/2
                self.fisher_information_matrix_copy[:, :] = jnp.array(truncFisher.tolist(), dtype='float64')
                
                if self.verbose:
                    truncated = jnp.abs(S)/maxev<svals_thresh #jnp.array([1 if jnp.abs(s)/maxev>svals_thresh else 0 for s in S ]
                    print('%s singular values truncated' %(truncated.sum()))
            else:
                Sinv = mpmath.matrix(jnp.array([1/s for s in S ]).astype('float64'))
                St = S
            
            cc = V.T * mpmath.diag([s for s in Sinv]) * U.T
            
            ### END OF INVERSION METHODS
            cc = (cc+cc.T)/2
            
            ws_matrix = jnp.array(ws.tolist(), dtype='float64')
            if jnp.sum(ws_matrix) > 0:
                # Undo the reweighting
                if self.verbose:
                    print("Reweighting was true, now undoing reweighting")
                inverse = ws * cc * ws
            else:
                inverse = cc
                
            inverse = jnp.array(inverse.tolist(), dtype='float64')
            
        eps = self.compute_inversion_error(self.fisher_information_matrix_copy, inverse, only_diagonal=True)
        print("Inversion error: diagonal only")
        print(eps)
        self.eps = eps
        
        self.inverse = inverse 
        return inverse
        
    def get_tuned_mass_matrix(self,
                              clip_left: float = 0,
                              clip_right: float = 1e-2,
                              desired_eps_tc: float = 1e-7):
        
        if self.inverse is None:
            print("Inversion not computed yet, returning ones")
            return jnp.eye(self.n_dim)
        
        inverse_diagonal = jnp.diag(self.inverse)
        if self.verbose:
            print("inverse_diagonal")
            print(inverse_diagonal)
        mass_matrix_diagonal = jnp.sqrt(inverse_diagonal)
        if self.verbose:
            print("mass_matrix_diagonal")
            print(mass_matrix_diagonal)
        
        # TODO implement these properly:
        self.divide_by_prior_range = False
        if self.divide_by_prior_range:
            if self.verbose:
                print("INFO: Dividing by prior range")
            
            mass_matrix_diagonal = mass_matrix_diagonal / self.prior_range
            
        # Rescale the mass matrix based on t_c
        if desired_eps_tc is not None:
            tc_idx = self.naming.index("t_c")
            initial_eps_tc = mass_matrix_diagonal[tc_idx]
            
            factor = desired_eps_tc / initial_eps_tc
            mass_matrix_diagonal = factor * mass_matrix_diagonal
        
        # Clip values to specified range
        mass_matrix_diagonal = jnp.clip(mass_matrix_diagonal, clip_left, clip_right)
        
        # Finally, convert from diagonal to matrix
        mass_matrix = jnp.diag(mass_matrix_diagonal)
        self.mass_matrix = mass_matrix
        
        if self.verbose:
            print("Summary")
            for name, value in zip(self.naming, mass_matrix_diagonal):
                print(f"Parameter: {name}, value mass matrix diagonal: {value}")
        
        return mass_matrix
    
    # @staticmethod
    def normalize_matrix(self, matrix: Array):
        """
        Note: matrix must be mpmath matrix.

        Args:
            matrix (Array): Given matrix
        """
        
        # Some return values to avoid errors
        matrix_normalized = jnp.zeros((self.n_dim, self.n_dim))
        cond = 0
        ws = jnp.zeros(self.n_dim)
        
        # Compute the eigenvalues
        E, _ = mpmath.eigh(matrix)
        E = jnp.array(E.tolist(), dtype='float64')
        
        if jnp.any(E < 0):
            print('Matrix is not positive definite, has a negative eigenvalue!')
            print("Eigenvaleus:")
            print(E)

        cond = jnp.max(jnp.abs(E))/jnp.min(jnp.abs(E))
        if self.verbose:
            print('Condition of original matrix: %s' %cond)
            
        try:
            # Normalize by the diagonal
            ws =  mpmath.diag([1 / mpmath.sqrt(matrix[i, i]) for i in range(self.n_dim) ])
            if self.verbose:
                print("ws")
                print(ws)

            matrix_normalized = ws * matrix * ws
            # Conditioning of the new Fisher
            EE, _ = mpmath.eigh(matrix_normalized)
            E = jnp.array(EE.tolist(), dtype='float64')
            cond = jnp.max(jnp.abs(E))/jnp.min(jnp.abs(E))
            if self.verbose:
                print('Condition of the new matrix: %s' %cond)
            reweighted = True
    
        except ZeroDivisionError:
            print('The Fisher matrix has a zero element on the diagonal. The normalization procedure will not be applied. Consider using a prior.')
            matrix_normalized = matrix
            
        return matrix_normalized, cond, ws
            
    
    #################
    ### UTILITIES ###
    #################
    
    # TODO move this to a different place?
    
    @staticmethod
    def noise_weighted_inner_product(h: Array, g: Array, detector: Detector, df: float):
        return 4 * jnp.sum((jnp.conj(h) * g) / detector.psd * df).real
    
    @staticmethod
    def compute_inversion_error(matrix, inverse, only_diagonal = False):
        n = jnp.shape(matrix)[0]
        errors = jnp.abs(inverse[:, :] @ matrix[:, :] - jnp.eye(n))
        if only_diagonal:
            errors_diagonal = jnp.diag(errors)
            return jnp.max(errors_diagonal)
        else:
            return jnp.max(errors)