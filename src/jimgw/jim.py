from jaxtyping import Array
import jax
import jax.numpy as jnp

from flowMC.sampler.Sampler import Sampler
from flowMC.sampler.MALA import MALA
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline, RQSpline
from flowMC.nfmodel.mlp import MLP
from flowMC.utils.EvolutionaryOptimizer import EvolutionaryOptimizer
from flowMC.utils.PRNG_keys import initialize_rng_keys



from jimgw.prior import Prior
from jimgw.likelihood import LikelihoodBase
from jimgw.hyperparameters import jim_default_hyperparameters

import numpy as np

class Jim(object):
    """
    Master class for interfacing with flowMC
    """

    # TODO make hyperparameters more grouped together
    def __init__(self, likelihood: LikelihoodBase, prior: Prior, **kwargs):
        self.Likelihood = likelihood
        self.Prior = prior
        
        # Set and override any given hyperparameters, and save as attribute
        self.hyperparameters = jim_default_hyperparameters
        hyperparameter_names = list(self.hyperparameters.keys())
        
        for key, value in kwargs.items():
            if key in hyperparameter_names:
                self.hyperparameters[key] = value
        
        for key, value in self.hyperparameters.items():
            setattr(self, key, value)
            
        self.rng_key_set = initialize_rng_keys(self.hyperparameters["n_chains"], seed=self.hyperparameters["seed"])
        local_sampler = MALA(self.posterior, True, self.local_sampler_arg)
        
        ### New attempt
        key = jax.random.PRNGKey(123456789)
        key, conditioner_key= jax.random.split(key)
        n_features = self.Prior.n_dim
        hidden_size = self.hidden_size
        num_bins = self.num_bins
        conditioner = MLP([n_features]+hidden_size+ [n_features*(num_bins*3+1)], conditioner_key, scale=1e-2, activation=jax.nn.tanh)
        model = RQSpline(conditioner, -10.0, 10.0)
        
        ### Old code
        # TODO swap here
        model = MaskedCouplingRQSpline(self.Prior.n_dim, self.num_layers, self.hidden_size, self.num_bins, self.rng_key_set[-1])
        
        self.Sampler = Sampler(
            self.Prior.n_dim,
            self.rng_key_set,
            None,
            local_sampler,
            model,
            **kwargs)
        
        

    def maximize_likelihood(self, bounds: tuple[Array,Array], seed = 92348):
        bounds = jnp.array(bounds).T
        key = jax.random.PRNGKey(seed)
        
        set_nwalkers = self.n_walkers_maximize_likelihood
        n_loops = self.n_loops_maximize_likelihood
        # set_nwalkers = set_nwalkers
        
        initial_guess = self.Prior.sample(key, set_nwalkers)

        y = lambda x: -self.posterior(x, None)
        y = jax.jit(jax.vmap(y))
        print("Compiling likelihood function")
        y(initial_guess)
        print("Done compiling")

        print("Starting the optimizer")
        optimizer = EvolutionaryOptimizer(self.Prior.n_dim, verbose = True)
        state = optimizer.optimize(y, bounds, n_loops=n_loops)
        best_fit_params = optimizer.get_result()[0]
        # Save best fit params to jim object
        self.best_fit_params = best_fit_params
        return best_fit_params
    
    def posterior(self, params: Array, data: dict):
        named_params = self.Prior.add_name(params, transform_name=True, transform_value=True)
        return self.Likelihood.evaluate(named_params, data) + self.Prior.log_prob(params)

    def sample(self, key: jax.random.PRNGKey,
               initial_guess: Array = None):
        if initial_guess is None:
            initial_guess = self.Prior.sample(key, self.Sampler.n_chains)
        self.Sampler.sample(initial_guess, None)

    def print_summary(self, save: bool=True) -> None:
        """
        Generate summary of the run
        """

        # Return value will be a long string
        summary_str = ""        

        pretrain_summary = self.Sampler.get_sampler_state("pretraining")
        train_summary = self.Sampler.get_sampler_state("training")
        production_summary = self.Sampler.get_sampler_state("production")

        pretraining_chain: Array = pretrain_summary["chains"]
        pretraining_log_prob: Array = pretrain_summary["log_prob"]
        pretraining_local_acceptance: Array = pretrain_summary["local_accs"]
        
        training_chain: Array = train_summary["chains"]
        training_log_prob: Array = train_summary["log_prob"]
        training_local_acceptance: Array = train_summary["local_accs"]
        training_global_acceptance: Array = train_summary["global_accs"]
        training_loss: Array = train_summary["loss_vals"]

        production_chain: Array = production_summary["chains"]
        production_log_prob: Array = production_summary["log_prob"]
        production_local_acceptance: Array = production_summary["local_accs"]
        production_global_acceptance: Array = production_summary["global_accs"]

        if self.Sampler.n_loop_pretraining > 0:
            summary_str += "Pretraining summary\n"
            summary_str += '=' * 10 + "\n"
            for index in range(len(self.Prior.naming)):
                summary_str += f"{self.Prior.naming[index]}: {pretraining_chain[:, :, index].mean():.3f} +/- {pretraining_chain[:, :, index].std():.3f}\n"
            summary_str += f"Log probability: {pretraining_log_prob.mean():.3f} +/- {pretraining_log_prob.std():.3f}\n"
            summary_str += f"Local acceptance: {pretraining_local_acceptance.mean():.3f} +/- {pretraining_local_acceptance.std():.3f}\n"

        if self.Sampler.n_loop_training > 0 and self.Sampler.use_global:
            summary_str += "Training summary\n"
            summary_str += '=' * 10 + "\n"
            for index in range(len(self.Prior.naming)):
                summary_str += f"{self.Prior.naming[index]}: {training_chain[:, :, index].mean():.3f} +/- {training_chain[:, :, index].std():.3f}\n"
            summary_str += f"Log probability: {training_log_prob.mean():.3f} +/- {training_log_prob.std():.3f}\n"
            summary_str += f"Local acceptance: {training_local_acceptance.mean():.3f} +/- {training_local_acceptance.std():.3f}\n"
            summary_str += f"Global acceptance: {training_global_acceptance.mean():.3f} +/- {training_global_acceptance.std():.3f}\n"
            summary_str += f"Max loss: {training_loss.max():.3f}, Min loss: {training_loss.min():.3f}\n"

        if self.Sampler.n_loop_production > 0:
            summary_str += "Production summary\n"
            summary_str += '=' * 10 + "\n"
            for index in range(len(self.Prior.naming)):
                summary_str += f"{self.Prior.naming[index]}: {production_chain[:, :, index].mean():.3f} +/- {production_chain[:, :, index].std():.3f}\n"
            summary_str += f"Log probability: {production_log_prob.mean():.3f} +/- {production_log_prob.std():.3f}\n"
            summary_str += f"Local acceptance: {production_local_acceptance.mean():.3f} +/- {production_local_acceptance.std():.3f}\n"
            summary_str += f"Global acceptance: {production_global_acceptance.mean():.3f} +/- {production_global_acceptance.std():.3f}\n"

        print(summary_str)
        if save:
            file_path = self.Sampler.outdir_name + "summary_output.txt"
            with open(file_path, "w") as file:
                file.write(summary_str)
            print(f"Summary has been written to {file_path}")
            

    def get_samples(self, training: bool = False) -> dict:
        """
        Get the samples from the sampler

        Args:
            training (bool, optional): If True, return the training samples. Defaults to False.

        Returns:
            Array: Samples
        """
        if training:
            chains = self.Sampler.get_sampler_state("training")["chains"]
        else:
            chains = self.Sampler.get_sampler_state("production")["chains"]

        chains = self.Prior.add_name(chains.transpose(2,0,1), transform_name=True)
        return chains
    
    def save_hyperparameters(self):
        import json
        
        # TODO automatically change any ArrayImpl to np array?
        if "step_size" in self.hyperparameters["local_sampler_arg"].keys():
            self.hyperparameters["local_sampler_arg"]["step_size"] = np.asarray(self.hyperparameters["local_sampler_arg"]["step_size"])
        
        hyperparameters_dict = {"flowmc": self.Sampler.hyperparameters,
                                "jim": self.hyperparameters}
        
        print(hyperparameters_dict)
        
        name = self.Sampler.outdir_name + "hyperparams.json"
        with open(name, 'w') as file:
            json.dump(hyperparameters_dict, file)

    def plot(self):
        pass