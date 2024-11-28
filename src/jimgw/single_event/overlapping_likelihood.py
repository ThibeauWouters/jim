import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from astropy.time import Time
# from flowMC.utils.EvolutionaryOptimizer import EvolutionaryOptimizer
from jaxtyping import Array, Float
from scipy.interpolate import interp1d

from jimgw.single_event.detector import Detector
from jimgw.prior import Prior
from jimgw.single_event.waveform import Waveform
import time
from jimgw.single_event.likelihood import SingleEventLiklihood, HeterodynedTransientLikelihoodFD

class DoubleTransientLikelihoodFD(SingleEventLiklihood):
    def __init__(
        self,
        detectors: list[Detector],
        waveform: Waveform,
        trigger_time: float = 0,
        duration: float = 4,
        post_trigger_duration: float = 2,
        **kwargs,
    ) -> None:
        self.detectors = detectors
        assert jnp.all(
            jnp.array(
                [
                    (self.detectors[0].frequencies == detector.frequencies).all()  # type: ignore
                    for detector in self.detectors
                ]
            )
        ), "The detectors must have the same frequency grid"
        self.frequencies = self.detectors[0].frequencies  # type: ignore
        self.waveform = waveform
        self.trigger_time = trigger_time
        self.required_keys = []
        self.required_keys += [f"{k}_1" for k in waveform.required_keys]
        self.required_keys += [f"{k}_2" for k in waveform.required_keys]
        self.required_keys += ["t_c_1", "psi_1", "ra_1", "dec_1"]
        self.required_keys += ["t_c_2", "psi_2", "ra_2", "dec_2"]
        
        self.gmst = (
            Time(trigger_time, format="gps").sidereal_time("apparent", "greenwich").rad
        )

        self.trigger_time = trigger_time
        self.duration = duration
        self.post_trigger_duration = post_trigger_duration
        self.kwargs = kwargs
        
        # the fixing_parameters is expected to be a dictionary
        # with key as parameter name and value is the fixed value
        # e.g. {'M_c': 1.1975, 't_c': 0}
        if "fixing_parameters" in self.kwargs:
            print("Note: likelihood will fix some parameters!")
            fixing_parameters = self.kwargs["fixing_parameters"]
            print(f"Parameters are fixed {fixing_parameters}")
            self.fixing_func = lambda x: {**x, **fixing_parameters}
        else:
            self.fixing_func = lambda x: x

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

    def extract_params(
        self, params: dict[str, Float]
    ) -> tuple[dict[str, Float], dict[str, Float]]:
        """
        Extract the parameters for the two components of the binary.
        """
        params_1 = {key[:-2]: params[key] for key in params if key[-1] == "1"}
        params_2 = {key[:-2]: params[key] for key in params if key[-1] == "2"}
        params_1["gmst"] = self.gmst
        params_2["gmst"] = self.gmst
        return params_1, params_2

    def evaluate_original(self, 
                          params_1: dict[str, Float], 
                          params_2: dict[str, Float], 
                          h_sky_1: dict[str, Float[Array, " n_dim"]],
                          h_sky_2: dict[str, Float[Array, " n_dim"]],
                          detectors: list[Detector],
                          freqs: Float[Array, " n_dim"],
                          align_time_1: Float,
                          align_time_2: Float,
                          **kwargs,) -> Float:
        """
        Evaluate the likelihood for a given set of parameters.
        """
        log_likelihood = 0.0
        df = freqs[1] - freqs[0]

        for detector in detectors:
            waveform_dec_1 = (
                detector.fd_response(freqs, h_sky_1, params_1)
                * align_time_1
            )
            waveform_dec_2 = (
                detector.fd_response(freqs, h_sky_2, params_2)
                * align_time_2
            )
            match_filter_SNR_1 = (
                4
                * jnp.sum(
                    (jnp.conj(waveform_dec_1) * detector.data) / detector.psd * df
                ).real
            )
            match_filter_SNR_2 = (
                4
                * jnp.sum(
                    (jnp.conj(waveform_dec_2) * detector.data) / detector.psd * df
                ).real
            )
            cross_term_1_2 = (
                4
                * jnp.sum(
                    (jnp.conj(waveform_dec_1) * waveform_dec_2) / detector.psd * df
                ).real
            )
            optimal_SNR_1 = (
                4
                * jnp.sum(
                    jnp.conj(waveform_dec_1) * waveform_dec_1 / detector.psd * df
                ).real
            )
            optimal_SNR_2 = (
                4
                * jnp.sum(
                    jnp.conj(waveform_dec_2) * waveform_dec_2 / detector.psd * df
                ).real
            )
            log_likelihood += (
                match_filter_SNR_1
                - optimal_SNR_1 / 2
                + match_filter_SNR_2
                - optimal_SNR_2 / 2
                - cross_term_1_2
            )
        return log_likelihood
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        """
        Evaluate the likelihood for a given set of parameters.
        """
        frequencies = self.frequencies
        params["gmst"] = self.gmst
        # adjust the params due to fixing parameters
        params = self.fixing_func(params)
        params_1, params_2 = self.extract_params(params)
        # evaluate the waveform as usual
        waveform_sky_1 = self.waveform(frequencies, params_1)
        waveform_sky_2 = self.waveform(frequencies, params_2)
        align_time_1 = jnp.exp(
            -1j * 2 * jnp.pi * frequencies * (self.epoch + params["t_c_1"])
        )
        align_time_2 = jnp.exp(
            -1j * 2 * jnp.pi * frequencies * (self.epoch + params["t_c_2"])
        )
        log_likelihood = self.evaluate_original(
            params_1,
            params_2,
            waveform_sky_1,
            waveform_sky_2,
            self.detectors,
            frequencies,
            align_time_1,
            align_time_2,
            **self.kwargs,
        )
        return log_likelihood

class HeterodynedDoubleTransientLikelihoodFD(DoubleTransientLikelihoodFD):
    n_bins: int  # Number of bins to use for the likelihood
    ref_params: dict  # Reference parameters for the likelihood
    freq_grid_low: Array  # Heterodyned frequency grid
    freq_grid_center: Array  # Heterodyned frequency grid at the center of the bin
    waveform_low_ref_1: dict[
        str, Float[Array, " n_bin"]
    ]  # Reference waveform at the low edge of the frequency bin, keyed by detector name
    waveform_center_ref_1: dict[
        str, Float[Array, " n_bin"]
    ]  # Reference waveform at the center of the frequency bin, keyed by detector name
    waveform_low_ref_2: dict[
        str, Float[Array, " n_bin"]
    ]  # Reference waveform at the low edge of the frequency bin, keyed by detector name
    waveform_center_ref_2: dict[
        str, Float[Array, " n_bin"]
    ]  # Reference waveform at the center of the frequency bin, keyed by detector name
    A0_array: dict[
        str, Float[Array, " n_bin"]
    ]  # A0 array for the likelihood, keyed by detector name
    A1_array: dict[
        str, Float[Array, " n_bin"]
    ]  # A1 array for the likelihood, keyed by detector name
    B0_array: dict[
        str, Float[Array, " n_bin"]
    ]  # B0 array for the likelihood, keyed by detector name
    B1_array: dict[
        str, Float[Array, " n_bin"]
    ]  # B1 array for the likelihood, keyed by detector name

    def __init__(
        self,
        detectors: list[Detector],
        waveform: Waveform,
        prior: Prior,
        bounds: Float[Array, " n_dim 2"],
        n_bins: int = 100,
        trigger_time: float = 0,
        duration: float = 4,
        post_trigger_duration: float = 2,
        popsize: int = 100,
        n_loops: int = 2000,
        ref_params=None,
        save_binning_scheme: bool = False,
        save_binning_scheme_location: str = "./",
        save_binning_scheme_name: str = "freq_grid",
        reference_waveform: Waveform = None,
        outdir_name: str = "./outdir/",
        **kwargs,
    ) -> None:

        super().__init__(
            detectors, waveform, trigger_time, duration, post_trigger_duration, **kwargs
        )

        if reference_waveform is None:
            reference_waveform = self.waveform

        self.reference_waveform = reference_waveform
        self.outdir_name = outdir_name

        print("Initializing heterodyned double transient likelihood..")

        # Get the original frequency grid

        frequency_original = self.frequencies
        # Get the grid of the relative binning scheme (contains the final endpoint)
        # and the center points
        
        # TODO: the binning scheme assumes a single inspiral, could this impact the results?
        freq_grid, self.freq_grid_center = self.make_binning_scheme(
            np.array(frequency_original), n_bins
        )
        if save_binning_scheme:
            filename = f"{save_binning_scheme_location}{save_binning_scheme_name}.npz"
            print(f"Saving the relative binning scheme to {filename}...")
            np.savez(
                filename, freq_grid=freq_grid, freq_grid_center=self.freq_grid_center
            )

        self.freq_grid_low = freq_grid[:-1]

        if ref_params is not None:
            print("Setting relative binning with given reference parameters:")
            self.ref_params = ref_params
        else:
            print("Finding reference parameters with evosax..")
            self.ref_params = self.maximize_likelihood(
                bounds=bounds, prior=prior, popsize=popsize, n_loops=n_loops
            )

        print("Ref params used:")
        print(self.ref_params)

        print("Constructing reference waveforms..")

        self.ref_params["gmst"] = self.gmst

        self.waveform_low_ref_1 = {}
        self.waveform_center_ref_1 = {}
        self.waveform_low_ref_2 = {}
        self.waveform_center_ref_2 = {}
        self.A0_array = {}
        self.A1_array = {}
        self.B0_array = {}
        self.B1_array = {}

        params_1, params_2 = self.extract_params(self.ref_params)
        h_sky_1 = self.reference_waveform(frequency_original, params_1)
        h_sky_2 = self.reference_waveform(frequency_original, params_2)

        # Get frequency masks to be applied, for both original
        # and heterodyne frequency grid -- check both WFs
        h_amp = jnp.sum(
            jnp.array([jnp.abs(h_sky_1[key] * h_sky_2[key]) for key in h_sky_1.keys()]), axis=0
        )
        f_valid = frequency_original[jnp.where(h_amp > 0)[0]]
        f_max = jnp.max(f_valid)
        f_min = jnp.min(f_valid)

        mask_heterodyne_grid = jnp.where((freq_grid <= f_max) & (freq_grid >= f_min))[0]
        mask_heterodyne_low = jnp.where(
            (self.freq_grid_low <= f_max) & (self.freq_grid_low >= f_min)
        )[0]
        mask_heterodyne_center = jnp.where(
            (self.freq_grid_center <= f_max) & (self.freq_grid_center >= f_min)
        )[0]
        freq_grid = freq_grid[mask_heterodyne_grid]
        self.freq_grid_low = self.freq_grid_low[mask_heterodyne_low]
        self.freq_grid_center = self.freq_grid_center[mask_heterodyne_center]

        # Assure frequency grids have same length
        if len(self.freq_grid_low) > len(self.freq_grid_center):
            self.freq_grid_low = self.freq_grid_low[: len(self.freq_grid_center)]

        h_sky_low_1 = self.reference_waveform(self.freq_grid_low, params_1)
        h_sky_center_1 = self.reference_waveform(self.freq_grid_center, params_1)
        
        h_sky_low_2 = self.reference_waveform(self.freq_grid_low, params_2)
        h_sky_center_2 = self.reference_waveform(self.freq_grid_center, params_2)

        # Get phase shifts to align time of coalescence
        align_time_1 = jnp.exp(
            -1j
            * 2
            * jnp.pi
            * frequency_original
            * (self.epoch + self.ref_params["t_c_1"])
        )
        align_time_low_1 = jnp.exp(
            -1j
            * 2
            * jnp.pi
            * self.freq_grid_low
            * (self.epoch + self.ref_params["t_c_1"])
        )
        align_time_center_1 = jnp.exp(
            -1j
            * 2
            * jnp.pi
            * self.freq_grid_center
            * (self.epoch + self.ref_params["t_c_1"])
        )
        
        align_time_2 = jnp.exp(
            -1j
            * 2
            * jnp.pi
            * frequency_original
            * (self.epoch + self.ref_params["t_c_2"])
        )
        align_time_low_2 = jnp.exp(
            -1j
            * 2
            * jnp.pi
            * self.freq_grid_low
            * (self.epoch + self.ref_params["t_c_2"])
        )
        align_time_center_2 = jnp.exp(
            -1j
            * 2
            * jnp.pi
            * self.freq_grid_center
            * (self.epoch + self.ref_params["t_c_2"])
        )
        
        self.align_time_1 = align_time_1
        self.align_time_2 = align_time_2
        
        self.align_time_low_1 = align_time_low_1
        self.align_time_low_2 = align_time_low_2
        
        self.align_time_center_1 = align_time_center_1
        self.align_time_center_2 = align_time_center_2

        for detector in self.detectors:
            # Get the reference waveforms
            waveform_ref_1 = (
                detector.fd_response(frequency_original, h_sky_1, params_1)
                * align_time_1
            )
            waveform_ref_2 = (
                detector.fd_response(frequency_original, h_sky_2, params_2)
                * align_time_2
            )
            self.waveform_low_ref_1[detector.name] = (
                detector.fd_response(self.freq_grid_low, h_sky_low_1, params_1)
                * align_time_low_1
            )
            self.waveform_center_ref_1[detector.name] = (
                detector.fd_response(
                    self.freq_grid_center, h_sky_center_1, params_1
                )
                * align_time_center_1
            )
            self.waveform_low_ref_2[detector.name] = (
                detector.fd_response(self.freq_grid_low, h_sky_low_2, params_2)
                * align_time_low_2
            )
            self.waveform_center_ref_2[detector.name] = (
                detector.fd_response(
                    self.freq_grid_center, h_sky_center_2, params_2
                )
                * align_time_center_2
            )
            A0, A1, B0, B1 = HeterodynedTransientLikelihoodFD.compute_coefficients(
                detector.data,
                waveform_ref_1 + waveform_ref_2,
                detector.psd,
                frequency_original,
                freq_grid,
                self.freq_grid_center,
            )

            self.A0_array[detector.name] = A0[mask_heterodyne_center]
            self.A1_array[detector.name] = A1[mask_heterodyne_center]
            self.B0_array[detector.name] = B0[mask_heterodyne_center]
            self.B1_array[detector.name] = B1[mask_heterodyne_center]

    def make_binning_scheme(
        self, freqs: npt.NDArray[np.float_], n_bins: int, chi: float = 1
    ) -> tuple[Float[Array, " n_bins+1"], Float[Array, " n_bins"]]:
        """
        Make a binning scheme based on the maximum phase difference between the
        frequencies in the array.

        Parameters
        ----------
        freqs: Float[Array, "dim"]
            Array of frequencies to be binned.
        n_bins: int
            Number of bins to be used.
        chi: float = 1
            The chi parameter used in the phase difference calculation.

        Returns
        -------
        f_bins: Float[Array, "n_bins+1"]
            The bin edges.
        f_bins_center: Float[Array, "n_bins"]
            The bin centers.
        """

        phase_diff_array = self.max_phase_diff(freqs, freqs[0], freqs[-1], chi=chi)
        bin_f = interp1d(phase_diff_array, freqs)
        f_bins = np.array([])
        for i in np.linspace(phase_diff_array[0], phase_diff_array[-1], n_bins + 1):
            f_bins = np.append(f_bins, bin_f(i))
        f_bins_center = (f_bins[:-1] + f_bins[1:]) / 2
        return jnp.array(f_bins), jnp.array(f_bins_center)
    
    @staticmethod
    def max_phase_diff(
        f: npt.NDArray[np.float_],
        f_low: float,
        f_high: float,
        chi: Float = 1.0,
    ):
        """
        Compute the maximum phase difference between the frequencies in the array.

        Parameters
        ----------
        f: Float[Array, "n_dim"]
            Array of frequencies to be binned.
        f_low: float
            Lower frequency bound.
        f_high: float
            Upper frequency bound.
        chi: float
            Power law index.

        Returns
        -------
        Float[Array, "n_dim"]
            Maximum phase difference between the frequencies in the array.
        """

        gamma = np.arange(-5, 6, 1) / 3.0
        f = np.repeat(f[:, None], len(gamma), axis=1)
        f_star = np.repeat(f_low, len(gamma))
        f_star[gamma >= 0] = f_high
        return 2 * np.pi * chi * np.sum((f / f_star) ** gamma * np.sign(gamma), axis=1)
    
    def maximize_likelihood(
        self,
        bounds: Float[Array, " n_dim 2"],
        prior: Prior,
        popsize: int = 200,
        n_loops: int = 2000,
    ):
        raise NotImplementedError
        
        # TODO: check if this works with the newer jax version
        # def y(x):
        #     return -self.evaluate_original(prior.transform(prior.add_name(x)), {})

        # start_time = time.time()
        # y = jax.jit(jax.vmap(y))

        # print("Starting the optimizer")
        # optimizer = EvolutionaryOptimizer(len(bounds), popsize=popsize, verbose=True)
        # _ = optimizer.optimize(y, bounds, n_loops=n_loops)
        # best_fit = optimizer.get_result()[0]
        # end_time = time.time()

        # elapsed_time = end_time - start_time
        # with open(f"{self.outdir_name}runtime_evosax.txt", "w") as f:
        #     f.write(str(elapsed_time))
        # print(
        #     f"Optimization time: {elapsed_time} seconds, {elapsed_time / 60} minutes."
        # )
        # return prior.transform(prior.add_name(best_fit))

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        log_likelihood = 0
        frequencies_low = self.freq_grid_low
        frequencies_center = self.freq_grid_center
        params["gmst"] = self.gmst
        params = self.fixing_func(params)
        params_1, params_2 = self.extract_params(params)
        waveform_sky_low_1 = self.waveform(frequencies_low, params_1)
        waveform_sky_center_1 = self.waveform(frequencies_center, params_1)
        waveform_sky_low_2 = self.waveform(frequencies_low, params_2)
        waveform_sky_center_2 = self.waveform(frequencies_center, params_2)
        align_time_low_1 = jnp.exp(
            -1j * 2 * jnp.pi * frequencies_low * (self.epoch + params["t_c_1"])
        )
        align_time_center_1 = jnp.exp(
            -1j * 2 * jnp.pi * frequencies_center * (self.epoch + params["t_c_1"])
        )
        align_time_low_2 = jnp.exp(
            -1j * 2 * jnp.pi * frequencies_low * (self.epoch + params["t_c_2"])
        )
        align_time_center_2 = jnp.exp(
            -1j * 2 * jnp.pi * frequencies_center * (self.epoch + params["t_c_2"])
        )
        for detector in self.detectors:
            waveform_low_1 = (
                detector.fd_response(frequencies_low, waveform_sky_low_1, params_1)
                * align_time_low_1
            )
            waveform_center_1 = (
                detector.fd_response(frequencies_center, waveform_sky_center_1, params_1)
                * align_time_center_1
            )
            waveform_low_2 = (
                detector.fd_response(frequencies_low, waveform_sky_low_2, params_2)
                * align_time_low_2
            )
            waveform_center_2 = (
                detector.fd_response(frequencies_center, waveform_sky_center_2, params_2)
                * align_time_center_2
            )

            waveform_center = waveform_center_1 + waveform_center_2
            waveform_low = waveform_low_1 + waveform_low_2
            
            waveform_center_ref = self.waveform_center_ref_1[detector.name] + self.waveform_center_ref_2[detector.name]
            waveform_low_ref = self.waveform_low_ref_1[detector.name] + self.waveform_low_ref_2[detector.name]
            
            r0 = waveform_center / waveform_center_ref
            r1 = (waveform_low / waveform_low_ref - r0) / (
                frequencies_low - frequencies_center)
            
            match_filter_SNR = jnp.sum(
                self.A0_array[detector.name] * (r0).conj()
                + self.A1_array[detector.name] * (r1).conj()
            )
            optimal_SNR = jnp.sum(
                self.B0_array[detector.name] * jnp.abs(r0) ** 2
                + 2 * self.B1_array[detector.name] * ((r0) * (r1).conj()).real
            )
            log_likelihood += (match_filter_SNR - optimal_SNR / 2).real

        return log_likelihood