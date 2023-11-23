# import os
# import shutil
# from scipy.signal.windows import tukey
# import time
# from jimgw.jim import Jim
# from jimgw.detector import H1, L1, V1
# from jimgw.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
# from jimgw.waveform import RippleIMRPhenomD
# from jimgw.prior import Uniform
# import jax.numpy as jnp
# import jax
# jax.config.update("jax_enable_x64", True)

# from gwpy.timeseries import TimeSeries
# import urllib.request
# import numpy as np
# import matplotlib.pyplot as plt
# import corner

# # TODO move!!!
# default_corner_kwargs = dict(bins=40, 
#                         smooth=1., 
#                         show_titles=False,
#                         label_kwargs=dict(fontsize=16),
#                         title_kwargs=dict(fontsize=16), 
#                         color="blue",
#                         # quantiles=[],
#                         # levels=[0.9],
#                         plot_density=True, 
#                         plot_datapoints=False, 
#                         fill_contours=True,
#                         max_n_ticks=4, 
#                         min_n_ticks=3,
#                         save=False)

# params = {
#     "axes.labelsize": 30,
#     "axes.titlesize": 30,
#     "text.usetex": True,
#     "font.family": "serif",
# }
# plt.rcParams.update(params)

# labels = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$d_{\rm{L}}/{\rm Mpc}$',
#                r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

# ###########################################
# ########## First we grab data #############
# ###########################################

# total_time_start = time.time()

# gps = 1187008882.43

# T = 128
# # start = gps - T/2
# # end   = gps + T/2
# fmin = 20.0
# fmax = 2048.0
# psd_pad = 16 # doesn't matter if we overwrite the PSD from an external file

# ifos = ["H1", "L1"]
# H1.load_data(gps, T, 2, fmin, fmax, psd_pad=psd_pad, tukey_alpha=0.2)
# L1.load_data(gps, T, 2, fmin, fmax, psd_pad=psd_pad, tukey_alpha=0.2)
# V1.load_data(gps, T, 2, fmin, fmax, psd_pad=psd_pad, tukey_alpha=0.2)

# base_url = "https://raw.githubusercontent.com/ThibeauWouters/gw-datasets/main/"

# outdir_name = "./outdir/"

# def get_psd(filename):
#     print(f"Fetching PSD for {filename}")
#     psd_file = base_url + filename
    
#     with urllib.request.urlopen(psd_file) as response:
#         data = response.read().decode('utf-8')
        
#         lines = data.split('\n')
#         f, psd = [], []

#         for line in lines:
#             if line.strip():  # Check if the line is not empty
#                 columns = line.split()
#                 f.append(float(columns[0]))
#                 psd.append(float(columns[1]))
#         print(f"Shapes: {np.shape(f)}, {np.shape(psd)}")
    
#     return f, psd

# override_psd = True
# if override_psd:
#     psd_pad = 0
#     print("Overriding PSD")
#     ### Override the PSD
#     # Taking PSD values from Peter:
#     ## H1

#     # We fetch the data separately in order to build the same frequency array, in order to get the same mask
#     data_td = TimeSeries.fetch_open_data("H1", gps - T, gps + 2, cache=True)
#     segment_length = data_td.duration.value
#     n = len(data_td)
#     delta_t = data_td.dt.value
#     freq = jnp.fft.rfftfreq(n, delta_t)
#     freq = freq[(freq>fmin)&(freq<fmax)]
    
#     f, psd = get_psd("h1_psd.txt")
    
#     # Get at specific frequencies for jim
#     psd = np.interp(freq, f, psd)
#     H1.psd = psd

#     ## L1
#     data_td = TimeSeries.fetch_open_data("L1", gps - T, gps + 2, cache=True)
#     segment_length = data_td.duration.value
#     n = len(data_td)
#     delta_t = data_td.dt.value
#     freq = jnp.fft.rfftfreq(n, delta_t)
#     freq = freq[(freq>fmin)&(freq<fmax)]
            
#     f, psd = get_psd("l1_psd.txt")
    
#     # Get at specific frequencies for jim
#     psd = np.interp(freq, f, psd)
#     L1.psd = psd
    
#     ## V1
#     data_td = TimeSeries.fetch_open_data("V1", gps - T, gps + 2, cache=True)
#     segment_length = data_td.duration.value
#     n = len(data_td)
#     delta_t = data_td.dt.value
#     freq = jnp.fft.rfftfreq(n, delta_t)
#     freq = freq[(freq>fmin)&(freq<fmax)]
            
#     f, psd = get_psd("v1_psd.txt")
    
#     # Get at specific frequencies for jim
#     psd = np.interp(freq, f, psd)
#     V1.psd = psd


# ### Check the data

# ifos = [H1, L1, V1]
# names = ["H1", "L1", "V1"]

# tukey_alpha = 0.2

# for detector, name in zip(ifos, names):
    
#     # Get data
    
#     # f, d = detector.frequencies, detector.data
#     # d = detector.data
    
#     data_td = TimeSeries.fetch_open_data(name, gps - T, gps + 2, cache=True)
#     segment_length = data_td.duration.value
#     n = len(data_td)
#     delta_t = data_td.dt.value
#     data = jnp.fft.rfft(jnp.array(data_td.value) * tukey(n, tukey_alpha))*delta_t
#     freq = jnp.fft.rfftfreq(n, delta_t)
#     start_psd = int(trigger_time) - gps_start_pad - psd_pad # What does Int do here?
#     end_psd = int(trigger_time) + gps_end_pad + psd_pad

#     print("Fetching PSD data...")
#     psd_data_td = TimeSeries.fetch_open_data(self.name, start_psd, end_psd, cache=True)
#     psd = psd_data_td.psd(fftlength=segment_length).value # TODO: Check whether this is sright.