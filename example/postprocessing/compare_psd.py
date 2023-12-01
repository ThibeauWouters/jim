import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# jim
from jimgw.jim import Jim
from jimgw.detector import L1, V1
from jimgw.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.waveform import RippleTaylorF2
from jimgw.prior import Uniform
from jimgw.fisher_information_matrix import FisherInformationMatrix
# ripple
# flowmc
from flowMC.utils.PRNG_keys import initialize_rng_keys
# jax
import jax.numpy as jnp
import jax

# others
import time
import numpy as np
from lal import GreenwichMeanSiderealTime
jax.config.update("jax_enable_x64", True)
from astropy.time import Time

import shutil
import numpy as np
import matplotlib.pyplot as plt
import corner

### Data locations:

data_location = "../GW190425_TaylorF2/data/"

data_dict = {"L1":{"data": data_location + "L-L1_HOFT_C01_T1700406_v3-1240211456-4096.gwf",
                   "psd": data_location + "glitch_median_PSD_forLI_L1_srate8192.txt",
                   "channel": "DCS-CALIB_STRAIN_CLEAN_C01_T1700406_v3"},
            "V1":{"data": data_location + "V-V1Online_T1700406_v3-1240214000-2000.gwf",
                    "psd": data_location + "glitch_median_PSD_forLI_V1_srate8192.txt",
                    "channel": "Hrec_hoft_16384Hz_T1700406_v3"}
}

params = {
    "axes.labelsize": 22,
    "axes.titlesize": 22,
    "text.usetex": True,
    "font.family": "serif",
}
plt.rcParams.update(params)

# new data: from https://gwosc.org/eventapi/html/O3_Discovery_Papers/GW190425/v1/
new_data_location = "../GW190425_TaylorF2/new_data/"

new_data_dict = {"L1":{"data": new_data_location + "L-L1_HOFT_C01_T1700406_v3-1240211456-4096.gwf",
                   "psd": new_data_location + "L1-psd.dat",
                   "channel": "DCS-CALIB_STRAIN_CLEAN_C01"},
            "V1":{"data": new_data_location + "V-V1Online_T1700406_v3-1240214000-2000.gwf",
                    "psd": new_data_location + "V1-psd.dat",
                    "channel": "Hrec_hoft_16384Hz"}
}

### Get the PSDs:

psd_dict = {"L1":{},"V1":{}}
for name, my_dict in zip(["Old PSD","New PSD"],[data_dict,new_data_dict]):
    for ifo in psd_dict.keys():
        filename = my_dict[ifo]["psd"]
        print(f"Reading data from {filename}")
        f, psd = np.genfromtxt(filename).T
        psd_dict[ifo][name] = (f, psd)

### Plot it

colors = ["red", "blue"]

for ifo in psd_dict.keys():
    plt.figure()
    for i, name in enumerate(psd_dict[ifo].keys()):
        f, psd = psd_dict[ifo][name]
        # Limit for frequencies above 20 
        mask = f>20
        f = f[mask]
        psd = psd[mask]
        plt.loglog(f, psd, label=name, color=colors[i])
    plt.legend()
    plt.title(ifo)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.savefig(f"./outdir/psd_comparison_{ifo}.png")
    plt.close()