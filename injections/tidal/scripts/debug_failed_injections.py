import psutil
p = psutil.Process()
p.cpu_affinity([0])

import json 
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the failed injections
failed_injections_list = [137, 149, 72, 58, 144, 130, 84, 155, 134, 126, 152, 138]

outdir = "/home/thibeau.wouters/public_html/jim_injections/tests_TaylorF2_19_01_2024/"

# Go over the subdirectories of the failed injections
for failed_injection_idx in failed_injections_list:
    failed_injection_dir = f"{outdir}injection_{failed_injection_idx}/"
    print(f"Failed injection: {failed_injection_dir}")

    # Load the json file
    json_file = f"{failed_injection_dir}config.json"
    with open(json_file, "r") as f:
        data = json.load(f)
    # print(data)

    # # Load the posterior samples
    # posterior_samples_file = f"{failed_injection_dir}/posterior_samples.dat"
    # posterior_samples = np.loadtxt(posterior_samples_file)
    # print(posterior_samples.shape)

    # Read the snr.csv file
    snr_file = f"{failed_injection_dir}/snr.csv"
    # snr_data = np.loadtxt(snr_file, )
    df = pd.read_csv(snr_file)
    snr_data = np.array(df["snr"])
    network_snr_data = snr_data[-1]
    print(snr_data)
    