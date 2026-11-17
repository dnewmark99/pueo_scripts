import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ROOT
import scipy
import os
import json
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import functools
from functools import reduce
import glob
from scipy.optimize import minimize
import pandas as pd
import random

def sum_dicts(path):
    fnames = [name for name in glob.glob(path)]
    fnames = sorted(fnames)

    first_pass = True
    for f in fnames:
        with open(f) as ff:
            temp_data = json.load(ff)

            if first_pass == True:
                data = {key: np.asarray(value, dtype=float) for key, value in temp_data.items()}
                first_pass = False

            else:
                for key in temp_data:
                    data[key] += np.asarray(temp_data[key], dtype=float)
    return data

def plot_wfs(run_a = 719, run_b = 1420):
    ### we have already accumulated our wfs
    ### we need to read in jsons, accumulated across jsons, and plot per channel

    ### first get total number of entries for both runs
    run_a_entry_path = f'/scratch/midway3/darcynewmark/beamformer_testing/btsselectedevents_run{run_a}.json'
    with open(run_a_entry_path) as f:
        run_a_num_entries = len(json.load(f))

    run_b_entry_path = f'/scratch/midway3/darcynewmark/beamformer_testing/btsselectedevents_run{run_b}.json'
    with open(run_b_entry_path) as f:
        run_b_num_entries = len(json.load(f))

    ### read in our data
    path_a = f'/scratch/midway3/darcynewmark/beamformer_testing/accumulatedbtswfs_run{run_a}_chunk*.json'
    path_b = f'/scratch/midway3/darcynewmark/beamformer_testing/accumulatedbtswfs_run{run_b}_chunk*.json'

    data_a = sum_dicts(path_a)
    data_b = sum_dicts(path_b)

    ### now we can plot!
    for key in data_a:
        this_key = key
        this_key_data_a = data_a[key]
        this_key_data_b = data_b[key]

        times = np.arange(len(this_key_data_a))*(1/3.0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.minorticks_on()
        ax.plot(times, this_key_data_a / run_a_num_entries, color = '#21b2aa', label = f'Run {run_a}')
        ax.plot(times, this_key_data_b / run_b_num_entries, color = '#c71585', label = f'Run {run_b}')
        ax.set_title(f'Beamformer Test -- Antenna {key}', fontsize = 20)
        ax.set_xlabel(f'Time [ns]', fontsize = 20)
        ax.set_ylabel(f'Average Voltage [mV]', fontsize = 20)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.legend(fontsize=16, frameon=False)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/accumwf_{key}_run{run_a}_run{run_b}.png', dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    plot_wfs()

