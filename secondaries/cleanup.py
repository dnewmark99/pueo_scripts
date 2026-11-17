import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import os
import json
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import glob

def find_expected_files(base_path = '/scratch/midway3/darcynewmark/secondaries/multipulse_metrics/jsons', max_array = 850):
    missing_arrays = []
    for a in range(max_array):
        this_file_path = f'{base_path}/secondary_metrics_{a}.json'
        if not os.path.exists(this_file_path):
            missing_arrays.append(a)
    print(f'missing_arrays = {missing_arrays}')

if __name__ == "__main__":
    find_expected_files()
