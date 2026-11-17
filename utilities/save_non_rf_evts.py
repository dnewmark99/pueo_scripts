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
from scipy.optimize import curve_fit
import argparse

import pueoTALON.pueoPythonAnalyzer.waveform as waveform
import pueoTALON.pueoPythonAnalyzer.plotter as pltr
import pueoTALON.pueoPythonAnalyzer.antenna as antenna
import pueoTALON.pueoPythonAnalyzer.correlator as correlator
import antennaUtils.response as antResponse
import pueoTALON.pueoPythonAnalyzer.utils as utils
import pueoTALON.pueoPythonAnalyzer.geography as geography

ROOT.gSystem.Load("libpueoEvent.so")


def save_force_triggers(run_start = 783, run_end = 808):
    ### grab entries that are from force triggers
    force_runs = []
    force_entries = []
    rf_runs = []
    rf_entries = []
    for run in range(run_start, run_end+1):
        channels = waveform.Channels(run, 0) ### make channels object
        N_entry = channels.dataset.N() ### get total number of entries per run

        for ev in range(N_entry):
            channels.load_event(ev) ### load our event
            trigger_type = channels.dataset.header().trigType 
            if trigger_type != 1:
                ### 1 is main instrument RF trigger...anything else is a force trigger!
                force_runs.append(run)
                force_entries.append(ev)
            else:
                rf_runs.append(run)
                rf_entries.append(ev)


    ### now let's save our list of force triggers!
    saving_path = f'/scratch/midway3/darcynewmark/useful/nonrftriggers_run{run_start}_run{run_end}.json'
    data = {
        'run': force_runs,
        'entry': force_entries,
    }
    with open(saving_path, 'w') as f:
        json.dump(data, f, indent=4)

    ### and save list of rf triggers!
    saving_path = f'/scratch/midway3/darcynewmark/useful/rftriggers_run{run_start}_run{run_end}.json'
    data = {
        'run': rf_runs,
        'entry': rf_entries,
    }
    with open(saving_path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    save_force_triggers(run_start = 783, run_end = 1402)
