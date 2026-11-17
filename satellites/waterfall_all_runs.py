import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
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
import time

import pueoTALON.pueoPythonAnalyzer.eventSummary as summary
import pueoTALON.pueoPythonAnalyzer.waveform as waveform
import pueoTALON.pueoPythonAnalyzer.plotter as pltr
import pueoTALON.pueoPythonAnalyzer.antenna as antenna
import pueoTALON.pueoPythonAnalyzer.correlator as correlator
import antennaUtils.response as antResponse
import pueoTALON.pueoPythonAnalyzer.utils as utils
import pueoTALON.pueoPythonAnalyzer.geography as geography

ROOT.gSystem.Load("libpueoEvent.so")

def mask_runs_with_no_muos_files(runs):
    mask = []
    for r in runs:
        if r < 1105:
            mask.append(True)
        else:
            mask.append(False)
    return np.asarray(mask)

def save_waterfalls(array, ev_per_array = 4000, ground_cal_name = 'TD'):
    if ground_cal_name == 'TD':
        source_loc_name = 'taylor_dome'

    all_runs = []
    all_entries = []
    all_flags = []
    all_groundcal_pol = []

    ### use some force triggers from ALL runs!
    path = f'/scratch/midway3/darcynewmark/useful/nonrftriggers_run783_run1402.json'
    with open(path) as f:
        force_dict = json.load(f)
    force_runs = np.asarray(force_dict['run'])
    force_evs = np.asarray(force_dict['entry'])
    ### mask out runs w no muos root files for now
    muos_mask = mask_runs_with_no_muos_files(force_runs)
    force_runs = force_runs[muos_mask]
    force_evs = force_evs[muos_mask]
    total_forcetrigger_events = len(force_evs)
    [all_runs.append(int(r)) for r in force_runs]
    [all_entries.append(int(e)) for e in force_evs]
    [all_flags.append('Force') for x in range(len(force_runs))]
    [all_groundcal_pol.append(0) for x in range(len(force_runs))]

    ### also get some ground cal events
    groundcal_events = np.genfromtxt(os.environ['PUEO_ANALYSIS_TOOLS'] + f'/pueoTALON/lookup/my{ground_cal_name}_list.csv', delimiter=',', names=True, dtype=None)
    groundcal_runs = np.asarray(groundcal_events['run'])
    groundcal_evs = np.asarray(groundcal_events['entry_number'])
    groundcal_pol = np.asarray(groundcal_events['pol'])
    total_groundcal_events = len(groundcal_runs)
    [all_runs.append(int(r)) for r in groundcal_runs]
    [all_entries.append(int(e)) for e in groundcal_evs]
    [all_flags.append(ground_cal_name) for x in range(len(groundcal_runs))]
    [all_groundcal_pol.append(p) for p in groundcal_pol]

    print(f'total of {len(all_runs)} events to process!')
    print(f'total of {total_groundcal_events} ground cal and {total_forcetrigger_events} force triggers events !')

    ### let's chunk up run, entry for this array!
    all_runs = np.asarray(all_runs)
    all_entries = np.asarray(all_entries)
    all_flags = np.asarray(all_flags)
    all_groundcal_pol = np.asarray(all_groundcal_pol)
    start_idx = array * ev_per_array
    end_idx = (array + 1) * ev_per_array
    print(f'for array = {array}, processing {start_idx} to {end_idx}')

    if start_idx > len(all_runs):
        print(f'oops! start idx is more than total number of events! bye bye!')
        return

    this_array_runs = all_runs[start_idx:end_idx]
    this_array_entries = all_entries[start_idx:end_idx]
    this_array_flags = all_flags[start_idx:end_idx]
    this_array_groundcal_pol = all_groundcal_pol[start_idx:end_idx]

    ### some general things to save
    run = []
    entry = []
    flag = []
    groundcal_pol = []
    time_per_event = []

    ### waterfall things
    power_filtered_vpol = []
    power_unfiltered_vpol = []
    power_filtered_hpol = []
    power_unfiltered_hpol = []

    ### now loop over our data and check!
    prev_run = None
    for r in range(len(this_array_runs)):
        start = time.perf_counter()

        this_run = int(this_array_runs[r])
        this_ev = int(this_array_entries[r])
        this_flag = this_array_flags[r]
        this_groundcal_pol = this_array_groundcal_pol[r]

        ### check if we have a correlator object for this run
        if prev_run == None or prev_run != this_run:
            ### set up correlator
            corr = correlator.Correlator(run = this_run)
            prev_run = this_run

            ### make our masks to use later
            channel_names = np.asarray(corr.ChannelGeo.MI_ringphipol)
            mask_H = np.array([int(ch[:2]) <= 4 and int(ch[2:4]) <= 24 and ch[-1] == 'H' for ch in channel_names])
            mask_V = np.array([int(ch[:2]) <= 4 and int(ch[2:4]) <= 24 and ch[-1] == 'V' for ch in channel_names])

        ### load this event
        corr.ChannelGeo.load_event(this_ev)

        ### get some flags
        this_event_summary = ROOT.PueoEventSummary()
        qual = corr.ChannelGeo.qualityCheck(this_event_summary)
        this_blast_flag = this_event_summary.flags.isPayloadBlast
        this_sat_flag = this_event_summary.flags.isSaturated
        this_deadtime_flag = this_event_summary.flags.isInDeadtime
        this_low_channel_rms_flag = this_event_summary.flags.lowChannelThermal

        if not any([this_blast_flag, this_sat_flag, this_deadtime_flag, this_low_channel_rms_flag]):
            ### get our filtered and un-filtered wfs
            corr.ChannelGeo.load_waveforms()
            unfilt_wfs = np.asarray(corr.ChannelGeo.waveforms_unfilt)
            filt_wfs = np.asarray(corr.ChannelGeo.waveforms)
            times = np.arange(unfilt_wfs.shape[-1]) * (1/3)

            ### now let's get the fft
            unfilt_f, unfilt_vf = utils.fourierTransform(times, unfilt_wfs)
            filt_f, filt_vf = utils.fourierTransform(times, filt_wfs)

            ### now let's save!!!
            power_filtered_vpol.append(np.real(utils.volt_to_dB(np.mean(filt_vf[mask_V], axis=0))))
            power_unfiltered_vpol.append(np.real(utils.volt_to_dB(np.mean(unfilt_vf[mask_V], axis=0))))
            power_filtered_hpol.append(np.real(utils.volt_to_dB(np.mean(filt_vf[mask_H], axis=0))))
            power_unfiltered_hpol.append(np.real(utils.volt_to_dB(np.mean(unfilt_vf[mask_H], axis=0))))

            ### finally, save our general info
            end = time.perf_counter()
            total_time = end - start

            run.append(this_run)
            entry.append(this_ev)
            flag.append(this_flag)
            groundcal_pol.append(this_groundcal_pol)
            time_per_event.append(total_time)

    ### now save this all to a json!
    saving_path = f'/scratch/midway3/darcynewmark/satellites/jsons/waterfall_all_runs_{array}.json'
    data = {
            'run' : np.asarray(run).tolist(),
            'entry' : np.asarray(entry).tolist(),
            'flag' : np.asarray(flag).tolist(),
            'groundcal_pol' : np.asarray(groundcal_pol).tolist(),
            'time_per_event' : np.asarray(time_per_event).tolist(),
            'power_filtered_vpol' : np.asarray(power_filtered_vpol).tolist(),
            'power_unfiltered_vpol' : np.asarray(power_unfiltered_vpol).tolist(),
            'power_filtered_hpol' : np.asarray(power_filtered_hpol).tolist(),
            'power_unfiltered_hpol' : np.asarray(power_unfiltered_hpol).tolist(),
    }
    with open(saving_path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--array", type = int, default = 0)
    args = parser.parse_args()

    save_waterfalls(args.array)
