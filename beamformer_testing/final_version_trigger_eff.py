import numpy as np
import matplotlib
import matplotlib as mpl
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
from scipy import special
from postrunmapping import run_attn_dict, BAD_RUNS
import csv

import pueoTALON.pueoPythonAnalyzer.waveform as waveform
import pueoTALON.pueoPythonAnalyzer.plotter as pltr
import pueoTALON.pueoPythonAnalyzer.antenna as antenna
import pueoTALON.pueoPythonAnalyzer.correlator as correlator
import antennaUtils.response as antResponse
import pueoTALON.pueoPythonAnalyzer.utils as utils
import pueoTALON.pueoPythonAnalyzer.geography as geography

ROOT.gSystem.Load("libpueoEvent.so")
os.environ["PUEO_ROOT_DATA"] = ("/project/cozzyd/postflight/rootified/")

all_16ch = np.asarray(['0101V', '0201V', '0301V', '0401V',
                       '0102V', '0202V', '0302V', '0402V',
                       '0103V', '0203V', '0303V', '0403V',
                       '0104V', '0204V', '0304V', '0404V'])
noise_only_channels = np.asarray(['0101V', '0102V', '0103V', '0104V',])
signal_only_mask = ~np.isin(all_16ch, noise_only_channels) 

def parse_daqhsk(file_name = 'daqhsk.root'):
    os.environ["PUEO_ROOT_DATA"] = ("/project/cozzyd/postflight/rootified/")
    PUEO_ROOT_DATA = os.environ["PUEO_ROOT_DATA"]
    daqhsk_file = f'{PUEO_ROOT_DATA}{file_name}'

    f = ROOT.TFile.Open(daqhsk_file)
    tree = f.Get("daqhskTree")

    trigger_time = []
    l2_masked = []
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        daqhsk = tree.daqhsk

        full_l2_readout_time_seconds = daqhsk.l2_readout_time + (daqhsk.l2_readout_timeNsecs * 1e-9)
        l2_enable_mask = daqhsk.l2_enable_mask
        l2_mask = functools.reduce(np.logical_and, [bool((l2_enable_mask >> i) & 1) for i in range(24)]) ### since we only connected AMPAs to channels used in this test,

        ### save our data
        trigger_time.append(full_l2_readout_time_seconds)
        l2_masked.append(l2_mask)

    return np.asarray(trigger_time), np.asarray(l2_masked)

def grab_data_json(runs, base_path = '/scratch/midway3/darcynewmark/beamformer_testing'):

    data = {}
    for run in runs:
        path = f'{base_path}/beamformerdata_run{run}_chunk*.json'
        fnames = [name for name in glob.glob(path)]
        fnames = sorted(fnames)

        ### read in our data
        for f in fnames:
            with open(f) as ff:
                temp_data = json.load(ff)

                ### check if this is the first dict we're loading in
                if len(data) == 0:
                    ### save keys to our dict
                    for key in temp_data:
                        data[key] = []

                ### now let's combine our dicts
                for key in data:
                    data[key].extend(temp_data[key])

    return data

def mode(inputData, axis=None, dtype=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """

    if axis is not None:
        fnc = lambda x: mode(x, dtype=dtype)
        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        def _hsm(data):
            if data.size == 1:
                return data[0]
            elif data.size == 2:
                return data.mean()
            elif data.size == 3:
                i1 = data[1] - data[0]
                i2 = data[2] - data[1]
                if i1 < i2:
                    return data[:2].mean()
                elif i2 > i1:
                    return data[1:].mean()
                else:
                    return data[1]
            else:
                wMin = data[-1] - data[0]
                N = int(data.size / 2 + data.size % 2)
                j = None
                for i in range(0, N):
                    w = data[i+N-1] - data[i]
                    if w < wMin:
                        wMin = w
                        j = i
                if j is None:
                    return data.mean()
                return _hsm(data[j:j+N])

        data = inputData.ravel()
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if dtype is not None:
            data = data.astype(dtype)

        # The data need to be sorted for this to work
        data = np.sort(data)

        # Find the mode
        dataMode = _hsm(data)

    return dataMode

def get_linear_pred(x, m, b):
    return (m * x) + b

def linear_fit_func(params, data):
    x = data[0]
    y = data[1]
    m, b = params
    pred = get_linear_pred(x, m, b)
    return np.sum((y - pred)**2)

def get_erf_pred(x, a, b):
    return 0.5 * (1 + special.erf(a * (x - b)))

def erf_fit_func(params, data):
    x = data[0]
    y = data[1]
    y_err = data[2]
    a, b = params
    pred = get_erf_pred(x, a, b)
    return np.sum(((y - pred) / y_err)**2)

def fit_times(run_start, run_end, subsec_err = 5e-4):
    runs = list(range(run_start, run_end + 1))
    runs = [r for r in runs if r not in BAD_RUNS]
    data = grab_data_json(runs)

    ### now extract time data
    trigger_time = np.asarray(data["trigger_time"])
    trigger_subsec = np.asarray(data["trigger_subsec"])

    ### mask our bad trigger subseconds
    valid_mask = np.logical_and(trigger_subsec >= 0, trigger_subsec <= 1)
    trigger_time = trigger_time[valid_mask]
    trigger_subsec = trigger_subsec[valid_mask]

    ### take mod 0.1 of subsec and subtract off global min trigger time
    if run_start in [1630, 1643, 1746, 1721]:
        print(f'shifting trigger subsecond fmod for run_start = {run_start}!')
        trigger_subsec_fmod = (trigger_subsec % 0.1 - 0.05) % 0.1
    else:
        trigger_subsec_fmod = trigger_subsec % 0.1
    global_min_time = np.amin(trigger_time)
    trigger_time -= global_min_time

    ### plot our data!
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    ax.scatter(trigger_time, trigger_subsec_fmod, s = 1, color = 'black', alpha = 0.1)
    ax.set_title(f'Beamformer Test -- Run {run_start} to {run_end}', fontsize = 20)
    ax.set_xlabel(f'Trigger Time [s]', fontsize = 20)
    ax.set_ylabel(f'Event Subsecond (% 0.1) [s]', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/fitanttnss_subsecond_vs_triggertime_r{run_start}_r{run_end}.png', dpi=300)
    plt.close(fig)

    ### now let's fit our data!!!
    nbins = 100
    x_min, x_max = np.amin(trigger_time), np.amax(trigger_time)
    x_bin_width = (x_max - x_min) / nbins
    x_data_bin_edges = np.linspace(x_min - x_bin_width, x_max + x_bin_width, nbins + 3)
    x_data_bin_centers = (x_data_bin_edges[:-1] + x_data_bin_edges[1:]) / 2
    x_bin_idx = np.digitize(trigger_time, x_data_bin_edges) - 1

    ### get the mode of y-axis data in every x-axis bin
    mode_y = np.full(len(x_data_bin_edges) - 1, 0.0)

    for i in range(len(mode_y)):
        y_slice = trigger_subsec_fmod[x_bin_idx == i]
        if run_start < 800:
            ### ldb runs are~700
            y_mask = np.logical_and(y_slice > 0.01, y_slice < 0.015)
            y_slice = y_slice[y_mask]
        if len(y_slice) > 0:
            mode_y[i] = mode(y_slice)

    ### remove the padded edge bins for 1D plots
    x_data_bin_centers = x_data_bin_centers[1:-1]
    mode_y = mode_y[1:-1]
    if run_start < 800:
        ### ldb runs are~700
        max_x_fitting_val = 1000
    elif run_start == 1502:
        ### uchicago data, runs ~ 1400
        max_x_fitting_val = 8000
    else:
        max_x_fitting_val = 2000

    valid_mask = np.logical_and(mode_y > 0, x_data_bin_centers < max_x_fitting_val)
    x_data_bin_centers = x_data_bin_centers[valid_mask]
    mode_y = mode_y[valid_mask]

    ### get values to select subsecond based on
    data = [x_data_bin_centers, mode_y]
    res = minimize(linear_fit_func, x0=[3e-7, np.mean(mode_y)], args=data, tol = 1e-16, method='L-BFGS-B', bounds=[ (-1e-3,1e-3), (0.0,0.1)], options = {"maxiter":1e7})
    best_fit_m, best_fit_b = res.x
    print(f'!!!!!!!!!! across all beamformer data, best fit slope = {best_fit_m} and best fit intercept = {best_fit_b} !!!!!!!!!!!!')

    ### select based on our best fit line
    subsec_fit = best_fit_m * trigger_time + best_fit_b
    residual = trigger_subsec_fmod - subsec_fit
    event_selection_mask = np.abs(residual) <= subsec_err
    n_selected_events = np.sum(event_selection_mask)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    ax.scatter(trigger_time[~event_selection_mask], trigger_subsec_fmod[~event_selection_mask], s = 1, color = 'black', alpha = 0.1)
    ax.scatter(trigger_time[event_selection_mask], trigger_subsec_fmod[event_selection_mask], s = 1, color = 'red', alpha = 0.1)
    ax.plot(trigger_time, subsec_fit, color = 'green', linewidth = 0.5)
    ax.set_title(f'Beamformer Test -- Run {run_start} to {run_end}', fontsize = 20)
    ax.set_xlabel(f'Trigger Time [s]', fontsize = 20)
    ax.set_ylabel(f'Event Subsecond (% 0.1) [s]', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/fitanttnss_subsecond_vs_triggertime_r{run_start}_r{run_end}_fitted.png', dpi=300)
    plt.close(fig)

    return best_fit_m, best_fit_b, global_min_time

def select_beamformer_events(run, best_fit_m, best_fit_b, global_min_time, subsec_err = 5e-4, clip_window = 5):
    ### select beamformer events based on trigger time and trigger subsecond
    ### remove clip_window seconds from start of the run

    ### read in our data
    data = grab_data_json([run])
    trigger_time = np.asarray(data["trigger_time"])
    trigger_subsec = np.asarray(data["trigger_subsec"])
    entry = np.asarray(data["entry"])
    wf_peak_to_peak = np.asarray(data["wf_peak_to_peak"])
    wf_peak_to_peak_noupsample = np.asarray(data["wf_peak_to_peak_noupsample"])
    he_max = np.asarray(data["he_max"])
    wf_rms = np.asarray(data["wf_rms"])
    wf_median = np.asarray(data["wf_median"])
    wf_mode = np.asarray(data["wf_mode"])
    wf_mad = np.asarray(data["wf_mad"])
    he_rms = np.asarray(data["he_rms"])
    he_median = np.asarray(data["he_median"])
    he_mode = np.asarray(data["he_mode"])
    he_mad = np.asarray(data["he_mad"])
    run_start_time = np.amin(trigger_time)

    ### mask our bad trigger subseconds + beginning of run
    valid_mask = functools.reduce(np.logical_and, [trigger_subsec >= 0, trigger_subsec <= 1, trigger_time >= (run_start_time + clip_window), ])
    trigger_time = trigger_time[valid_mask]
    trigger_subsec = trigger_subsec[valid_mask]
    entry = entry[valid_mask]
    clipped_run_start_time = np.amin(trigger_time)
    run_end_time = np.amax(trigger_time)
    wf_peak_to_peak = wf_peak_to_peak[valid_mask]
    wf_peak_to_peak_noupsample = wf_peak_to_peak_noupsample[valid_mask]
    he_max = he_max[valid_mask]
    wf_rms = wf_rms[valid_mask]
    wf_median = wf_median[valid_mask]
    wf_mode = wf_mode[valid_mask]
    wf_mad = wf_mad[valid_mask]
    he_rms = he_rms[valid_mask]
    he_median = he_median[valid_mask]
    he_mode = he_mode[valid_mask]
    he_mad = he_mad[valid_mask]

    ### take mod 0.1 of subsec and subtract off global_min_time
    if run in [1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641,
               1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654,
               1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756,
               1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732,]:
        trigger_subsec_fmod = (trigger_subsec % 0.1 - 0.05) % 0.1
    else:
        trigger_subsec_fmod = trigger_subsec % 0.1
    trigger_time -= global_min_time

    ### select based on our best fit line
    subsec_fit = best_fit_m * trigger_time + best_fit_b
    residual = trigger_subsec_fmod - subsec_fit
    event_selection_mask = np.abs(residual) <= subsec_err

    ### before we're done, let's get the accidental number of events
    n_non_bts_events = np.sum(~event_selection_mask)
    bts_area = (2 * subsec_err) * (run_end_time - clipped_run_start_time)
    total_area = 0.1 * (run_end_time - clipped_run_start_time)
    non_bts_area = total_area - bts_area
    total_accident_coinc = n_non_bts_events * (bts_area / non_bts_area)
    
    ### let's also save bts entries to a json
    bts_event_entries = entry[event_selection_mask]
    saving_path = f'/scratch/midway3/darcynewmark/beamformer_testing/btsselectedevents_run{run}.json'
    with open(saving_path, 'w') as f:
        json.dump(bts_event_entries.tolist(), f)

    ### return signal strength metrics of bts events, noise metrics for non-bts events, and times/num accidentals
    return wf_peak_to_peak[event_selection_mask], wf_rms[~event_selection_mask], clipped_run_start_time, run_end_time, total_accident_coinc


def get_trigger_eff(run_start, run_end, rate = 10):
    ### using all our data, let's now fit for beamformer event time distribution
    best_fit_m, best_fit_b, global_min_time = fit_times(run_start, run_end)
    
    ### now, let's get our housekeeping information to account for time when the trigger was disabled
    daqhsk_all_times, daqhsk_all_l2_masks = parse_daqhsk()
    daqhsk_times = daqhsk_all_times[1:] ## throw away first entry so we always have the previous time the mask was valid over
    daqhsk_l2_masks = daqhsk_all_l2_masks[1:]
    daqhsk_previous_times = daqhsk_all_times[:-1]
    
    ### now loop over each run to get eff and snr!
    trigger_eff = []
    trigger_eff_err = []
    wf_snr = []
    wf_snr_error = []
    all_attenuations = []
    reference_snr = None
    reference_snr_err = None
    reference_attn = None
    for run in range(run_start, run_end+1):
        ### step 1 -- get number of bts events and accidentals
        this_wf_p2p, this_wf_rms, run_start_time, run_end_time, n_accidentals = select_beamformer_events(run, best_fit_m, best_fit_b, global_min_time)
        observed_number_of_events = len(this_wf_p2p) - n_accidentals 

        ### step 2 -- get total number of expected events
        ### to first order, this is total data collection time * rate
        
        ## get run dt (in seconds)!
        dt = run_end_time - run_start_time
        
        ## now adjust for phi sector masking
        overlap_start = np.maximum(daqhsk_previous_times, run_start_time)
        overlap_end = np.minimum(daqhsk_times, run_end_time)
        false_mask = (~daqhsk_l2_masks & (overlap_end > overlap_start))
        false_duration = np.sum(overlap_end[false_mask] - overlap_start[false_mask])
        dt -= false_duration

        ## now we can get the expected number of events!
        expected_number_of_events = dt * rate

        ### step 3 -- calculate SNR!!!
        current_atten = run_attn_dict[run]
        if reference_snr == None:
            ### ok we need to calculate SNR per channel --> average --> scale by 0.75 to get our reference
            avg_wf_rms_per_channel = np.mean(this_wf_rms, axis=0)
            avg_wf_rms_signal = avg_wf_rms_per_channel[signal_only_mask]
            wf_snr_per_event_channel = (this_wf_p2p / (2 * avg_wf_rms_signal))
            avg_wf_snr = np.mean(wf_snr_per_event_channel)
            avg_wf_snr_err = (np.std(wf_snr_per_event_channel, ddof=1) / np.sqrt(wf_snr_per_event_channel.size))

            ### dont forget to scale :)
            scale_factor = 0.75
            avg_wf_snr *= scale_factor
            avg_wf_snr_err *= scale_factor
        
            ### save to our reference variables
            reference_snr = avg_wf_snr
            reference_snr_err = avg_wf_snr_err
            reference_attn = current_atten

            print(f'for the reference, avg_wf_snr = {avg_wf_snr}, avg_wf_snr_err = {avg_wf_snr_err}')

        ### now let's get the snr based on scaling
        scaled_wf_snr = reference_snr * 10**(-current_atten / 20.0)
        scaled_wf_snr_err = reference_snr_err * 10**(-current_atten / 20.0) 

        ### finally, get eff and save!
        eff = observed_number_of_events / expected_number_of_events
        eff_error = np.sqrt(eff * (1 - eff) / expected_number_of_events)

        trigger_eff.append(eff)
        trigger_eff_err.append(eff_error)
        wf_snr.append(scaled_wf_snr)
        wf_snr_error.append(scaled_wf_snr_err)
        all_attenuations.append(current_atten)
        
        print(f'for run = {run}, eff = {eff*100:0.2f}% (err = {eff_error*100:0.2f}%), scaled snr = {scaled_wf_snr:0.2f} (err = {scaled_wf_snr_err:0.4f})')
    print(f'trigger_eff = {np.asarray(trigger_eff).tolist()}')
    print(f'trigger_eff_err = {np.asarray(trigger_eff_err).tolist()}')
    print(f'wf_snr = {np.asarray(wf_snr).tolist()}')

if __name__ == "__main__":
    get_trigger_eff(run_start = 1776, run_end = 1789)
