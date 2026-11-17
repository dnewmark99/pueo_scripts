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
from scipy import special
from postrunmapping import run_attn_dict

import pueoTALON.pueoPythonAnalyzer.waveform as waveform
import pueoTALON.pueoPythonAnalyzer.plotter as pltr
import pueoTALON.pueoPythonAnalyzer.antenna as antenna
import pueoTALON.pueoPythonAnalyzer.correlator as correlator
import antennaUtils.response as antResponse
import pueoTALON.pueoPythonAnalyzer.utils as utils
import pueoTALON.pueoPythonAnalyzer.geography as geography

ROOT.gSystem.Load("libpueoEvent.so")
os.environ["PUEO_ROOT_DATA"] = ("/project/cozzyd/postflight/rootified/")

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
    a, b = params
    pred = get_erf_pred(x, a, b)
    return np.sum((y - pred)**2)

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
        #print(f'found {len(fnames)} files for run = {run}!')

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

def fit_times(run_start, run_end, subsec_err = 5e-4):
    runs = list(range(run_start, run_end + 1))
    data = grab_data_json(runs)

    ### now extract time data
    trigger_time = np.asarray(data["trigger_time"])
    trigger_subsec = np.asarray(data["trigger_subsec"])

    ### mask our bad trigger subseconds
    valid_mask = np.logical_and(trigger_subsec >= 0, trigger_subsec <= 1)
    trigger_time = trigger_time[valid_mask]
    trigger_subsec = trigger_subsec[valid_mask]

    ### take mod 0.1 of subsec and subtract off global min trigger time
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
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/preprocesseddata_subsecond_vs_triggertime_r{run_start}_r{run_end}.png', dpi=300)
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
    else:
        ### uchicago data, runs ~ 1400
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
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/preprocesseddata_subsecond_vs_triggertime_r{run_start}_r{run_end}_fitted.png', dpi=300)
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
    peak_to_peak = np.asarray(data["peak_to_peak"])
    rms_vals = np.asarray(data["rms_vals"])
    he_peak = np.asarray(data["he_peak"])
    he_mad = np.asarray(data["he_mad"])
    run_start_time = np.amin(trigger_time)

    ### mask our bad trigger subseconds + beginning of run
    valid_mask = functools.reduce(np.logical_and, [trigger_subsec >= 0, trigger_subsec <= 1, trigger_time >= (run_start_time + clip_window), ])
    trigger_time = trigger_time[valid_mask]
    trigger_subsec = trigger_subsec[valid_mask]
    entry = entry[valid_mask]
    peak_to_peak = peak_to_peak[valid_mask]
    rms_vals = rms_vals[valid_mask]
    he_peak = he_peak[valid_mask]
    he_mad = he_mad[valid_mask]
    clipped_run_start_time = np.amin(trigger_time)
    run_end_time = np.amax(trigger_time)

    ### take mod 0.1 of subsec and subtract off global_min_time
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

    ### quick plooooot!
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    ax.hist(trigger_subsec, range = (0,1), bins = 400, linewidth = 1.5, histtype='step', linestyle = '-', color = 'black')
    ax.set_title(f'Beamformer Test -- Run {run}', fontsize = 20)
    ax.set_xlabel(f'Event Subsecond [s]', fontsize = 20)
    ax.set_ylabel(f'Number of Events', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/preprocesseddata_subsecond_hist_r{run}.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    ax.scatter(trigger_time[~event_selection_mask], trigger_subsec_fmod[~event_selection_mask], s = 1, color = 'black', alpha = 0.1)
    ax.scatter(trigger_time[event_selection_mask], trigger_subsec_fmod[event_selection_mask], s = 1, color = 'red', alpha = 0.1)
    ax.plot(trigger_time, subsec_fit, color = 'green', linewidth = 0.5)
    ax.set_title(f'Beamformer Test -- Run {run}', fontsize = 20)
    ax.set_xlabel(f'Trigger Time [s]', fontsize = 20)
    ax.set_ylabel(f'Event Subsecond (% 0.1) [s]', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/preprocesseddata_subsecond_vs_triggertime_only_r{run}.png', dpi=300)
    plt.close(fig)

    ### let's also save bts entries to a json
    bts_event_entries = entry[event_selection_mask]
    saving_path = f'/scratch/midway3/darcynewmark/beamformer_testing/btsselectedevents_run{run}.json'
    with open(saving_path, 'w') as f:
        json.dump(bts_event_entries.tolist(), f)

    ### return p2p of selected events and rms of sideband events along with start and end time of run
    return peak_to_peak[event_selection_mask], rms_vals[~event_selection_mask], clipped_run_start_time, run_end_time, total_accident_coinc, he_peak[event_selection_mask], he_mad[~event_selection_mask]

def get_eff_curve(run_start = 1407, run_end = 1418, rate = 10):
    ### we need to select beamformer pulse events, calculate total number of expected events, and calculate snr --> that should give us snr vs trigger efficiency!

    corr_daqhsk = False
    if run_start > 800:
        corr_daqhsk = True

    if corr_daqhsk:
        ### first, let's get our housekeeping information
        daqhsk_all_times, daqhsk_all_l2_masks = parse_daqhsk()
        daqhsk_times = daqhsk_all_times[1:] ## throw away first entry so we always have the previous time the mask was valid over
        daqhsk_l2_masks = daqhsk_all_l2_masks[1:]
        daqhsk_previous_times = daqhsk_all_times[:-1]

    ### using all our data, let's now fit for beamformer event time distribution
    best_fit_m, best_fit_b, global_min_time = fit_times(run_start, run_end)

    ### now loop over each run to get eff and snr!
    trigger_eff = []
    trigger_eff_err = []
    reference_snr = None
    reference_he_metric = None
    reference_atten = None
    all_atten = []
    calc_snr = []
    scaled_snr = []
    calc_he_metric = []
    scaled_he_metric = []
    for run in range(run_start, run_end+1):
        ### step 1 -- select beamformer pulse events (and therefore remaining events are noise sample)
        bts_p2p, noise_rms, run_start_time, run_end_time, accidental_rate, bts_he_peak, noise_he_mad = select_beamformer_events(run, best_fit_m, best_fit_b, global_min_time)
        observed_number_of_events = len(bts_p2p) - accidental_rate

        ### step 2 -- get total number of expected events
        ### to first order, this is total data collection time * rate
        ### additionally, need to account for phi sector masking

        ## get run dt (in seconds)!
        dt = run_end_time - run_start_time

        if corr_daqhsk:
            ## now adjust for phi sector masking
            overlap_start = np.maximum(daqhsk_previous_times, run_start_time)
            overlap_end = np.minimum(daqhsk_times, run_end_time)
            false_mask = (~daqhsk_l2_masks & (overlap_end > overlap_start))
            false_duration = np.sum(overlap_end[false_mask] - overlap_start[false_mask])
            dt -= false_duration

        ## now we can get the expected number of events!
        expected_number_of_events = dt * rate
        print(f'for run = {run}, observed {len(bts_p2p)} bts events with {accidental_rate:0.2f} accidentals (expected {expected_number_of_events} total BTS events --> eff = {(observed_number_of_events / expected_number_of_events)*100:0.2f}%)')

        ### step 3 -- calculate SNR + he metric
        avg_p2p_per_event = np.mean(bts_p2p, axis = 1)
        overall_avg_p2p = np.mean(avg_p2p_per_event)
        avg_rms_per_event = np.mean(noise_rms, axis = 1)
        overall_avg_rms = np.mean(avg_rms_per_event)
        this_calc_snr = overall_avg_p2p / (2*overall_avg_rms)
        this_calc_snr *= 0.75

        avg_he_max_per_event = np.mean(bts_he_peak, axis = 1)
        overall_avg_he_max = np.mean(avg_he_max_per_event)
        avg_he_mad_per_event = np.mean(noise_he_mad, axis = 1)
        overall_avg_he_mad = np.mean(avg_he_mad_per_event)
        this_calc_he_metric = overall_avg_he_max / overall_avg_he_mad
        this_calc_he_metric *= 0.75

        ### check for scaling
        current_atten = run_attn_dict[run]
        if reference_snr == None:
            reference_snr = this_calc_snr
            reference_he_metric = this_calc_he_metric
            reference_atten = current_atten
            this_scaled_snr = this_calc_snr
            this_scaled_he_metric = this_calc_he_metric
        else:
            this_scaled_snr = reference_snr * np.sqrt(10**(-(current_atten - reference_atten) / 10))
            this_scaled_he_metric = reference_he_metric * np.sqrt(10**(-(current_atten - reference_atten) / 10))

        ### finally, get eff and save!
        eff = observed_number_of_events / expected_number_of_events
        eff_error = np.sqrt(eff * (1 - eff) / expected_number_of_events)

        trigger_eff.append(eff)
        trigger_eff_err.append(eff_error)
        scaled_snr.append(this_scaled_snr)
        calc_snr.append(this_calc_snr)
        calc_he_metric.append(this_calc_he_metric)
        scaled_he_metric.append(this_scaled_he_metric)
        all_atten.append(current_atten)

    #### let's make a quick scatter plot!
    #fig, ax = plt.subplots(figsize=(10, 6))
    #ax.minorticks_on()
    #ax.scatter(calc_snr, snr, s = 10, color = 'black')
    #line_start = min(np.amin(calc_snr), np.amin(snr))
    #line_end = max(np.amax(calc_snr), np.amax(snr))
    #ax.plot([line_start,line_end], [line_start,line_end], color = 'black', alpha = 0.5)
    #ax.set_title(f'Beamformer Test -- Run {run_start} to {run_end}', fontsize = 20)
    #ax.set_xlabel(f'Calculated SNR', fontsize = 20)
    #ax.set_ylabel(f'Scaled SNR', fontsize = 20)
    #ax.tick_params(axis='both', which='both', labelsize=20)
    #ax.set_xlim(line_start - 0.1, line_end + 0.1)
    #ax.set_ylim(line_start - 0.1, line_end + 0.1)
    #fig.tight_layout()
    #fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/preprocesseddata_calc_vs_scaled_snr_r{run_start}_r{run_end}.png', dpi=300)
    #plt.close(fig)

    #fig, ax = plt.subplots(figsize=(10, 6))
    #ax.minorticks_on()
    #ax.scatter(all_atten, calc_snr, s = 10, color = 'black')
    #ax.set_title(f'Beamformer Test -- Run {run_start} to {run_end}', fontsize = 20)
    #ax.set_xlabel(f'Attenuation [dB]', fontsize = 20)
    #ax.set_ylabel(f'Calculated SNR', fontsize = 20)
    #ax.tick_params(axis='both', which='both', labelsize=20)
    #fig.tight_layout()
    #fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/preprocesseddata_atten_vs_calc_snr_r{run_start}_r{run_end}.png', dpi=300)
    #plt.close(fig)

    return trigger_eff, trigger_eff_err, calc_snr, scaled_snr, calc_he_metric, scaled_he_metric, all_atten

def repeat_ldb():
    uc_snr, uc_eff, uc_eff_err = get_eff_curve(run_start = 1420, run_end = 1431)
    uc2_snr, uc2_eff, uc2_eff_err = get_eff_curve(run_start = 1433, run_end = 1444)
    ldb_snr, ldb_eff, ldb_eff_err = get_eff_curve(run_start = 719, run_end = 730)

    ### let's also grab the LDB test data for comparison
    ldb_data_path = '/scratch/midway3/darcynewmark/beamformer_testing/BeamformerEfficiencyLDBTest.csv'
    ldb_reported_data = pd.read_csv(ldb_data_path)
    ldb_reported_snr = ldb_reported_data["SNR"]
    ldb_reported_trigger_eff = ldb_reported_data[" efficiency"]
    ldb_reported_trigger_eff_err = ldb_reported_data[" Eff_error"]

    all_snr_data = [uc_snr, uc2_snr, ldb_snr, ldb_reported_snr]
    all_eff_data = [uc_eff, uc2_eff, ldb_eff, ldb_reported_trigger_eff]
    all_eff_error = [uc_eff_err, uc2_eff_err, ldb_eff_err, ldb_reported_trigger_eff_err]
    all_colors = ['#21b2aa', 'orange', '#c71585', 'black']
    all_labels = ['UChicago Sept 2026', 'UChicago Sept 2026 -- Flight Cables', 'LDB Dec 2025 (Reprocessed)', 'LDB Dec 2025']

    ### now we can plot our data!!!
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()

    for a in range(len(all_snr_data)):
        ### let's also fit our data!
        data = [all_snr_data[a], all_eff_data[a]]
        res = minimize(erf_fit_func, x0=[3, 1.5], args=data, tol = 1e-16, method='L-BFGS-B', bounds=[ (0,10), (0,10)], options = {"maxiter":1e7})
        best_fit_a, best_fit_b = res.x
        fine_grained_x = np.linspace(0.5, 3.5, 100)
        best_fit_erf = get_erf_pred(fine_grained_x, best_fit_a, best_fit_b)

        ### plot data + fit
        ax.errorbar(all_snr_data[a], all_eff_data[a], yerr=all_eff_error[a], fmt='o', color=all_colors[a], linestyle='none', ms = 2, capsize=4, label = all_labels[a])
        ax.plot(fine_grained_x, best_fit_erf, color=all_colors[a], linestyle = '--', alpha = 0.5)

    ax.set_title(f'Beamformer Test', fontsize = 20)
    ax.set_xlabel(f'Scaled Single Antenna SNR', fontsize = 20)
    ax.set_ylabel(f'Efficiency', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.grid(True, which='major', axis='both', linewidth=0.5, color='gray', alpha = 0.5)
    ax.legend(fontsize=16, frameon=False)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/preprocesseddata_trigger_eff_runcomparison.png', dpi=300)
    plt.close(fig)

def vary_el():
    el_min15_snr, el_min15_eff, el_min15_eff_err = get_eff_curve(run_start = 1433, run_end = 1444)
    el_min10_snr, el_min10_eff, el_min10_eff_err = get_eff_curve(run_start = 1446, run_end = 1457)
    #el_min5_snr, el_min5_eff, el_min5_eff_err = get_eff_curve(run_start = 1459, run_end = 1475)
    el_min5_snr, el_min5_eff, el_min5_eff_err = get_eff_curve(run_start = 1464, run_end = 1475)

    all_snr_data = [el_min15_snr, el_min10_snr, el_min5_snr]
    all_eff_data = [el_min15_eff, el_min10_eff, el_min5_eff]
    all_eff_error = [el_min15_eff_err, el_min10_eff_err, el_min5_eff_err]
    all_colors = ['black', '#21b2aa', '#c71585']
    all_labels = [r'$\phi = 5^\circ,\ \theta = -15^\circ$', r'$\phi = 5^\circ,\ \theta = -10^\circ$', r'$\phi = 5^\circ,\ \theta = -5^\circ$']

    ### now we can plot our data!!!
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()

    for a in range(len(all_snr_data)):
        ### let's also fit our data!
        data = [all_snr_data[a], all_eff_data[a]]
        res = minimize(erf_fit_func, x0=[3, 1.5], args=data, tol = 1e-16, method='L-BFGS-B', bounds=[ (0,10), (0,10)], options = {"maxiter":1e7})
        best_fit_a, best_fit_b = res.x
        fine_grained_x = np.linspace(0.5, 3.5, 100)
        best_fit_erf = get_erf_pred(fine_grained_x, best_fit_a, best_fit_b)

        ### plot data + fit
        ax.errorbar(all_snr_data[a], all_eff_data[a], yerr=all_eff_error[a], fmt='o', color=all_colors[a], linestyle='none', ms = 2, capsize=4, label = all_labels[a])
        ax.plot(fine_grained_x, best_fit_erf, color=all_colors[a], linestyle = '--', alpha = 0.5)

    ax.set_title(f'UC Sept 2026 Beamformer Test', fontsize = 20)
    ax.set_xlabel(f'Scaled Single Antenna SNR', fontsize = 20)
    ax.set_ylabel(f'Efficiency', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.grid(True, which='major', axis='both', linewidth=0.5, color='gray', alpha = 0.5)
    ax.legend(fontsize=16, frameon=False)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/preprocesseddata_trigger_eff_varyel.png', dpi=300)
    plt.close(fig)

def check_bugged_trigger():
    ldb_snr, ldb_eff, ldb_eff_err = get_eff_curve(run_start = 719, run_end = 730)
    uc_snr, uc_eff, uc_eff_err = get_eff_curve(run_start = 1433, run_end = 1444)
    ucbugged_snr, ucbugged_eff, ucbugged_eff_err = get_eff_curve(run_start = 1489, run_end = 1500)

    all_snr_data = [ldb_snr, uc_snr, ucbugged_snr]
    all_eff_data = [ldb_eff, uc_eff, ucbugged_eff]
    all_eff_error = [ldb_eff_err, uc_eff_err, ucbugged_eff_err]
    all_colors = ['black', '#21b2aa', '#c71585',]
    all_labels = ['LDB Dec 2025 (Reprocessed)', 'UC Sept 2026', 'UC Sept 2026 (BUGGED TRIGGER SETUP)']

    ### now we can plot our data!!!
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()

    for a in range(len(all_snr_data)):
        ### let's also fit our data!
        data = [all_snr_data[a], all_eff_data[a]]
        res = minimize(erf_fit_func, x0=[3, 1.5], args=data, tol = 1e-16, method='L-BFGS-B', bounds=[ (0,10), (0,10)], options = {"maxiter":1e7})
        best_fit_a, best_fit_b = res.x
        fine_grained_x = np.linspace(0.5, 3.5, 100)
        best_fit_erf = get_erf_pred(fine_grained_x, best_fit_a, best_fit_b)

        ### plot data + fit
        ax.errorbar(all_snr_data[a], all_eff_data[a], yerr=all_eff_error[a], fmt='o', color=all_colors[a], linestyle='none', ms = 2, capsize=4, label = all_labels[a])
        ax.plot(fine_grained_x, best_fit_erf, color=all_colors[a], linestyle = '--', alpha = 0.5)

    ax.set_title(r'Beamformer Test -- $\phi = 5^\circ,\ \theta = -15^\circ$', fontsize = 20)
    ax.set_xlabel(f'Scaled Single Antenna SNR', fontsize = 20)
    ax.set_ylabel(f'Efficiency', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.grid(True, which='major', axis='both', linewidth=0.5, color='gray', alpha = 0.5)
    ax.legend(fontsize=16, frameon=False)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/preprocesseddata_trigger_eff_buggedtrigger.png', dpi=300)
    plt.close(fig)

def compare_metrics():
    ### ok we need to read in our data and some snr to he based metric
    ### as well as look at scaling with attenuation
    ### we probably want to select beamformer events and return snr, he, and attn
    ldb_trigger_eff, ldb_trigger_eff_err, ldb_calc_snr, ldb_scaled_snr, ldb_calc_he_metric, ldb_scaled_he_metric, ldb_all_atten = get_eff_curve(run_start = 719, run_end = 730)
    uc_trigger_eff, uc_trigger_eff_err, uc_calc_snr, uc_scaled_snr, uc_calc_he_metric, uc_scaled_he_metric, uc_all_atten = get_eff_curve(run_start = 1433, run_end = 1444)
    uc2_trigger_eff, uc2_trigger_eff_err, uc2_calc_snr, uc2_scaled_snr, uc2_calc_he_metric, uc2_scaled_he_metric, uc2_all_atten = get_eff_curve(run_start = 1477, run_end = 1487)

    ### let's make some scatter plots
    scatter_x = [[ldb_all_atten, uc_all_atten, uc2_all_atten], [ldb_all_atten, uc_all_atten, uc2_all_atten]]
    scatter_y = [[ldb_calc_snr, uc_calc_snr, uc2_calc_snr], [ldb_calc_he_metric, uc_calc_he_metric, uc2_calc_he_metric]]
    colors = [['black', '#21b2aa', '#c71585'], ['black', '#21b2aa', '#c71585']]
    data_labels = [['LDB', 'UC', 'UC (Small Attenuation)'], ['LDB', 'UC', 'UC (Small Attenuation)']]
    saving_name = ['attn_vs_calcsnr', 'attn_vs_calchemetric']
    x_labels = ['Attenuation [dB]', 'Attenuation [dB]']
    y_labels = ['Calculated SNR', 'Calculated HE Metric']

    for p in range(len(saving_name)):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.minorticks_on()

        for i in range(len(scatter_x[p])):
            ax.scatter(scatter_x[p][i], scatter_y[p][i], s = 10, color = colors[p][i], label = data_labels[p][i])

        ax.set_title(r'Beamformer Test -- $\phi = 5^\circ,\ \theta = -15^\circ$', fontsize = 20)
        ax.set_xlabel(x_labels[p], fontsize = 20)
        ax.set_ylabel(y_labels[p], fontsize = 20)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.legend(fontsize=16, frameon=False)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/preprocesseddata_{saving_name[p]}.png', dpi=300)
        plt.close(fig)

def check_bts_pulse_shapes():
    ldb_trigger_eff, ldb_trigger_eff_err, ldb_calc_snr, ldb_scaled_snr, ldb_calc_he_metric, ldb_scaled_he_metric, ldb_all_atten = get_eff_curve(run_start = 719, run_end = 730)
    old_pulse_shape_trigger_eff, old_pulse_shape_trigger_eff_err, old_pulse_shape_calc_snr, old_pulse_shape_scaled_snr, old_pulse_shape_calc_he_metric, old_pulse_shape_scaled_he_metric, old_pulse_shape_all_atten = get_eff_curve(run_start = 1433, run_end = 1444)
    new_pulse_shape_trigger_eff, new_pulse_shape_trigger_eff_err, new_pulse_shape_calc_snr, new_pulse_shape_scaled_snr, new_pulse_shape_calc_he_metric, new_pulse_shape_scaled_he_metric, new_pulse_shape_all_atten = get_eff_curve(run_start = 1512, run_end = 1523)

    all_snr_data = [ldb_scaled_snr, old_pulse_shape_scaled_snr, new_pulse_shape_scaled_snr]
    all_eff_data = [ldb_trigger_eff, old_pulse_shape_trigger_eff, new_pulse_shape_trigger_eff]
    all_eff_error = [ldb_trigger_eff_err, old_pulse_shape_trigger_eff_err, new_pulse_shape_trigger_eff_err]
    all_colors = ['#c71585', '#21b2aa', 'orange',]
    all_labels = ['LDB Data (Reprocessed)', 'UC Sept 2026', 'UC Sept 2026 -- Updated BTS Pulse Shapes',]

    ### now we can plot our data!!!
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()

    for a in range(len(all_snr_data)):
        ### let's also fit our data!
        data = [all_snr_data[a], all_eff_data[a]]
        res = minimize(erf_fit_func, x0=[3, 1.5], args=data, tol = 1e-16, method='L-BFGS-B', bounds=[ (0,10), (0,10)], options = {"maxiter":1e7})
        best_fit_a, best_fit_b = res.x
        fine_grained_x = np.linspace(0.5, 3.5, 100)
        best_fit_erf = get_erf_pred(fine_grained_x, best_fit_a, best_fit_b)

        ### plot data + fit
        ax.errorbar(all_snr_data[a], all_eff_data[a], yerr=all_eff_error[a], fmt='o', color=all_colors[a], linestyle='none', ms = 2, capsize=4, label = all_labels[a])
        ax.plot(fine_grained_x, best_fit_erf, color=all_colors[a], linestyle = '--', alpha = 0.5)

    ax.set_title(r'Beamformer Test -- $\phi = 5^\circ,\ \theta = -15^\circ$', fontsize = 20)
    ax.set_xlabel(f'Scaled Single Antenna SNR', fontsize = 20)
    ax.set_ylabel(f'Efficiency', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.grid(True, which='major', axis='both', linewidth=0.5, color='gray', alpha = 0.5)
    ax.legend(fontsize=16, frameon=False)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/preprocesseddata_trigger_eff_checkpulseshape.png', dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    #repeat_ldb()
    #vary_el()
    #check_bugged_trigger()
    #compare_metrics()
    check_bts_pulse_shapes()
