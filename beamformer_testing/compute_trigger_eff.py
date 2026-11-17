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

def get_pred(x, m, b):
    return (m * x) + b

def fit_func(params, data):
    x = data[0]
    y = data[1]
    m, b = params
    pred = get_pred(x, m, b)
    return np.sum((y - pred)**2)

def parse_daqhsk(file_name = 'daqhsk.root'):
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

def fit_all_beamformer_time_data(channels, run_start, run_end):
    ### fit all runs trigger time vs trigger subsecond to get beamformer events
    subsecond_fmod = []
    trigger_time = []
    for run in range(run_start, run_end+1):
        channels.load_run_event(int(run), 0)
        ev_N  = channels.dataset.N()

        for ev in range(ev_N):
            if ev == 0:
                channels.dataset.firstEvent()
            else:
                channels.dataset.nextEvent()
            if channels.dataset.header().trigType == 1:
                subsec = channels.subsecond()
                event_time = channels.triggerTime()
                if subsec > 0:
                    subsecond_fmod.append(subsec % 0.1)
                    trigger_time.append(event_time)

    subsecond_fmod = np.asarray(subsecond_fmod)
    trigger_time = np.asarray(trigger_time)
    global_min_time = np.amin(trigger_time)
    trigger_time -= global_min_time

    ### let's bin the data and see what the mode is per x-bin
    nbins = 100
    x_min, x_max = np.amin(trigger_time), np.amax(trigger_time)
    x_bin_width = (x_max - x_min) / nbins
    x_data_bin_edges = np.linspace(x_min - x_bin_width, x_max + x_bin_width, nbins + 3)
    x_data_bin_centers = (x_data_bin_edges[:-1] + x_data_bin_edges[1:]) / 2
    x_bin_idx = np.digitize(trigger_time, x_data_bin_edges) - 1

    ### get the mode of y-axis data in every x-axis bin
    mode_y = np.full(len(x_data_bin_edges) - 1, 0.0)

    for i in range(len(mode_y)):
        y_slice = subsecond_fmod[x_bin_idx == i]
        if len(y_slice) > 0:
            mode_y[i] = mode(y_slice)

    ### remove the padded edge bins for 1D plots
    x_data_bin_centers = x_data_bin_centers[1:-1]
    mode_y = mode_y[1:-1]
    valid_mask = np.logical_and(mode_y > 0.03, x_data_bin_centers < 2000)
    x_data_bin_centers = x_data_bin_centers[valid_mask]
    mode_y = mode_y[valid_mask]

    ### get values to select subsecond based on
    data = [x_data_bin_centers, mode_y]
    res = minimize(fit_func, x0=[3e-7, np.mean(mode_y)], args=data, tol = 1e-16, method='L-BFGS-B', bounds=[ (-1e-3,1e-3), (0.04,0.06)], options = {"maxiter":1e7})
    best_fit_m, best_fit_b = res.x
    print(f'!!!!!!!!!! across all beamformer data, best fit slope = {best_fit_m} and best fit intercept = {best_fit_b} !!!!!!!!!!!!')

    ### select based on our best fit line
    subsec_fit = best_fit_m * trigger_time + best_fit_b
    subsec_err = 5e-4
    residual = subsecond_fmod - subsec_fit
    event_selection_mask = np.abs(residual) <= subsec_err
    n_selected_events = np.sum(event_selection_mask)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    ax.scatter(trigger_time, subsecond_fmod, s = 1, color = 'black', alpha = 0.5)
    ax.scatter(trigger_time[event_selection_mask], subsecond_fmod[event_selection_mask], s = 1, color = 'red', alpha = 0.5)
    ax.plot(trigger_time, subsec_fit, color = 'green', linewidth = 0.5)
    ax.set_title(f'Beamformer Test -- Run {run_start} to {run_end}', fontsize = 20)
    ax.set_xlabel(f'Trigger Time [s]', fontsize = 20)
    ax.set_ylabel(f'Event Subsecond (% 0.1) [s]', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/eventselectioncheck_subsecond_vs_triggertime_allruns.png', dpi=300)
    plt.close(fig)

    return best_fit_m, best_fit_b, global_min_time

def select_beamformer_events(channels, run, best_fit_m, best_fit_b, global_min_time, clip_window = 5):
    ### select beamformer events based on trigger time and trigger subsecond
    ### remove clip_window seconds from start of the run
    ### return number of events, run start time, run end time

    ### get the run start time (before clipping)!
    channels.load_run_event(int(run), 0)
    raw_run_start_time = channels.triggerTime()
    clipped_run_start_time = None
    run_end_time = None

    ### now make trigger time vs event subsecond distribution to select events
    subsecond_fmod = []
    trigger_time = []
    channels.load_run_event(int(run), 0)
    ev_N  = channels.dataset.N()
    all_entries = []

    for ev in range(ev_N):
        if ev == 0:
            channels.dataset.firstEvent()
        else:
            channels.dataset.nextEvent()
        if channels.dataset.header().trigType == 1:
            subsec = channels.subsecond()
            event_time = channels.triggerTime()
            if (run == 1409 and (event_time - global_min_time) > 940):
                continue
            if subsec > 0:
                ### make sure we're not too close to the ends of our run window
                if event_time > (raw_run_start_time + clip_window):
                    subsecond_fmod.append(subsec % 0.1)
                    trigger_time.append(event_time)
                    all_entries.append(ev)

                    ### make sure we updaed our clipped run start time
                    if clipped_run_start_time == None:
                        clipped_run_start_time = event_time

                    ### update our run end time
                    run_end_time = event_time

    all_entries = np.asarray(all_entries)
    subsecond_fmod = np.asarray(subsecond_fmod)
    trigger_time = np.asarray(trigger_time)
    trigger_time -= global_min_time

    ### select based on our best fit line
    subsec_fit = best_fit_m * trigger_time + best_fit_b
    subsec_err = 5e-4
    residual = subsecond_fmod - subsec_fit
    event_selection_mask = np.abs(residual) <= subsec_err
    n_selected_events = np.sum(event_selection_mask)
    selected_event_entries = all_entries[event_selection_mask]
    sideband_event_entries = all_entries[~event_selection_mask]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    ax.scatter(trigger_time, subsecond_fmod, s = 1, color = 'black', alpha = 0.5)
    ax.scatter(trigger_time[event_selection_mask], subsecond_fmod[event_selection_mask], s = 1, color = 'red', alpha = 0.5)
    ax.plot(trigger_time, subsec_fit, color = 'green', linewidth = 0.5)
    ax.set_title(f'Beamformer Test -- Run {run}', fontsize = 20)
    ax.set_xlabel(f'Trigger Time [s]', fontsize = 20)
    ax.set_ylabel(f'Event Subsecond (% 0.1) [s]', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/eventselectioncheck_subsecond_vs_triggertime_r{run}.png', dpi=300)
    plt.close(fig)

    return n_selected_events, clipped_run_start_time, run_end_time, selected_event_entries, sideband_event_entries

def calculate_snr(channels, run, selected_event_entries, sideband_event_entries):
    ### first thing's first, let's use the side band entries to calculate the noise rate
    print(f'for run = {run}, sideband_event_entries = {sideband_event_entries[:10]}')
    noise_rms_values = []
    desired_ch = np.asarray(['0201V', '0301V', '0401V', '0202V', '0302V', '0402V', '0203V', '0303V', '0403V', '0204V', '0304V', '0404V'])
    desired_ch_mask = np.isin(channels.MI_ringphipol, desired_ch)
    for e in sideband_event_entries:
        channels.load_run_event(int(run), int(e)) ### load our event
        channels.load_waveforms(filter = False) ### load waveforms
        unfilt_wfs = channels.waveforms_unfilt[desired_ch_mask]
        unfilt_wfs_rms = utils.RMS(unfilt_wfs)
        [noise_rms_values.append(r) for r in unfilt_wfs_rms]

    noise_rms_values = np.asarray(noise_rms_values)
    avg_noise_rms = np.mean(noise_rms_values)

    ### now let's get our signal rms

    return 0

def get_eff(rate = 10):
    ### we need to select beamformer pulse events, calculate total number of expected events, and calculate snr --> that should give us snr vs trigger efficiency!

    ### first, let's get our housekeeping information
    daqhsk_all_times, daqhsk_all_l2_masks = parse_daqhsk()
    daqhsk_times = daqhsk_all_times[1:] ## throw away first entry so we always have the previous time the mask was valid over
    daqhsk_l2_masks = daqhsk_all_l2_masks[1:]
    daqhsk_previous_times = daqhsk_all_times[:-1]

    ### each run is at a different attenuation level so should be the same snr per run
    run_start = 1407
    run_end = 1418
    channels = waveform.Channels(run_start, 0)

    ### now fit time vs subsecond plot to select beamformer events
    best_fit_m, best_fit_b, global_min_time = fit_all_beamformer_time_data(channels, run_start, run_end)

    snr = []
    trigger_eff = []
    trigger_eff_error = []
    for run in range(run_start, run_end+1):
        ### Step 1 -- select beamformer pulse events
        n_events, run_start_time, run_end_time, selected_event_entries, sideband_event_entries = select_beamformer_events(channels, run, best_fit_m, best_fit_b, global_min_time)

        ### Step 2 -- get total number of expected events
        ### to first order, this is total data collection time * rate
        ### additionally, need to account for phi sector masking

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

        print(f'for run {run}, observed {n_events} and expected {expected_number_of_events} events (eff = {(n_events/expected_number_of_events)*100:0.3f}%)')

        ### Step 3 -- Calculate SNR
        #avg_snr = calculate_snr(channels, run, selected_event_entries, sideband_event_entries)
        break

if __name__ == "__main__":
    get_eff()
