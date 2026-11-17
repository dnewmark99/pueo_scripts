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

def read_newscope_cvs(t_filename, v_filename, column=0):
    s= open(t_filename)
    csv_s= csv.reader(s)
    t = []
    t_begin, t_end, steps=0,0,0
    for row in csv_s:
        if str(row[0][0:7])=='XStart:':
            t_begin= float(row[0][7:-1])
        if str(row[0][0:6])=='XStop:':
            t_end= float(row[0][6:-1])
        if str(row[0][0:13])=='RecordLength:':
            steps= int(row[0][13:-1])
    step= (t_end-t_begin)/steps
    t= np.arange(t_begin* 1.e9, t_end* 1.e9,step* 1.e9)

    v = open(v_filename)
    csv_v= csv.reader(v)
    wave = []
    for row in csv_v:
        wave.append(row[column])

    return t, np.array(wave, dtype=float)

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

def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of
    a given width

    Assumes that x is sorted numpy array.
    """
    n = len(x)
    cred_mass = 1.0 - alpha

    interval_idx_inc = int(np.floor(cred_mass * n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx + interval_idx_inc]
    return hdi_min, hdi_max

def hpd(x, alpha=0.05, transform=lambda x: x):
    """Calculate highest posterior density (HPD) of array for given alpha. The HPD is the
    minimum width Bayesian credible interval (BCI).

    :Arguments:
      x : Numpy array
          An array containing MCMC samples
      alpha : float
          Desired probability of type I error (defaults to 0.05)
      transform : callable
          Function to transform data (defaults to identity)

    """
    # Make a copy of trace
    x = transform(np.copy(x))

    # For multivariate node
    if x.ndim > 1:

        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:] + [0])
        dims = np.shape(tx)

        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1] + (2,))

        for index in make_indices(dims[:-1]):

            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])

            # Append to list
            intervals[index] = calc_min_interval(sx, alpha)

        # Transpose back before returning
        return np.array(intervals)

    else:
        # Sort univariate node
        sx = np.sort(x)

        return np.array(calc_min_interval(sx, alpha))

def interval(arr, proportion=scipy.special.erf(1.0/np.sqrt(2.0)), min=None, max=None):
    """
    Compute distribution mode and the HPD that contains `proportion` of the mass.
    """
    x = hpd(arr, alpha=(1.0-proportion))
    if x[0] <= np.amin(arr):
        if min is not None:
            x[0] = min
    if x[-1] <= np.amax(arr):
        if max is not None:
            x[1] = max
    return x[0], mode(arr), x[1]

def get_linear_pred(x, m, b):
    return (m * x) + b

def linear_fit_func(params, data):
    x = data[0]
    y = data[1]
    m, b = params
    pred = get_linear_pred(x, m, b)
    return np.sum((y - pred)**2)

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
    if not np.all(np.isfinite(pred)):
        return 1e30
    return np.sum((y - pred)**2)

def get_attn_pred(x, snr0, a):
    return snr0 * 10**(-x/a)

def attn_fit_func(params, data):
    x = data[0]
    y = data[1]
    snr0, a = params
    pred = get_attn_pred(x, snr0, a)
    return np.sum((y - pred)**2)

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
    
    NON_bts_event_entries = entry[~event_selection_mask]
    saving_path = f'/scratch/midway3/darcynewmark/beamformer_testing/NONbtsselectedevents_run{run}.json'
    with open(saving_path, 'w') as f:
        json.dump(NON_bts_event_entries.tolist(), f)

    ### return signal strength metrics of bts events, noise metrics for non-bts events, and times/num accidentals
    return (wf_peak_to_peak[event_selection_mask], wf_peak_to_peak_noupsample[event_selection_mask], he_max[event_selection_mask],
            wf_rms[~event_selection_mask], wf_median[~event_selection_mask], wf_mode[~event_selection_mask], wf_mad[~event_selection_mask],
            he_rms[~event_selection_mask], he_median[~event_selection_mask], he_mode[~event_selection_mask], he_mad[~event_selection_mask],
            clipped_run_start_time, run_end_time, total_accident_coinc)

def fit_attn_sig_strength(run_start = 1502, run_end = 1524):
    ### using all our data, let's now fit for beamformer event time distribution
    best_fit_m, best_fit_b, global_min_time = fit_times(run_start, run_end)

    ### ok now we can pick out bts events -- calculate signal strength -- and get relationships with attenuation
    ### do this per run and save!!!
    run_attenuation = []
    avg_snr = []
    avg_hess = []
    avg_wf_p2p = []
    avg_he_max = []
    mode_wf_p2p = []
    mode_wf_p2p_noupsample = []
    mode_he_max = []
    onesig_lb_wf_p2p = []
    onesig_ub_wf_p2p = []
    onesig_lb_wf_p2p_noupsample = []
    onesig_ub_wf_p2p_noupsample = []
    onesig_lb_he_max = []
    onesig_ub_he_max = []
    #### save noise stuff per channel
    noise_mode_wf_rms = {}
    noise_mode_wf_mad = {}
    noise_mode_he_rms = {}
    noise_mode_he_mad = {}
    noise_mode_wf_rms_onesiglb = {}
    noise_mode_wf_mad_onesiglb = {}
    noise_mode_he_rms_onesiglb = {}
    noise_mode_he_mad_onesiglb = {}
    noise_mode_wf_rms_onesigub = {}
    noise_mode_wf_mad_onesigub = {}
    noise_mode_he_rms_onesigub = {}
    noise_mode_he_mad_onesigub = {}

    ### add our noise antennas to the dicts
    for ant in all_16ch:
        noise_mode_wf_rms[ant] = []
        noise_mode_wf_mad[ant] = []
        noise_mode_he_rms[ant] = []
        noise_mode_he_mad[ant] = []
        noise_mode_wf_rms_onesiglb[ant] = []
        noise_mode_wf_mad_onesiglb[ant] = []
        noise_mode_he_rms_onesiglb[ant] = []
        noise_mode_he_mad_onesiglb[ant] = []
        noise_mode_wf_rms_onesigub[ant] = []
        noise_mode_wf_mad_onesigub[ant] = []
        noise_mode_he_rms_onesigub[ant] = []
        noise_mode_he_mad_onesigub[ant] = []
    
    for run in range(run_start, run_end+1):
        run_attenuation.append(run_attn_dict[run])

        this_wf_p2p, this_wf_p2p_noupsample, this_he_max, this_wf_rms, this_wf_median, this_wf_mode, this_wf_mad, this_he_rms, this_he_median, this_he_mode, this_he_mad, run_start_time, run_end_time, n_accidentals = select_beamformer_events(run, best_fit_m, best_fit_b, global_min_time)
        print(f'for run = {run}, n_bts_events = {this_wf_p2p.shape[0]}, n_noise_events = {this_wf_median.shape[0]}, and n_accidentals = {n_accidentals}')

        ### for each attenuation run, let's plot all the data    
        ### we are considering either wf p2p or he max for signal strength metric (i think he max will make more sense)
        x_data = this_wf_p2p.ravel() 
        y_data = this_he_max.ravel() 

        ### set up bin edges
        nbins = 80

        x_min, x_max = np.amin(x_data), np.amax(x_data)
        x_bin_width = (x_max - x_min) / nbins

        x_data_bin_edges = np.linspace(x_min - x_bin_width, x_max + x_bin_width, nbins + 3)

        y_min, y_max = np.amin(y_data), np.amax(y_data)
        y_bin_width = (y_max - y_min) / nbins

        y_data_bin_edges = np.linspace(y_min - y_bin_width, y_max + y_bin_width, nbins + 3)

        ### make our matrix
        M, _, _ = np.histogram2d(x_data, y_data, bins=[x_data_bin_edges, y_data_bin_edges])

        ### plot!
        cm = plt.get_cmap("Spectral_r")
        lin_norm = matplotlib.colors.LogNorm(vmin=1, vmax=np.amax(M.T))

        fig, ax = plt.subplots(figsize=(10, 7))
        mesh = ax.pcolormesh(x_data_bin_edges, y_data_bin_edges, M.T, cmap=cm, norm = lin_norm)
        cb = fig.colorbar(mesh, ax=ax)
        cb.ax.set_ylabel("Number of Events", fontsize = 20)
        cb.ax.minorticks_off()
        ax.set_title(f'BTS Run {run}, Attenuation = {run_attn_dict[run]}', fontsize = 20)
        ax.set_xlabel(f'Waveform Pk2Pk', fontsize = 20)
        ax.set_ylabel(f'Hilbert Envelope Max.', fontsize = 20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        fig.tight_layout()
        plt.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/fitanttnss_twod_wfp2p_vs_hemax_r{run}.png', dpi=300)
        plt.close(fig)

        ### plot 1D distributions as well
        xdata_onesig_lb, xdata_onesig_mode, xdata_onesig_ub = interval(x_data)
        xdata_width = ((xdata_onesig_ub - xdata_onesig_lb) / xdata_onesig_mode)*100
        
        ydata_onesig_lb, ydata_onesig_mode, ydata_onesig_ub = interval(y_data)
        ydata_width = ((ydata_onesig_ub - ydata_onesig_lb) / ydata_onesig_mode)*100
        
        xdata_noupsample = this_wf_p2p_noupsample.ravel()
        xdata_noupsample_onesig_lb, xdata_noupsample_onesig_mode, xdata_noupsample_onesig_ub = interval(xdata_noupsample)
        xdata_noupsample_width = ((xdata_noupsample_onesig_ub - xdata_noupsample_onesig_lb) / xdata_noupsample_onesig_mode)*100
       
        ### save mode and widths
        mode_wf_p2p.append(xdata_onesig_mode)
        onesig_lb_wf_p2p.append(xdata_onesig_mode - xdata_onesig_lb)
        onesig_ub_wf_p2p.append(xdata_onesig_ub - xdata_onesig_mode)
        
        mode_wf_p2p_noupsample.append(xdata_noupsample_onesig_mode)
        onesig_lb_wf_p2p_noupsample.append(xdata_noupsample_onesig_mode - xdata_noupsample_onesig_lb)
        onesig_ub_wf_p2p_noupsample.append(xdata_noupsample_onesig_ub - xdata_noupsample_onesig_mode)
        
        mode_he_max.append(ydata_onesig_mode)
        onesig_lb_he_max.append(ydata_onesig_mode - ydata_onesig_lb)
        onesig_ub_he_max.append(ydata_onesig_ub - ydata_onesig_mode)

        ### back to plotting!!!
        min_bin_range = min(np.amin(x_data), np.amin(y_data))
        max_bin_range = max(np.amax(x_data), np.amax(y_data))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.minorticks_on()
        ax.hist(xdata_noupsample, bins = 2*nbins, range = (min_bin_range, max_bin_range), linewidth = 1.5, histtype='step', linestyle = '-', color = 'black',
                label = rf'Waveform Pk2Pk (No Upsampling), $1\sigma$ = [{xdata_noupsample_onesig_lb:0.2f}, {xdata_noupsample_onesig_ub:0.2f}]'f'\nFrxn Width = {xdata_noupsample_width:0.2f}%')
        ax.hist(x_data, bins = 2*nbins, range = (min_bin_range, max_bin_range), linewidth = 1.5, histtype='step', linestyle = '-', color = '#009E73',
                label = rf'Waveform Pk2Pk, $1\sigma$ = [{xdata_onesig_lb:0.2f}, {xdata_onesig_ub:0.2f}]'f'\nFrxn Width = {xdata_width:0.2f}%')
        ax.hist(y_data, bins = 2*nbins, range = (min_bin_range, max_bin_range), linewidth = 1.5, histtype='step', linestyle = '-', color = '#D55E00',
                label = rf'Hilbert Envelope Max., $1\sigma$ = [{ydata_onesig_lb:0.2f}, {ydata_onesig_ub:0.2f}]'f'\nFrxn Width = {ydata_width:0.2f}%')
        ax.axvline(xdata_noupsample_onesig_lb, color = 'black', alpha = 0.5)
        ax.axvline(xdata_noupsample_onesig_ub, color = 'black', alpha = 0.5)
        ax.axvline(xdata_onesig_lb, color = '#009E73', alpha = 0.5)
        ax.axvline(xdata_onesig_ub, color = '#009E73', alpha = 0.5)
        ax.axvline(ydata_onesig_lb, color = '#D55E00', alpha = 0.5)
        ax.axvline(ydata_onesig_ub, color = '#D55E00', alpha = 0.5)
        ax.set_title(f'BTS Run {run}, Attenuation = {run_attn_dict[run]}', fontsize = 20)
        ax.set_xlabel(f'Metric [mV]', fontsize = 20)
        ax.set_ylabel(f'Number of Events', fontsize = 20)
        ax.legend(fontsize=14, frameon=False)
        ax.tick_params(axis='both', which='both', labelsize=20)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/fitanttnss_oned_wfp2p_vs_hemax_r{run}.png', dpi=300)
        plt.close(fig)

        ### now let's plot some noise metrics, just say oned distributions for now
        wf_noise_data = [this_wf_rms, this_wf_median, this_wf_mode, this_wf_mad]
        he_noise_data = [this_he_rms, this_he_median, this_he_mode, this_he_mad]
        labels = ['RMS', 'Median', 'Mode', 'Median Absolute Deviation']
        bin_scale = [4, 4, 1, 4]
        saving_name = ['rms', 'median', 'mode', 'mad']
        
        for n in range(len(wf_noise_data)):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.minorticks_on()
            
            ### plot each channel individually
            n_channels = wf_noise_data[n].shape[1]
            cmap = plt.get_cmap('Spectral_r')
            norm = mpl.colors.Normalize(vmin=0, vmax=n_channels - 1)
            for c in range(n_channels):
                color = cmap(norm(c))
                x_data = np.asarray(wf_noise_data[n][:,c])
                y_data = np.asarray(he_noise_data[n][:,c])
                
                xdata_onesig_lb, xdata_onesig_mode, xdata_onesig_ub = interval(x_data)
                ydata_onesig_lb, ydata_onesig_mode, ydata_onesig_ub = interval(y_data)
                xdata_width = ((xdata_onesig_ub - xdata_onesig_lb) / xdata_onesig_mode)*100
                ydata_width = ((ydata_onesig_ub - ydata_onesig_lb) / ydata_onesig_mode)*100

                min_bin_range = min(np.amin(x_data), np.amin(y_data))
                max_bin_range = max(np.amax(x_data), np.amax(y_data))
                nbins = int((max_bin_range-min_bin_range)*bin_scale[n])
                
                ax.hist(x_data, bins = nbins, range = (min_bin_range, max_bin_range), linewidth = 1.5, histtype='step', linestyle = '--', color = color,)
                ax.hist(y_data, bins = nbins, range = (min_bin_range, max_bin_range), linewidth = 1.5, histtype='step', linestyle = '-', color = color,
                        label = rf'{all_16ch[c]}, Frxn Width = {xdata_width:0.2f}% / {ydata_width:0.2f}%')
           
                ### let's also save to our dicts for plotting later!
                x_data_onesig_lb, x_data_onesig_mode, x_data_onesig_ub = interval(x_data)
                y_data_onesig_lb, y_data_onesig_mode, y_data_onesig_ub = interval(y_data)
                ant = all_16ch[c]

                if labels[n] == 'RMS':
                    noise_mode_wf_rms[ant].append(x_data_onesig_mode)
                    noise_mode_he_rms[ant].append(y_data_onesig_mode)
                    noise_mode_wf_rms_onesiglb[ant].append(x_data_onesig_mode - x_data_onesig_lb)
                    noise_mode_he_rms_onesiglb[ant].append(y_data_onesig_mode - y_data_onesig_lb)
                    noise_mode_wf_rms_onesigub[ant].append(x_data_onesig_ub - x_data_onesig_mode)
                    noise_mode_he_rms_onesigub[ant].append(y_data_onesig_ub - y_data_onesig_mode)

                if labels[n] == 'Median Absolute Deviation':
                    noise_mode_wf_mad[ant].append(x_data_onesig_mode)
                    noise_mode_he_mad[ant].append(y_data_onesig_mode)
                    noise_mode_wf_mad_onesiglb[ant].append(x_data_onesig_mode - x_data_onesig_lb)
                    noise_mode_he_mad_onesiglb[ant].append(y_data_onesig_mode - y_data_onesig_lb)
                    noise_mode_wf_mad_onesigub[ant].append(x_data_onesig_ub - x_data_onesig_mode)
                    noise_mode_he_mad_onesigub[ant].append(y_data_onesig_ub - y_data_onesig_mode)

            ax.set_title(f'BTS Run {run}, Attenuation = {run_attn_dict[run]}', fontsize = 20)
            ax.set_xlabel(f'{labels[n]} [mV]', fontsize = 20)
            ax.set_ylabel(f'Number of Events', fontsize = 20)
            ax.legend(fontsize=8, frameon=False)
            ax.tick_params(axis='both', which='both', labelsize=20)
            fig.tight_layout()
            fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/fitanttnss_noise_oned_{saving_name[n]}_wf_vs_he_r{run}.png', dpi=300)
            plt.close(fig)

        ### then save average snr and average hess (for fitting)
        #### ok after looking at everything -- i think hilbert evenelope max / 2 rms makes the most sense for alternative metric
        ### so let's calculate that per attenuation
        avg_wf_rms_per_channel = np.mean(this_wf_rms, axis=0)
        avg_wf_p2p_per_channel = np.mean(this_wf_p2p, axis=0)
        avg_wf_snr = np.mean(avg_wf_p2p_per_channel / (2*avg_wf_rms_per_channel[signal_only_mask]))
        
        avg_he_rms_per_channel = np.mean(this_he_rms, axis=0)
        avg_he_max_per_channel = np.mean(this_he_max, axis=0)
        avg_he_snr = np.mean(avg_he_max_per_channel / (2*avg_he_rms_per_channel[signal_only_mask]))

        ### dont forget to scale :)
        avg_wf_snr *= 0.75
        avg_he_snr *= 0.75

        avg_snr.append(avg_wf_snr)
        avg_hess.append(avg_he_snr)
        avg_wf_p2p.append(np.mean(this_wf_p2p))
        avg_he_max.append(np.mean(this_he_max))

    ### plot attn vs some things
    run_attenuation = np.asarray(run_attenuation)
    y_data = [mode_wf_p2p_noupsample, mode_wf_p2p, mode_he_max]
    y_onesig_lb = [onesig_lb_wf_p2p_noupsample, onesig_lb_wf_p2p, onesig_lb_he_max]
    y_onesig_ub = [onesig_ub_wf_p2p_noupsample, onesig_ub_wf_p2p, onesig_ub_he_max]
    y_labels = ['Waveform Pk2Pk2 (No Upsampling)', 'Waveform Pk2Pk2', 'Hilbert Envelope Max.']
    colors = ['black', '#009E73', '#D55E00']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    for y in range(len(y_data)):
        this_y = np.asarray(y_data[y])
        this_y_onesig_lb = np.asarray(y_onesig_lb[y])
        this_y_onesig_ub = np.asarray(y_onesig_ub[y])
        ax.plot(run_attenuation, this_y, color = colors[y], linewidth = 1.5, label = f'{y_labels[y]}')
        ax.fill_between(run_attenuation, this_y - this_y_onesig_lb, this_y + this_y_onesig_ub, color=colors[y], alpha=0.2)
    ax.set_title(f'BTS Attenuation vs Mode of Signal Strength Metric', fontsize = 20)
    ax.set_xlabel(f'BTS Attenuation [dB]', fontsize = 20)
    ax.set_ylabel(r'Metric (Mode $\pm$ $1\sigma$)', fontsize = 20)
    ax.legend(fontsize=20, frameon=False)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/fitanttnss_attn_vs_metricmode.png', dpi=300)
    plt.close(fig)

    ### plot attn vs noise metrics
    fig_he_noise, ax_he_noise = plt.subplots(figsize=(10, 6))
    fig_wf_noise, ax_wf_noise = plt.subplots(figsize=(10, 6))
    ax_he_noise.minorticks_on()
    ax_wf_noise.minorticks_on()
    n_channels = len(all_16ch) 
    cmap = plt.get_cmap('Spectral_r')
    norm = mpl.colors.Normalize(vmin=0, vmax=n_channels - 1)
    
    counter = 0
    for c in (all_16ch):
        color = cmap(norm(counter))
        this_c_wf_rms_mode = np.asarray(noise_mode_wf_rms[c])
        this_c_wf_rms_onesiglb = np.asarray(noise_mode_wf_rms_onesiglb[c])
        this_c_wf_rms_onesigub = np.asarray(noise_mode_wf_rms_onesigub[c])
        ax_wf_noise.plot(run_attenuation, this_c_wf_rms_mode, color = color, linewidth = 1.5, linestyle = '--',label = f'{c}, Waveform RMS')
        ax_wf_noise.fill_between(run_attenuation, this_c_wf_rms_mode - this_c_wf_rms_onesiglb, this_c_wf_rms_mode + this_c_wf_rms_onesigub, color=color, alpha=0.2)
        
        this_c_wf_mad_mode = np.asarray(noise_mode_wf_mad[c])
        this_c_wf_mad_onesiglb = np.asarray(noise_mode_wf_mad_onesiglb[c])
        this_c_wf_mad_onesigub = np.asarray(noise_mode_wf_mad_onesigub[c])
        ax_wf_noise.plot(run_attenuation, this_c_wf_mad_mode, color = color, linewidth = 1.5, linestyle = '-', label = f'{c}, Waveform MAD')
        ax_wf_noise.fill_between(run_attenuation, this_c_wf_mad_mode - this_c_wf_mad_onesiglb, this_c_wf_mad_mode + this_c_wf_mad_onesigub, color=color, alpha=0.2)
        
        this_c_he_rms_mode = np.asarray(noise_mode_he_rms[c])
        this_c_he_rms_onesiglb = np.asarray(noise_mode_he_rms_onesiglb[c])
        this_c_he_rms_onesigub = np.asarray(noise_mode_he_rms_onesigub[c])
        ax_he_noise.plot(run_attenuation, this_c_he_rms_mode, color = color, linewidth = 1.5, linestyle = '--', label = f'{c}, HE RMS')
        ax_he_noise.fill_between(run_attenuation, this_c_he_rms_mode - this_c_he_rms_onesiglb, this_c_he_rms_mode + this_c_he_rms_onesigub, color=color, alpha=0.2)
        
        this_c_he_mad_mode = np.asarray(noise_mode_he_mad[c])
        this_c_he_mad_onesiglb = np.asarray(noise_mode_he_mad_onesiglb[c])
        this_c_he_mad_onesigub = np.asarray(noise_mode_he_mad_onesigub[c])
        ax_he_noise.plot(run_attenuation, this_c_he_mad_mode, color = color, linewidth = 1.5, linestyle = '-', label = f'{c}, HE MAD')
        ax_he_noise.fill_between(run_attenuation, this_c_he_mad_mode - this_c_he_mad_onesiglb, this_c_he_mad_mode + this_c_he_mad_onesigub, color=color, alpha=0.2)
        
        counter += 1

    ax_he_noise.set_title(f'BTS Attenuation vs Mode of Noise Metric', fontsize = 20)
    ax_he_noise.set_xlabel(f'BTS Attenuation [dB]', fontsize = 20)
    ax_he_noise.set_ylabel(r'Metric (Mode $\pm$ $1\sigma$)', fontsize = 20)
    ax_he_noise.legend(fontsize=6, frameon=False)
    ax_he_noise.tick_params(axis='both', which='both', labelsize=20)
    fig_he_noise.tight_layout()
    fig_he_noise.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/fitanttnss_attn_vs_noisemetric_he.png', dpi=300)
    plt.close(fig_he_noise)
    
    ax_wf_noise.set_title(f'BTS Attenuation vs Mode of Noise Metric', fontsize = 20)
    ax_wf_noise.set_xlabel(f'BTS Attenuation [dB]', fontsize = 20)
    ax_wf_noise.set_ylabel(r'Metric (Mode $\pm$ $1\sigma$)', fontsize = 20)
    ax_wf_noise.legend(fontsize=6, frameon=False)
    ax_wf_noise.tick_params(axis='both', which='both', labelsize=20)
    fig_wf_noise.tight_layout()
    fig_wf_noise.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/fitanttnss_attn_vs_noisemetric_wf.png', dpi=300)
    plt.close(fig_wf_noise)

    #### now...what we're all here for...fit attenuation vs our signal strength metrics!!!
    y_data = [avg_snr, avg_hess]
    y_labels = ['Waveform SNR', 'Hilbert Envelope SNR']
    colors = ['#009E73', '#D55E00']
    best_fit_snr0s = []
    best_fit_as = []

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    attn_fit_start = 0
    attn_fit_end = 7
    for y in range(len(y_data)):
        this_y = np.asarray(y_data[y])
        
        ### quick fit to the data
        fit_mask = np.logical_and(run_attenuation >= attn_fit_start, run_attenuation <= attn_fit_end)
        data = [run_attenuation[fit_mask], this_y[fit_mask]]
        res = minimize(attn_fit_func, x0=[this_y[0], 20], args=data, tol = 1e-20, method='L-BFGS-B', bounds=[ (this_y[0]/10,this_y[0]*100), (10,30)], options = {"maxiter":1e7})
        best_fit_snr0 = res.x[0]
        best_fit_a = res.x[1]
        print(f'best fit snr0 = {best_fit_snr0}, best fit a = {best_fit_a}')
        
        interp_attn = np.linspace(np.amin(run_attenuation),np.amax(run_attenuation), 100)
        prediction = get_attn_pred(interp_attn, best_fit_snr0, best_fit_a)
        ax.scatter(run_attenuation, this_y, color = colors[y], s = 30, label = rf'{y_labels[y]} = {best_fit_snr0:.2f}$\times10^{{-A/{best_fit_a:.2f}}}$')
        ax.plot(interp_attn, prediction, color = colors[y], linewidth = 1.5, alpha = 0.5)
        best_fit_snr0s.append(best_fit_snr0)
        best_fit_as.append(best_fit_a)

        ### do a test fit where we force exponent scaling to 20...
        res = minimize(attn_fit_func, x0=[this_y[0], 20], args=data, tol = 1e-20, method='L-BFGS-B', bounds=[ (this_y[0]/10,this_y[0]*100), (20,20)], options = {"maxiter":1e7})
        best_fit_snr0 = res.x[0]
        best_fit_a = res.x[1]
        prediction = get_attn_pred(interp_attn, best_fit_snr0, best_fit_a)
        ax.plot(interp_attn, prediction, color = colors[y], linewidth = 1.5, linestyle = '--', alpha = 0.5, label = rf'{y_labels[y]} = {best_fit_snr0:.2f}$\times10^{{-A/{best_fit_a:.2f}}}$')

    ax.axvspan(attn_fit_start, attn_fit_end, color = 'black', alpha = 0.15)
    ax.text(1.5,1, 'Fit Region', fontsize = 20)
    ax.set_title(f'BTS Attenuation vs Signal Strength Metric', fontsize = 20)
    ax.set_xlabel(f'BTS Attenuation [dB]', fontsize = 20)
    ax.set_ylabel(r'SNR', fontsize = 20)
    ax.legend(fontsize=18, frameon=False)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/fitanttnss_attn_vs_snrhess.png', dpi=300)
    plt.close(fig)
    
    #### for sanity check, do the same fit on just wf p2p and he max
    y_data = [avg_wf_p2p, avg_he_max]
    y_labels = ['Waveform Pk2Pk', 'Hilbert Envelope Max']
    colors = ['#009E73', '#D55E00']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    for y in range(len(y_data)):
        this_y = np.asarray(y_data[y])
        
        ### quick fit to the data
        fit_mask = np.logical_and(run_attenuation >= attn_fit_start, run_attenuation <= attn_fit_end)
        data = [run_attenuation[fit_mask], this_y[fit_mask]]
        res = minimize(attn_fit_func, x0=[this_y[0], 20], args=data, tol = 1e-20, method='L-BFGS-B', bounds=[ (this_y[0]/10,this_y[0]*100), (10,30)], options = {"maxiter":1e7})
        best_fit_snr0 = res.x[0]
        best_fit_a = res.x[1]
        print(f'best fit snr0 = {best_fit_snr0}, best fit a = {best_fit_a}')
        
        interp_attn = np.linspace(np.amin(run_attenuation),np.amax(run_attenuation), 100)
        prediction = get_attn_pred(interp_attn, best_fit_snr0, best_fit_a)
        ax.scatter(run_attenuation, this_y, color = colors[y], s = 30, label = rf'{y_labels[y]} = {best_fit_snr0:.2f}$\times10^{{-A/{best_fit_a:.2f}}}$')
        ax.plot(interp_attn, prediction, color = colors[y], linewidth = 1.5, alpha = 0.5)
        
        ### do a test fit where we force exponent scaling to 20...
        res = minimize(attn_fit_func, x0=[this_y[0], 20], args=data, tol = 1e-20, method='L-BFGS-B', bounds=[ (this_y[0]/10,this_y[0]*100), (20,20)], options = {"maxiter":1e7})
        best_fit_snr0 = res.x[0]
        best_fit_a = res.x[1]
        prediction = get_attn_pred(interp_attn, best_fit_snr0, best_fit_a)
        ax.plot(interp_attn, prediction, color = colors[y], linewidth = 1.5, linestyle = '--', alpha = 0.5, label = rf'{y_labels[y]} = {best_fit_snr0:.2f}$\times10^{{-A/{best_fit_a:.2f}}}$')

    ax.axvspan(attn_fit_start, attn_fit_end, color = 'black', alpha = 0.15)
    ax.text(1.5,150, 'Fit Region', fontsize = 20)
    ax.set_title(f'BTS Attenuation vs Signal Strength', fontsize = 20)
    ax.set_xlabel(f'BTS Attenuation [dB]', fontsize = 20)
    ax.set_ylabel(f'Signal Strength [mV]', fontsize = 20)
    ax.legend(fontsize=18, frameon=False)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/fitanttnss_attn_vs_sigstrength.png', dpi=300)
    plt.close(fig)

    return best_fit_snr0s[0], best_fit_as[0], best_fit_snr0s[1], best_fit_as[1]

def get_trigger_eff(wf_snr0, wf_a, he_snr0, he_a, run_start = 1512, run_end = 1524, rate = 10):
    ### using all our data, let's now fit for beamformer event time distribution
    best_fit_m, best_fit_b, global_min_time = fit_times(run_start, run_end)
    
    ### now loop over each run to get eff and snr!
    trigger_eff = []
    trigger_eff_err = []
    wf_snr = []
    he_snr = []
    all_attenuations = []
    for run in range(run_start, run_end+1):
        ### step 1 -- get number of bts events and accidentals
        this_wf_p2p, this_wf_p2p_noupsample, this_he_max, this_wf_rms, this_wf_median, this_wf_mode, this_wf_mad, this_he_rms, this_he_median, this_he_mode, this_he_mad, run_start_time, run_end_time, n_accidentals = select_beamformer_events(run, best_fit_m, best_fit_b, global_min_time)
        observed_number_of_events = len(this_wf_p2p) - n_accidentals 

        ### step 2 -- get total number of expected events
        ### to first order, this is total data collection time * rate

        ## get run dt (in seconds)!
        dt = run_end_time - run_start_time

        ## now we can get the expected number of events!
        expected_number_of_events = dt * rate
        print(f'for run = {run}, observed {len(this_wf_p2p)} bts events with {n_accidentals:0.2f} accidentals (expected {expected_number_of_events} total BTS events --> eff = {(observed_number_of_events / expected_number_of_events)*100:0.2f}%)')

        ### step 3 -- calculate wf + he SNR based on the fit!!!
        current_atten = run_attn_dict[run]
        this_run_wf_snr = wf_snr0 * 10**(-current_atten/wf_a)
        this_run_he_snr = he_snr0 * 10**(-current_atten/he_a)

        ### finally, get eff and save!
        eff = observed_number_of_events / expected_number_of_events
        eff_error = np.sqrt(eff * (1 - eff) / expected_number_of_events)

        if eff < 0:
            trigger_eff.append(0)
            trigger_eff_err.append(0)
        else:
            trigger_eff.append(eff)
            trigger_eff_err.append(eff_error)
        wf_snr.append(this_run_wf_snr)
        he_snr.append(this_run_he_snr)
        all_attenuations.append(current_atten)

    #print(f'for {run_start} to {run_end}, trigger_eff = {np.asarray(trigger_eff).tolist()}, trigger_eff_err = {np.asarray(trigger_eff_err).tolist()}, wf_snr = {np.asarray(wf_snr).tolist()}, and he_snr = {np.asarray(he_snr).tolist()}')
    return trigger_eff, trigger_eff_err, wf_snr, he_snr, all_attenuations

def plot_trigger_eff_for_paper(wf_snr0, wf_a, he_snr0, he_a):
    ### get trigger eff for uc + ldb data
    uc_trigger_eff, uc_trigger_eff_err, uc_wf_snr, uc_he_snr, all_attenuations = get_trigger_eff(wf_snr0, wf_a, he_snr0, he_a, run_start = 1512, run_end = 1523)
    ldb_trigger_eff, ldb_trigger_eff_err, ldb_wf_snr, ldb_he_snr, all_attenuations = get_trigger_eff(wf_snr0, wf_a, he_snr0, he_a, run_start = 719, run_end = 730)

    ### plot!!!
    all_wf_snr_data = [uc_wf_snr, ldb_wf_snr]
    all_he_snr_data = [uc_he_snr, ldb_he_snr]
    all_eff_data = [uc_trigger_eff, ldb_trigger_eff]
    all_eff_error = [uc_trigger_eff_err, ldb_trigger_eff_err]
    all_colors = ['#009E73', '#D55E00']
    all_labels = ['UChicago (September 2026)', 'LDB (December 2025)']

    ### now we can plot our data!!!
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()

    for a in range(len(all_wf_snr_data)):
        ### plot data
        ax.errorbar(all_wf_snr_data[a], all_eff_data[a], yerr=all_eff_error[a], fmt='o', color=all_colors[a], linestyle='none', ms = 2, capsize=4, label = all_labels[a])

    ax.set_xlabel(f'Scaled Single Antenna SNR (Waveform)', fontsize = 20)
    ax.set_ylabel(f'Efficiency [%]', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.grid(True, which='major', axis='both', linewidth=0.5, color='gray', alpha = 0.5)

    ### add hilbert envelope x-axis
    ax2 = ax.twiny()

    for a in range(len(all_he_snr_data)):
        ax2.errorbar(all_he_snr_data[a], all_eff_data[a], yerr=all_eff_error[a], fmt='s', color=all_colors[a], linestyle='none', ms=2, capsize=4, alpha = 0)

    ax2.set_xlabel('Scaled Single Antenna SNR (Hilbert Envelope)', fontsize=20)
    ax2.tick_params(axis='x', which='both', labelsize=20)

    ax.legend(fontsize=16, frameon=False)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/fitanttnss_trigger_eff_runcomparison.png', dpi=300)
    plt.close(fig)

def fit_bts_scope_traces(filtered):
    ### take our bts scope traces and fit attn vs p2p/he max
    #attns = [0,1,2,3,4,5,6,7,8,9,10]
    attns = [0,1,2,3,4,5,6,7,8,]
    wf_p2ps = []
    he_maxs = []
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    n_attns = len(attns) 
    cmap = plt.get_cmap('Spectral_r')
    norm = mpl.colors.Normalize(vmin=0, vmax=n_attns - 1)
    for a in attns:
        if filtered:
            wf_file = f'/scratch/midway3/darcynewmark/beamformer_testing/avgBTS_ch6_{a}dB_bandpassfilter.Wfm.csv'
        else:
            wf_file = f'/scratch/midway3/darcynewmark/beamformer_testing/avgBTS_ch6_{a}dB.Wfm.csv'
        settings_file = wf_file.replace(".Wfm", "")
        time, wf = read_newscope_cvs(settings_file, wf_file, column = 0)
        wf *= 1e3 ### V to mV
        wf_mode = mode(wf)
        wf -= wf_mode
        color = cmap(norm(a))
        ax.plot(time, wf, color = color, linewidth = 1.5, label = f'Attn = {a} dB')
   
        ### save wf peak to peak + he max
        this_wf_p2p = np.amax(wf) - np.amin(wf)
        this_he_max = np.amax(utils.hilbert_envelope(wf))
        wf_p2ps.append(this_wf_p2p)
        he_maxs.append(this_he_max)

    if filtered:
        ax.set_title(f'Average BTS Scope Traces (Channel 6) After Bandpass Filter', fontsize = 20)
    else:
        ax.set_title(f'Average BTS Scope Traces (Channel 6)', fontsize = 20)
    ax.set_xlabel(f'Time [ns]', fontsize = 20)
    ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
    if filtered:
        ax.set_xlim(-4,15)
    else:
        ax.set_xlim(-1.5, 2)
    ax.legend(fontsize=10, frameon=False)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    if filtered:
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/fitanttnss_avgBTS_scope_traces_bandpassfilter.png', dpi=300)
    else:
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/fitanttnss_avgBTS_scope_traces.png', dpi=300)
    plt.close(fig)

    ### plot attn vs signal strength!
    attns = np.asarray(attns)
    y_data = [wf_p2ps, he_maxs]
    y_labels = ['Waveform Pk2Pk', 'Hilbert Envelope Max']
    colors = ['#009E73', '#D55E00']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    for y in range(len(y_data)):
        this_y = np.asarray(y_data[y])
        
        ### quick fit to the data
        fit_mask = attns < 8
        data = [attns[fit_mask], this_y[fit_mask]]
        res = minimize(attn_fit_func, x0=[this_y[0], 20], args=data, tol = 1e-20, method='L-BFGS-B', bounds=[ (this_y[0]-2,this_y[0]+2), (10,30)], options = {"maxiter":1e7})
        best_fit_snr0 = res.x[0]
        best_fit_a = res.x[1]
        print(f'best fit snr0 = {best_fit_snr0}, best fit a = {best_fit_a}')
        
        interp_attn = np.linspace(np.amin(attns), np.amax(attns), 100)
        prediction = get_attn_pred(interp_attn, best_fit_snr0, best_fit_a)
        ax.scatter(attns, this_y, color = colors[y], s = 30, label = rf'{y_labels[y]} = {best_fit_snr0:.2f}$\times10^{{-A/{best_fit_a:.2f}}}$')
        ax.plot(interp_attn, prediction, color = colors[y], linewidth = 1.5, alpha = 0.5)

    ax.axvspan(0,7, color = 'black', alpha = 0.15)
    ax.text(1.5,5, 'Fit Region', fontsize = 20)
    if filtered:
        ax.set_title(f'Average BTS Scope Traces (Channel 6) After Bandpass Filter', fontsize = 20)
    else:
        ax.set_title(f'Average BTS Scope Traces (Channel 6)', fontsize = 20)
    ax.set_xlabel(f'BTS Attenuation [dB]', fontsize = 20)
    ax.set_ylabel(f'Signal Strength [mV]', fontsize = 20)
    ax.legend(fontsize=18, frameon=False)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    if filtered:
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/fitanttnss_avgBTS_attn_vs_sigstrength_bandpassfilter.png', dpi=300)
    else:
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/fitanttnss_avgBTS_attn_vs_sigstrength.png', dpi=300)
    plt.close(fig)

def compare_az_scan_data(wf_snr0, wf_a, he_snr0, he_a):
    ### let's get our eff for scan over azimuth
    run_ranges = [(1526, 1537), (1539, 1550), (1552, 1563), (1565, 1576), (1578, 1589), (1591, 1602), (1604, 1615), (1617, 1627), (1630, 1641), (1643, 1654)]
    azimuths = [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    
    n_azs = len(azimuths) 
    cmap = plt.get_cmap('Spectral_r')
    norm = mpl.colors.Normalize(vmin = 0, vmax= n_azs - 1)
    fiftypercenteff_snr = []

    for a in range(len(azimuths)):
        this_az = azimuths[a]
        this_run_start = run_ranges[a][0]
        this_run_end = run_ranges[a][1]
        this_color = cmap(norm(a))

        trigger_eff, trigger_eff_err, wf_snr, he_snr, all_attenuations = get_trigger_eff(wf_snr0, wf_a, he_snr0, he_a, run_start = this_run_start, run_end = this_run_end)
        
        ### let's also fit our data!
        data = [wf_snr, trigger_eff]
        res = minimize(erf_fit_func, x0=[1.0, 2], args=data, tol = 1e-16, method='L-BFGS-B', bounds=[ (0,10), (0,10)], options = {"maxiter":1e7})
        best_fit_a, best_fit_b = res.x
        fine_grained_x = np.linspace(np.amin(wf_snr)-1, np.amax(wf_snr)+1, 100)
        best_fit_erf = get_erf_pred(fine_grained_x, best_fit_a, best_fit_b)
        print(f'for azimuth = {this_az}, best_fit_a = {best_fit_a}, best_fit_b = {best_fit_b}')

        ### plot data + fit
        ax.errorbar(wf_snr, trigger_eff, yerr=trigger_eff_err, fmt='o', color=this_color, linestyle='none', ms = 2, capsize=4, label = rf'$\phi = {this_az}^\circ$, best fit a = {best_fit_a:0.2f}, best fit b = {best_fit_b:0.2f}')
        ax.plot(fine_grained_x, best_fit_erf, color=this_color, linestyle = '--', alpha = 0.5)

        fiftypercenteff_snr.append(best_fit_b)

    ax.set_xlabel(f'Scaled Single Antenna SNR (Waveform)', fontsize = 20)
    ax.set_ylabel(f'Efficiency [%]', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.grid(True, which='major', axis='both', linewidth=0.5, color='gray', alpha = 0.5)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/fitanttnss_trigger_eff_azimuthscan.png', dpi=300)
    plt.close(fig)


    ### make another plot of az vs 50% eff point
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    ax.scatter(azimuths, fiftypercenteff_snr, color = 'black', s = 30)
    ax.set_xlabel(f'Azimuth [deg]', fontsize = 20)
    ax.set_ylabel(f'50% Effiency SNR (Best Fit B)', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.grid(True, which='major', axis='both', linewidth=0.5, color='gray', alpha = 0.5)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/fitanttnss_fiftypercentriggereff_vs_azimuth.png', dpi=300)
    plt.close(fig)

def save_bts_events(run_start, run_end):
    ### using all our data, let's now fit for beamformer event time distribution
    best_fit_m, best_fit_b, global_min_time = fit_times(run_start, run_end)
    
    ### now loop over each run to select events and save !!!
    for run in range(run_start, run_end+1):
        if run in BAD_RUNS:
            continue
        this_wf_p2p, this_wf_p2p_noupsample, this_he_max, this_wf_rms, this_wf_median, this_wf_mode, this_wf_mad, this_he_rms, this_he_median, this_he_mode, this_he_mad, run_start_time, run_end_time, n_accidentals = select_beamformer_events(run, best_fit_m, best_fit_b, global_min_time)

if __name__ == "__main__":
    #fit_bts_scope_traces(filtered=False)
    #fit_bts_scope_traces(filtered=True)
    #wf_snr0, wf_a, he_snr0, he_a = fit_attn_sig_strength()
    #plot_trigger_eff_for_paper(wf_snr0, wf_a, he_snr0, he_a)
    #compare_az_scan_data(wf_snr0, wf_a, he_snr0, he_a)
    #save_bts_events(run_start = 1680, run_end = 1690)
    #save_bts_events(run_start = 1746, run_end = 1756)
    #save_bts_events(run_start = 1721, run_end = 1732)
    #save_bts_events(run_start = 1758, run_end = 1774)
    save_bts_events(run_start = 1502, run_end = 1524)
    #save_bts_events(run_start = 1776, run_end = 1789)

