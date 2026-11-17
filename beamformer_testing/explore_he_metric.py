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
from scipy import stats

import pueoTALON.pueoPythonAnalyzer.waveform as waveform
import pueoTALON.pueoPythonAnalyzer.plotter as pltr
import pueoTALON.pueoPythonAnalyzer.antenna as antenna
import pueoTALON.pueoPythonAnalyzer.correlator as correlator
import antennaUtils.response as antResponse
import pueoTALON.pueoPythonAnalyzer.utils as utils
import pueoTALON.pueoPythonAnalyzer.geography as geography

ROOT.gSystem.Load("libpueoEvent.so")

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
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/hemetric_subsecond_vs_triggertime_r{run_start}_r{run_end}.png', dpi=300)
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
        max_x_fitting_val = 8000

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
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/hemetric_subsecond_vs_triggertime_r{run_start}_r{run_end}_fitted.png', dpi=300)
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

    ### take mod 0.1 of subsec and subtract off global_min_time
    trigger_subsec_fmod = trigger_subsec % 0.1
    trigger_time -= global_min_time

    ### select based on our best fit line
    subsec_fit = best_fit_m * trigger_time + best_fit_b
    residual = trigger_subsec_fmod - subsec_fit
    event_selection_mask = np.abs(residual) <= subsec_err
    
    ### let's also save bts entries to a json
    bts_event_entries = entry[event_selection_mask]
    saving_path = f'/scratch/midway3/darcynewmark/beamformer_testing/btsselectedevents_run{run}.json'
    with open(saving_path, 'w') as f:
        json.dump(bts_event_entries.tolist(), f)

    ### return p2p of selected events and rms of sideband events along with start and end time of run
    return peak_to_peak[event_selection_mask], rms_vals[event_selection_mask], he_peak[event_selection_mask], he_mad[event_selection_mask], entry[event_selection_mask]

def plot_he_metric(run_start = 1502, run_end = 1524):
    all_16ch = np.asarray(['0101V', '0201V', '0301V', '0401V',
                           '0102V', '0202V', '0302V', '0402V',
                           '0103V', '0203V', '0303V', '0403V',
                           '0104V', '0204V', '0304V', '0404V'])
    noise_only_channels = np.asarray(['0101V', '0102V', '0103V', '0104V',])
    noise_only_mask = np.isin(all_16ch, noise_only_channels)

    ### using all our data, let's now fit for beamformer event time distribution
    best_fit_m, best_fit_b, global_min_time = fit_times(run_start, run_end)

    ### now loop over each run to plot he metric stuff!
    for run in range(run_start, run_end+1):
        bts_p2p, bts_rms, bts_he_peak, bts_he_mad, bts_entries = select_beamformer_events(run, best_fit_m, best_fit_b, global_min_time)

        ### now lets plot some things -- p2p vs he max, noise rms vs he mad, and snr vs he metric
        bts_p2p_per_channel = bts_p2p
        bts_he_peak_per_channel = bts_he_peak
        bts_rms_noise = np.mean(bts_rms[:, noise_only_mask], axis=1)
        bts_he_mad_noise = np.mean(bts_he_mad[:, noise_only_mask], axis=1)

        snr = bts_p2p_per_channel / (2 * bts_rms_noise[:, np.newaxis])
        he = bts_he_peak_per_channel / bts_he_mad_noise[:, np.newaxis]

        all_x_data = [bts_p2p_per_channel.ravel(), bts_rms_noise.ravel(), snr.ravel()]
        all_y_data = [bts_he_peak_per_channel.ravel(), bts_he_mad_noise.ravel(), he.ravel()]
        x_axes_labels = ['Single Wf Pk2Pk', 'Noise RMS', 'SNR']
        y_axes_labels = ['Single HE Max', 'HE MedAbsDev', 'HESS']
        saving_name = ['bts_p2p_vs_bts_he_peak', 'noise_rms_vs_noise_he_mad', 'snr_vs_he_metric']

        for x in range(len(all_x_data)):
            x_data = np.asarray(all_x_data[x])
            y_data = np.asarray(all_y_data[x])

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
            cb.ax.set_ylabel("Number of Events", fontsize = 16)
            cb.ax.minorticks_off()
            ax.set_title(f'BTS Run {run}, Attenuation = {run_attn_dict[run]}', fontsize = 16)
            ax.set_xlabel(f'{x_axes_labels[x]}', fontsize = 16)
            ax.set_ylabel(f'{y_axes_labels[x]}', fontsize = 16)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=14)
            fig.tight_layout()
            plt.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/hemetric_twod_{saving_name[x]}_r{run}.png', dpi=300)
            plt.close(fig)

            if x_axes_labels[x] == 'SNR':
                ### plot 1D distributions as well
                xdata_onesig_lb, xdata_onesig_mode, xdata_onesig_ub = interval(x_data)
                ydata_onesig_lb, ydata_onesig_mode, ydata_onesig_ub = interval(y_data)
                xdata_width = ((xdata_onesig_ub - xdata_onesig_lb) / xdata_onesig_mode)*100
                ydata_width = ((ydata_onesig_ub - ydata_onesig_lb) / ydata_onesig_mode)*100
                
                min_bin_range = min(np.amin(x_data), np.amin(y_data))
                max_bin_range = max(np.amax(x_data), np.amax(y_data))

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.minorticks_on()
                ax.hist(x_data, bins = 2*nbins, range = (min_bin_range, max_bin_range), linewidth = 1.5, histtype='step', linestyle = '-', color = '#009E73',
                        label = rf'{x_axes_labels[x]}, $1\sigma$ = [{xdata_onesig_lb:0.2f}, {xdata_onesig_ub:0.2f}]'f'\nFrxn Width = {xdata_width:0.2f}%')
                ax.hist(y_data, bins = 2*nbins, range = (min_bin_range, max_bin_range), linewidth = 1.5, histtype='step', linestyle = '-', color = '#D55E00',
                        label = rf'{y_axes_labels[x]}, $1\sigma$ = [{ydata_onesig_lb:0.2f}, {ydata_onesig_ub:0.2f}]'f'\nFrxn Width = {ydata_width:0.2f}%')
                ax.axvline(xdata_onesig_lb, color = '#009E73', alpha = 0.5)
                ax.axvline(xdata_onesig_ub, color = '#009E73', alpha = 0.5)
                ax.axvline(ydata_onesig_lb, color = '#D55E00', alpha = 0.5)
                ax.axvline(ydata_onesig_ub, color = '#D55E00', alpha = 0.5)
                ax.set_title(f'BTS Run {run}, Attenuation = {run_attn_dict[run]}', fontsize = 16)
                ax.set_xlabel(f'Metric', fontsize = 16)
                ax.set_ylabel(f'Number of Events', fontsize = 16)
                ax.legend(fontsize=16, frameon=False)
                ax.tick_params(axis='both', which='both', labelsize=20)
                fig.tight_layout()
                fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/hemetric_oned_{saving_name[x]}_r{run}.png', dpi=300)
                plt.close(fig)

            ### quick check on oddballs
            if run == 1502 and x_axes_labels[x] == 'SNR':
                oddballs_mask = np.any(snr < 6, axis=1)
                print(f'found {np.sum(oddballs_mask)} oddball entries!')
                print(f'entries = {bts_entries[oddballs_mask]}')

        break

def plot_wfs(run = 1502, n_events = 10):
    ### lets plot some wfs and hilbert envelopes!
    ### get bts entries for this run
    this_run_entry_path = f'/scratch/midway3/darcynewmark/beamformer_testing/btsselectedevents_run{run}.json'
    with open(this_run_entry_path) as f:
        all_entries = np.asarray(json.load(f))
    
    if run < 800:
        os.environ["PUEO_ROOT_DATA"] = ("/project/cozzyd/LDB/rootified/")
    else:
        os.environ["PUEO_ROOT_DATA"] = ("/project/cozzyd/postflight/rootified/")
    
    channels = waveform.Channels(run, 0)
    all_16ch = np.asarray(['0101V', '0201V', '0301V', '0401V',
                           '0102V', '0202V', '0302V', '0402V',
                           '0103V', '0203V', '0303V', '0403V',
                           '0104V', '0204V', '0304V', '0404V'])
    noise_only_channels = np.asarray(['0101V', '0102V', '0103V', '0104V',])
    noise_only_mask = np.isin(channels.MI_ringphipol, noise_only_channels)
    dt = 1/3.0
    time_window = 40
    num_rows = 4
    num_cols = 4

    ### and let's just plot the raw data
    #oddball_entries = [145, 197, 505, 2214, 3335, 4445, 4492, 5038, 5410, 5861, 6008, 6738, 6810, 6846, 7344, 7398, 7548, 7867]
    for n in range(n_events):
        this_entry = int(all_entries[n])
        #this_entry = int(oddball_entries[n])
        channels.load_run_event(run, this_entry) ### load our event
        channels.load_waveforms(filter = False) ### load waveforms
        unfilt_wfs = channels.waveforms_unfilt
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows), layout="constrained")
        ### let's get the noise wfs
        noise_wfs = unfilt_wfs[noise_only_mask]
        noise_he = utils.hilbert_envelope(noise_wfs)

        noise_wf_rms = np.mean(utils.RMS(noise_wfs))
        noise_he_rms = np.mean(utils.RMS(noise_he))
        noise_he_median = np.mean(np.median(noise_he, axis = -1))
        noise_he_mode = np.mean(mode(noise_he, axis = -1))
        noise_he_mad = np.mean(stats.median_abs_deviation(noise_he, axis=-1))
         
        for i, ant in enumerate(all_16ch):
            row = i // num_cols
            col = i % num_cols
            ant_mask = np.isin(channels.MI_ringphipol, np.asarray([ant]))
            this_wf = unfilt_wfs[ant_mask].flatten() ### flatten to 1D
            this_he = utils.hilbert_envelope(this_wf) 
            this_times = np.arange(len(this_wf)) * dt
            if ant not in noise_only_channels:
                ### lets get snr and hess
                snr = (np.amax(this_wf) - np.amin(this_wf)) / (2*noise_wf_rms) 
                hess = np.amax(this_he) / noise_he_mad
                time_at_max = this_times[np.argmax(this_wf)]
                time_mask = np.logical_and(this_times > (time_at_max - time_window), this_times < (time_at_max + time_window))
                axes[row, col].plot(this_times[time_mask], this_wf[time_mask], color='#009E73', ls = '-')
                axes[row, col].plot(this_times[time_mask], this_he[time_mask], color='#D55E00', ls = '-')
                #axes[row, col].text(0.54, 0.8, f'SNR={snr:0.2f}', transform=axes[row, col].transAxes, color='#009E73', fontsize = 18)
                #axes[row, col].text(0.54, 0.7, f'HESS={hess:0.2f}', transform=axes[row, col].transAxes, color='#D55E00', fontsize = 18)
                axes[row, col].axhline(noise_he_rms*2, color = 'green', ls = '--')
                axes[row, col].axhline(noise_he_median, color = 'black', ls = '--')
                axes[row, col].axhline(noise_he_mode, color = 'red', ls = '--')
                axes[row, col].axhline(noise_he_mad, color = 'blue', ls = '--')
            else:
                axes[row, col].plot(this_times, this_wf, color='#009E73', ls = '-')
                axes[row, col].plot(this_times, this_he, color='#D55E00', ls = '-')
                axes[row, col].axhline(utils.RMS(this_he)*2, color = 'green', ls = '-', label = 'HE 2*RMS')
                axes[row, col].axhline(np.median(this_he), color = 'black', ls = '-', label = 'HE Median')
                axes[row, col].axhline(mode(this_he), color = 'red', ls = '-', label = 'HE Mode')
                axes[row, col].axhline(stats.median_abs_deviation(this_he), color = 'blue', ls = '-', label = 'HE MAD')
            axes[row, col].text(0.7, 0.9, f'{ant}', transform=axes[row, col].transAxes, fontsize = 20)

        axes[0, 0].legend(fontsize=12, frameon=False)   
        fig.supxlabel("Time (ns)", fontsize=40)
        fig.supylabel("Voltage (mV)", fontsize=40)
        plt.tight_layout()
        plt.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/hemetric_wfs_r{run}_e{this_entry}.png', dpi=300)
        plt.close()

        

if __name__ == "__main__":
    #plot_he_metric(run_start = 1502, run_end = 1524)
    plot_wfs(run = 1502)
    #plot_wfs(run = 1508)
    #plot_wfs(run = 1516)


