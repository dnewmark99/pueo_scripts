import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import ROOT
import scipy
import argparse
import os
import json
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import functools
from functools import reduce
import glob
from scipy.optimize import minimize, minimize_scalar
import pandas as pd
import random
from scipy import special
from postrunmapping import run_attn_dict, BAD_RUNS
from scipy import stats
import csv
from scipy.stats import f
from scipy.stats import norm
from fit_attn_vs_sigstrength import fit_times, select_beamformer_events

import pueoTALON.pueoPythonAnalyzer.waveform as waveform
import pueoTALON.pueoPythonAnalyzer.plotter as pltr
import pueoTALON.pueoPythonAnalyzer.antenna as antenna
import pueoTALON.pueoPythonAnalyzer.correlator as correlator
import antennaUtils.response as antResponse
import pueoTALON.pueoPythonAnalyzer.utils as utils
import pueoTALON.pueoPythonAnalyzer.geography as geography

ROOT.gSystem.Load("libpueoEvent.so")
os.environ["PUEO_ROOT_DATA"] = ("/project/cozzyd/postflight/rootified/")

only_12ch = np.asarray(['0201V', '0301V', '0401V',
                        '0202V', '0302V', '0402V',
                        '0203V', '0303V', '0403V',
                        '0204V', '0304V', '0404V'])
all_16ch = np.asarray(['0101V', '0201V', '0301V', '0401V',
                       '0102V', '0202V', '0302V', '0402V',
                       '0103V', '0203V', '0303V', '0403V',
                       '0104V', '0204V', '0304V', '0404V'])

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


def get_template(run, selected_ch):
    ### read in the averaged wfs
    path = f'/scratch/midway3/darcynewmark/beamformer_testing/accumulatedbtswfs_run{run}.json'
    with open(path) as f:
        avg_accumulated_wf = json.load(f)
    this_wf = np.asarray(avg_accumulated_wf[selected_ch])
    return this_wf

def cross_corr_events_with_template(run, template, selected_ch, upsample_factor = 10):
    ### ok we have selected BTS events based on trigger time and subsecond distributions
    ### let's cross correlate the selected events with the template and look at distribution of corrs and max lags

    ### first, get entries for bts selected events
    this_run_entry_path = f'/scratch/midway3/darcynewmark/beamformer_testing/btsselectedevents_run{run}.json'
    with open(this_run_entry_path) as f:
        entries = np.asarray(json.load(f))

    ### now loop over all events and cross correlate each event with the template
    ### save max lag and cross corr value for bts and non bts events
    bts_events_max_corr = []
    bts_events_max_lag = []
    
    channels = waveform.Channels(run, 0)
    wf_mask = np.isin(channels.MI_ringphipol, np.asarray([selected_ch])) ### mask out our special channel
    for idx_counter, ev in enumerate(entries):
        channels.load_event(int(ev)) ### load our event
        channels.load_waveforms(filter = False) ### load waveforms
        unfilt_wfs = channels.waveforms_unfilt
        
        this_wf = unfilt_wfs[wf_mask].flatten() ### flatten to 1D
        up_this_wf = scipy.signal.resample(this_wf, len(this_wf)*upsample_factor, axis=-1) ### upsample this wf for accumulation

        ### now cross correlate!
        corr = scipy.signal.correlate(template, up_this_wf, mode='full', method='fft')
        lags = scipy.signal.correlation_lags(len(template), len(up_this_wf), mode='full')
        max_corr = np.amax(corr)
        max_lag = lags[np.argmax(corr)]
       
        ### save
        bts_events_max_corr.append(max_corr)
        bts_events_max_lag.append(max_lag)

    ### now do again but with NON bts events
    this_run_entry_path = f'/scratch/midway3/darcynewmark/beamformer_testing/NONbtsselectedevents_run{run}.json'
    with open(this_run_entry_path) as f:
        NONentries = np.asarray(json.load(f))

    ### now loop over all events and cross correlate each event with the template
    ### save max lag and cross corr value for bts and non bts events
    NONbts_events_max_corr = []
    NONbts_events_max_lag = []
    
    channels = waveform.Channels(run, 0)
    wf_mask = np.isin(channels.MI_ringphipol, np.asarray([selected_ch])) ### mask out our special channel
    for idx_counter, ev in enumerate(NONentries):
        channels.load_event(int(ev)) ### load our event
        channels.load_waveforms(filter = False) ### load waveforms
        unfilt_wfs = channels.waveforms_unfilt
        
        this_wf = unfilt_wfs[wf_mask].flatten() ### flatten to 1D
        up_this_wf = scipy.signal.resample(this_wf, len(this_wf)*upsample_factor, axis=-1) ### upsample this wf for accumulation

        ### now cross correlate!
        corr = scipy.signal.correlate(template, up_this_wf, mode='full', method='fft')
        lags = scipy.signal.correlation_lags(len(template), len(up_this_wf), mode='full')
        max_corr = np.amax(corr)
        max_lag = lags[np.argmax(corr)]
       
        ### save
        NONbts_events_max_corr.append(max_corr)
        NONbts_events_max_lag.append(max_lag)

    ### now saveeeeee
    saving_path = f'/scratch/midway3/darcynewmark/beamformer_testing/templatecorrbtsselectedevents_run{run}.json'
    data = {
        'entry': np.asarray(entries).tolist(),
        'max_corr': np.asarray(bts_events_max_corr).tolist(),
        'max_lag': np.asarray(bts_events_max_lag).tolist(),
    }
    with open(saving_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    saving_path = f'/scratch/midway3/darcynewmark/beamformer_testing/templatecorrNONbtsselectedevents_run{run}.json'
    data = {
        'entry': np.asarray(NONentries).tolist(),
        'max_corr': np.asarray(NONbts_events_max_corr).tolist(),
        'max_lag': np.asarray(NONbts_events_max_lag).tolist(),
    }
    with open(saving_path, 'w') as f:
        json.dump(data, f, indent=4)

def plot_cross_corrs(run_start, run_end, selected_ch):
    
    for run in range(run_start, run_end+1):
        print(f'on run = {run}')
        if run in BAD_RUNS:
            continue
        
        ### get attenuation
        this_attn = run_attn_dict[run]

        ### get max correlation + max lag value for bts events
        bts_event_path = f'/scratch/midway3/darcynewmark/beamformer_testing/templatecorrbtsselectedevents_run{run}.json'
        with open(bts_event_path) as f:
            bts_dict = json.load(f)
            bts_max_corr = np.asarray(bts_dict['max_corr'])
            bts_max_lag = np.asarray(bts_dict['max_lag'])

        ### get max correlation + max lag value for NON bts events
        NONbts_event_path = f'/scratch/midway3/darcynewmark/beamformer_testing/templatecorrNONbtsselectedevents_run{run}.json'
        with open(NONbts_event_path) as f:
            NONbts_dict = json.load(f)
            NONbts_max_corr = np.asarray(NONbts_dict['max_corr'])
            NONbts_max_lag = np.asarray(NONbts_dict['max_lag'])

        ### plot!
        max_corr_bins = (min(np.amin(bts_max_corr), np.amin(NONbts_max_corr)), max(np.amax(bts_max_corr), np.amax(NONbts_max_corr)))
        max_corr_nbins = 100 
        max_lag_bins = (min(np.amin(bts_max_lag), np.amin(NONbts_max_lag)), max(np.amax(bts_max_lag), np.amax(NONbts_max_lag)))
        max_lag_nbins = 300 

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.minorticks_on()
        ax.hist(bts_max_corr, bins = max_corr_nbins, range = max_corr_bins, linewidth = 1.5, histtype='step', linestyle = '-', color = '#009E73', label = rf'BTS Selected Events')
        ax.hist(NONbts_max_corr, bins = max_corr_nbins, range = max_corr_bins, linewidth = 1.5, histtype='step', linestyle = '-', color = '#D55E00', label = rf'NON BTS Selected Events')
        nonbtsmax = np.amax(NONbts_max_corr)
        ax.axvline(nonbtsmax, linewidth = 1.5, color = '#D55E00', label = f'NON BTS Selected Events Max Corr = {nonbtsmax:0.2e}')
        ax.set_title(f'Run {run}, Ant {selected_ch}, Attn = {this_attn} dB', fontsize = 20)
        ax.set_xlabel(f'Max Correlation With Template', fontsize = 20)
        ax.set_ylabel(f'Counts', fontsize = 20)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.legend(fontsize=14, frameon=False)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/single_channel_debug_wnoise_maxcorr_run{run}_{selected_ch}.png', dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.minorticks_on()
        ### let's also get the platau for our max lags in data
        bts_max_lag_lb, bts_max_lag_mode, bts_max_lag_ub = interval(bts_max_lag)
        ax.hist(bts_max_lag, bins = max_lag_nbins, range = max_lag_bins, linewidth = 1.5, histtype='step', linestyle = '-', color = '#009E73', label = rf'BTS Selected Events, Mode = {bts_max_lag_mode:0.2f}, 1 Sig = [{bts_max_lag_lb:0.2f}, {bts_max_lag_ub:0.2f}]')
        ax.hist(NONbts_max_lag, bins = max_lag_nbins, range = max_lag_bins, linewidth = 1.5, histtype='step', linestyle = '-', color = '#D55E00', label = rf'NON BTS Selected Events')
        ax.axvline(bts_max_lag_mode, linewidth = 1.5,  linestyle = '-', color = '#009E73')
        ax.axvline(bts_max_lag_mode - bts_max_lag_lb, linewidth = 1.5,  linestyle = '--', color = '#009E73')
        ax.axvline(bts_max_lag_mode + bts_max_lag_ub, linewidth = 1.5,  linestyle = '--', color = '#009E73')
        ax.set_title(f'Run {run}, Ant {selected_ch}, Attn = {this_attn} dB', fontsize = 20)
        ax.set_xlabel(f'Max Lag With Template', fontsize = 20)
        ax.set_ylabel(f'Counts', fontsize = 20)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.legend(fontsize=14, frameon=False)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/single_channel_debug_wnoise_maxlag_run{run}_{selected_ch}.png', dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--array", type = int, default = 0)
    args = parser.parse_args()
    
    selected_ch = '0201V'
    save_data = False
    if save_data:
        run = 1502
        template = get_template(run = run, selected_ch = selected_ch)
        
        run_ranges = (1502, 1524)
        all_runs = list(range(run_ranges[0], run_ranges[1] + 1))
        all_runs = [r for r in all_runs if r not in BAD_RUNS]
        run = all_runs[args.array]
        cross_corr_events_with_template(run = run, template = template, selected_ch = selected_ch)

    ### now plot distros!
    plot_cross_corrs(run_start = 1502, run_end = 1524, selected_ch = selected_ch)

