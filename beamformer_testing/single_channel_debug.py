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

def get_attn_pred(x, snr0, a):
    return snr0 * 10**(-x/a)

def attn_fit_func(params, data):
    x = data[0]
    y = data[1]
    snr0, a = params
    pred = get_attn_pred(x, snr0, a)
    if len(data) > 2:
        y_err = data[2]
        return np.sum(((y - pred) / y_err)**2)
    else:
        return np.sum((y - pred)**2)

def gauss(x, mu, sigma, scale):
    return scale * (1 / np.sqrt(2*np.pi*sigma*sigma)) * np.exp(-((x - mu)**2)/(2*sigma*sigma))

def gauss_fit_func(params, data):
    x = data[0]
    y = data[1]
    mu, sigma, scale = params
    pred = gauss(x, mu, sigma, scale)
    mask = y > 0
    return np.sum((y[mask] - pred[mask])**2 / y[mask])

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

def fit_lucas_data():
    csv_path = f'/scratch/midway3/darcynewmark/beamformer_testing/lucas_pk_atten.csv'
    df = pd.read_csv(csv_path)
    attn = df.iloc[:, 0].to_numpy()
    wfpk2pk = df.iloc[:, 1].to_numpy()
    
    ### let's fit!!!
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    
    all_attn = np.asarray(attn)
    all_wfpk2pk = np.asarray(wfpk2pk)

    fit_start = 5
    fit_end = 13
    fit_mask = np.logical_and(all_attn >= fit_start, all_attn <= fit_end)
    data = [all_attn[fit_mask], all_wfpk2pk[fit_mask]]
    
    res = minimize(attn_fit_func, x0=[all_wfpk2pk[0], 20], args=data, tol = 1e-12, method='L-BFGS-B', bounds=[ (all_wfpk2pk[0]*0.5, all_wfpk2pk[0]*1.5), (10,30)], options = {"maxiter":1e7})
    best_fit_snr0 = res.x[0]
    best_fit_a = res.x[1]
    
    cov = np.asarray(res.hess_inv.todense())
    snr0_err, a_err = np.sqrt(np.diag(cov))
    print(f"snr0 = {best_fit_snr0:.4f} +/- {snr0_err:.4f}")
    print(f"a = {best_fit_a:.4f} +/- {a_err:.4f}")

    interp_attn = np.linspace(np.amin(all_attn), np.amax(all_attn), 100)
    prediction = get_attn_pred(interp_attn, best_fit_snr0, best_fit_a)
    ax.scatter(all_attn, all_wfpk2pk, color = '#009E73', s = 20, label = 'Lucas Data')
    ax.plot(interp_attn, prediction, color = '#009E73', linewidth = 1.5, linestyle = '-', label=rf'SNR = ({best_fit_snr0:.2f} $\pm$ {snr0_err:.2f})'rf'$\times 10^{{-A/({best_fit_a:.2f} \pm {a_err:.2f})}}$'rf' (Fitting [{fit_start}, {fit_end}] dB)')

    ax.axvspan(fit_start,fit_end, color = '#009E73', alpha = 0.15)
    ax.set_xlabel(f'Attenuation [dB]', fontsize = 20)
    ax.set_ylabel(f'WkPk2Pk [mV]', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.legend(fontsize=14, frameon=False)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/fit_lucas_data.png', dpi=300)
    plt.close(fig)

def get_trigger_eff(run_start, run_end, rate = 10):
    ### using all our data, let's now fit for beamformer event time distribution
    best_fit_m, best_fit_b, global_min_time = fit_times(run_start, run_end)
    
    ### now loop over each run to get eff and snr!
    trigger_eff = []
    trigger_eff_err = []
    for run in range(run_start, run_end+1):
        current_atten = run_attn_dict[run]
        ### step 1 -- get number of bts events and accidentals
        this_wf_p2p, this_wf_p2p_noupsample, this_he_max, this_wf_rms, this_wf_median, this_wf_mode, this_wf_mad, this_he_rms, this_he_median, this_he_mode, this_he_mad, run_start_time, run_end_time, n_accidentals = select_beamformer_events(run, best_fit_m, best_fit_b, global_min_time)
        observed_number_of_events = len(this_wf_p2p) - n_accidentals 

        ### step 2 -- get total number of expected events
        ### to first order, this is total data collection time * rate
        ## get run dt (in seconds)!
        dt = run_end_time - run_start_time

        ## now we can get the expected number of events!
        expected_number_of_events = dt * rate

        ### finally, get eff and save!
        eff = observed_number_of_events / expected_number_of_events
        eff_error = np.sqrt(eff * (1 - eff) / expected_number_of_events)

        if eff < 0:
            trigger_eff.append(0)
            trigger_eff_err.append(0)
        else:
            trigger_eff.append(eff)
            trigger_eff_err.append(eff_error)

        print(f'at run = {run}, attn = {current_atten}, eff = {eff*100:0.2f}% +- {eff_error*100:0.2f}%')

    return trigger_eff, trigger_eff_err

def compare_avg_to_distro(run_start, run_end, selected_ch, fitting_ranges = [(5,13),], fitting_linestyles = ['-']):
    all_attns = []
    avg_wf_p2p = []
    avg_distro_wf_p2p = []
    avg_distro_wf_p2p_err = []
    ch_idx = np.where(only_12ch == selected_ch)[0][0]

    for run in range(run_start, run_end+1):
        if run in BAD_RUNS:
            continue

        ### get the attenuation for this run
        this_attn = run_attn_dict[run]

        ### get mie p2p from the averaged wfs
        path = f'/scratch/midway3/darcynewmark/beamformer_testing/accumulatedbtswfs_run{run}.json'
        with open(path) as f:
            avg_accumulated_wf = json.load(f)
        this_wf = np.asarray(avg_accumulated_wf[selected_ch])
        this_mie_p2p = np.amax(this_wf) - np.amin(this_wf)

        ### and get distribution of p2p for selected bts events!
        this_run_entry_path = f'/scratch/midway3/darcynewmark/beamformer_testing/btsselectedevents_run{run}.json'
        with open(this_run_entry_path) as f:
            entries = np.asarray(json.load(f))
        
        data = grab_data_json([run])
        entry = np.asarray(data["entry"])
        wf_peak_to_peak = np.asarray(data["wf_peak_to_peak"])
        this_ch_wf_peak_to_peak = wf_peak_to_peak[:, ch_idx]

        ### mask to select bts events
        bts_events_mask = np.isin(entry, entries)
        all_bts_events_wfp2p = this_ch_wf_peak_to_peak[bts_events_mask]

        ### get average + error
        dist_avg = np.mean(all_bts_events_wfp2p)
        error_on_average = np.std(all_bts_events_wfp2p, ddof=1)
        print(f'avg = {dist_avg}, err = {error_on_average}, N events = {len(entries)}')

        ### now let's plot!
        min_bin_range = np.amin(all_bts_events_wfp2p)
        max_bin_range = np.amax(all_bts_events_wfp2p) 
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.minorticks_on()
        ax.hist(all_bts_events_wfp2p, bins = 160, range = (min_bin_range, max_bin_range), linewidth = 1.5, histtype='step', linestyle = '-', color = '#009E73', label = rf'Waveform Pk2Pk (10x Upsampling)')
        ax.axvline(dist_avg, color = '#009E73', linewidth = 1.5, linestyle = '--', label = rf'Distribution Avg WfPk2Pk = {dist_avg:0.2f} $\pm$ {error_on_average:0.2f} mV')
        ax.axvline(dist_avg - error_on_average, color = '#009E73', linewidth = 1.5, linestyle = '--')
        ax.axvline(dist_avg + error_on_average, color = '#009E73', linewidth = 1.5, linestyle = '--')
        ax.axvline(this_mie_p2p, color = 'black', linewidth = 1.5, linestyle = '--', label = f'Avg WfPk2Pk = {this_mie_p2p:0.2f} mV')
        ax.set_title(f'Wf Pk2Pk, Run {run}, Ant {selected_ch}, Attn = {this_attn} dB', fontsize = 20)
        ax.set_xlabel(f'Waveform Peak to Peak [mV]', fontsize = 20)
        ax.set_ylabel(f'Counts', fontsize = 20)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.legend(fontsize=14, frameon=False)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/single_channel_debug_wfpk2pk_run{run}_{selected_ch}.png', dpi=300)
        plt.close(fig)
        
        ### now save
        avg_wf_p2p.append(this_mie_p2p)
        all_attns.append(this_attn)
        avg_distro_wf_p2p.append(dist_avg)
        avg_distro_wf_p2p_err.append(error_on_average)
    
    all_attns = np.asarray(all_attns)
    avg_wf_p2p = np.asarray(avg_wf_p2p)
    avg_distro_wf_p2p = np.asarray(avg_distro_wf_p2p)
    avg_distro_wf_p2p_err = np.asarray(avg_distro_wf_p2p_err)

    ### plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    
    ### add some fits
    fit_data = [avg_distro_wf_p2p, avg_wf_p2p]
    colors = ['#009E73',  'black']
    labels = ['Distribution Avg WfPk2Pk', 'Avg WfPk2Pk']
        
    for ff in range(len(fitting_ranges)):
        fit_start = fitting_ranges[ff][0]
        fit_end = fitting_ranges[ff][1]

        for y in range(len(fit_data)):
            y_data = fit_data[y]
            
            fit_mask = np.logical_and(all_attns >= fit_start, all_attns <= fit_end)
            data = [all_attns[fit_mask], y_data[fit_mask]]
            if y == 0:
                ax.errorbar(all_attns, y_data, yerr=avg_distro_wf_p2p_err, fmt='o', color=colors[y], markersize=4, capsize=3,)
                data = [all_attns[fit_mask], y_data[fit_mask], avg_distro_wf_p2p_err[fit_mask]]

            res = minimize(attn_fit_func, x0=[y_data[0], 20], args=data, tol = 1e-12, method='L-BFGS-B', bounds=[ (y_data[0]*0.5,y_data[0]*1.5), (10,30)], options = {"maxiter":1e7})
            best_fit_snr0 = res.x[0]
            best_fit_a = res.x[1]
            print(f'best fit snr0 = {best_fit_snr0}, best fit a = {best_fit_a}')

            cov = np.asarray(res.hess_inv.todense())
            snr0_err, a_err = np.sqrt(np.diag(cov))
            print(f"snr0 = {best_fit_snr0:.4f} +/- {snr0_err:.4f}")
            print(f"a = {best_fit_a:.4f} +/- {a_err:.4f}")

            interp_attn = np.linspace(np.amin(all_attns), np.amax(all_attns), 100)
            prediction = get_attn_pred(interp_attn, best_fit_snr0, best_fit_a)
            ax.scatter(all_attns, y_data, color = colors[y], s=10,)
            ax.plot(interp_attn, prediction, color = colors[y], linewidth = 1.5, linestyle = fitting_linestyles[ff], label=rf'{labels[y]} = ({best_fit_snr0:.2f} $\pm$ {snr0_err:.2f})'rf'$\times 10^{{-A/({best_fit_a:.2f} \pm {a_err:.2f})}}$'rf' (Fitting [{fit_start}, {fit_end}] dB)')
            #ax.axvspan(fit_start,fit_end, color = colors[y], alpha = 0.15)
       
            #### let's also force the fit to be a/20
            #res = minimize(attn_fit_func, x0=[y_data[0], 20], args=data, tol = 1e-20, method='L-BFGS-B', bounds=[ (y_data[0]*0.5,y_data[0]*1.5), (20,20)], options = {"maxiter":1e7})
            #best_fit_snr0 = res.x[0]
            #best_fit_a = res.x[1]
            #print(f'best fit snr0 = {best_fit_snr0}, best fit a = {best_fit_a}')
            #prediction = get_attn_pred(interp_attn, best_fit_snr0, best_fit_a)
            #ax.plot(interp_attn, prediction, color = colors[y], linewidth = 1.5, linestyle = '--',label = rf'{labels[y]} = {best_fit_snr0:.2f}$\times10^{{-A/{best_fit_a:.2f}}}$ (Fitting [{fit_start}, {fit_end}] dB)')

    ax.set_title(f'Run {run_start} to {run_end}, Ant {selected_ch}', fontsize = 20)
    ax.set_xlabel(f'BTS Attenuation [dB]', fontsize = 20)
    ax.set_ylabel(f'Avg Waveform Pk2Pk [mV]', fontsize = 20)
    ax.legend(fontsize=10, frameon=False)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.set_yscale('log')
    ax.grid(True, which='both', axis='both', linewidth=0.5, color='gray', alpha = 0.5)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/single_channel_debug_avgwf_attn_vs_sigstrength_r{run_start}_r{run_end}_{selected_ch}.png', dpi=300)
    plt.close(fig)

def fit_distros(run_start, run_end, selected_ch, trigger_eff, trigger_eff_err):
    ch_idx = np.where(only_12ch == selected_ch)[0][0]
    noise_ch_idx = np.where(all_16ch == selected_ch)[0][0]
    all_attn = []
    all_signal_sigma = []
    all_avg_snr = []
    all_avg_snr_err = []
    all_signal_mean = []
    all_signal_stddev = []
    for run in range(run_start, run_end+1):
        if run in BAD_RUNS:
            continue

        ### get the attenuation for this run
        this_attn = run_attn_dict[run]

        ### get mie p2p from the averaged wfs
        path = f'/scratch/midway3/darcynewmark/beamformer_testing/accumulatedbtswfs_run{run}.json'
        with open(path) as f:
            avg_accumulated_wf = json.load(f)
        this_wf = np.asarray(avg_accumulated_wf[selected_ch])
        this_mie_p2p = np.amax(this_wf) - np.amin(this_wf)

        ### and get distribution of p2p for selected bts events!
        this_run_entry_path = f'/scratch/midway3/darcynewmark/beamformer_testing/btsselectedevents_run{run}.json'
        with open(this_run_entry_path) as f:
            entries = np.asarray(json.load(f))
        
        data = grab_data_json([run])
        entry = np.asarray(data["entry"])
        wf_peak_to_peak = np.asarray(data["wf_peak_to_peak"])
        wf_rms = np.asarray(data["wf_rms"])
        this_ch_wf_peak_to_peak = wf_peak_to_peak[:, ch_idx]
        this_ch_noise_wf_rms = wf_rms[:, noise_ch_idx]

        ### mask to select bts events
        bts_events_mask = np.isin(entry, entries)
        all_bts_events_wfp2p = this_ch_wf_peak_to_peak[bts_events_mask]

        ### get average + error
        dist_avg = np.mean(all_bts_events_wfp2p)
        error_on_average = np.std(all_bts_events_wfp2p, ddof=1)
        
        ### and finally get distribution of p2p for NON selected bts events!
        this_run_entry_path = f'/scratch/midway3/darcynewmark/beamformer_testing/NONbtsselectedevents_run{run}.json'
        with open(this_run_entry_path) as f:
            NONentries = np.asarray(json.load(f))
        
        NONbts_events_mask = np.isin(entry, NONentries)
        NONbts_events_wfp2p = this_ch_wf_peak_to_peak[NONbts_events_mask]
        NONbts_events_wfrms = this_ch_noise_wf_rms[NONbts_events_mask]

        ### calculate rms
        signal_rms = utils.RMS(all_bts_events_wfp2p)
        noise_rms = utils.RMS(NONbts_events_wfp2p)

        ### let's fit our signal distribution distributions with gaussians!
        nbins = 160
        min_bin_range = min(np.amin(all_bts_events_wfp2p), np.amin(NONbts_events_wfp2p))
        max_bin_range = max(np.amax(all_bts_events_wfp2p), np.amax(NONbts_events_wfp2p))
        
        binned_signal_wfp2p_counts, bin_edges = np.histogram(all_bts_events_wfp2p, bins = nbins, range = (min_bin_range, max_bin_range))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        max_distro = np.amax(binned_signal_wfp2p_counts)
        mean_distro = np.mean(all_bts_events_wfp2p)
        stddev_distro = np.std(all_bts_events_wfp2p)
        bounds = [(mean_distro/10, mean_distro*10), (stddev_distro/10, stddev_distro*10), (0, 1e6)]
        p0 = [mean_distro, stddev_distro, max_distro]
        res = minimize(gauss_fit_func, x0=p0, args=[bin_centers, binned_signal_wfp2p_counts], tol = 1e-12, method='L-BFGS-B', bounds = bounds,  options = {"maxiter":1e7})
        best_fit_signal_mu, best_fit_signal_sigma, best_fit_signal_scale = res.x

        print(f'for run = {run}, attn = {this_attn}, wf peak to peak distribution best fit mu = {best_fit_signal_mu}')

        ### now let's plot!
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.minorticks_on()
        ax.hist(all_bts_events_wfp2p, bins = nbins, range = (min_bin_range, max_bin_range), linewidth = 1.5, histtype='step', linestyle = '-', color = '#009E73', label = rf'BTS Selected Events, Mu = {best_fit_signal_mu:0.2f}, Sigma = {best_fit_signal_sigma:0.2f}')
        ax.hist(NONbts_events_wfp2p, bins = nbins, range = (min_bin_range, max_bin_range), linewidth = 1.5, histtype='step', linestyle = '-', color = '#D55E00', label = rf'NON BTS Selected Events')
        ax.plot(bin_centers, gauss(bin_centers, best_fit_signal_mu, best_fit_signal_sigma, best_fit_signal_scale), linewidth = 1.5, linestyle = '-', color = '#009E73')
        ax.axvline(best_fit_signal_mu,  linewidth = 1.5, linestyle = '-', color = '#009E73')
        ax.axvline(best_fit_signal_mu - best_fit_signal_sigma,  linewidth = 1.5, linestyle = '--', color = '#009E73')
        ax.axvline(best_fit_signal_mu + best_fit_signal_sigma,  linewidth = 1.5, linestyle = '--', color = '#009E73')
        #ax.axvline(np.mean(all_bts_events_wfp2p) - np.std(all_bts_events_wfp2p), linewidth = 1.5, linestyle = ':', color = '#009E73', label = f'BTS Selected Events Mean +- StdDev = {np.mean(all_bts_events_wfp2p):0.2f} +- {np.std(all_bts_events_wfp2p):0.2f}')
        #ax.axvline(np.mean(all_bts_events_wfp2p) + np.std(all_bts_events_wfp2p), linewidth = 1.5, linestyle = ':', color = '#009E73', )
        #ax.axvline(np.mean(NONbts_events_wfp2p) - np.std(NONbts_events_wfp2p), linewidth = 1.5, linestyle = ':', color = '#D55E00', label = f'NON BTS Selected Events Mean +- StdDev = {np.mean(NONbts_events_wfp2p):0.2f} +- {np.std(NONbts_events_wfp2p):0.2f}')
        #ax.axvline(np.mean(NONbts_events_wfp2p) + np.std(NONbts_events_wfp2p), linewidth = 1.5, linestyle = ':', color = '#D55E00',)
        ax.set_title(f'Wf Pk2Pk, Run {run}, Ant {selected_ch}, Attn = {this_attn} dB', fontsize = 20)
        ax.set_xlabel(f'Waveform Peak to Peak [mV]', fontsize = 20)
        ax.set_ylabel(f'Counts', fontsize = 20)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.legend(fontsize=14, frameon=False)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/single_channel_debug_wnoise_wfpk2pk_run{run}_{selected_ch}.png', dpi=300)
        plt.close(fig)
        
        ### let's also plot the noise rms!!!!
        nbins = 50
        min_bin_range = np.amin(NONbts_events_wfrms) 
        max_bin_range = np.amax(NONbts_events_wfrms)
        
        binned_noise_wfrms_counts, bin_edges = np.histogram(NONbts_events_wfrms, bins = nbins, range = (min_bin_range, max_bin_range))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        max_distro = np.amax(binned_noise_wfrms_counts)
        mean_distro = np.mean(NONbts_events_wfrms)
        stddev_distro = np.std(NONbts_events_wfrms)
        bounds = [(mean_distro/10, mean_distro*10), (stddev_distro/10, stddev_distro*10), (0, 1e6)]
        p0 = [mean_distro, stddev_distro, max_distro]
        res = minimize(gauss_fit_func, x0=p0, args=[bin_centers, binned_noise_wfrms_counts], tol = 1e-12, method='L-BFGS-B', bounds = bounds,  options = {"maxiter":1e7})
        best_fit_noise_mu, best_fit_noise_sigma, best_fit_noise_scale = res.x

        ### now let's plot!
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.minorticks_on()
        ax.hist(NONbts_events_wfrms, bins = nbins, range = (min_bin_range, max_bin_range), linewidth = 1.5, histtype='step', linestyle = '-', color = '#D55E00', label = rf'NON BTS Selected Events, Mu = {best_fit_noise_mu:0.2f}, Sigma = {best_fit_noise_sigma:0.2f}')
        ax.plot(bin_centers, gauss(bin_centers, best_fit_noise_mu, best_fit_noise_sigma, best_fit_noise_scale), linewidth = 1.5, linestyle = '-', color = '#D55E00')
        ax.axvline(best_fit_noise_mu,  linewidth = 1.5, linestyle = '-', color = '#D55E00')
        ax.axvline(best_fit_noise_mu - best_fit_noise_sigma,  linewidth = 1.5, linestyle = '--', color = '#D55E00')
        ax.axvline(best_fit_noise_mu + best_fit_noise_sigma,  linewidth = 1.5, linestyle = '--', color = '#D55E00')
        #ax.axvline(np.mean(NONbts_events_wfrms) - np.std(NONbts_events_wfrms), linewidth = 1.5, linestyle = ':', color = '#D55E00', label = f'NON BTS Selected Events Mean +- StdDev = {np.mean(NONbts_events_wfrms):0.2f} +- {np.std(NONbts_events_wfrms):0.2f}')
        #ax.axvline(np.mean(NONbts_events_wfrms) + np.std(NONbts_events_wfrms), linewidth = 1.5, linestyle = ':', color = '#D55E00', )
        ax.set_title(f'Noise Wf RMS, Run {run}, Ant {selected_ch}, Attn = {this_attn} dB', fontsize = 20)
        ax.set_xlabel(f'Noise Events Wf RMS [mV]', fontsize = 20)
        ax.set_ylabel(f'Counts', fontsize = 20)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.legend(fontsize=14, frameon=False)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/single_channel_debug_wnoise_wfrms_run{run}_{selected_ch}.png', dpi=300)
        plt.close(fig)

        ### now let's put it all together to calculate snr with uncertainties!!!
        avg_snr = best_fit_signal_mu / (2 * best_fit_noise_mu)
        avg_snr_err = avg_snr * np.sqrt((best_fit_signal_sigma/best_fit_signal_mu)**2 + (best_fit_noise_sigma/best_fit_noise_mu)**2) 
        
        ### save
        all_attn.append(this_attn)
        all_signal_sigma.append(best_fit_signal_sigma)
        all_avg_snr.append(avg_snr)
        all_avg_snr_err.append(avg_snr_err)
        all_signal_mean.append(np.mean(all_bts_events_wfp2p))
        all_signal_stddev.append(np.std(all_bts_events_wfp2p))

        #print(f'at this_attn = {this_attn}, wf p2p = {best_fit_signal_mu}, noise rms = {best_fit_noise_mu}, so snr = {avg_snr} and err = {avg_snr_err}')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    ax.scatter(all_attn, all_signal_sigma, color = '#009E73', s = 20, label = 'BTS Selected Events Best Fit Sigma')
    ax.scatter(all_attn, all_signal_stddev, color = '#D55E00', s = 20, label = 'BTS Selected Events StdDev')
    ax.set_title(f'Ant {selected_ch}', fontsize = 20)
    ax.set_xlabel(f'Attenuation [dB]', fontsize = 20)
    ax.set_ylabel(f'Signal Events Wf Pk2Pk Distribution [mV]', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.legend(fontsize=14, frameon=False)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/single_channel_debug_wnoise_rms_{selected_ch}.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    ax.scatter(all_attn, all_avg_snr, color = '#009E73', s = 20, label = 'BTS Selected Events')
    ax.errorbar(all_attn, all_avg_snr, yerr=all_avg_snr_err, fmt='o', color='#009E73', markersize=4, capsize=3,)
    
    ### let's also fit!!!
    all_attn = np.asarray(all_attn)
    all_avg_snr = np.asarray(all_avg_snr)
    all_avg_snr_err = np.asarray(all_avg_snr_err)

    fit_start = 4
    fit_end = 8
    fit_mask = np.logical_and(all_attn >= fit_start, all_attn <= fit_end)
    data = [all_attn[fit_mask], all_avg_snr[fit_mask]]
    
    res = minimize(attn_fit_func, x0=[all_avg_snr[0], 20], args=data, tol = 1e-12, method='L-BFGS-B', bounds=[ (all_avg_snr[0]*0.5, all_avg_snr[0]*1.5), (10,30)], options = {"maxiter":1e7})
    best_fit_snr0 = res.x[0]
    best_fit_a = res.x[1]
    
    cov = np.asarray(res.hess_inv.todense())
    snr0_err, a_err = np.sqrt(np.diag(cov))
    print(f"snr0 = {best_fit_snr0:.4f} +/- {snr0_err:.4f}")
    print(f"a = {best_fit_a:.4f} +/- {a_err:.4f}")

    interp_attn = np.linspace(np.amin(all_attn), np.amax(all_attn), 100)
    prediction = get_attn_pred(interp_attn, best_fit_snr0, best_fit_a)
    ax.plot(interp_attn, prediction, color = '#009E73', linewidth = 1.5, linestyle = '-', label=rf'SNR = ({best_fit_snr0:.2f} $\pm$ {snr0_err:.2f})'rf'$\times 10^{{-A/({best_fit_a:.2f} \pm {a_err:.2f})}}$'rf' (Fitting [{fit_start}, {fit_end}] dB)')
   
    ### add a band around the best fit
    df_dsnr0 = prediction / best_fit_snr0
    df_da = prediction * np.log(10) * interp_attn / best_fit_a**2
    prediction_var = (df_dsnr0**2 * cov[0, 0] + df_da**2 * cov[1, 1] + 2 * df_dsnr0 * df_da * cov[0, 1])
    prediction_err = np.sqrt(np.maximum(prediction_var, 0))
    ax.fill_between(interp_attn, prediction - prediction_err, prediction + prediction_err, color='#009E73', alpha=0.2,)

    ax.axvspan(fit_start,fit_end, color = '#009E73', alpha = 0.15)
    ax.set_title(f'Ant {selected_ch}', fontsize = 20)
    ax.set_xlabel(f'Attenuation [dB]', fontsize = 20)
    ax.set_ylabel(f'SNR', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.legend(fontsize=14, frameon=False)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/single_channel_debug_wnoise_snr_{selected_ch}.png', dpi=300)
    plt.close(fig)

    ### now make a version of this fit that is snr (with errors) vs trigger eff
    scaled_snr = get_attn_pred(all_attn, best_fit_snr0, best_fit_a)
    df_dsnr0 = scaled_snr / best_fit_snr0
    df_da = scaled_snr * np.log(10) * all_attn / best_fit_a**2
    scaled_snr_var = (df_dsnr0**2 * cov[0, 0] + df_da**2 * cov[1, 1] + 2 * df_dsnr0 * df_da * cov[0, 1])
    scaled_snr_err = np.sqrt(np.maximum(scaled_snr_var, 0))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    ax.scatter(scaled_snr, trigger_eff, color = 'black', s = 20,)
    ax.errorbar(scaled_snr, trigger_eff, xerr=scaled_snr_err, yerr=trigger_eff_err, fmt='o', color='black', markersize=4, capsize=3,)
    
    ax.set_title(f'Ant {selected_ch}', fontsize = 20)
    ax.set_xlabel(f'Single Antenna SNR', fontsize = 20)
    ax.set_ylabel(f'Trigger Efficiency [%]', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.grid(True, which='both', axis='both', linewidth=0.5, color='gray', alpha = 0.5)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/single_channel_debug_wnoise_triggereffvssnr_{selected_ch}.png', dpi=300)
    fig.tight_layout()
    ax.set_xlim(0,6)
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/single_channel_debug_wnoise_triggereffvssnr_{selected_ch}_zoom.png', dpi=300)
    plt.close(fig)

    print(f'scaled_snr = {np.asarray(scaled_snr).tolist()}')
    print(f'trigger_eff = {np.asarray(trigger_eff).tolist()}')
    print(f'scaled_snr_err = {np.asarray(scaled_snr_err).tolist()}')
    print(f'trigger_eff_err = {np.asarray(trigger_eff_err).tolist()}')
    print(f'all_signal_mean = {np.asarray(all_signal_mean).tolist()}')
    print(f'all_signal_stddev = {np.asarray(all_signal_stddev).tolist()}')

if __name__ == "__main__":
    trigger_eff, trigger_eff_err = get_trigger_eff(run_start = 1502, run_end = 1524)
    fit_distros(run_start = 1502, run_end = 1524, selected_ch = '0201V', trigger_eff = trigger_eff, trigger_eff_err = trigger_eff_err)
    
    #compare_avg_to_distro(run_start = 1502, run_end = 1524, selected_ch = '0201V', fitting_ranges = [(5,8), ], fitting_linestyles = ['-',])
    #compare_avg_to_distro(run_start = 1776, run_end = 1789, selected_ch = '0201V', fitting_ranges = [(0,2),])
    #fit_lucas_data()
