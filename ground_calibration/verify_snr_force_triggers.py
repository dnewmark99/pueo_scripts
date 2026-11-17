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

def plot_force_triggers(n_events = 10):
    ### ok we need to grab run + event number for 10 force triggers
    ### then look at rms calculation in the snr

    ### let's just use one run for now
    N_peaks = 1
    run = 784
    corr = correlator.Correlator(run = run)
    geography_helper = geography.GeographyHelper(root_file=os.environ['PUEO_ANALYSIS_TOOLS'] + "/pueoTALON/lookup/abxtwo.root", root_source="ABX-Two")

    ### grab entries that are from force triggers
    entries = []
    for n in range(0, n_events):
        force_entry = corr.ChannelGeo.dataset.nextMinBiasEvent()
        entries.append(force_entry)

    ### now we can go through our force triggers and plot snr-related quantities
    for entry in entries:
        corr.ChannelGeo.load_run_event(int(run),  int(entry))
        event_time = corr.ChannelGeo.triggerTime()
        geography_helper.updatePosition(event_time, max_time_delta = 3)

        #make skymap and find peak location
        poli = 'V'
        skymap, peak_el, peak_az, _, zskymap, peak_zel, peak_zaz, _, all_zel, all_zaz  = corr.reconstruct(poli, zoom_angle_range=7, N_peaks=N_peaks)

        ### get the coherent summed wf around the max peak
        cohsum, overlap_start, overlap_end = corr.coherent_sum(peak_el[0], peak_az[0],polarization=poli, return_overlap_boundaries=True)
        cohsum_times = np.arange(cohsum.size)*corr.dt
        overlap_start_time = overlap_start * corr.dt
        overlap_end_time = overlap_end * corr.dt

        ### get the snr using second half of wf
        fixed_snr, fixed_window_start, fixed_window_end = utils.SNR(cohsum, dynamic_rms_window = False, return_window = True, valid_window = (overlap_start, overlap_end))

        ### now get the snr using a dynamic window
        dynamic_snr, dynamic_window_start, dynamic_window_end = utils.SNR(cohsum, dynamic_rms_window = True, return_window = True, valid_window = (overlap_start, overlap_end))

        ### finally, plot!
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.minorticks_on()
        ax.plot(cohsum_times, cohsum, color = 'black', label = 'Coherent Summed Wf')
        ax.axvspan(fixed_window_start*corr.dt, fixed_window_end*corr.dt,  color = '#21b2aa', alpha = 0.3, label = f'Fixed Window SNR = {fixed_snr:0.2f}')
        ax.axvspan(dynamic_window_start*corr.dt, dynamic_window_end*corr.dt,  color = '#c71585', alpha = 0.3, label = f'Dynamic Window SNR = {dynamic_snr:0.2f}')
        ax.axvline(overlap_start_time, color = 'green', linestyle = '--')
        ax.axvline(overlap_end_time, color = 'green', linestyle = '--')
        ax.set_title(f'Force Trigger Coherent Summed Wf, Run = {run}, Entry = {entry}')
        ax.set_xlabel(f'Time [ns]', fontsize = 20)
        ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
        ax.set_xlim(0, cohsum.size*corr.dt)
        ax.tick_params(axis='both', which='both', labelsize=16)
        ax.legend(fontsize=16, frameon=False)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/ground_cal/forcetrigger_snr_test_run={run}_entry={entry}.png', dpi = 300)
        plt.close(fig)

def compare_rms(n_events = int(1000)):
    N_peaks = 1
    run = 784
    corr = correlator.Correlator()
    geography_helper = geography.GeographyHelper(root_file=os.environ['PUEO_ANALYSIS_TOOLS'] + "/pueoTALON/lookup/abxtwo.root", root_source="ABX-Two")

    ### grab entries that are from force triggers
    runs = []
    entries = []
    for n in range(0, n_events):
        force_entry = corr.ChannelGeo.dataset.nextMinBiasEvent()
        force_run = corr.ChannelGeo.run

        entries.append(force_entry)
        runs.append(force_run)

    ### now we can go through our force triggers and plot snr-related quantities
    rms_second_half = []
    rms_all = []
    rms_onethird_dynamic = []

    for e in range(len(entries)):
        corr.ChannelGeo.load_run_event(int(runs[e]),  int(entries[e]))
        event_time = corr.ChannelGeo.triggerTime()
        geography_helper.updatePosition(event_time, max_time_delta = 3)

        #make skymap and find peak location
        poli = 'V'
        skymap, peak_el, peak_az, _, zskymap, peak_zel, peak_zaz, _, all_zel, all_zaz  = corr.reconstruct(poli, zoom_angle_range=7, N_peaks=N_peaks)

        ### get the coherent summed wf around the max peak
        cohsum, overlap_start, overlap_end = corr.coherent_sum(peak_el[0], peak_az[0],polarization=poli, return_overlap_boundaries=True)
        cohsum_times = np.arange(cohsum.size)*corr.dt
        overlap_start_time = overlap_start * corr.dt
        overlap_end_time = overlap_end * corr.dt

        ### get the snr using second half of wf
        fixed_snr, fixed_rms = utils.SNR(cohsum, dynamic_rms_window = False, valid_window = (overlap_start, overlap_end), return_rms = True)

        ### now get the snr using a dynamic window
        dynamic_snr, dynamic_window_rms = utils.SNR(cohsum, dynamic_rms_window = True, dynamic_range_fraction = 0.3, valid_window = (overlap_start, overlap_end), return_rms = True)

        ### finally get the snr using the entire wf!
        wholewindow_snr, wholewindow_window_rms = utils.SNR(cohsum, dynamic_rms_window = True, dynamic_range_fraction = 1, valid_window = (overlap_start, overlap_end), return_rms = True)

        ### save!
        rms_second_half.append(fixed_rms)
        rms_all.append(wholewindow_window_rms)
        rms_onethird_dynamic.append(dynamic_window_rms)

    ### save data
    saving_path = os.environ['PUEO_ANALYSIS_TOOLS'] + f'/pueoTALON/lookup/scratch/forcetrigger_rms_comparison.json'
    data = {
            'rms_second_half': rms_second_half,
            'rms_all': rms_all,
            'rms_onethird_dynamic': rms_onethird_dynamic,
    }
    with open(saving_path, 'w') as f:
        json.dump(data, f, indent=4)

    ### plot!!!
    nbins = 20
    bin_edges = np.linspace(0, 2, nbins+1)
    rms_second_half = np.asarray(rms_second_half)
    rms_all = np.asarray(rms_all)
    rms_onethird_dynamic = np.asarray(rms_onethird_dynamic)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    ax.hist(rms_all, bins = bin_edges, linewidth = 1.5, histtype='step', linestyle = '-', color = 'black', label = f'Entire Wf')
    ax.hist(rms_second_half, bins = bin_edges, linewidth = 1.5, histtype='step', linestyle = '-', color = '#21b2aa', label = 'Second 1/2 of Wf')
    ax.hist(rms_onethird_dynamic, bins = bin_edges, linewidth = 1.5, histtype='step', linestyle = '-', color = '#c71585', label = '1/3 Dynamic Window of Wf')
    ax.set_xlabel(f'RMS', fontsize = 20)
    ax.set_ylabel(f'Number of Events', fontsize = 20)
    ax.legend(fontsize=20, frameon=False)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/ground_cal/forcetrigger_rms_comparison.png', dpi=300)
    plt.close(fig)

def plot_after_the_fact():
    ### grab our json
    path = os.environ['PUEO_ANALYSIS_TOOLS'] + f'/pueoTALON/lookup/scratch/forcetrigger_rms_comparison.json'
    with open(path) as file:
        data = json.load(file)

    ### plot!!!
    nbins = 40
    bin_edges = np.linspace(0.23, 0.4, nbins+1)
    rms_second_half = np.asarray(data["rms_second_half"])
    rms_all = np.asarray(data["rms_all"])
    rms_onethird_dynamic = np.asarray(data["rms_onethird_dynamic"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    ax.hist(rms_all, bins = bin_edges, linewidth = 1.5, histtype='step', linestyle = '-', color = 'black', label = f'Entire Wf, Mode = {mode(rms_all):0.3f}')
    ax.hist(rms_second_half, bins = bin_edges, linewidth = 1.5, histtype='step', linestyle = '-', color = '#21b2aa', label = f'Second 1/2 of Wf, Mode = {mode(rms_second_half):0.3f}')
    ax.hist(rms_onethird_dynamic, bins = bin_edges, linewidth = 1.5, histtype='step', linestyle = '-', color = '#c71585', label = f'1/3 Dynamic Window of Wf, Mode = {mode(rms_onethird_dynamic):0.3f}')
    ax.set_xlabel(f'RMS', fontsize = 20)
    ax.set_ylabel(f'Number of Events', fontsize = 20)
    ax.legend(fontsize=12, frameon=False)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/ground_cal/forcetrigger_rms_comparison.png', dpi=300)
    plt.close(fig)

    ### also plot the ratio!
    nbins = 20
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    ax.hist((rms_second_half - rms_all) / rms_all, bins = nbins, linewidth = 1.5, histtype='step', linestyle = '-', color = '#21b2aa', label = f'(Second 1/2 - Entire Wf) / Entire Wf')
    ax.hist((rms_onethird_dynamic - rms_all) / rms_all, bins = nbins, linewidth = 1.5, histtype='step', linestyle = '-', color = '#c71585', label = f'(1/3 Dynamic - Entire Wf) / Entire Wf')
    ax.set_xlabel(f'RMS Comparison', fontsize = 20)
    ax.set_ylabel(f'Number of Events', fontsize = 20)
    ax.legend(fontsize=16, frameon=False)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/ground_cal/forcetrigger_rms_comparison_ratio.png', dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    plot_force_triggers(n_events = 1)
    #compare_rms()
    #plot_after_the_fact()
