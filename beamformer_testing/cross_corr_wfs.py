import numpy as np
import ROOT
import os
import json
import glob
import argparse
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt

import pueoTALON.pueoPythonAnalyzer.waveform as waveform
import pueoTALON.pueoPythonAnalyzer.plotter as pltr
import pueoTALON.pueoPythonAnalyzer.antenna as antenna
import pueoTALON.pueoPythonAnalyzer.correlator as correlator
import antennaUtils.response as antResponse
import pueoTALON.pueoPythonAnalyzer.utils as utils
import pueoTALON.pueoPythonAnalyzer.geography as geography

ROOT.gSystem.Load("libpueoEvent.so")

def cross_corr(ref_wf, this_wf, window_size = 10, pad_samples = 50, upsample_factor = 4, run = 1420, entry = 1, ref_ch = '0303V', this_ch = '0303V'):
    ### first we need to zero pad -- then cross corr -- then upsample
    ### return the max + lag of the cross correlation

    ### let's window our pulse first
    ### calculate hilbert envelope -- select around the peak
    ref_wf_he = utils.hilbert_envelope(ref_wf)
    this_wf_he = utils.hilbert_envelope(this_wf)
    sample_dt = 1/3.0
    wf_times = np.arange(len(ref_wf_he))*sample_dt

    ref_wf_max_he = wf_times[np.argmax(ref_wf_he)]
    this_wf_max_he = wf_times[np.argmax(this_wf_he)]

    #### ploooot for check
    #fig, ax = plt.subplots(figsize=(10, 6))
    #ax.minorticks_on()
    #ax.plot(wf_times, ref_wf_he, color = '#21b2aa', label = f'{ref_ch}')
    #ax.plot(wf_times, this_wf_he, color = '#c71585', label = f'{this_ch}')
    #ax.axvline(ref_wf_max_he, color = '#21b2aa', linestyle = '--')
    #ax.axvline(this_wf_max_he, color = '#c71585', linestyle = '--')
    #ax.axvspan(ref_wf_max_he - window_size, ref_wf_max_he + window_size, color = '#21b2aa', alpha = 0.3)
    #ax.axvspan(this_wf_max_he - window_size, this_wf_max_he + window_size, color = '#c71585', alpha = 0.3)
    #ax.set_title(f'HE Windowing Test -- Antenna {ref_ch} X Antenna {this_ch}', fontsize = 20)
    #ax.set_xlabel(f'Time [ns]', fontsize = 20)
    #ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
    #ax.tick_params(axis='both', which='both', labelsize=20)
    #ax.legend(fontsize=16, frameon=False)
    #fig.tight_layout()
    #fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/crosscorrelationtest_he_r{run}_e{entry}_{ref_ch}_{this_ch}.png', dpi=300)
    #plt.close(fig)

    ### let's window our wfs
    window_size_samples = int(window_size / sample_dt)
    ref_wf_window_center = np.argmax(ref_wf_he)
    this_wf_window_center = np.argmax(this_wf_he)

    ref_start = max(0, ref_wf_window_center - window_size_samples)
    ref_end = min(len(ref_wf), ref_wf_window_center + window_size_samples)

    this_start = max(0, this_wf_window_center - window_size_samples)
    this_end = min(len(this_wf), this_wf_window_center + window_size_samples)

    ref_wf = ref_wf[ref_start:ref_end]
    this_wf = this_wf[this_start:this_end]

    ### zero padding
    padded_ref_wf = np.pad(ref_wf, (pad_samples, pad_samples))
    padded_this_wf = np.pad(this_wf, (pad_samples, pad_samples))

    ### upsample
    up_ref_wf = scipy.signal.resample(padded_ref_wf, len(padded_ref_wf)*upsample_factor)
    up_this_wf = scipy.signal.resample(padded_this_wf, len(padded_this_wf)*upsample_factor)

    ### correlating
    corr = scipy.signal.correlate(up_ref_wf, up_this_wf, mode='full', method='fft') #calculate cross-correlation using FFT method for speed
    lags = scipy.signal.correlation_lags(len(up_ref_wf), len(up_this_wf), mode='full')
    lag_dt = sample_dt / upsample_factor

    #### ploooot for check
    #fig, ax = plt.subplots(figsize=(10, 6))
    #ax.minorticks_on()
    #ax.plot(lags_upsample, corr_upsample, color = 'black')
    #ax.axvline(max_lag_ns, color = 'green', linestyle = '--')
    #ax.set_title(f'Cross Correlation Test -- Antenna {ref_ch} X Antenna {this_ch}', fontsize = 20)
    #ax.set_xlabel(f'Time [ns]', fontsize = 20)
    #ax.set_ylabel(f'Voltage Time [mV ns]', fontsize = 20)
    #ax.tick_params(axis='both', which='both', labelsize=20)
    #fig.tight_layout()
    #fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/crosscorrelationtest_r{run}_e{entry}_{ref_ch}_{this_ch}.png', dpi=300)
    #plt.close(fig)

    return lags[np.argmax(corr)] * lag_dt

def save_data(array):
    ### we need to run over the desired run for the desired chunk
    runs = [719, 1420] ## just 2 runs for now
    n_chunks = 10

    run_idx = array // n_chunks
    chunk = array % n_chunks

    run = runs[run_idx]

    if run < 800:
        ### ldb data
        os.environ["PUEO_ROOT_DATA"] = ("/project/cozzyd/LDB/rootified/")
    else:
        ### uchicago data
        os.environ["PUEO_ROOT_DATA"] = ("/project/cozzyd/postflight/rootified/")

    ### now let's chunk up entries for our desired run
    this_run_entry_path = f'/scratch/midway3/darcynewmark/beamformer_testing/btsselectedevents_run{run}.json'
    with open(this_run_entry_path) as f:
        all_entries = np.asarray(json.load(f))

    entry_chunks = np.array_split(all_entries, n_chunks)
    entries = entry_chunks[chunk]

    print(f'for array = {array}, run = {run}, chunk = {chunk}, running over entries:')
    print(entries)

    ### now we can loop over our events and calculate what we need!
    ### used phi sectors 1-4 V-pol for the beamformer tests
    channels = waveform.Channels(run, 0)
    only_12ch = np.asarray(['0201V', '0301V', '0401V',
                            '0202V', '0302V', '0402V',
                            '0203V', '0303V', '0403V',
                            '0204V', '0304V', '0404V'])
    reference_channel = '0303V'
    mi_ringphipol = np.asarray(channels.MI_ringphipol)
    reference_channel_mask = mi_ringphipol == reference_channel

    max_lag_dict = {}
    for c in only_12ch:
        max_lag_dict[c] = []

    for ev in entries:
        channels.load_run_event(int(run), int(ev)) ### load our event
        channels.load_waveforms(filter = False) ### load waveforms
        unfilt_wfs = channels.waveforms_unfilt
        ref_wf = unfilt_wfs[reference_channel_mask].flatten() ### flatten to 1D
        for c in only_12ch:
            ch_mask = mi_ringphipol == c
            this_wf = unfilt_wfs[ch_mask].flatten() ### flatten to 1D

            ### now cross correlate!
            max_lag = cross_corr(ref_wf, this_wf)

            ### save!
            max_lag_dict[c].append(max_lag)

    ### now save!
    saving_path = f'/scratch/midway3/darcynewmark/beamformer_testing/ccmaxlags_run{run}_chunk{chunk}.json'
    with open(saving_path, 'w') as f:
        json.dump(max_lag_dict, f, indent=4)

def read_data(run):
    data = {}
    path = f'/scratch/midway3/darcynewmark/beamformer_testing/ccmaxlags_run{run}_chunk*.json'
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

def plot_data():
    ### read in our data and plot distributions!
    ldb_data = read_data(run = 719)
    uc_data = read_data(run = 1420)
    ref_ch = '0303V'
    nbins = 25
    bin_edges = np.linspace(-2, 2, nbins+1)

    for key in ldb_data:
        this_ch = key
        this_ch_ldb_data = ldb_data[key]
        this_ch_uc_data = uc_data[key]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.minorticks_on()
        ax.hist(this_ch_ldb_data, bins = bin_edges, density = True, linewidth = 1.5, histtype='step', linestyle = '-', color = '#21b2aa', label = f'LDB Data')
        ax.hist(this_ch_uc_data, bins = bin_edges, density = True, linewidth = 1.5, histtype='step', linestyle = '-', color = '#c71585', label = 'UC Data')
        ax.set_title(f'{ref_ch} x {this_ch}')
        ax.set_xlabel(f'Cross Correlation Max Lag [ns]', fontsize = 20)
        ax.set_ylabel(f'Normalized Number of Events', fontsize = 20)
        ax.legend(fontsize=20, frameon=False)
        ax.tick_params(axis='both', which='both', labelsize=20)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/ccdistribution_{ref_ch}_{this_ch}.png', dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--array", type = int, default = 0)
    args = parser.parse_args()

    #save_data(args.array)
    plot_data()




