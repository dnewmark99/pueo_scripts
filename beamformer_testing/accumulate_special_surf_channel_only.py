import numpy as np
import ROOT
import os
import json
import glob
import argparse
import scipy.signal
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from postrunmapping import run_attn_dict, BAD_RUNS

import pueoTALON.pueoPythonAnalyzer.waveform as waveform
import pueoTALON.pueoPythonAnalyzer.plotter as pltr
import pueoTALON.pueoPythonAnalyzer.antenna as antenna
import pueoTALON.pueoPythonAnalyzer.correlator as correlator
import antennaUtils.response as antResponse
import pueoTALON.pueoPythonAnalyzer.utils as utils
import pueoTALON.pueoPythonAnalyzer.geography as geography

ROOT.gSystem.Load("libpueoEvent.so")

run_reference_event_idx = {1746: 0, 1747: 0, 1748: 0, 1749: 0, 1750: 0, 1751: 1, 1752: 0, 1753: 0, 1754: 0, 1755: 4, 1756: 0}

def get_attn_pred(x, snr0, a):
    return snr0 * 10**(-x/a)

def attn_fit_func(params, data):
    x = data[0]
    y = data[1]
    snr0, a = params
    pred = get_attn_pred(x, snr0, a)
    return np.sum((y - pred)**2)


def check_reference_wfs(run_start = 1746, run_end = 1756):
    
    os.environ["PUEO_ROOT_DATA"] = ("/project/cozzyd/postflight/rootified/")
    special_surf_channel = '0101H'
    sample_dt = 1/3.0
    upsample_factor = 10

    for run in range(run_start, run_end+1):
        ### plot the first bts selected event in this run to make sure it looks good
        this_run_entry_path = f'/scratch/midway3/darcynewmark/beamformer_testing/btsselectedevents_run{run}.json'
        with open(this_run_entry_path) as f:
            all_entries = np.asarray(json.load(f))

        channels = waveform.Channels(run, 0)
        reference_entry = all_entries[run_reference_event_idx[run]]
        channels.load_event(int(reference_entry)) ### load our event
        channels.load_waveforms(filter = False) ### load waveforms
        unfilt_wfs = channels.waveforms_unfilt
        
        wf_mask = np.isin(channels.MI_ringphipol, np.asarray([special_surf_channel])) ### mask out our special channel
        this_wf = unfilt_wfs[wf_mask].flatten() ### flatten to 1D
        up_this_wf = scipy.signal.resample(this_wf, len(this_wf) * upsample_factor) ### upsample this wf for accumulation
        this_times = np.arange(len(up_this_wf)) * sample_dt / upsample_factor
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.minorticks_on()
        ax.plot(this_times, up_this_wf, color = 'black',)
        ax.set_title(f'Reference Wf, AMPA --> 0101H, Run {run}', fontsize = 20)
        ax.set_xlabel(f'Time [ns]', fontsize = 20)
        ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.legend(fontsize=16, frameon=False)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/referenceaccumwf_run{run}_0101H.png', dpi=300)
        plt.close(fig)

def save_wfs(array):
    ### we need to run over the desired run for the desired chunk
    run_ranges = [(1746, 1756)]
    chunks = [1]
    
    ### figure out which run this array belongs to
    array_remaining = array

    for (run_start, run_end), total_chunks in zip(run_ranges, chunks):

        n_runs = run_end - run_start + 1
        n_arrays = n_runs * total_chunks

        if array_remaining < n_arrays:
            ### this array belongs to this run range
            run_index = array_remaining // total_chunks
            chunk = array_remaining % total_chunks

            run = run_start + run_index
            break

        ### skip this entire run range
        array_remaining -= n_arrays

    else:
        raise ValueError(f"Array index {array} is out of range")

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

    entry_chunks = np.array_split(all_entries, total_chunks)
    entries = entry_chunks[chunk]

    print(f'for array = {array}, run = {run}, chunk = {chunk}, running over {len(entries)} entries out of {len(all_entries)} total entries')

    ### now we can loop over our events and calculate what we need!
    special_surf_channel = '0101H'
    sample_dt = 1/3.0
    upsample_factor = 10
    pad_samples = 1500
    cross_corr_start_time = 40
    cross_corr_end_time = 90
    channels = waveform.Channels(run, 0)
    n_samples_around_max_lag_up = 2000
    accumulated_wfs = np.zeros(2 * n_samples_around_max_lag_up) 
    n_accumulated_wfs = np.zeros(2 * n_samples_around_max_lag_up, dtype=int)
    reference_wf = None
    this_reference_entry_idx = run_reference_event_idx[run]

    for idx_counter, ev in enumerate(entries):
        if idx_counter < this_reference_entry_idx:
            continue
    
        channels.load_event(int(ev)) ### load our event
        channels.load_waveforms(filter = False) ### load waveforms
        unfilt_wfs = channels.waveforms_unfilt
        
        wf_mask = np.isin(channels.MI_ringphipol, np.asarray([special_surf_channel])) ### mask out our special channel
        this_wf = unfilt_wfs[wf_mask].flatten() ### flatten to 1D
        up_this_wf = scipy.signal.resample(this_wf, len(this_wf) * upsample_factor) ### upsample this wf for accumulation
        this_times = np.arange(len(this_wf)) * sample_dt

        ### window our wfs before cross correlating
        window_mask = ((this_times > cross_corr_start_time) & (this_times < cross_corr_end_time))
        window_start_idx = np.where(window_mask)[0][0]

        this_window = this_wf[window_mask]
       
        ### pad and upsample this wf for correlation
        padded_wf_2 = np.pad(this_window, (pad_samples, pad_samples))
        up_wf_2 = scipy.signal.resample(padded_wf_2, len(padded_wf_2) * upsample_factor)

        ### check for our reference wf
        if reference_wf is None:
            print(f'for run = {run}, using event idx {idx_counter} as reference wf!')
            reference_wf = this_wf.copy()
            reference_window = this_window.copy()

            padded_wf_1 = np.pad(reference_window, (pad_samples, pad_samples))
            up_wf_1 = scipy.signal.resample(padded_wf_1, len(padded_wf_1) * upsample_factor)

            up_reference_window = scipy.signal.resample(reference_window,len(reference_window) * upsample_factor)
            reference_peak_idx_up = (window_start_idx * upsample_factor + np.argmax(np.abs(up_reference_window)))

            max_lag_upsampled = 0
        
        else:
            corr = scipy.signal.correlate(up_wf_1, up_wf_2, mode='full', method='fft')
            lags = scipy.signal.correlation_lags(len(up_wf_1), len(up_wf_2), mode='full')
            max_lag_upsampled = lags[np.argmax(corr)]

        ### need to convert max lag in the windowed wfs to our entire wfs 
        this_wf_max_lag_idx_up = int(round(reference_peak_idx_up - max_lag_upsampled))
        
        left_bound = max(0,this_wf_max_lag_idx_up - n_samples_around_max_lag_up)
        right_bound = min(len(up_this_wf),this_wf_max_lag_idx_up + n_samples_around_max_lag_up)

        padded_left = (left_bound - (this_wf_max_lag_idx_up - n_samples_around_max_lag_up))
        padded_right = padded_left + (right_bound - left_bound)

        accumulated_event = np.zeros(2 * n_samples_around_max_lag_up)
        accumulated_event[padded_left:padded_right] = (up_this_wf[left_bound:right_bound])

        ### saaavvveeee!
        accumulated_wfs += accumulated_event 
        n_accumulated_wfs[padded_left:padded_right] += 1

    ### now save!
    avg_accumulated_wf = np.divide(accumulated_wfs, n_accumulated_wfs, out=np.zeros_like(accumulated_wfs), where=n_accumulated_wfs > 0)
    saving_path = f'/scratch/midway3/darcynewmark/beamformer_testing/accumulatedbtswfs_run{run}_chunk{chunk}.json'
    with open(saving_path, 'w') as f:
        json.dump({
                    'avg_accumulated_wf': avg_accumulated_wf.tolist(),
                    'accumulated_wfs': accumulated_wfs.tolist(),
                    'n_accumulated_wfs': n_accumulated_wfs.tolist(),
                    'reference_wf': reference_wf.tolist(),
                    'upsample_factor': upsample_factor,
                }, f, indent=4)

def plot_wfs(run_start = 1746, run_end = 1756):
    ### we have already accumulated our wfs
    all_attns = []
    all_avg_wf_pk2pk = []
    for run in range(run_start, run_end+1):
        ### read in our data
        path = f'/scratch/midway3/darcynewmark/beamformer_testing/accumulatedbtswfs_run{run}_chunk*.json'
        fnames = [name for name in glob.glob(path)]
        
        with open(fnames[0]) as f:
            data = json.load(f)

        avg_accumulated_wf = np.asarray(data['avg_accumulated_wf'])
        times = np.arange(len(avg_accumulated_wf))*(1/3.0)/10

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.minorticks_on()
        ax.plot(times, avg_accumulated_wf, color = 'black',)
        ax.set_title(f'Avg Wf, AMPA --> 0101H, Run {run}', fontsize = 20)
        ax.set_xlabel(f'Time [ns]', fontsize = 20)
        ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.legend(fontsize=16, frameon=False)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/accumwf_run{run}_0101H.png', dpi=300)
        plt.close(fig)

        ### also save some stuff for plotting
        this_attn = run_attn_dict[run]
        this_avg_wf_pk2pk = np.amax(avg_accumulated_wf) - np.amin(avg_accumulated_wf)

        all_attns.append(this_attn)
        all_avg_wf_pk2pk.append(this_avg_wf_pk2pk)

    all_attns = np.asarray(all_attns)
    all_avg_wf_pk2pk = np.asarray(all_avg_wf_pk2pk)
    
    ### plot attn vs signal strength!
    y_data = [all_avg_wf_pk2pk]
    y_labels = ['Avg WfPk2Pk', ]
    colors = ['#009E73',]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    fit_start = 0
    fit_end = 6
    for y in range(len(y_data)):
        this_y = np.asarray(y_data[y])
        
        ### quick fit to the data
        fit_mask = np.logical_and(all_attns >= fit_start, all_attns <= fit_end)
        data = [all_attns[fit_mask], this_y[fit_mask]]
        res = minimize(attn_fit_func, x0=[this_y[0], 20], args=data, tol = 1e-20, method='L-BFGS-B', bounds=[ (this_y[0]-2,this_y[0]+2), (10,30)], options = {"maxiter":1e7})
        best_fit_snr0 = res.x[0]
        best_fit_a = res.x[1]
        print(f'best fit snr0 = {best_fit_snr0}, best fit a = {best_fit_a}')
        
        interp_attn = np.linspace(np.amin(all_attns), np.amax(all_attns), 100)
        prediction = get_attn_pred(interp_attn, best_fit_snr0, best_fit_a)
        ax.scatter(all_attns, this_y, color = colors[y], s = 30, label = rf'{y_labels[y]} = {best_fit_snr0:.2f}$\times10^{{-A/{best_fit_a:.2f}}}$')
        ax.plot(interp_attn, prediction, color = colors[y], linewidth = 1.5, alpha = 0.5)

    ax.axvspan(fit_start,fit_end, color = 'black', alpha = 0.15)
    ax.text(fit_start+0.5,np.amin(prediction), 'Fit Region', fontsize = 20)
    ax.set_title(f'Avg Wf, AMPA --> 0101H', fontsize = 20)
    ax.set_xlabel(f'BTS Attenuation [dB]', fontsize = 20)
    ax.set_ylabel(f'Signal Strength [mV]', fontsize = 20)
    ax.legend(fontsize=18, frameon=False)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/avgwf_attn_vs_sigstrength_ampa_test.png', dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--array", type = int, default = 0)
    args = parser.parse_args()
    
    #check_reference_wfs(run_start = 1746, run_end = 1756)
    #save_wfs(args.array)
    plot_wfs(run_start = 1746, run_end = 1756)

