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
os.environ["PUEO_ROOT_DATA"] = ("/project/cozzyd/postflight/rootified/")

only_12ch = np.asarray(['0201V', '0301V', '0401V',
                        '0202V', '0302V', '0402V',
                        '0203V', '0303V', '0403V',
                        '0204V', '0304V', '0404V'])
channel_special_attn = {'0201V': '31 dB', '0301V': '32 dB', '0401V': '39 dB',
                        '0202V': '36 dB', '0302V': '33 dB', '0402V': '30 dB',
                        '0203V': '30 dB', '0303V': '30 dB', '0403V': '30 dB',
                        '0204V': '30 dB', '0304V': '30 dB', '0404V': '30 dB',
                        }

#reference_event_idx = 2
reference_event_idx = 0

def get_attn_pred(x, snr0, a):
    return snr0 * 10**(-x/a)

def attn_fit_func(params, data):
    x = data[0]
    y = data[1]
    snr0, a = params
    pred = get_attn_pred(x, snr0, a)
    return np.sum((y - pred)**2)

def grab_reference_wfs(run):
    
    sample_dt = 1/3.0
    upsample_factor = 10
    reference_wfs = {}

    for c in range(len(only_12ch)):
        this_run_entry_path = f'/scratch/midway3/darcynewmark/beamformer_testing/btsselectedevents_run{run}.json'
        with open(this_run_entry_path) as f:
            all_entries = np.asarray(json.load(f))

        channels = waveform.Channels(run, 0)
        reference_entry = all_entries[reference_event_idx]
        channels.load_event(int(reference_entry)) ### load our event
        channels.load_waveforms(filter = False) ### load waveforms
        unfilt_wfs = channels.waveforms_unfilt
        
        wf_mask = np.isin(channels.MI_ringphipol, np.asarray([only_12ch[c]])) ### mask out our channel of interest 
        this_wf = unfilt_wfs[wf_mask].flatten() ### flatten to 1D
        up_this_wf = scipy.signal.resample(this_wf, len(this_wf)*upsample_factor, axis=-1) ### upsample this wf for accumulation
        this_times = np.arange(len(up_this_wf)) * sample_dt / upsample_factor
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.minorticks_on()
        ax.plot(this_times, up_this_wf, color = 'black',)
        ax.set_title(f'Reference Wf, Run {run}, Ant {only_12ch[c]}', fontsize = 20)
        ax.set_xlabel(f'Time [ns]', fontsize = 20)
        ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.legend(fontsize=16, frameon=False)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/referenceaccumwf_run{run}_{only_12ch[c]}.png', dpi=300)
        plt.close(fig)

        ### save our reference wf
        reference_wfs[only_12ch[c]] = this_wf.copy() 
    
    return reference_wfs

def save_wfs(array, reference_wfs):
    ### we need to run over the desired run for the desired chunk
    #run_ranges = (1721, 1732)
    #run_ranges = (1758, 1774)
    #run_ranges = (1502, 1524)
    run_ranges = (1776, 1789)
    all_runs = list(range(run_ranges[0], run_ranges[1] + 1))
    all_runs = [r for r in all_runs if r not in BAD_RUNS]
    run = all_runs[array]

    ### now get all bts selected runs
    this_run_entry_path = f'/scratch/midway3/darcynewmark/beamformer_testing/btsselectedevents_run{run}.json'
    with open(this_run_entry_path) as f:
        entries = np.asarray(json.load(f))

    print(f'for run = {run}, accumulating {len(entries)} events!')

    ### now we can loop over our events and calculate what we need!
    sample_dt = 1/3.0
    upsample_factor = 10
    pad_samples = 1500
    cross_corr_start_time = 40
    cross_corr_end_time = 90
    channels = waveform.Channels(run, 0)
    n_samples_before_max_lag_up = 1000
    n_samples_after_max_lag_up = 3000
    accumulated_wfs = {}
    n_accumulated_wfs = {}

    for c in only_12ch:
        accumulated_wfs[c] = np.zeros(n_samples_before_max_lag_up+n_samples_after_max_lag_up) 
        n_accumulated_wfs[c] = np.zeros(n_samples_before_max_lag_up+n_samples_after_max_lag_up, dtype=int)
    
    ### let's get our reference wf information
    reference_up_wf = {}
    reference_peak_idx = {}
    for c in only_12ch:
        reference_wf = reference_wfs[c]    
        
        ### window around our reference wf
        this_times = np.arange(len(reference_wf)) * sample_dt
        window_mask = ((this_times > cross_corr_start_time) & (this_times < cross_corr_end_time))
        window_start_idx = np.where(window_mask)[0][0]
        reference_window = reference_wf[window_mask]
       
        padded_wf_1 = np.pad(reference_window, (pad_samples, pad_samples))
        up_wf_1 = scipy.signal.resample(padded_wf_1, len(padded_wf_1)*upsample_factor, axis=-1) ### upsample this wf for accumulation

        up_reference_window = scipy.signal.resample(reference_window, len(reference_window)*upsample_factor, axis=-1)
        reference_peak_idx_up = (window_start_idx * upsample_factor + np.argmax(np.abs(up_reference_window)))

        ### save!
        reference_up_wf[c] = up_wf_1.copy() 
        reference_peak_idx[c] = reference_peak_idx_up

    ### now loop over our events to accumulate!
    for idx_counter, ev in enumerate(entries):
        channels.load_event(int(ev)) ### load our event
        channels.load_waveforms(filter = False) ### load waveforms
        unfilt_wfs = channels.waveforms_unfilt
        
        for c in only_12ch:
            wf_mask = np.isin(channels.MI_ringphipol, np.asarray([c])) ### mask out our special channel
            this_wf = unfilt_wfs[wf_mask].flatten() ### flatten to 1D
            up_this_wf = scipy.signal.resample(this_wf, len(this_wf)*upsample_factor, axis=-1) ### upsample this wf for accumulation
            this_times = np.arange(len(this_wf)) * sample_dt

            ### window our wfs before cross correlating
            window_mask = ((this_times > cross_corr_start_time) & (this_times < cross_corr_end_time))
            window_start_idx = np.where(window_mask)[0][0]

            this_window = this_wf[window_mask]
           
            ### pad and upsample this wf for correlation
            padded_wf_2 = np.pad(this_window, (pad_samples, pad_samples))
            up_wf_2 = scipy.signal.resample(padded_wf_2, len(padded_wf_2)*upsample_factor, axis=-1)

            ### get our reference info
            up_wf_1 = reference_up_wf[c]
            reference_peak_idx_up = reference_peak_idx[c]

            ### correlate!!!
            corr = scipy.signal.correlate(up_wf_1, up_wf_2, mode='full', method='fft')
            lags = scipy.signal.correlation_lags(len(up_wf_1), len(up_wf_2), mode='full')
            max_lag_upsampled = lags[np.argmax(corr)]

            ### need to convert max lag in the windowed wfs to our entire wfs 
            this_wf_max_lag_idx_up = int(round(reference_peak_idx_up - max_lag_upsampled))
            
            left_bound = max(0,this_wf_max_lag_idx_up - n_samples_before_max_lag_up)
            right_bound = min(len(up_this_wf),this_wf_max_lag_idx_up + n_samples_after_max_lag_up)

            padded_left = (left_bound - (this_wf_max_lag_idx_up - n_samples_before_max_lag_up))
            padded_right = padded_left + (right_bound - left_bound)

            accumulated_event = np.zeros(n_samples_before_max_lag_up+n_samples_after_max_lag_up)
            accumulated_event[padded_left:padded_right] = (up_this_wf[left_bound:right_bound])

            ### check the alignment
            actual_output_peak = np.argmax(np.abs(accumulated_event))
            delta_peak = np.abs(actual_output_peak - n_samples_before_max_lag_up)
            if delta_peak > 100:
                print(f'excluding run = {run}, event = {ev} because the expected peak idx = {n_samples_before_max_lag_up} and the actual peak idx = {actual_output_peak}')
                continue

            ### saaavvveeee!
            accumulated_wfs[c] += accumulated_event 
            n_accumulated_wfs[c][padded_left:padded_right] += 1

    ### now save!
    avg_accumulated_wf = {}
    for c in only_12ch:
        avg_accumulated_wf[c] = (np.divide(accumulated_wfs[c], n_accumulated_wfs[c], out=np.zeros_like(accumulated_wfs[c]), where=n_accumulated_wfs[c] > 0)).tolist()

    saving_path = f'/scratch/midway3/darcynewmark/beamformer_testing/accumulatedbtswfs_run{run}.json'
    with open(saving_path, 'w') as f:
        json.dump(avg_accumulated_wf, f, indent=4)

def plot_wfs(run_start, run_end, var_attn = True):
    ### we have already accumulated our wfs
    all_attns = []
    all_avg_wf_pk2pk = {}
    for c in only_12ch:
        all_avg_wf_pk2pk[c] = []
    
    for run in range(run_start, run_end+1):
        
        ### read in our data
        path = f'/scratch/midway3/darcynewmark/beamformer_testing/accumulatedbtswfs_run{run}.json'
        
        if not os.path.exists(path):
            continue
        
        with open(path) as f:
            avg_accumulated_wf = json.load(f)

        for c in only_12ch:
            this_wf = avg_accumulated_wf[c]
            times = np.arange(len(this_wf))*(1/3.0)/10

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.minorticks_on()
            ax.plot(times, this_wf, color = 'black',)
            ax.set_title(f'Avg Wf, Run {run}, Ant {c}', fontsize = 20)
            ax.set_xlabel(f'Time [ns]', fontsize = 20)
            ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
            ax.tick_params(axis='both', which='both', labelsize=20)
            fig.tight_layout()
            fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/accumwf_run{run}_{c}.png', dpi=300)
            plt.close(fig)
        
            ### also save some stuff for plotting
            this_avg_wf_pk2pk = np.amax(this_wf) - np.amin(this_wf)
            all_avg_wf_pk2pk[c].append(this_avg_wf_pk2pk)
        
        ### also save some stuff for plotting
        this_attn = run_attn_dict[run]
        all_attns.append(this_attn)
    
    all_attns = np.asarray(all_attns)
    
    ### plot attn vs signal strength!
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    fit_start = 0
    fit_end = 5
    n_channels = len(only_12ch)
    cmap = plt.get_cmap('Spectral_r')
    norm = mpl.colors.Normalize(vmin=0, vmax=n_channels - 1)
    counter = 0
    for c in only_12ch:
        this_y = np.asarray(all_avg_wf_pk2pk[c])
        color = cmap(norm(counter))
        
        ### quick fit to the data
        fit_mask = np.logical_and(all_attns >= fit_start, all_attns <= fit_end)
        data = [all_attns[fit_mask], this_y[fit_mask]]
        res = minimize(attn_fit_func, x0=[this_y[0], 20], args=data, tol = 1e-20, method='L-BFGS-B', bounds=[ (this_y[0]-2,this_y[0]+2), (10,30)], options = {"maxiter":1e7})
        best_fit_snr0 = res.x[0]
        best_fit_a = res.x[1]
        print(f'best fit snr0 = {best_fit_snr0}, best fit a = {best_fit_a}')
        
        interp_attn = np.linspace(np.amin(all_attns), np.amax(all_attns), 100)
        prediction = get_attn_pred(interp_attn, best_fit_snr0, best_fit_a)
        if var_attn:
            label = rf'{c} WfPk2Pk ({channel_special_attn[c]}) = {best_fit_snr0:.2f}$\times10^{{-A/{best_fit_a:.2f}}}$'
        else:
            label = rf'{c} WfPk2Pk = {best_fit_snr0:.2f}$\times10^{{-A/{best_fit_a:.2f}}}$'
        ax.scatter(all_attns, this_y, color = color, s = 30, label = label)
        ax.plot(interp_attn, prediction, color = color, linewidth = 1.5, alpha = 0.5)

        counter += 1

    ax.axvspan(fit_start,fit_end, color = 'black', alpha = 0.15)
    ax.text(fit_start,np.amin(prediction), 'Fit Region', fontsize = 20)
    ax.set_title(f'Per-Channel Attenuation Scaling, Run {run_start} to {run_end}', fontsize = 20)
    ax.set_xlabel(f'BTS Attenuation [dB]', fontsize = 20)
    ax.set_ylabel(f'Signal Strength [mV]', fontsize = 20)
    ax.legend(fontsize=10, frameon=False, loc = 'upper right')
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/avgwf_attn_vs_sigstrength_r{run_start}_r{run_end}.png', dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--array", type = int, default = 0)
    args = parser.parse_args()
    
    #reference_wfs = grab_reference_wfs(run = 1721)
    #save_wfs(args.array, reference_wfs)
    #plot_wfs(run_start = 1721, run_end = 1732)
    
    #reference_wfs = grab_reference_wfs(run = 1758)
    #save_wfs(args.array, reference_wfs)
    #plot_wfs(run_start = 1758, run_end = 1774, var_attn=False)
    
    #reference_wfs = grab_reference_wfs(run = 1502)
    #save_wfs(args.array, reference_wfs)
    #plot_wfs(run_start = 1502, run_end = 1524, var_attn=False)

    reference_wfs = grab_reference_wfs(run = 1776)
    save_wfs(args.array, reference_wfs)
    #plot_wfs(run_start = 1776, run_end = 1789, var_attn=False)
