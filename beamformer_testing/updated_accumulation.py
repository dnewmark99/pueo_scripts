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

def get_attn_pred(x, snr0, a):
    return snr0 * 10**(-x/a)

def attn_fit_func(params, data):
    x = data[0]
    y = data[1]
    snr0, a = params
    pred = get_attn_pred(x, snr0, a)
    return np.sum((y - pred)**2)

def grab_template_wf(run, special_reference_channel):
    ### use previously averaged wfs (created by accumulate_bts_wfs.py) as reference wf
    path = f'/scratch/midway3/darcynewmark/beamformer_testing/accumulatedbtswfs_run{run}.json'
    with open(path) as f:
        avg_accumulated_wf = json.load(f)
    this_wf = np.asarray(avg_accumulated_wf[special_reference_channel])
    return this_wf

def save_wfs(array, template_wf, special_reference_channel):
    ### we need to run over the desired run for the desired chunk
    run_ranges = (1502, 1524)
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
    accumulated_wfs_all_entries = {}
    accumulated_wfs = {}
    n_accumulated_wfs = {}

    for c in only_12ch:
        accumulated_wfs_all_entries[c] = [[] for n in range(n_samples_before_max_lag_up+n_samples_after_max_lag_up)]  
        accumulated_wfs[c] = np.zeros(n_samples_before_max_lag_up+n_samples_after_max_lag_up) 
        n_accumulated_wfs[c] = np.zeros(n_samples_before_max_lag_up+n_samples_after_max_lag_up, dtype=int)
   
    ### grab the peak of the template wf
    template_peak_idx_up = int(np.argmax(template_wf))

    ### the template is already upsampled but not windowed or zero padded -- so let's add that
    ### windowing +-(cross_corr_end_time-cross_corr_start_time)/2 relative to the peak of the template
    window_length_up = int(round((cross_corr_end_time - cross_corr_start_time) / sample_dt * upsample_factor))
    template_window_before_up = window_length_up // 2
    template_window_after_up = window_length_up - template_window_before_up

    template_start_idx_up = template_peak_idx_up - template_window_before_up
    template_end_idx_up = template_peak_idx_up + template_window_after_up
    template_window = template_wf[template_start_idx_up:template_end_idx_up]

    ### now pad template
    pad_samples_up = pad_samples * upsample_factor
    padded_template_wf = np.pad(template_window,(pad_samples_up, pad_samples_up))
    template_peak_idx_padded_up = pad_samples_up + template_window_before_up

    ### now loop over our events to accumulate!
    for idx_counter, ev in enumerate(entries):
        channels.load_event(int(ev)) ### load our event
        channels.load_waveforms(filter = False) ### load waveforms
        unfilt_wfs = channels.waveforms_unfilt
        
        ### let's first grab out accumlation alignment based on our special reference ch
        ref_mask = np.isin(channels.MI_ringphipol, np.asarray([special_reference_channel])) ### mask out our special channel
        ref_wf = unfilt_wfs[ref_mask].flatten() ### flatten to 1D
        up_ref_wf = scipy.signal.resample(ref_wf, len(ref_wf)*upsample_factor, axis=-1) ### upsample this wf for accumulation
        
        ### window our reference wf
        window_start_idx = int(round(cross_corr_start_time / sample_dt))
        window_end_idx = int(round(cross_corr_end_time / sample_dt))
        this_window = ref_wf[window_start_idx:window_end_idx]

        ### pad + upsample our reference wf
        padded_ref_wf = np.pad(this_window, (pad_samples, pad_samples))
        up_ref_wf = scipy.signal.resample(padded_ref_wf, len(padded_ref_wf)*upsample_factor, axis=-1)

        ### now we can cross correlate with our template wf!
        corr = scipy.signal.correlate(padded_template_wf, up_ref_wf, mode='full', method='fft')
        lags = scipy.signal.correlation_lags(len(padded_template_wf), len(up_ref_wf), mode='full')
        max_lag_upsampled = lags[np.argmax(corr)]
        event_peak_idx_in_padded_up = (template_peak_idx_padded_up - max_lag_upsampled)
        event_reference_idx_up = int(window_start_idx * upsample_factor + event_peak_idx_in_padded_up - pad_samples_up)
       
        ### now we can loop over channels and accumulate relative to our reference idx!
        for c in only_12ch:
            wf_mask = np.isin(channels.MI_ringphipol, np.asarray([c])) ### mask out our special channel
            this_wf = unfilt_wfs[wf_mask].flatten() ### flatten to 1D
            up_this_wf = scipy.signal.resample(this_wf, len(this_wf)*upsample_factor, axis=-1) ### upsample this wf for accumulation
            
            left_bound = max(0, event_reference_idx_up - n_samples_before_max_lag_up)
            right_bound = min(len(up_this_wf), event_reference_idx_up + n_samples_after_max_lag_up)

            padded_left = (left_bound - (event_reference_idx_up - n_samples_before_max_lag_up))
            padded_right = padded_left + (right_bound - left_bound)

            accumulated_event = np.zeros(n_samples_before_max_lag_up+n_samples_after_max_lag_up)
            accumulated_event[padded_left:padded_right] = (up_this_wf[left_bound:right_bound])

            ### saaavvveeee!
            accumulated_wfs[c] += accumulated_event 
            n_accumulated_wfs[c][padded_left:padded_right] += 1

    ### now save!
    avg_accumulated_wf = {}
    for c in only_12ch:
        avg_accumulated_wf[c] = (np.divide(accumulated_wfs[c], n_accumulated_wfs[c], out=np.zeros_like(accumulated_wfs[c]), where=n_accumulated_wfs[c] > 0)).tolist()

    saving_path = f'/scratch/midway3/darcynewmark/beamformer_testing/accumulatedbtswfssingledt_run{run}.json'
    with open(saving_path, 'w') as f:
        json.dump(avg_accumulated_wf, f, indent=4)

def compare_accum(run_start = 1502, run_end = 1524):
    ### we have already accumulated our wfs
    ### plot old vs new accumulation methods
    all_attns = []
    all_avg_wf_pk2pk_old = {}
    all_avg_wf_pk2pk_new = {}
    for c in only_12ch:
        all_avg_wf_pk2pk_old[c] = []
        all_avg_wf_pk2pk_new[c] = []

    for run in range(run_start, run_end+1):
        
        ### read in our OLD accumulated data
        old_path = f'/scratch/midway3/darcynewmark/beamformer_testing/accumulatedbtswfs_run{run}.json'
        with open(old_path) as f:
            old_avg_accumulated_wf = json.load(f)
        
        ### now read in NEW accumulated data
        new_path = f'/scratch/midway3/darcynewmark/beamformer_testing/accumulatedbtswfssingledt_run{run}.json'
        with open(new_path) as f:
            new_avg_accumulated_wf = json.load(f)

        for c in only_12ch:
            this_old_wf = old_avg_accumulated_wf[c]
            this_new_wf = new_avg_accumulated_wf[c]
            times = np.arange(len(this_old_wf))*(1/3.0)/10

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.minorticks_on()
            ax.plot(times, this_old_wf, color = 'black', label = f'Old Accumulation, Pk2Pk = {(np.amax(this_old_wf) - np.amin(this_old_wf)):0.2f}')
            ax.plot(times, this_new_wf, color = '#D55E00', label = f'New Accumulation, Pk2Pk = {(np.amax(this_new_wf) - np.amin(this_new_wf)):0.2f}')
            ax.set_title(f'Avg Wf, Run {run}, Ant {c}', fontsize = 20)
            ax.set_xlabel(f'Time [ns]', fontsize = 20)
            ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
            ax.tick_params(axis='both', which='both', labelsize=20)
            ax.legend(fontsize=18, frameon=False)
            fig.tight_layout()
            fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/compareaccumwf_run{run}_{c}.png', dpi=300)
            plt.close(fig)
        
            ### also save some stuff for plotting
            all_avg_wf_pk2pk_old[c].append(np.amax(this_old_wf) - np.amin(this_old_wf))
            all_avg_wf_pk2pk_new[c].append(np.amax(this_new_wf) - np.amin(this_new_wf))
        
        ### also save some stuff for plotting
        this_attn = run_attn_dict[run]
        all_attns.append(this_attn)
    
    ### plot attn vs signal strength!
    all_attns = np.asarray(all_attns)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    fit_start = 11
    fit_end = 15
    n_channels = len(only_12ch)
    cmap = plt.get_cmap('Spectral_r')
    norm = mpl.colors.Normalize(vmin=0, vmax=n_channels - 1)
    counter = 0
    for c in only_12ch:
        this_y_old = np.asarray(all_avg_wf_pk2pk_old[c])
        this_y_new = np.asarray(all_avg_wf_pk2pk_new[c])
        color = cmap(norm(counter))
        
        ### quick fit to the data
        fit_mask = np.logical_and(all_attns >= fit_start, all_attns <= fit_end)
        data = [all_attns[fit_mask], this_y_old[fit_mask]]
        res = minimize(attn_fit_func, x0=[this_y_old[0], 20], args=data, tol = 1e-20, method='L-BFGS-B', bounds=[ (this_y_old[0]-2,this_y_old[0]+2), (10,30)], options = {"maxiter":1e7})
        best_fit_snr0 = res.x[0]
        best_fit_a = res.x[1]
        print(f'for OLD accumulation, best fit snr0 = {best_fit_snr0}, best fit a = {best_fit_a}')
        
        interp_attn = np.linspace(np.amin(all_attns), np.amax(all_attns), 100)
        prediction = get_attn_pred(interp_attn, best_fit_snr0, best_fit_a)
        label = rf'{c} WfPk2Pk OLD = {best_fit_snr0:.2f}$\times10^{{-A/{best_fit_a:.2f}}}$'
        ax.scatter(all_attns, this_y_old, color = color, s = 20, label = label)
        ax.plot(interp_attn, prediction, color = color, linestyle = '--', linewidth = 1.5, alpha = 0.5)
        
        
        ### quick fit to the data
        fit_mask = np.logical_and(all_attns >= fit_start, all_attns <= fit_end)
        data = [all_attns[fit_mask], this_y_new[fit_mask]]
        res = minimize(attn_fit_func, x0=[this_y_new[0], 20], args=data, tol = 1e-20, method='L-BFGS-B', bounds=[ (this_y_new[0]-2,this_y_new[0]+2), (10,30)], options = {"maxiter":1e7})
        best_fit_snr0 = res.x[0]
        best_fit_a = res.x[1]
        print(f'for NEW accumulation, best fit snr0 = {best_fit_snr0}, best fit a = {best_fit_a}')
        
        interp_attn = np.linspace(np.amin(all_attns), np.amax(all_attns), 100)
        prediction = get_attn_pred(interp_attn, best_fit_snr0, best_fit_a)
        label = rf'{c} WfPk2Pk NEW = {best_fit_snr0:.2f}$\times10^{{-A/{best_fit_a:.2f}}}$'
        ax.scatter(all_attns, this_y_new, color = color, s = 20, marker = '^', label = label)
        ax.plot(interp_attn, prediction, color = color, linestyle = '-', linewidth = 1.5, alpha = 0.5)

        counter += 1

    ax.axvspan(fit_start,fit_end, color = 'black', alpha = 0.15)
    ax.text(fit_start,np.amin(prediction), 'Fit Region', fontsize = 20)
    ax.set_title(f'Per-Channel Attenuation Scaling, Run {run_start} to {run_end}', fontsize = 20)
    ax.set_xlabel(f'BTS Attenuation [dB]', fontsize = 20)
    ax.set_ylabel(f'Signal Strength [mV]', fontsize = 20)
    ax.legend(fontsize=6, frameon=False, loc = 'upper right')
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/compareavgwf_attn_vs_sigstrength_r{run_start}_r{run_end}.png', dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--array", type = int, default = 0)
    args = parser.parse_args()
    
    special_reference_channel = '0201V'
    #template_wfs = grab_template_wf(run = 1502, special_reference_channel = special_reference_channel)
    #save_wfs(args.array, template_wfs, special_reference_channel)
    compare_accum() 


