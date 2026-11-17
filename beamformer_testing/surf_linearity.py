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
from scipy.optimize import minimize
import pandas as pd
import random
from scipy import special
from postrunmapping import run_attn_dict, BAD_RUNS
from scipy import stats
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

only_12ch = np.asarray(['0201V', '0301V', '0401V',
                        '0202V', '0302V', '0402V',
                        '0203V', '0303V', '0403V',
                        '0204V', '0304V', '0404V'])

channel_special_attn = {'0201V': '31 dB', '0301V': '32 dB', '0401V': '39 dB',
                        '0202V': '36 dB', '0302V': '33 dB', '0402V': '30 dB',
                        '0203V': '30 dB', '0303V': '30 dB', '0403V': '30 dB',
                        '0204V': '30 dB', '0304V': '30 dB', '0404V': '30 dB',
                        }

def grab_data_json(runs, descrip_path = 'surflinearitydata', base_path = '/scratch/midway3/darcynewmark/beamformer_testing'):

    data = {}
    for run in runs:
        path = f'{base_path}/{descrip_path}_run{run}_chunk*.json'
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

def get_attn_pred(x, snr0, a):
    return snr0 * 10**(-x/a)

def attn_fit_func(params, data):
    x = data[0]
    y = data[1]
    snr0, a = params
    pred = get_attn_pred(x, snr0, a)
    return np.sum((y - pred)**2)

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

def fourierTransform(time, voltage):
    freq=np.fft.rfftfreq(time.shape[-1],time[1]-time[0])
    amp_f= np.fft.rfft(voltage)
    voltage_ft= np.abs(amp_f)*np.exp(np.unwrap(np.angle(amp_f))*1j)
    return freq, voltage_ft

def volt_to_dB(voltage_ratio):
    dB = np.zeros_like(voltage_ratio)
    dB[np.abs(voltage_ratio) > 1e-10] = 20*np.log10(np.abs(voltage_ratio[np.abs(voltage_ratio) > 1e-10]))
    return dB

def plot_wfs(run, n_events = 10):
    ### lets plot some wfs
    
    channels = waveform.Channels(run, 0)
    all_16ch = np.asarray(['0101', '0201', '0301', '0401',
                           '0102', '0202', '0302', '0402',
                           '0103', '0203', '0303', '0403',
                           '0104', '0204', '0304', '0404'])
    dt = 1/3.0
    num_rows = 4
    num_cols = 4

    ### and let's just plot the raw data
    for n in range(n_events):
        this_entry = n 
        channels.load_event(this_entry) ### load our event
        channels.load_waveforms(filter = False) ### load waveforms
        unfilt_wfs = channels.waveforms_unfilt
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows), layout="constrained")
         
        for i, ant in enumerate(all_16ch):
            row = i // num_cols
            col = i % num_cols
            vpol_ant_mask = np.isin(channels.MI_ringphipol, np.asarray([f'{ant}V']))
            hpol_ant_mask = np.isin(channels.MI_ringphipol, np.asarray([f'{ant}H']))
            vpol_wf = unfilt_wfs[vpol_ant_mask].flatten()
            hpol_wf = unfilt_wfs[hpol_ant_mask].flatten()
            this_times = np.arange(len(vpol_wf)) * dt
            
             ### also do the fourier transform
            f_vpol, vf_vpol = fourierTransform(this_times, vpol_wf)
            f_hpol, vf_hpol = fourierTransform(this_times, hpol_wf)
            
            axes[row, col].plot(this_times, vpol_wf, color='#009E73', ls = '-')
            axes[row, col].plot(this_times, hpol_wf, color='#D55E00', ls = '-')
            #axes[row, col].plot(f_vpol, volt_to_dB(np.abs(vf_vpol)), color='#009E73', ls = '-')
            #axes[row, col].plot(f_hpol, volt_to_dB(np.abs(vf_hpol)), color='#D55E00', ls = '-')
            axes[row, col].text(0.6, 0.9, f'Ant {ant}', transform=axes[row, col].transAxes, fontsize = 20)

        fig.supxlabel("Time [ns]", fontsize=40)
        fig.supylabel("Voltage [mV]", fontsize=40)
        plt.tight_layout()
        plt.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/surflin_r{run}_e{this_entry}.png', dpi=300)
        plt.close()

def accumulate_wfs(run, n_events = None):
    ### set up channels object and get total number of entries in this run
    channels = waveform.Channels(run, 0)
    total_N_entry = channels.dataset.N()
    
    ### set up array to save accumulated wf to
    n_accum_samples_before_peak = 300
    n_accum_samples_after_peak = 300
    accumulated_wf = np.zeros(n_accum_samples_before_peak+n_accum_samples_after_peak)

    ### now start accumulating!
    e = 0
    for ev in range(total_N_entry):
        channels.load_event(ev) ### load our event
        channels.load_waveforms(filter = False) ### load waveforms
        unfilt_wfs = channels.waveforms_unfilt
        upsampled_unfilt_wfs = scipy.signal.resample(unfilt_wfs, len(unfilt_wfs[0])*10, axis=-1)  ### upsample our unfiltered wfs!
        ant_mask = np.isin(channels.MI_ringphipol, np.asarray([f'0101H'])) ### our special signal is 0101H 
        this_wf = upsampled_unfilt_wfs[ant_mask].flatten()

        ### let's see if we want to accumulate this event!
        this_wf_peak_val = np.amax(this_wf)
        this_wf_peak_sample = np.argmax(this_wf)

        if this_wf_peak_val > 5:
            ### ok this seems to be a bts signal! savvveee!
            if n_events == None or e < n_events:
                left_bound = max(0,this_wf_peak_sample-n_accum_samples_before_peak)
                right_bound = min(len(this_wf), this_wf_peak_sample+n_accum_samples_after_peak)
                padded_left = (left_bound - (this_wf_peak_sample - n_accum_samples_before_peak))
                padded_right = padded_left + (right_bound - left_bound)
                accumulated_event = np.zeros(n_accum_samples_before_peak + n_accum_samples_after_peak)
                accumulated_event[padded_left:padded_right] = this_wf[left_bound:right_bound]
                accumulated_wf += accumulated_event
                e += 1

        if n_events is not None and e >= n_events:
            break

    ### let's plot!
    dt_accum = (1/3.0) / 10.0
    times = (np.arange(len(accumulated_wf)) - n_accum_samples_before_peak) * dt_accum
    accum_wf_peak_time = times[np.argmax(accumulated_wf)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()

    ### get bts wf!
    wf_file = f'/scratch/midway3/darcynewmark/beamformer_testing/avgBTS_ch3_teedamp_0dB.Wfm.csv'
    settings_file = wf_file.replace(".Wfm", "")
    bts_time, wf = read_newscope_cvs(settings_file, wf_file, column = 0)
    wf *= 1e3 ### V to mV
    wf_mode = mode(wf)
    wf -= wf_mode
    bts_peak_time = bts_time[np.argmax(wf)]
    bts_time_shifted = bts_time - bts_peak_time
    ax.plot(bts_time_shifted, -1*wf, color = '#009E73', linewidth = 1.5, label = f'Average Scope Trace')
    
    ### back to normal plotting!
    ax.plot(times, accumulated_wf / e, color = 'black', label = f'Average 0101H')
    ax.set_xlabel(f'Time [ns]', fontsize = 20)
    ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.legend(fontsize=16, frameon=False)
    #ax.set_xlim(20,66)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/surflin_accumulated_wf.png', dpi=300)
    plt.close(fig)

def fit_bts_scope_traces(bts_descrip = 'ch3_linearity_test'):
    ### take our bts scope traces and fit attn vs p2p/he max
    attns = [0,1,2,3,4,5,6,7,8,9,10]
    wf_p2ps = []
    he_maxs = []
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    n_attns = len(attns) 
    cmap = plt.get_cmap('Spectral_r')
    norm = mpl.colors.Normalize(vmin=0, vmax=n_attns - 1)
    for a in attns:
        wf_file = f'/scratch/midway3/darcynewmark/beamformer_testing/avgBTS_{bts_descrip}_{a}dB.Wfm.csv'
        settings_file = wf_file.replace(".Wfm", "")
        time, wf = read_newscope_cvs(settings_file, wf_file, column = 0)
        wf *= 1e3 ### V to mV
        wf_mode = mode(wf)
        wf -= wf_mode
        color = cmap(norm(a))
        time -= time[np.argmax(wf)]
        ax.plot(time, wf, color = color, linewidth = 1.5, label = f'Attn = {a} dB')
   
        ### save wf peak to peak + he max
        this_wf_p2p = np.amax(wf) - np.amin(wf)
        this_he_max = np.amax(utils.hilbert_envelope(wf))
        wf_p2ps.append(this_wf_p2p)
        he_maxs.append(this_he_max)

    ax.set_title(f'Average BTS Scope Traces (Channel 3)', fontsize = 20)
    ax.set_xlabel(f'Time [ns]', fontsize = 20)
    ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
    ax.set_xlim(-6, 20)
    ax.legend(fontsize=10, frameon=False)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/surflin_avgBTS_scope_traces_{bts_descrip}.png', dpi=300)
    plt.close(fig)

    ### plot attn vs signal strength!
    attns = np.asarray(attns)
    y_data = [wf_p2ps, he_maxs]
    y_labels = ['Waveform Pk2Pk', 'Hilbert Envelope Max']
    colors = ['#009E73', '#D55E00']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    fit_start = 0
    fit_end = 7
    for y in range(len(y_data)):
        this_y = np.asarray(y_data[y])
        
        ### quick fit to the data
        fit_mask = np.logical_and(attns >= fit_start, attns <= fit_end)
        data = [attns[fit_mask], this_y[fit_mask]]
        res = minimize(attn_fit_func, x0=[this_y[0], 20], args=data, tol = 1e-20, method='L-BFGS-B', bounds=[ (this_y[0]-2,this_y[0]+2), (10,30)], options = {"maxiter":1e7})
        best_fit_snr0 = res.x[0]
        best_fit_a = res.x[1]
        print(f'best fit snr0 = {best_fit_snr0}, best fit a = {best_fit_a}')
        
        interp_attn = np.linspace(np.amin(attns), np.amax(attns), 100)
        prediction = get_attn_pred(interp_attn, best_fit_snr0, best_fit_a)
        ax.scatter(attns, this_y, color = colors[y], s = 30, label = rf'{y_labels[y]} = {best_fit_snr0:.2f}$\times10^{{-A/{best_fit_a:.2f}}}$')
        ax.plot(interp_attn, prediction, color = colors[y], linewidth = 1.5, alpha = 0.5)

    ax.axvspan(fit_start,fit_end, color = 'black', alpha = 0.15)
    ax.text(fit_start+0.5,np.amin(prediction), 'Fit Region', fontsize = 20)
    ax.set_title(f'Average BTS Scope Traces (Channel 3)', fontsize = 20)
    ax.set_xlabel(f'BTS Attenuation [dB]', fontsize = 20)
    ax.set_ylabel(f'Signal Strength [mV]', fontsize = 20)
    ax.legend(fontsize=18, frameon=False)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/surflin_avgBTS_attn_vs_sigstrength_{bts_descrip}.png', dpi=300)
    plt.close(fig)

def compare_response(run_start, run_end, bts_descrip = 'ch3_linearity_test', bts = True, title_descrip = 'SURF', use_bts_events = False):
    ### let's read in our data for each run, make distribution of signal strengths, then find the mode for the ''signal'' case and compare to bts data!!!
    all_attns = []
    all_surf_data_wf_pk2pk = []
    all_surf_data_he_max = []
    all_bts_scope_wf_pk2pk = []
    all_bts_scope_he_max = []

    for run in range(run_start, run_end+1):
        if run in BAD_RUNS:
            continue

        ### let's read in our surf data
        this_attn = run_attn_dict[run]
        data = grab_data_json([run])
        this_entry = np.asarray(data["entry"])
        this_wf_peak_to_peak = np.asarray(data["wf_peak_to_peak"])
        this_he_max = np.asarray(data["he_max"])
        this_wf_rms = np.asarray(data["wf_rms"])
        
        if use_bts_events:
            ### get bts selected events for this run
            this_run_entry_path = f'/scratch/midway3/darcynewmark/beamformer_testing/btsselectedevents_run{run}.json'
            with open(this_run_entry_path) as f:
                all_bts_selected_entries = np.asarray(json.load(f))
        
            bts_mask = np.asarray([True if e in all_bts_selected_entries else False for e in this_entry])
            mode_wf_pk2pk_data = this_wf_peak_to_peak[bts_mask]
            mode_he_max_data = this_he_max[bts_mask]

        else:
            mode_wf_pk2pk_data = this_wf_peak_to_peak[this_wf_peak_to_peak > 175]
            mode_he_max_data = this_he_max[this_he_max > 100]
        
        #this_wf_pk2pk_mode = mode(mode_wf_pk2pk_data)
        #this_he_max_mode = mode(mode_he_max_data)
        this_wf_pk2pk_mode = np.mean(mode_wf_pk2pk_data)
        this_he_max_mode = np.mean(mode_he_max_data)
        
        ### plot distribiutions and mode to seeeee 
        min_bin_range = min(np.amin(this_wf_peak_to_peak), np.amin(this_he_max))
        max_bin_range = max(np.amax(this_wf_peak_to_peak), np.amax(this_he_max))
        nbins = int(max_bin_range/2) 

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.minorticks_on()
        if use_bts_events:
            ax.hist(mode_wf_pk2pk_data, bins = nbins, range = (min_bin_range, max_bin_range), linewidth = 1.5, histtype='step', linestyle = '-', color = '#009E73', label = rf'Waveform Pk2Pk, Avg = {this_wf_pk2pk_mode:0.2f}')
            ax.hist(mode_he_max_data, bins = nbins, range = (min_bin_range, max_bin_range), linewidth = 1.5, histtype='step', linestyle = '-', color = '#D55E00', label = rf'Hilbert Envelope Max., Avg = {this_he_max_mode:0.2f}')
        else:
            ax.hist(this_wf_peak_to_peak, bins = nbins, range = (min_bin_range, max_bin_range), linewidth = 1.5, histtype='step', linestyle = '-', color = '#009E73', label = rf'Waveform Pk2Pk, Mode = {this_wf_pk2pk_mode:0.2f}')
            ax.hist(this_he_max, bins = nbins, range = (min_bin_range, max_bin_range), linewidth = 1.5, histtype='step', linestyle = '-', color = '#D55E00', label = rf'Hilbert Envelope Max., Mode = {this_he_max_mode:0.2f}')
        ax.axvline(this_wf_pk2pk_mode, color = '#009E73', alpha = 0.5)
        ax.axvline(this_he_max_mode, color = '#D55E00', alpha = 0.5)
        ax.set_title(f'{title_descrip} Run {run}, Attenuation = {this_attn}', fontsize = 20)
        ax.set_xlabel(f'Signal Strength [mV]', fontsize = 20)
        ax.set_ylabel(f'Number of Events', fontsize = 20)
        ax.legend(fontsize=14, frameon=False)
        ax.tick_params(axis='both', which='both', labelsize=20)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/surflin_oned_wfp2p_vs_hemax_r{run}.png', dpi=300)
        plt.close(fig)

        ### also plot rms to seeee
        rms_of_sig_events = this_wf_rms[this_wf_peak_to_peak > 175]
        min_bin_range = np.amin(rms_of_sig_events) 
        max_bin_range = np.amax(rms_of_sig_events)
        nbins = int(max_bin_range*10) 
        rms_mode = mode(rms_of_sig_events)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.minorticks_on()
        ax.hist(rms_of_sig_events, bins = nbins, range = (min_bin_range, max_bin_range), linewidth = 1.5, histtype='step', linestyle = '-', color = '#009E73', label = rf'Waveform RMS, Mode = {rms_mode:0.2f}')
        ax.axvline(rms_mode, color = '#009E73', alpha = 0.5)
        ax.set_title(f'{title_descrip} Run {run}, Attenuation = {this_attn}', fontsize = 20)
        ax.set_xlabel(f'RMS [mV]', fontsize = 20)
        ax.set_ylabel(f'Number of Events', fontsize = 20)
        ax.legend(fontsize=14, frameon=False)
        ax.tick_params(axis='both', which='both', labelsize=20)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/surflin_oned_wfrms_r{run}.png', dpi=300)
        plt.close(fig)

        this_bts_wf_pk2pk = 0
        this_bts_he_max = 0
        if bts:
            ### now let's get corresponding BTS signal strength
            bts_wf_file = f'/scratch/midway3/darcynewmark/beamformer_testing/avgBTS_{bts_descrip}_{this_attn}dB.Wfm.csv'
            bts_settings_file = bts_wf_file.replace(".Wfm", "")
            bts_time, bts_wf = read_newscope_cvs(bts_settings_file, bts_wf_file, column = 0)
            bts_wf *= 1e3 ### V to mV
            bts_wf_mode = mode(bts_wf)
            bts_wf -= bts_wf_mode
            this_bts_wf_pk2pk = np.amax(bts_wf) - np.amin(bts_wf)
            this_bts_he_max = np.amax(utils.hilbert_envelope(bts_wf))

        ### now save everybody!!!
        all_attns.append(this_attn)
        all_surf_data_wf_pk2pk.append(this_wf_pk2pk_mode)
        all_surf_data_he_max.append(this_he_max_mode)
        all_bts_scope_wf_pk2pk.append(this_bts_wf_pk2pk)
        all_bts_scope_he_max.append(this_bts_he_max)

    all_attns = np.asarray(all_attns)
    if bts:
        ### ok now let's put it all together and plot attn vs bts sig strength and surf data sig strength
        all_surf_data_wf_pk2pk = np.asarray(all_surf_data_wf_pk2pk)
        all_surf_data_he_max = np.asarray(all_surf_data_he_max)
        all_bts_scope_wf_pk2pk = np.asarray(all_bts_scope_wf_pk2pk)
        all_bts_scope_he_max = np.asarray(all_bts_scope_he_max)
        
        fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax.minorticks_on()
        ax.plot(all_attns, all_surf_data_wf_pk2pk, color = '#009E73', linestyle = '-', linewidth = 1.5, label = 'SURF Wf Pk2Pk')
        ax.plot(all_attns, all_bts_scope_wf_pk2pk, color = '#009E73', linestyle = '--', linewidth = 1.5, label = 'BTS Wf Pk2Pk')
        ax.plot(all_attns, all_surf_data_he_max, color = '#D55E00', linestyle = '-', linewidth = 1.5, label = 'SURF HE Max')
        ax.plot(all_attns, all_bts_scope_he_max, color = '#D55E00', linestyle = '--', linewidth = 1.5, label = 'BTS HE Max')
        
        ### plot the residual
        wf_res = all_surf_data_wf_pk2pk / all_bts_scope_wf_pk2pk
        he_res = all_surf_data_he_max / all_bts_scope_he_max
        ax_res.plot(all_attns, wf_res, color = '#009E73', linestyle = '-', linewidth = 1.5, label = 'Wf Pk2Pk Ratio')
        ax_res.plot(all_attns, he_res, color = '#D55E00', linestyle = '-', linewidth = 1.5, label = 'HE Max Ratio')
        ax_res.set_ylabel(f'SURF / BTS', )
        ax_res.legend(fontsize=16, frameon=False)
        ax_res.grid(True, which='major', axis='both', linewidth=0.5, color='gray', alpha = 0.5)
        
        ax.set_title(f'SURF Linearity Test', fontsize = 20)
        ax_res.set_xlabel(f'BTS Attenuation [dB]', fontsize = 20)
        ax.set_ylabel(r'Signal Strength [mV]', fontsize = 20)
        ax.legend(fontsize=18, frameon=False)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax_res.tick_params(axis='both', which='both', labelsize=20)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/surflin_attn_vs_signalstrength_r{run_start}_r{run_end}.png', dpi=300)
        plt.close(fig)

    ### let's also fit attn vs signal strength!
    y_data = [all_surf_data_wf_pk2pk, all_surf_data_he_max]
    y_labels = ['0101H Waveform Pk2Pk', '0101H Hilbert Envelope Max']
    colors = ['#009E73', '#D55E00']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    attn_fit_start = 0
    atten_fit_end = 6
    for y in range(len(y_data)):
        this_y = np.asarray(y_data[y])
        
        ### quick fit to the data
        fit_mask = np.logical_and(all_attns >= attn_fit_start, all_attns <= atten_fit_end)
        data = [all_attns[fit_mask], this_y[fit_mask]]
        res = minimize(attn_fit_func, x0=[this_y[0], 20], args=data, tol = 1e-20, method='L-BFGS-B', bounds=[ (this_y[0]-2,this_y[0]+2), (10,50)], options = {"maxiter":1e7})
        best_fit_snr0 = res.x[0]
        best_fit_a = res.x[1]
        print(f'best fit snr0 = {best_fit_snr0}, best fit a = {best_fit_a}')
        
        interp_attn = np.linspace(np.amin(all_attns), np.amax(all_attns), 100)
        prediction = get_attn_pred(interp_attn, best_fit_snr0, best_fit_a)
        ax.scatter(all_attns, this_y, color = colors[y], s = 30, label = rf'{y_labels[y]} = {best_fit_snr0:.2f}$\times10^{{-A/{best_fit_a:.2f}}}$')
        ax.plot(interp_attn, prediction, color = colors[y], linewidth = 1.5, alpha = 0.5)

    ax.axvspan(attn_fit_start, atten_fit_end, color = 'black', alpha = 0.15)
    ax.text(attn_fit_start + 0.1, np.amin(prediction)+10, 'Fit Region', fontsize = 20)
    ax.set_title(f'{title_descrip} Test', fontsize = 20)
    ax.set_xlabel(f'BTS Attenuation [dB]', fontsize = 20)
    ax.set_ylabel(f'Signal Strength [mV]', fontsize = 20)
    ax.legend(fontsize=16, frameon=False)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/surflin_surf_attn_vs_sigstrength_r{run_start}_r{run_end}.png', dpi=300)
    plt.close(fig)

def spot_check_surf(run_start, run_end, n_events = 16):
    
    special_surf_channel = '0101H'
    dt = 1/3.0
    num_cols = int(np.ceil(np.sqrt(n_events)))
    num_rows = int(np.ceil(n_events / num_cols))

    for run in range(run_start, run_end+1):
        if run in BAD_RUNS:
            continue
        ### make a plot of surf 0101H channel for n_events per run
        plotted_events = 0
        current_entry = 0
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows), layout="constrained")
        channels = waveform.Channels(run, 0)

        while plotted_events < n_events:
            this_entry = current_entry
            current_entry += 1
            channels.load_event(this_entry) ### load our event
            channels.load_waveforms(filter = False) ### load waveforms
            unfilt_wfs = channels.waveforms_unfilt
            upsampled_unfilt_wfs = scipy.signal.resample(unfilt_wfs, len(unfilt_wfs[0])*10, axis=-1) ### upsample our unfiltered wfs!
            wf_mask = np.isin(channels.MI_ringphipol, np.asarray([special_surf_channel])) ### mask out our special channel
            this_wf = upsampled_unfilt_wfs[wf_mask].flatten() ### flatten to 1D
            this_times = np.arange(len(this_wf)) * dt / 10

            ### get hilbert envelope and wf pk2pk
            hilbert_env = utils.hilbert_envelope(this_wf)
            hilenv_max = np.max(hilbert_env)
            wf_p2p = np.max(this_wf) - np.min(this_wf)

            if wf_p2p < 100:
                continue

            ### ok this event is big enough! let's plot!
            row = plotted_events // num_cols
            col = plotted_events % num_cols

            axes[row, col].plot(this_times, this_wf, color='#009E73', ls = '-')
            axes[row, col].plot(this_times, hilbert_env, color='#D55E00', ls = '-')
            axes[row, col].set_xlim(50, 120)
            axes[row, col].text(0.3, 0.9, f'{special_surf_channel},r={run},e={this_entry}', transform=axes[row, col].transAxes, fontsize = 18)
            axes[row, col].text(0.3, 0.8, f'WfPk2Pk={wf_p2p:0.2f}', transform=axes[row, col].transAxes, fontsize = 18)
            axes[row, col].text(0.3, 0.7, f'HEMax={hilenv_max:0.2f}', transform=axes[row, col].transAxes, fontsize = 18)

            plotted_events += 1

        fig.supxlabel("Time [ns]", fontsize=40)
        fig.supylabel("Voltage [mV]", fontsize=40)
        plt.tight_layout()
        plt.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/surflin_spotcheck_r{run}.png', dpi=300)
        plt.close()

def check_bts_per_event(n_events = 60):
    ### plot distribution of bts wf pk 2 pk
    bts_pk2pk = []
    bts_he_max = []
    for e in range(n_events):
        bts_wf_file = f'/scratch/midway3/darcynewmark/beamformer_testing/BTS_ch3_0dB_e{e}.Wfm.csv'
        bts_settings_file = bts_wf_file.replace(".Wfm", "")
        bts_time, bts_wf = read_newscope_cvs(bts_settings_file, bts_wf_file, column = 0)
        bts_wf *= 1e3 ### V to mV
        bts_wf_mode = mode(bts_wf)
        bts_wf -= bts_wf_mode
        this_bts_wf_pk2pk = np.amax(bts_wf) - np.amin(bts_wf)
        this_bts_he_max = np.amax(utils.hilbert_envelope(bts_wf))

        bts_pk2pk.append(this_bts_wf_pk2pk)
        bts_he_max.append(this_bts_he_max)
    
    min_bin_range = min(np.amin(bts_pk2pk), np.amin(bts_he_max))
    max_bin_range = max(np.amax(bts_pk2pk), np.amax(bts_he_max))
    nbins = int(max_bin_range/22) 

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    ax.hist(bts_pk2pk, bins = nbins, range = (min_bin_range, max_bin_range), linewidth = 1.5, histtype='step', linestyle = '-', color = '#009E73', label = rf'Waveform Pk2Pk')
    ax.hist(bts_he_max, bins = nbins, range = (min_bin_range, max_bin_range), linewidth = 1.5, histtype='step', linestyle = '-', color = '#D55E00', label = rf'Hilbert Envelope Max.')
    ax.set_title(f'Individual BTS Scope Traces, Attn = 0', fontsize = 20)
    ax.set_xlabel(f'Signal Strength [mV]', fontsize = 20)
    ax.set_ylabel(f'Number of Events', fontsize = 20)
    ax.legend(fontsize=14, frameon=False)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/surflin_individualbtsscope_sigstrength.png', dpi=300)
    plt.close(fig)

def sanity_checks(run_start, run_end, clip_window = 5):
    for run in range(run_start, run_end+1):
        if run in BAD_RUNS:
            continue

        ### let's read in our surf data
        this_attn = run_attn_dict[run]
        data = grab_data_json([run])
        this_wf_peak_to_peak = np.asarray(data["wf_peak_to_peak"])
        this_he_max = np.asarray(data["he_max"])
        this_trigger_subsec = np.asarray(data["trigger_subsec"])
        this_trigger_time = np.asarray(data["trigger_time"])
        run_start_time = np.amin(this_trigger_time)

        ### mask out early trigger times
        valid_mask = functools.reduce(np.logical_and, [this_trigger_subsec >= 0, this_trigger_subsec <= 1, this_trigger_time >= (run_start_time + clip_window), ])
        this_wf_peak_to_peak = this_wf_peak_to_peak[valid_mask]
        this_he_max = this_he_max[valid_mask]
        this_trigger_subsec = this_trigger_subsec[valid_mask]
        this_trigger_time = this_trigger_time[valid_mask]
        this_trigger_time -= np.amin(this_trigger_time)

        ### let's make trigger time vs signal strength metric 2d plots
        x_data = [this_trigger_time, this_trigger_time]
        y_data = [this_wf_peak_to_peak, this_he_max]
        x_label = ['Trigger Time [s]', 'Trigger Time [s]']
        y_label = ['Waveform Pk2Pk [mV]', 'Hilbert Envelope Max [mV]']
        saving_names = ['trigger_time_vs_wfpk2pk', 'trigger_time_vs_hemax']

        for i in range(len(x_data)):
            this_x_data = x_data[i]
            this_y_data = y_data[i]
        
            mask = this_y_data > 100
            this_x_data = this_x_data[mask]
            this_y_data = this_y_data[mask]

            ### set up bin edges
            nbins = 50
            x_min, x_max = np.amin(this_x_data), np.amax(this_x_data)
            x_bin_width = (x_max - x_min) / nbins
            x_data_bin_edges = np.linspace(x_min - x_bin_width, x_max + x_bin_width, nbins + 3)
            y_min, y_max = np.amin(this_y_data), np.amax(this_y_data)
            y_bin_width = (y_max - y_min) / nbins
            y_data_bin_edges = np.linspace(y_min - y_bin_width, y_max + y_bin_width, nbins + 3)

            ### make our matrix
            M, _, _ = np.histogram2d(this_x_data, this_y_data, bins=[x_data_bin_edges, y_data_bin_edges])

            ### plot!
            cm = plt.get_cmap("Spectral_r")
            lin_norm = matplotlib.colors.LogNorm(vmin=1, vmax=np.amax(M.T))

            fig, ax = plt.subplots(figsize=(10, 7))
            mesh = ax.pcolormesh(x_data_bin_edges, y_data_bin_edges, M.T, cmap=cm, norm = lin_norm)
            cb = fig.colorbar(mesh, ax=ax)
            cb.ax.set_ylabel("Number of Events", fontsize = 20)
            cb.ax.minorticks_off()
            ax.set_title(f'SURF Test Run {run}, Attenuation = {run_attn_dict[run]}', fontsize = 20)
            ax.set_xlabel(x_label[i], fontsize = 20)
            ax.set_ylabel(y_label[i], fontsize = 20)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=14)
            fig.tight_layout()
            plt.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/surflin_twod_{saving_names[i]}_r{run}.png', dpi=300)
            plt.close(fig)

def plot_sig_strength_distros(run_start, run_end, var_attn = False):
   
    all_attns = []
    per_channel_avg_wfpk2pk = {}
    for c in only_12ch:
        per_channel_avg_wfpk2pk[c] = []

    for run in range(run_start, run_end+1):
        if run in BAD_RUNS:
            continue
        
        ### get bts selected events for this run
        this_run_entry_path = f'/scratch/midway3/darcynewmark/beamformer_testing/btsselectedevents_run{run}.json'
        with open(this_run_entry_path) as f:
            all_bts_selected_entries = np.asarray(json.load(f))

        ### let's read in our surf data
        this_attn = run_attn_dict[run]
        data = grab_data_json([run], descrip_path = 'beamformerdata')
        entry = np.asarray(data["entry"])
        this_wf_peak_to_peak = (np.asarray(data["wf_peak_to_peak"]))
        this_he_max = (np.asarray(data["he_max"]))

        ### only look at bts selected events
        bts_mask = np.asarray([True if e in all_bts_selected_entries else False for e in entry])
        this_wf_peak_to_peak = this_wf_peak_to_peak[bts_mask]
        this_he_max = this_he_max[bts_mask]

        ### plot distribiutions to seeeee 
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.minorticks_on()
        
        n_channels = len(only_12ch)
        cmap = plt.get_cmap('Spectral_r')
        norm = mpl.colors.Normalize(vmin=0, vmax=n_channels - 1)
        
        for c in range(n_channels):
            color = cmap(norm(c))
            wf_data = np.asarray(this_wf_peak_to_peak[:,c])
            he_data = np.asarray(this_he_max[:,c])
        
            #min_bin_range = min(np.amin(wf_data), np.amin(he_data))
            #max_bin_range = max(np.amax(wf_data), np.amax(he_data))
            min_bin_range = np.amin(wf_data)
            max_bin_range = np.amax(wf_data)
            nbins = int(max_bin_range/8) 

            if var_attn:
                label = f'{only_12ch[c]} WfPk2Pk ({channel_special_attn[only_12ch[c]]})'
            else:
                label = f'{only_12ch[c]} WfPk2Pk'
            ax.hist(wf_data, bins = nbins, range = (min_bin_range, max_bin_range), linewidth = 1.5, histtype='step', linestyle = '-', color = color, label = label)
            #ax.hist(he_data, bins = nbins, range = (min_bin_range, max_bin_range), linewidth = 1.5, histtype='step', linestyle = '--', color = color)
        
            ### save avg wf pk2pk to our dict
            per_channel_avg_wfpk2pk[only_12ch[c]].append(np.mean(wf_data))

        ax.set_title(f'BTS Selected Events Run {run}, Attenuation = {this_attn}', fontsize = 20)
        ax.set_xlabel(f'Signal Strength [mV]', fontsize = 20)
        ax.set_ylabel(f'Number of Events', fontsize = 20)
        ax.legend(fontsize=10, frameon=False)
        ax.tick_params(axis='both', which='both', labelsize=20)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/surflin_oned_wfp2p_r{run}.png', dpi=300)
        plt.close(fig)
    
        all_attns.append(this_attn)

    ### now let's fit attn vs avg wf pk2pk for EACH channel!
    all_attns = np.asarray(all_attns)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    
    n_channels = len(only_12ch)
    cmap = plt.get_cmap('Spectral_r')
    norm = mpl.colors.Normalize(vmin=0, vmax=n_channels - 1)
    
    fit_start_attn = 0
    fit_end_attn = 3
    for c in range(n_channels):
        color = cmap(norm(c))
        this_avg_wfpk2pk = np.asarray(per_channel_avg_wfpk2pk[only_12ch[c]])

        ### quick fit to the data
        fit_mask = np.logical_and(all_attns >= fit_start_attn, all_attns <= fit_end_attn)
        data = [all_attns[fit_mask], this_avg_wfpk2pk[fit_mask]]
        res = minimize(attn_fit_func, x0=[this_avg_wfpk2pk[0], 20], args=data, tol = 1e-20, method='L-BFGS-B', bounds=[ (this_avg_wfpk2pk[0]-2,this_avg_wfpk2pk[0]+2), (10,30)], options = {"maxiter":1e7})
        best_fit_snr0 = res.x[0]
        best_fit_a = res.x[1]
        print(f'best fit snr0 = {best_fit_snr0}, best fit a = {best_fit_a}')
        
        interp_attn = np.linspace(np.amin(all_attns), np.amax(all_attns), 100)
        prediction = get_attn_pred(interp_attn, best_fit_snr0, best_fit_a)
        if var_attn:
            label = rf'{only_12ch[c]} WfPk2Pk ({channel_special_attn[only_12ch[c]]}) = {best_fit_snr0:.2f}$\times10^{{-A/{best_fit_a:.2f}}}$'
        else:
            label = rf'{only_12ch[c]} WfPk2Pk = {best_fit_snr0:.2f}$\times10^{{-A/{best_fit_a:.2f}}}$'
        ax.scatter(all_attns, this_avg_wfpk2pk, color = color, s = 30, label = label)
        ax.plot(interp_attn, prediction, color = color, linewidth = 1.5, alpha = 0.5)

    ax.axvspan(fit_start_attn, fit_end_attn, color = 'black', alpha = 0.15)
    ax.text(fit_start_attn,np.amin(prediction), 'Fit Region', fontsize = 20)
    ax.set_title(f'Per-Channel Attenuation Scaling, Run {run_start} to {run_end}', fontsize = 20)
    ax.set_xlabel(f'BTS Attenuation [dB]', fontsize = 20)
    ax.set_ylabel(f'Signal Strength [mV]', fontsize = 20)
    ax.legend(fontsize=10, frameon=False, loc = 'upper right')
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/surflin_perchannel_attn_vs_sigstrength_r{run_start}_r{run_end}.png', dpi=300)
    plt.close(fig)

def check_rx_linearity():
    ### take our bts scope traces and fit attn vs p2p/he max
    attns = [0,1,2,3,4,5,6,7,8,9,10,]
    wf_p2ps = []
    he_maxs = []
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    n_attns = len(attns) 
    cmap = plt.get_cmap('Spectral_r')
    norm = mpl.colors.Normalize(vmin=0, vmax=n_attns - 1)
    for a in attns:
        wf_file = f'/scratch/midway3/darcynewmark/beamformer_testing/avgBTS_017_{a}dB.Wfm.csv'
        settings_file = wf_file.replace(".Wfm", "")
        time, wf = read_newscope_cvs(settings_file, wf_file, column = 0)
        wf *= 1e3 ### V to mV
        wf_mode = mode(wf)
        wf -= wf_mode
        color = cmap(norm(a))
        time -= time[np.argmax(wf)]
        ax.plot(time, wf, color = color, linewidth = 1.5, label = f'Attn = {a} dB')
   
        ### save wf peak to peak + he max
        this_wf_p2p = np.amax(wf) - np.amin(wf)
        this_he_max = np.amax(utils.hilbert_envelope(wf))
        wf_p2ps.append(this_wf_p2p)
        he_maxs.append(this_he_max)

    ax.set_title(f'Average BTS Scope Traces From Rx Box', fontsize = 20)
    ax.set_xlabel(f'Time [ns]', fontsize = 20)
    ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
    ax.set_xlim(-3, 6)
    ax.legend(fontsize=10, frameon=False, )
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/rxboxlin_avgBTS_scope_traces.png', dpi=300)
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
    ax.text(1.5,np.amin(prediction), 'Fit Region', fontsize = 20)
    ax.set_title(f'Average BTS Scope Traces From Rx Box', fontsize = 20)
    ax.set_xlabel(f'BTS Attenuation [dB]', fontsize = 20)
    ax.set_ylabel(f'Signal Strength [mV]', fontsize = 20)
    ax.legend(fontsize=18, frameon=False)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/rxboxlin_avgBTS_attn_vs_sigstrength.png', dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    #accumulate_wfs(run = 1665, n_events = 10)
    #fit_bts_scope_traces(bts_descrip = 'ch3_linearity_test')
    #fit_bts_scope_traces(bts_descrip = 'ch3_r1691')
    #plot_wfs(run = 1680)
    #plot_wfs(run = 1608)
    #plot_wfs(run = 1692)
    #plot_wfs(run = 1709)
    #plot_wfs(run = 1746)
    #compare_response(run_start = 1680, run_end = 1690)
    #compare_response(run_start = 1692, run_end = 1707, bts_descrip = 'ch3_r1691')
    #spot_check_surf(run_start = 1692, run_end = 1707)
    #check_bts_per_event()
    #sanity_checks(run_start = 1692, run_end = 1707)
    #plot_sig_strength_distros(run_start = 1502, run_end = 1524)
    #plot_sig_strength_distros(run_start = 1680, run_end = 1690)
    #check_rx_linearity()
    #compare_response(run_start = 1709, run_end = 1719, title_descrip = 'AMPA-->0101H', bts = False)
    #compare_response(run_start = 1746, run_end = 1756, title_descrip = 'AMPA-->0101H', bts = False, use_bts_events = True)
    #plot_sig_strength_distros(run_start = 1746, run_end = 1756)
    #fit_bts_scope_traces(bts_descrip = '30dBAMPA')
    plot_sig_strength_distros(run_start = 1721, run_end = 1732, var_attn = True)
