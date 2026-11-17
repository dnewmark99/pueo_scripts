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

def get_attn_pred(x, snr0, a):
    return snr0 * 10**(a*(-x/20.0))

def attn_fit_func(params, data):
    x = data[0]
    y = data[1]
    snr0, a = params
    pred = get_attn_pred(x, snr0, a)
    return np.sum((y - pred)**2)

def get_linear_pred(x, m, b):
    return (m * x) + b

def linear_fit_func(params, data):
    x = data[0]
    y = data[1]
    m, b = params
    pred = get_linear_pred(x, m, b)
    return np.sum((y - pred)**2)

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


def fit_bts_scope_traces():
    ### take our bts scope traces and fit attn vs p2p/he max
    attns = [0,1,2,3,4,5,6,7,8,9]
    wf_p2ps = []
    he_maxs = []
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    n_attns = len(attns) 
    cmap = plt.get_cmap('Spectral_r')
    norm = mpl.colors.Normalize(vmin=0, vmax=n_attns - 1)
    for a in attns:
        wf_file = f'/scratch/midway3/darcynewmark/beamformer_testing/ch_1_2_6_7_{a}dB.Wfm.csv'
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

    ax.set_title(f'Average BTS Scope Traces (Channel 1)', fontsize = 20)
    ax.set_xlabel(f'Time [ns]', fontsize = 20)
    ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
    ax.set_xlim(-6, 20)
    ax.legend(fontsize=10, frameon=False)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/powin_powout_avgBTS_scope_traces_ch1.png', dpi=300)
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
        res = minimize(attn_fit_func, x0=[this_y[0], 1], args=data, tol = 1e-20, method='L-BFGS-B', bounds=[ (this_y[0]-2,this_y[0]+2), (0.1,3)], options = {"maxiter":1e7})
        best_fit_snr0 = res.x[0]
        best_fit_a = res.x[1]
        print(f'best fit snr0 = {best_fit_snr0}, best fit a = {best_fit_a}')
        
        interp_attn = np.linspace(np.amin(attns), np.amax(attns), 100)
        prediction = get_attn_pred(interp_attn, best_fit_snr0, best_fit_a)
        ax.scatter(attns, this_y, color = colors[y], s = 30, label = rf'{y_labels[y]} = {best_fit_snr0:.2f}$\times10^{{{best_fit_a:.2f}\times-A/20}}$')
        ax.plot(interp_attn, prediction, color = colors[y], linewidth = 1.5, alpha = 0.5)

    ax.axvspan(fit_start,fit_end, color = 'black', alpha = 0.15)
    ax.text(fit_start+0.5,np.amin(prediction), 'Fit Region', fontsize = 20)
    ax.set_title(f'Average BTS Scope Traces (Channel 1)', fontsize = 20)
    ax.set_xlabel(f'BTS Attenuation [dB]', fontsize = 20)
    ax.set_ylabel(f'Signal Strength [mV]', fontsize = 20)
    ax.legend(fontsize=18, frameon=False)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/powin_powout_avgBTS_attn_vs_sigstrength_ch1.png', dpi=300)
    plt.close(fig)

def compare_data_w_scope(run_start, run_end):
    ### let's get our accumulated wf for every run, calculate p2p and power and compare with scope data!!!
    ### just do using 1 channel for the time being
    scope_channel_name = 'Ch 1'
    scope_channel_idx = 0
    mie_antenna = '0201V'
    scope_p2p = []
    mie_p2p = []
    all_attns = []

    for run in range(run_start, run_end+1):
        if run in BAD_RUNS:
            continue

        ### get the attenuation for this run
        this_attn = run_attn_dict[run]

        ### get scope p2p
        wf_file = f'/scratch/midway3/darcynewmark/beamformer_testing/ch_1_2_6_7_{this_attn}dB.Wfm.csv'
        settings_file = wf_file.replace(".Wfm", "")
        time, wf = read_newscope_cvs(settings_file, wf_file, column = scope_channel_idx)
        wf *= 1e3 ### V to mV
        wf_mode = mode(wf)
        wf -= wf_mode
        this_scope_p2p = np.amax(wf) - np.amin(wf)

        ### get mie p2p from the averaged wfs
        path = f'/scratch/midway3/darcynewmark/beamformer_testing/accumulatedbtswfs_run{run}.json'
        with open(path) as f:
            avg_accumulated_wf = json.load(f)
        this_wf = np.asarray(avg_accumulated_wf[mie_antenna])
        this_mie_p2p = np.amax(this_wf) - np.amin(this_wf)

        ### now save
        scope_p2p.append(this_scope_p2p)
        mie_p2p.append(this_mie_p2p)
        all_attns.append(this_attn)

    ### now plot!!!
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
        
    ### quick fit to the data
    scope_p2p = np.asarray(scope_p2p)
    mie_p2p = np.asarray(mie_p2p)
    all_attns = np.asarray(all_attns)
    data = [scope_p2p, mie_p2p]
    res = minimize(linear_fit_func, x0=[mie_p2p[0], 30], args=data, tol = 1e-20, method='L-BFGS-B', bounds=[ (mie_p2p[0]*0.001,mie_p2p[0]*2), (0,100)], options = {"maxiter":1e7})
    best_fit_m = res.x[0]
    best_fit_b = res.x[1]
    print(f'bounds = {(mie_p2p[0]*0.1,mie_p2p[0]*2)}, {(0,30)}')
    print(f'best fit m = {best_fit_m}, best fit b = {best_fit_b}')
    
    interp_x = np.linspace(np.amin(scope_p2p), np.amax(scope_p2p), 100)
    prediction = get_linear_pred(interp_x, best_fit_m, best_fit_b)
    ax.plot(interp_x, prediction, color = 'black', linewidth = 1.5, linestyle = '--', alpha = 0.5, label = f'Best Fit Vpp Out = {best_fit_m:0.2f} * Vpp In + {best_fit_b:0.2f}')
    ax.scatter(scope_p2p, mie_p2p, color = 'black', s=20, label = f'Data')
    
    ax.set_title(f'Vpp In vs Vpp Out, Run {run_start} to Run {run_end}, Ant {mie_antenna}', fontsize = 20)
    ax.set_xlabel(f'Vpp In [mV]', fontsize = 20)
    ax.set_ylabel(f'Vpp Out [mV]', fontsize = 20)
    ax.grid(True, which='both', axis='both', linewidth=0.5, color='gray', alpha = 0.5)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.legend(fontsize=16, frameon=False)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/powin_powout_vpp_r{run_start}_r{run_end}_{mie_antenna}_lin.png', dpi=300)
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/powin_powout_vpp_r{run_start}_r{run_end}_{mie_antenna}.png', dpi=300)
    plt.close(fig)

    ### let's also fit attn vs sig stregnth WITH errors!
    y_fitting_data = [mie_p2p, scope_p2p]
    y_labels = ['Vpp Out', 'Vpp In']
    fitting_ranges = [(0,3), (3,6), (6,9)]
    fitting_colors = ['black', '#009E73', '#D55E00']
    for y in range(len(y_fitting_data)):
        ### set up plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.minorticks_on()
        
        ### grab y data
        this_y = y_fitting_data[y]
        ax.scatter(all_attns, this_y, color = 'black', s = 30)

        ### fitttttt across different ranges
        for ff in range(len(fitting_ranges)):
            fit_start = fitting_ranges[ff][0]
            fit_end = fitting_ranges[ff][1]

            fit_mask = np.logical_and(all_attns >= fit_start, all_attns <= fit_end)
            data = [all_attns[fit_mask], this_y[fit_mask]]
            res = minimize(attn_fit_func, x0=[this_y[0], 1], args=data, tol = 1e-20, method='L-BFGS-B', bounds=[ (this_y[0]-2,this_y[0]+2), (0.1,3)], options = {"maxiter":1e7})
            best_fit_snr0 = res.x[0]
            best_fit_a = res.x[1]
            print(f'best fit snr0 = {best_fit_snr0}, best fit a = {best_fit_a}')

            interp_attn = np.linspace(np.amin(all_attns), np.amax(all_attns), 100)
            prediction = get_attn_pred(interp_attn, best_fit_snr0, best_fit_a)
            ax.plot(interp_attn, prediction, color = fitting_colors[ff], linewidth = 1.5, label = rf'{y_labels[y]} = {best_fit_snr0:.2f}$\times10^{{{best_fit_a:.2f}\times-A/20}}$ (Fitting [{fit_start}, {fit_end}] dB)')
            ax.axvspan(fit_start,fit_end, color = fitting_colors[ff], alpha = 0.15)

        ### now save!
        ax.set_title(f'{y_labels[y]} vs Attenuation Fit', fontsize = 20)
        ax.set_xlabel(f'BTS Attenuation [dB]', fontsize = 20)
        ax.set_ylabel(f'Signal Strength [mV]', fontsize = 20)
        ax.legend(fontsize=14, frameon=False)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.set_yscale('log')
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/powin_powout_attn_vs_{y_labels[y]}_fit.png', dpi=300)
        plt.close(fig)

def fit_only_data(run_start, run_end):
    ### let's get our accumulated wf for every run, calculate p2p and power and compare with scope data!!!
    ### just do using 1 channel for the time being
    mie_antenna = '0201V'
    mie_p2p = []
    all_attns = []

    for run in range(run_start, run_end+1):
        if run in BAD_RUNS:
            continue

        ### get the attenuation for this run
        this_attn = run_attn_dict[run]

        ### get mie p2p from the averaged wfs
        path = f'/scratch/midway3/darcynewmark/beamformer_testing/accumulatedbtswfs_run{run}.json'
        with open(path) as f:
            avg_accumulated_wf = json.load(f)
        this_wf = np.asarray(avg_accumulated_wf[mie_antenna])
        this_mie_p2p = np.amax(this_wf) - np.amin(this_wf)

        ### now save
        mie_p2p.append(this_mie_p2p)
        all_attns.append(this_attn)

    ### let's also fit attn vs sig stregnth WITH errors!
    mie_p2p = np.asarray(mie_p2p)
    all_attns = np.asarray(all_attns)
    y_fitting_data = [mie_p2p, ]
    y_labels = ['Vpp Out', ]
    fitting_ranges = [(0,3), (3,6), (6,9), (9,12)]
    fitting_colors = ['black', '#009E73', '#D55E00', 'magenta']
    for y in range(len(y_fitting_data)):
        ### set up plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.minorticks_on()
        
        ### grab y data
        this_y = y_fitting_data[y]
        ax.scatter(all_attns, this_y, color = 'black', s = 30)

        ### fitttttt across different ranges
        for ff in range(len(fitting_ranges)):
            fit_start = fitting_ranges[ff][0]
            fit_end = fitting_ranges[ff][1]

            fit_mask = np.logical_and(all_attns >= fit_start, all_attns <= fit_end)
            data = [all_attns[fit_mask], this_y[fit_mask]]
            res = minimize(attn_fit_func, x0=[this_y[0], 1], args=data, tol = 1e-20, method='L-BFGS-B', bounds=[ (this_y[0]-2,this_y[0]+2), (0.1,3)], options = {"maxiter":1e7})
            best_fit_snr0 = res.x[0]
            best_fit_a = res.x[1]
            print(f'best fit snr0 = {best_fit_snr0}, best fit a = {best_fit_a}')

            interp_attn = np.linspace(np.amin(all_attns), np.amax(all_attns), 100)
            prediction = get_attn_pred(interp_attn, best_fit_snr0, best_fit_a)
            ax.plot(interp_attn, prediction, color = fitting_colors[ff], linewidth = 1.5, label = rf'{y_labels[y]} = {best_fit_snr0:.2f}$\times10^{{{best_fit_a:.2f}\times-A/20}}$ (Fitting [{fit_start}, {fit_end}] dB)')
            ax.axvspan(fit_start,fit_end, color = fitting_colors[ff], alpha = 0.15)

        ### now save!
        ax.set_title(f'{y_labels[y]} vs Attenuation Fit', fontsize = 20)
        ax.set_xlabel(f'BTS Attenuation [dB]', fontsize = 20)
        ax.set_ylabel(f'Avg Wf Pk2Pk [mV]', fontsize = 20)
        ax.legend(fontsize=14, frameon=False)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.set_yscale('log')
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/powin_powout_attn_vs_{y_labels[y]}_fit.png', dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    #fit_bts_scope_traces()
    compare_data_w_scope(run_start = 1758, run_end = 1767)
    fit_only_data(run_start = 1758, run_end = 1774)
