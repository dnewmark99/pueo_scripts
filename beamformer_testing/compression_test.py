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
import random
from scipy import stats
import csv
from scipy.optimize import minimize

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

def get_linear_pred(x, m, b):
    return (m * x) + b

def linear_fit_func(params, data):
    x = data[0]
    y = data[1]
    m, b = params
    pred = get_linear_pred(x, m, b)
    return np.sum((y - pred)**2)

def get_dbm(wf, R = 50):
    P_avg_W = np.sum(wf**2) / R
    P_dBm = 10 * np.log10(P_avg_W / 1e-3)
    return P_dBm

def get_dbm_sine(wf, R = 50):
    vrms = (np.amax(wf) - np.amin(wf)) / (2*np.sqrt(2))
    p_dbm = 10 * np.log10((vrms**2/R) / 1e-3)
    return p_dbm

def look_at_bts_events(powin = True, powout = True):
    ### recorded data for "pow in" == BTS signal output scope traces
    ### recorded data for "pow out" == RF chain signal ouptut scope traces
    attns = [0,1,2,3,4,5,6,7,8,9,10]
    power_in = []
    power_out = []
    dbm_in = []
    dbm_out = []
    dbmsine_in = []
    dbmsine_out = []
    fig_powin, ax_powin = plt.subplots(figsize=(10, 6))
    ax_powin.minorticks_on()
    fig_powout, ax_powout = plt.subplots(figsize=(10, 6))
    ax_powout.minorticks_on()
    n_attns = len(attns) 
    cmap = plt.get_cmap('Spectral_r')
    norm = mpl.colors.Normalize(vmin=0, vmax=n_attns - 1)
    attenuation_dB = 30
    for a in attns:
        if powin:
            wf_file = f'/scratch/midway3/darcynewmark/beamformer_testing/avgBTS_ch1_{a}dB.Wfm.csv'
            settings_file = wf_file.replace(".Wfm", "")
            time, wf = read_newscope_cvs(settings_file, wf_file, column = 0)
            wf_mode = mode(wf)
            wf -= wf_mode
            wf *= 1e3 ### V to mV
            color = cmap(norm(a))
            vpp = np.amax(wf) - np.amin(wf)
            ax_powin.plot(time, wf, color = color, linewidth = 1.5, label = f'Attn = {a} dB, Vpp = {vpp:0.2f} mV')
            power_in.append(vpp)
            dbm = get_dbm(wf*1e-3*10**(-attenuation_dB / 20))
            dbm_in.append(dbm)
            dbmsine = get_dbm_sine(wf*1e-3*10**(-attenuation_dB / 20))
            dbmsine_in.append(dbmsine)
        if powout:
            wf_file = f'/scratch/midway3/darcynewmark/beamformer_testing/avgSURF_0201V_{a}dB.Wfm.csv'
            settings_file = wf_file.replace(".Wfm", "")
            time, wf = read_newscope_cvs(settings_file, wf_file, column = 0)
            wf_mode = mode(wf)
            wf -= wf_mode
            wf *= 1e3 ### V to mV
            color = cmap(norm(a))
            vpp = np.amax(wf) - np.amin(wf)
            ax_powout.plot(time, wf, color = color, linewidth = 1.5, label = f'Attn = {a} dB, Vpp = {vpp:0.2f} mV')
            power_out.append(vpp)
            dbm = get_dbm(wf*1e-3)
            dbm_out.append(dbm)
            dbmsine = get_dbm_sine(wf*1e-3)
            dbmsine_out.append(dbmsine)
    if powin: 
        ax_powin.set_title(f'Average BTS Scope Traces (Channel 3)', fontsize = 20)
        ax_powin.set_xlabel(f'Time [ns]', fontsize = 20)
        ax_powin.set_ylabel(f'Voltage [mV]', fontsize = 20)
        ax_powin.set_xlim(-2, 6)
        ax_powin.legend(fontsize=10, frameon=False)
        ax_powin.tick_params(axis='both', which='both', labelsize=20)
        fig_powin.tight_layout()
        fig_powin.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/compression_test_ch1_scope_traces.png', dpi=300)
        plt.close(fig_powin)
    if powout: 
        ax_powout.set_title(f'Average SURF 0201V Scope Traces', fontsize = 20)
        ax_powout.set_xlabel(f'Time [ns]', fontsize = 20)
        ax_powout.set_ylabel(f'Voltage [mV]', fontsize = 20)
        ax_powout.legend(fontsize=10, frameon=False)
        ax_powout.tick_params(axis='both', which='both', labelsize=20)
        ax_powout.set_xlim(65, 110)
        fig_powout.tight_layout()
        fig_powout.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/compression_test_avg_0201V_scope_traces.png', dpi=300)
        plt.close(fig_powout)

    if powin and powout:
        ### ok now let's plot power in vs power out!!!
        power_in = np.asarray(power_in)
        power_out = np.asarray(power_out)
        dbm_in = np.asarray(dbm_in)
        dbm_out = np.asarray(dbm_out)
        dbmsine_in = np.asarray(dbmsine_in)
        dbmsine_out = np.asarray(dbmsine_out)
        plotting_powin = power_in*10**(-attenuation_dB / 20)*1e3

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.minorticks_on()
        ax.scatter(plotting_powin, power_out, color = 'black', s = 10, label = 'Data')
        
        fig_dbm, ax_dbm = plt.subplots(figsize=(10, 6))
        ax_dbm.minorticks_on()
        ax_dbm.scatter(dbm_in, dbm_out, color = 'black', s = 10, label = 'Data')
        
        fig_dbmsine, ax_dbmsine = plt.subplots(figsize=(10, 6))
        ax_dbmsine.minorticks_on()
        ax_dbmsine.scatter(dbmsine_in, dbmsine_out, color = 'black', s = 10, label = 'Data')
        
        ### quick fit to the data
        fit_start = 200
        fit_end = 300
        fit_mask = np.logical_and(plotting_powin >= fit_start, plotting_powin <= fit_end)
        data = [plotting_powin[fit_mask], power_out[fit_mask]]
        res = minimize(linear_fit_func, x0=[1.4, 280], args=data, tol = 1e-20, method='L-BFGS-B', bounds=[ (0,400), (-500,500)], options = {"maxiter":1e7})
        best_fit_m = res.x[0]
        best_fit_b = res.x[1]
        
        interp_powin = np.linspace(np.amin(plotting_powin), np.amax(plotting_powin), 100)
        prediction = get_linear_pred(interp_powin, best_fit_m, best_fit_b)
        ax.plot(interp_powin, prediction, color = 'red', linewidth = 1.5, alpha = 0.5, label = f'y = {best_fit_m:0.2f} * x + {best_fit_b:0.2f}')

        ax.set_title(f'RF Chain Compression Test', fontsize = 20)
        ax.set_xlabel(f'Vpp In [uV]', fontsize = 20)
        ax.set_ylabel(f'Vpp Out [mV]', fontsize = 20)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.legend(fontsize=18, frameon=False)
        ax.grid(True, which='both', axis='both', linewidth=0.5, color='gray', alpha = 0.5)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/compression_test_powerin_vs_powerout.png', dpi=300)
        plt.close(fig)
        
        ### quick fit to the data
        fit_start = -95
        fit_end = -48
        fit_mask = np.logical_and(dbm_in >= fit_start, dbm_in <= fit_end)
        data = [dbm_in[fit_mask], dbm_out[fit_mask]]
        res = minimize(linear_fit_func, x0=[1.4, 50], args=data, tol = 1e-20, method='L-BFGS-B', bounds=[ (0,400), (-500,500)], options = {"maxiter":1e7})
        best_fit_m = res.x[0]
        best_fit_b = res.x[1]
        
        interp_powin = np.linspace(np.amin(dbm_in), np.amax(dbm_in), 100)
        prediction = get_linear_pred(interp_powin, best_fit_m, best_fit_b)
        ax_dbm.plot(interp_powin, prediction, color = 'red', linewidth = 1.5, alpha = 0.5, label = f'y = {best_fit_m:0.2f} * x + {best_fit_b:0.2f}')
        
        ax_dbm.set_title(f'RF Chain Compression Test', fontsize = 20)
        ax_dbm.set_xlabel(f'Power In [dBm]', fontsize = 20)
        ax_dbm.set_ylabel(f'Power Out [dBm]', fontsize = 20)
        ax_dbm.tick_params(axis='both', which='both', labelsize=20)
        ax_dbm.legend(fontsize=18, frameon=False)
        ax_dbm.grid(True, which='both', axis='both', linewidth=0.5, color='gray', alpha = 0.5)
        fig_dbm.tight_layout()
        fig_dbm.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/compression_test_powerin_vs_powerout_dbm.png', dpi=300)
        plt.close(fig_dbm)
        
        ### quick fit to the data
        fit_start = -95
        fit_end = -66.5
        fit_mask = np.logical_and(dbmsine_in >= fit_start, dbmsine_in <= fit_end)
        data = [dbmsine_in[fit_mask], dbmsine_out[fit_mask]]
        res = minimize(linear_fit_func, x0=[1.4, 50], args=data, tol = 1e-20, method='L-BFGS-B', bounds=[ (0,400), (-500,500)], options = {"maxiter":1e7})
        best_fit_m = res.x[0]
        best_fit_b = res.x[1]
        
        interp_powin = np.linspace(np.amin(dbmsine_in), np.amax(dbmsine_in), 100)
        prediction = get_linear_pred(interp_powin, best_fit_m, best_fit_b)
        ax_dbmsine.plot(interp_powin, prediction, color = 'red', linewidth = 1.5, alpha = 0.5, label = f'y = {best_fit_m:0.2f} * x + {best_fit_b:0.2f}')
        
        ax_dbmsine.set_title(f'RF Chain Compression Test (Assuming Sine Wave)', fontsize = 20)
        ax_dbmsine.set_xlabel(f'Power In [dBm]', fontsize = 20)
        ax_dbmsine.set_ylabel(f'Power Out [dBm]', fontsize = 20)
        ax_dbmsine.tick_params(axis='both', which='both', labelsize=20)
        ax_dbmsine.legend(fontsize=18, frameon=False)
        ax_dbmsine.grid(True, which='both', axis='both', linewidth=0.5, color='gray', alpha = 0.5)
        fig_dbmsine.tight_layout()
        fig_dbmsine.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/compression_test_powerin_vs_powerout_dbmsine.png', dpi=300)
        plt.close(fig_dbmsine)

if __name__ == "__main__":
    look_at_bts_events()

