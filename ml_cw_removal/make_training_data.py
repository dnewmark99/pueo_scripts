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
from scipy.optimize import curve_fit
import argparse

import pueoTALON.pueoPythonAnalyzer.eventSummary as summary
import pueoTALON.pueoPythonAnalyzer.waveform as waveform
import pueoTALON.pueoPythonAnalyzer.plotter as pltr
import pueoTALON.pueoPythonAnalyzer.antenna as antenna
import pueoTALON.pueoPythonAnalyzer.correlator as correlator
import antennaUtils.response as antResponse
import pueoTALON.pueoPythonAnalyzer.utils as utils
import pueoTALON.pueoPythonAnalyzer.geography as geography

ROOT.gSystem.Load("libpueoEvent.so")

def generate_cw(positions, n_samples, dt_ns, freq_GHz, amplitude, azimuth_deg, elevation_deg, phase,):
    ### consts
    c = 0.299792458
    t = np.arange(n_samples) * dt_ns

    ### get direction of source
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el),])

    ### get delays to each antenna
    delays = positions @ direction / c

    ### now make contious wave!
    cw = amplitude * np.cos(2 * np.pi * freq_GHz * (t[None, :] - delays[:, None]) + phase)

    return cw

def plot_csw(cohsum, contaminated_cohsum, times, this_run, this_ev):
    ### set up our plot
    fig, (ax, ax_freq) = plt.subplots(2, 1, figsize=(10, 12))
    ax.minorticks_on()
    ax_freq.minorticks_on()
    
    ### let's also get the fft
    f, vf = utils.fourierTransform(times, cohsum)
    power = utils.volt_to_dB(np.abs(vf))
    
    cont_f, cont_vf = utils.fourierTransform(times, contaminated_cohsum)
    cont_power = utils.volt_to_dB(np.abs(cont_vf))

    ### plot!
    ax.plot(times, cohsum, color = 'black', label = f'CSW')
    ax.plot(times, contaminated_cohsum, color = '#d62728', label = f'CSW + CW Contimination')
    ax_freq.plot(f, power, color = 'black', label = f'CSW FFT')
    ax_freq.plot(cont_f, cont_power, color = '#d62728', label = f'CSW + CW Contimination FFT')
    ax.set_title(f'CW Injection, Run = {this_run}, Ev = {this_ev}')
    ax.set_xlabel(f'Times [ns]', fontsize = 20)
    ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
    ax_freq.set_xlabel(f'Frequency [GHz]', fontsize = 20)
    ax_freq.set_ylabel(f'Gain [dB]', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax_freq.tick_params(axis='both', which='both', labelsize=20)
    ax.legend(fontsize=16, frameon=False)
    ax_freq.legend(fontsize=16, frameon=False)
    ax_freq.set_xlim(0,1.5)
    ax_freq.set_ylim(0,max(np.amax(power), np.amax(cont_power))*1.1)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/ml_cw_removal/cwinjection_{this_run}_e{this_ev}.png', dpi=300)
    plt.close(fig)
    
    print(f'plotted csw for run = {this_run} and ev = {this_ev}')

def test_cw_injection(n_ev = 2, poli = 'V'):
    ### let's grab some neutrino monte carlo runs, events
    path = f'/scratch/midway3/darcynewmark/useful/nonempty_neutrino_sim_oct2026_weightsandcurrrent.json'
    with open(path) as f:
        numc_dict = json.load(f)
    numc_runs = np.asarray(numc_dict['run'])
    numc_evs = np.asarray(numc_dict['entry'])
    numc_runs = [813, 783]
    numc_evs = [2372, 160133]
    total_numc_events = len(numc_evs)

    ### loop over a few events
    prev_run = None
    for n in range(n_ev):
        this_run = int(numc_runs[n])
        this_ev = int(numc_evs[n])
        
        ### check if this is a simulation set
        sim_flag = this_run > 180000

        ### check if we have a correlator object for this run
        if prev_run == None or prev_run != this_run:
            ### set up correlator + geography helper per run (to allow for handling of sim)
            corr = correlator.Correlator(run = this_run)
            prev_run = this_run
            
            ### make our masks to use later
            channel_names = np.asarray(corr.ChannelGeo.MI_ringphipol)
            mask_H = np.array([int(ch[:2]) <= 4 and int(ch[2:4]) <= 24 and ch[-1] == 'H' for ch in channel_names])
            mask_V = np.array([int(ch[:2]) <= 4 and int(ch[2:4]) <= 24 and ch[-1] == 'V' for ch in channel_names])

        ### load this event
        corr.ChannelGeo.load_event(this_ev)
        
        ### now reconstruct the directions and coherent sum
        skymap, peak_zen, peak_az, corr_vals, zoomedskymap, zoomedpeak_zen, zoomedpeak_az, zoomedcorr_vals, all_zoomedzen, all_zoomedaz = corr.reconstruct(poli, zoom_angle_range=7, N_peaks=1)
        main_peak_zen = zoomedpeak_zen[0][0]
        main_peak_el = 90 - main_peak_zen
        main_peak_az = zoomedpeak_az[0][0]

        ### get coherent sum
        _, _, _, cohsum, overlap_start, overlap_end = corr.coherent_sum(main_peak_zen, main_peak_az, polarization=poli, return_overlap_boundaries=True, return_all=True)
        cohsum = cohsum[overlap_start:overlap_end]
        times = np.arange(cohsum.size)*corr.dt

        ### now let's add a lil CW!
        event_wfs = np.asarray(corr.ChannelGeo.waveforms)
        n_samples = event_wfs.shape[-1]
        cw = generate_cw(positions=corr.ChannelGeo.Antennas.phase_centers, n_samples=n_samples, dt_ns=1/3., freq_GHz=1.2, amplitude=70, azimuth_deg=45, elevation_deg=20, phase=0)
        contaminated = event_wfs.copy()
        contaminated[mask_H] += cw[mask_H]
        contaminated[mask_V] += cw[mask_V]

        ### and make cohsum again!
        corr.ChannelGeo.waveforms_unfilt = contaminated
        _, _, _, contaminated_cohsum, overlap_start, overlap_end = corr.coherent_sum(main_peak_zen, main_peak_az, polarization=poli, return_overlap_boundaries=True, return_all=True)
        contaminated_cohsum = contaminated_cohsum[overlap_start:overlap_end]

        ### now plooot!
        plot_csw(cohsum, contaminated_cohsum, times, this_run, this_ev)

if __name__ == "__main__":
    test_cw_injection()
