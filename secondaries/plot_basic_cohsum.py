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

import pueoTALON.pueoPythonAnalyzer.waveform as waveform
import pueoTALON.pueoPythonAnalyzer.plotter as pltr
import pueoTALON.pueoPythonAnalyzer.antenna as antenna
import pueoTALON.pueoPythonAnalyzer.correlator as correlator
import antennaUtils.response as antResponse
import pueoTALON.pueoPythonAnalyzer.utils as utils
import pueoTALON.pueoPythonAnalyzer.geography as geography

ROOT.gSystem.Load("libpueoEvent.so")

def fourierTransform(time, voltage):
    freq=np.fft.rfftfreq(time.shape[-1],time[1]-time[0])
    amp_f= np.fft.rfft(voltage)
    voltage_ft= np.abs(amp_f)*np.exp(np.unwrap(np.angle(amp_f))*1j)
    return freq, voltage_ft

def volt_to_dB(voltage_ratio):
    dB = np.zeros_like(voltage_ratio)
    dB[np.abs(voltage_ratio) > 1e-10] = 20*np.log10(np.abs(voltage_ratio[np.abs(voltage_ratio) > 1e-10]))
    return dB

def plot_wfs(source_type = 'TD', n_events = 20):
    ### get a few runs, evs to plot
    if source_type == 'TD':
        groundcal_events = np.genfromtxt(os.environ['PUEO_ANALYSIS_TOOLS'] + f'/pueoTALON/lookup/my{source_type}_list.csv', delimiter=',', names=True, dtype=None)
        groundcal_runs = np.asarray(groundcal_events['run'])
        groundcal_evs = np.asarray(groundcal_events['entry_number'])
        runs = groundcal_runs[:n_events]
        evs = groundcal_evs[:n_events]
    if source_type == 'Force Triggers':
        path = f'/scratch/midway3/darcynewmark/useful/nonrftriggers_run783_run808.json'
        with open(path) as f:
            force_dict = json.load(f)
        force_runs = np.asarray(force_dict['run'])
        force_evs = np.asarray(force_dict['entry'])
        runs = force_runs[:n_events]
        evs = force_evs[:n_events]
    if source_type == 'NuMC':
        path = f'/scratch/midway3/darcynewmark/useful/nonempty_neutrino_sim_oct2026.json'
        with open(path) as f:
            numc_dict = json.load(f)
        numc_runs = np.asarray(numc_dict['run'])
        numc_evs = np.asarray(numc_dict['entry'])
        numc_pdg = np.asarray(numc_dict['nu_pdg'])
        runs = numc_runs[:n_events]
        evs = numc_evs[:n_events]
        pdgs = numc_pdg[:n_events]

    prev_run = None
    polarizations = ['V', 'H']
    colors = ['black', '#D55E00']
    for r in range(len(runs)):
        this_run = int(runs[r])
        this_ev = int(evs[r])

        ### check if this is a simulation set
        sim_flag = this_run > 180000

        ### check if we have a correlator object for this run
        if prev_run == None or prev_run != this_run:
            ### set up correlator + geography helper per run (to allow for handling of sim)
            corr = correlator.Correlator(run = this_run, sim = sim_flag)
            geography_helper = geography.GeographyHelper(root_file=os.environ['PUEO_ANALYSIS_TOOLS'] + "/pueoTALON/lookup/abxtwo.root", root_source="ABX-Two")
            prev_run = this_run

        ### load this event
        corr.ChannelGeo.load_event(this_ev)

        ### update geography helper
        event_time = corr.ChannelGeo.triggerTime()
        geography_helper.updatePosition(event_time, max_time_delta = 3)

        ### get some flags
        this_event_summary = ROOT.PueoEventSummary()
        qual = corr.ChannelGeo.qualityCheck(this_event_summary)
        this_blast_flag = this_event_summary.flags.isPayloadBlast
        this_sat_flag = this_event_summary.flags.isSaturated
        this_deadtime_flag = this_event_summary.flags.isInDeadtime

        if (this_blast_flag or this_sat_flag or this_deadtime_flag):
            print(f'oops! skipping this event! this_blast_flag = {this_blast_flag}, this_sat_flag = {this_sat_flag}, this_deadtime_flag = {this_deadtime_flag}')
            continue

        ### set up our plot
        fig, (ax, ax_freq) = plt.subplots(2, 1, figsize=(10, 12))
        ax.minorticks_on()
        ax_freq.minorticks_on()

        ### loop over each polarization
        max_v = []
        for idx, poli in enumerate(polarizations):
            ### now reconstruct the directions and coherent sum
            skymap, peak_zen, peak_az, corr_vals, zoomedskymap, zoomedpeak_zen, zoomedpeak_az, zoomedcorr_vals, all_zoomedzen, all_zoomedaz = corr.reconstruct(poli, zoom_angle_range=7, N_peaks=1)
            main_peak_zen = zoomedpeak_zen[0][0]
            main_peak_el = 90 - main_peak_zen
            main_peak_az = zoomedpeak_az[0][0]

            ### get coherent sum
            cohsum, overlap_start, overlap_end = corr.coherent_sum(main_peak_zen, main_peak_az, polarization=poli, return_overlap_boundaries=True)
            cohsum_snr = utils.SNR(cohsum, dynamic_rms_window = True, valid_window = (overlap_start, overlap_end))
            cohsum = cohsum[overlap_start:overlap_end]
            times = np.arange(cohsum.size)*corr.dt
            
            ### let's also get the fft
            f, vf = fourierTransform(times, cohsum)
            dt = times[1] - times[0]

            ### plot!
            ax.plot(times, cohsum, color = colors[idx], label = f'CSW {poli}-pol')
            ax_freq.plot(f, volt_to_dB(np.abs(vf*dt)), color = colors[idx], label = f'CSW FFT {poli}-pol')
            max_v.append(np.amax(volt_to_dB(np.abs(vf*dt))))

        if source_type == 'NuMC':
            ax.set_title(f'{source_type}, Run {this_run}, Event {this_ev} (PDG = {pdgs[r]})', fontsize = 20)
        else:
            ax.set_title(f'{source_type}, Run {this_run}, Event {this_ev}', fontsize = 20)
        ax.set_xlabel(f'Times [ns]', fontsize = 20)
        ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
        ax_freq.set_xlabel(f'Frequency [GHz]', fontsize = 20)
        ax_freq.set_ylabel(f'Gain [dB]', fontsize = 20)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax_freq.tick_params(axis='both', which='both', labelsize=20)
        ax.legend(fontsize=16, frameon=False)
        ax_freq.legend(fontsize=16, frameon=False)
        ax_freq.set_xlim(0,1.5)
        ax_freq.set_ylim(0,np.amax(max_v)*1.1)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/secondaries/ground_cal_csw/basic_csw_run{this_run}_e{this_ev}.png', dpi=300)
        plt.close(fig)
        
        print(f'plotted csw for run = {this_run} and ev = {this_ev}')

if __name__ == "__main__":
    plot_wfs('NuMC')
    plot_wfs('TD')
    plot_wfs('Force Triggers')
