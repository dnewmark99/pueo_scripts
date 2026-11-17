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

def mass_enclosed(data, fraction=0.68):
    ### normalize
    weights = np.clip(data, 0, None)
    weights /= np.sum(weights)

    ### get peak
    peak_idx = np.argmax(data)

    ### start from peak and expand outwards until sigma of total mass is enclosed
    lower_idx = peak_idx
    upper_idx = peak_idx
    enclosed_mass = weights[peak_idx]

    while enclosed_mass < fraction:

        left = weights[lower_idx - 1] if lower_idx > 0 else -1
        right = weights[upper_idx + 1] if upper_idx < len(data) - 1 else -1

        if left < 0 and right < 0:
            break

        if left >= right:
            lower_idx -= 1
            enclosed_mass += weights[lower_idx]
        else:
            upper_idx += 1
            enclosed_mass += weights[upper_idx]
    return lower_idx, upper_idx

def fwhm(data):
    ### get peak
    peak_idx = np.argmax(data)
    half_max = data[peak_idx] / 2

    ### find indices above half maximum
    above_half = np.where(data >= half_max)[0]

    lower_idx = above_half[0]
    upper_idx = above_half[-1]

    return lower_idx, upper_idx

def fourierTransform(time, voltage):
    freq=np.fft.rfftfreq(time.shape[-1],time[1]-time[0])
    amp_f= np.fft.rfft(voltage)
    voltage_ft= np.abs(amp_f)*np.exp(np.unwrap(np.angle(amp_f))*1j)
    return freq, voltage_ft

def volt_to_dB(voltage_ratio):
    dB = np.zeros_like(voltage_ratio)
    dB[np.abs(voltage_ratio) > 1e-10] = 20*np.log10(np.abs(voltage_ratio[np.abs(voltage_ratio) > 1e-10]))
    return dB

def testing(poli = 'V', trigger_window_start_time = 30, trigger_window_end_time = 80):
    ### ok for testing, let's take some runs, evs that we know have very large CW contaimination and try to tage
    ### also important to compare to some nice TD ground cal to make sure we don't remove good events
    ### let's try calculating stokes paramter, max power, max / median power, "smoothness" parameter

    good_runs = [783, 783, 783, 783, 783, 802, 802, 803, 803, 803]
    good_evs = [160133, 199655, 202171, 214699, 223807, 57371, 64679, 191541, 234998, 63639]

    bad_runs = [783, 783, 783, 783, 783, 784, 783, 783, 783, 783]
    bad_evs = [11252, 12173, 13143, 211505, 312928, 88382, 182967, 258, 283, 288]
    bad_runs = []
    bad_evs = []
    
    all_runs = good_runs + bad_runs
    all_evs = good_evs + bad_evs

    #all_runs = [813,813,813,813,813,813,813,813,813,813,813]
    #all_evs = [0,1,2,3,4,5,6,7,8,9,10]
    all_runs = [783, 813]
    all_evs = [199655, 4]

    opp_poli = 'H' if poli == 'V' else 'V'

    prev_run = None
    for r in range(len(all_runs)):
        this_run = int(all_runs[r])
        this_ev = int(all_evs[r])

        ### check if we have a correlator object for this run
        if prev_run == None or prev_run != this_run:
            ### set up correlator + geography helper per run (to allow for handling of sim)
            corr = correlator.Correlator(run = this_run)
            geography_helper = geography.GeographyHelper(root_file=os.environ['PUEO_ANALYSIS_TOOLS'] + "/pueoTALON/lookup/abxtwo.root", root_source="ABX-Two")
            prev_run = this_run

        ### load this event
        corr.ChannelGeo.load_event(this_ev)

        ### update geography helper
        event_time = corr.ChannelGeo.triggerTime()
        geography_helper.updatePosition(event_time = event_time, max_time_delta = 3)

        ### set up our plot
        fig, (ax, ax_freq, ax_stokes) = plt.subplots(3, 1, figsize=(10, 14))
        ax.minorticks_on()
        ax_freq.minorticks_on()
        ax_stokes.minorticks_on()

        ### now reconstruct the directions and coherent sum
        skymap, peak_zen, peak_az, corr_vals, zoomedskymap, zoomedpeak_zen, zoomedpeak_az, zoomedcorr_vals, all_zoomedzen, all_zoomedaz = corr.reconstruct(poli, zoom_angle_range=7, N_peaks=1, isolate_MUOS=True)
        main_peak_zen = zoomedpeak_zen[0][0]
        main_peak_el = 90 - main_peak_zen
        main_peak_az = zoomedpeak_az[0][0]

        ### get coherent sum
        cohsum, overlap_start, overlap_end = corr.coherent_sum(main_peak_zen, main_peak_az, polarization=poli, return_overlap_boundaries=True,
                                                                isolate_MUOS=True, )
        opp_cohsum, opp_overlap_start, opp_overlap_end = corr.coherent_sum(main_peak_zen, main_peak_az, polarization=opp_poli, return_overlap_boundaries=True,
                                                                            isolate_MUOS=True, )
        cohsum_snr = utils.SNR(cohsum, dynamic_rms_window = True, valid_window = (overlap_start, overlap_end))
        opp_cohsum_snr = utils.SNR(opp_cohsum, dynamic_rms_window = True, valid_window = (opp_overlap_start, opp_overlap_end))
        cohsum_start = max(overlap_start, opp_overlap_start)
        cohsum_end = min(overlap_end, opp_overlap_end)
        cohsum = cohsum[cohsum_start:cohsum_end]
        opp_cohsum = opp_cohsum[cohsum_start:cohsum_end]
        times = np.arange(cohsum.size)*corr.dt

        ### now stokes!
        if poli == 'V':
            I, Q, U, V = utils.stokes_parameters_time(opp_cohsum, cohsum)
        else:
            I, Q, U, V = utils.stokes_parameters_time(cohsum, opp_cohsum)

        ### get average linear and circular polarization fraction
        time_windows = [(0, len(cohsum)-1), ### whole wf
                        (int(trigger_window_start_time/corr.dt), int(trigger_window_end_time/corr.dt)), ### around trigger window
                        (np.argmax(I)-50, np.argmax(I)+100), ### close to peak
                        (np.argmax(I)-100, np.argmax(I)+200) ### close to peak
                       ]
        time_window_colors = ['#d62728', '#21b2aa', '#c71585', 'orange']
        time_window_descrip = ['Entire', 'Trigger', 'Smallest', 'SecondSmallest']

        for t in range(len(time_windows)):
            ttws, ttwe = time_windows[t]

            ### quick bounds check
            this_time_window_start = max(0, ttws)
            this_time_window_end = min(len(cohsum)-1, ttwe)

            ### now let's calculate stokes things!
            avg_I = np.mean(I[this_time_window_start:this_time_window_end])
            avg_Q = np.mean(Q[this_time_window_start:this_time_window_end])
            avg_U = np.mean(U[this_time_window_start:this_time_window_end])
            avg_V = np.mean(V[this_time_window_start:this_time_window_end])

            p_l = np.sqrt(avg_Q**2 + avg_U**2) / avg_I
            p_c = np.abs(avg_V) / avg_I

            ### plot
            start_time = this_time_window_start * corr.dt
            end_time = this_time_window_end * corr.dt
            print(f'start_time = {start_time}, end_time = {end_time}, and total dt = {end_time-start_time} for corr.dt = {corr.dt}')
            ax_stokes.axvline(start_time, color = time_window_colors[t], linestyle = '--', label = f'P_lin = {p_l:0.2f}, P_circ = {p_c:0.2f}, ({time_window_descrip[t]})')
            ax_stokes.axvline(end_time, color = time_window_colors[t], linestyle = '--')

        ### let's also get the fft
        f, vf = fourierTransform(times, cohsum)
        opp_f, opp_vf = fourierTransform(times, opp_cohsum)
        dt = times[1] - times[0]
        power = volt_to_dB(np.abs(vf*dt))
        opp_power = volt_to_dB(np.abs(opp_vf*dt))

        ### get max and median power
        max_power = np.amax(power)
        median_power = np.median(power[power>0])

        ### let's also get a "rough-ness" parameter
        first_order_roughness = np.sum(np.diff(power[power>0])**2)
        second_order_roughness = np.sum(np.diff(power[power>0], n=2)**2)

        ### plot!
        ax.plot(times, cohsum, color = 'black', label = f'CSW {poli}-pol, SNR = {cohsum_snr:0.2f}')
        ax.plot(times, opp_cohsum, color = '#d62728', alpha = 0.7, label = f'CSW {opp_poli}-pol, SNR = {opp_cohsum_snr:0.2f}')
        ax_freq.plot(f, power, color = 'black', label = f'CSW FFT {poli}-pol')
        ax_freq.plot(f, opp_power, color ='#d62728', label = f'CSW FFT {opp_poli}-pol')
        #ax_freq.plot(f, power, color = 'black', label = f'CSW FFT {poli}-pol, Max/Med = {max_power/median_power:0.2f}')
        #ax_freq.axhline(max_power, color = '#d62728', alpha = 0.5, label = f'Max Power = {max_power:0.2f}')
        #ax_freq.axhline(median_power, color = '#1e90ff', alpha = 0.5, label = f'Median Power = {median_power:0.2f}')
        #ax_freq.text(0.05, max_power*0.9, f'R1 = {first_order_roughness:0.2f}', fontsize = 16)
        #ax_freq.text(0.05, max_power*0.7, f'R2 = {second_order_roughness:0.2f}', fontsize = 16)
        ax_stokes.plot(times, I, color = 'black', label = f'I')
        ax_stokes.plot(times, Q, color = '#21b2aa', label = 'Q')
        ax_stokes.plot(times, U, color = '#c71585', label = 'U')
        ax_stokes.plot(times, V, color = 'orange', label = 'V')
        ax.set_title(f'Run = {this_run}, Ev = {this_ev}')
        ax.set_xlabel(f'Times [ns]', fontsize = 20)
        ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
        ax_freq.set_xlabel(f'Frequency [GHz]', fontsize = 20)
        ax_freq.set_ylabel(f'Gain [dB]', fontsize = 20)
        ax_stokes.set_xlabel(f'Times [ns]', fontsize = 20)
        ax_stokes.set_ylabel(f'Stokes Params [mV^2]', fontsize = 20)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax_freq.tick_params(axis='both', which='both', labelsize=20)
        ax_stokes.tick_params(axis='both', which='both', labelsize=20)
        ax.legend(fontsize=16, frameon=False)
        ax_freq.legend(fontsize=16, frameon=False)
        ax_freq.set_xlim(0,1.5)
        ax_freq.set_ylim(0,np.amax(volt_to_dB(np.abs(vf*dt)))*1.1)
        ax_stokes.legend(fontsize=16, frameon=False, loc = 'upper right')
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/satellites/tagcwexamplesmuosonly_{this_run}_e{this_ev}.png', dpi=300)
        plt.close(fig)

        print(f'plotted csw for run = {this_run} and ev = {this_ev}')

if __name__ == "__main__":
    testing()
