import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import ROOT
import os
import json
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import glob
import functools
from functools import reduce
from examine_stokes import grab_data_json

import pueoTALON.pueoPythonAnalyzer.waveform as waveform
import pueoTALON.pueoPythonAnalyzer.plotter as pltr
import pueoTALON.pueoPythonAnalyzer.antenna as antenna
import pueoTALON.pueoPythonAnalyzer.correlator as correlator
import antennaUtils.response as antResponse
import pueoTALON.pueoPythonAnalyzer.utils as utils
import pueoTALON.pueoPythonAnalyzer.geography as geography
import pueoTALON.pueoPythonAnalyzer.plotter as pltr

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

def plot_wfs(this_run, this_ev, color, title, outlier_value, outlier_name, trigger_window_start_time = 30, trigger_window_end_time = 80, poli = 'V', opp_poli = 'H'):
    no_spaces = title.replace(" ", "")
    
    ### check if this is a simulation set
    sim_flag = this_run > 180000
    
    ### update trigger window for simulation
    if sim_flag:
        trigger_window_start_time = 60
        trigger_window_end_time = 110

    ### make our correlator
    corr = correlator.Correlator(run = this_run, sim = sim_flag)
    geography_helper = geography.GeographyHelper(root_file=os.environ['PUEO_ANALYSIS_TOOLS'] + "/pueoTALON/lookup/abxtwo.root", root_source="ABX-Two")

    ### load this event
    corr.ChannelGeo.load_event(this_ev)

    ### update geography helper
    event_time = corr.ChannelGeo.triggerTime()
    geography_helper.updatePosition(event_time=event_time, max_time_delta = 3)

    ### set up our plot
    fig, (ax, ax_freq) = plt.subplots(2, 1, figsize=(10, 12))
    ax.minorticks_on()
    ax_freq.minorticks_on()

    ### now reconstruct the directions and coherent sum
    skymap, peak_zen, peak_az, corr_vals, zoomedskymap, zoomedpeak_zen, zoomedpeak_az, zoomedcorr_vals, all_zoomedzen, all_zoomedaz = corr.reconstruct(poli, zoom_angle_range=7, N_peaks=1)
    main_peak_zen = zoomedpeak_zen[0][0]
    main_peak_el = 90 - main_peak_zen
    main_peak_az = zoomedpeak_az[0][0]
    
    ### get coherent sum
    cohsum, overlap_start, overlap_end = corr.coherent_sum(main_peak_zen, main_peak_az, polarization=poli, return_overlap_boundaries=True, )
    opp_cohsum, opp_overlap_start, opp_overlap_end = corr.coherent_sum(main_peak_zen, main_peak_az, polarization=opp_poli, return_overlap_boundaries=True, )
    cohsum_snr = utils.SNR(cohsum, dynamic_rms_window = True, valid_window = (overlap_start, overlap_end))
    opp_cohsum_snr = utils.SNR(opp_cohsum, dynamic_rms_window = True, valid_window = (opp_overlap_start, opp_overlap_end))
    cohsum_start = max(overlap_start, opp_overlap_start)
    cohsum_end = min(overlap_end, opp_overlap_end)
    cohsum = cohsum[cohsum_start:cohsum_end]
    opp_cohsum = opp_cohsum[cohsum_start:cohsum_end]
    times = np.arange(cohsum.size)*corr.dt

    ### let's also get the fft
    f, vf = fourierTransform(times, cohsum)
    opp_f, opp_vf = fourierTransform(times, opp_cohsum)
    dt = times[1] - times[0]
    power = volt_to_dB(np.abs(vf*dt))
    opp_power = volt_to_dB(np.abs(opp_vf*dt))

    ### plot!
    ax.plot(times, cohsum, color = color, label = f'CSW {poli}-pol, SNR = {cohsum_snr:0.2f}')
    ax.plot(times, opp_cohsum, color = 'orange', alpha = 0.5, zorder = 10, label = f'CSW {opp_poli}-pol, SNR = {opp_cohsum_snr:0.2f}')
    ax.axvline(trigger_window_start_time, color = 'black', linestyle = '--', alpha = 0.5, label = f'{outlier_name} = {outlier_value:0.2f}')
    ax.axvline(trigger_window_end_time, color = 'black', linestyle = '--', alpha = 0.5,)
    ax_freq.plot(f, power, color = color, label = f'CSW FFT {poli}-pol')
    ax_freq.plot(f, opp_power, color = 'orange', alpha = 0.5, zorder = 10, label = f'CSW FFT {opp_poli}-pol')
    ax.set_title(f'{title}, Run = {this_run}, Ev = {this_ev}')
    ax.set_xlabel(f'Times [ns]', fontsize = 20)
    ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
    ax_freq.set_xlabel(f'Frequency [GHz]', fontsize = 20)
    ax_freq.set_ylabel(f'Gain [dB]', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax_freq.tick_params(axis='both', which='both', labelsize=20)
    ax.legend(fontsize=16, frameon=False)
    ax_freq.legend(fontsize=16, frameon=False)
    ax_freq.set_xlim(0,1.5)
    ax_freq.set_ylim(0,np.amax(volt_to_dB(np.abs(vf*dt)))*1.1)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/satellites/outlierplot_{no_spaces}_{this_run}_e{this_ev}.png', dpi=300)
    plt.close(fig)
    
    print(f'plotted csw for run = {this_run} and ev = {this_ev}')

def plot_outliers(n_events = 5):
    ### let's read in our json data
    data = grab_data_json()
    
    ### unpack it all
    run = np.asarray(data['run'])
    entry = np.asarray(data['entry'])
    flag = np.asarray(data['flag'])
    groundcal_pol = np.asarray(data['groundcal_pol'])
    numc_pdg = np.asarray(data['numc_pdg'])
    numc_weight = np.asarray(data['numc_weight'])
    numc_interaction_current = np.asarray(data['numc_interaction_current'])
    trigger_time = np.asarray(data['trigger_time'])
    trigger_subsecond = np.asarray(data['trigger_subsecond'])
    saturation_flag = np.asarray(data['saturation_flag'])
    blast_flag = np.asarray(data['blast_flag'])
    deadtime_flag = np.asarray(data['deadtime_flag'])
    low_channel_rms_flag = np.asarray(data['low_channel_rms_flag'])
    avg_lin_vpol = np.asarray(data['avg_lin_vpol'])
    avg_circ_vpol = np.asarray(data['avg_circ_vpol'])
    avg_pol_vpol = np.asarray(data['avg_pol_vpol'])
    avg_angle_vpol = np.asarray(data['avg_angle_vpol'])
    isolate_muos_avg_lin_vpol = np.asarray(data['isolate_muos_avg_lin_vpol'])
    isolate_muos_avg_circ_vpol = np.asarray(data['isolate_muos_avg_circ_vpol'])
    isolate_muos_avg_pol_vpol = np.asarray(data['isolate_muos_avg_pol_vpol'])
    isolate_muos_avg_angle_vpol = np.asarray(data['isolate_muos_avg_angle_vpol'])
    avg_lin_hpol = np.asarray(data['avg_lin_hpol'])
    avg_circ_hpol = np.asarray(data['avg_circ_hpol'])
    avg_pol_hpol = np.asarray(data['avg_pol_hpol'])
    avg_angle_hpol = np.asarray(data['avg_angle_hpol'])
    isolate_muos_avg_lin_hpol = np.asarray(data['isolate_muos_avg_lin_hpol'])
    isolate_muos_avg_circ_hpol = np.asarray(data['isolate_muos_avg_circ_hpol'])
    isolate_muos_avg_pol_hpol = np.asarray(data['isolate_muos_avg_pol_hpol'])
    isolate_muos_avg_angle_hpol = np.asarray(data['isolate_muos_avg_angle_hpol'])
    csw_snr_vpol = np.asarray(data['csw_snr_vpol'])
    csw_snr_vpol = csw_snr_vpol.reshape(-1, 2)[:, 1]
    csw_snr_hpol = np.asarray(data['csw_snr_hpol'])
    csw_snr_hpol = csw_snr_hpol.reshape(-1, 2)[:, 1]
  
    ### add basic saturation, deadtime, and blast mask
    valid_mask = functools.reduce(np.logical_and, [saturation_flag == False, blast_flag == False, deadtime_flag == False, low_channel_rms_flag == False, ])
    print(f'basic cuts on saturation, deadtime, blast, and low thermal remove {len(saturation_flag) - np.sum(valid_mask)} events!')
    print(f'left with {np.sum(valid_mask)} events after basic cuts!')
    run = run[valid_mask]
    entry = entry[valid_mask]
    flag = flag[valid_mask]
    groundcal_pol = groundcal_pol[valid_mask]
    numc_pdg = numc_pdg[valid_mask]
    numc_weight = numc_weight[valid_mask]
    numc_interaction_current = numc_interaction_current[valid_mask]
    trigger_time = trigger_time[valid_mask]
    trigger_subsecond = trigger_subsecond[valid_mask]
    saturation_flag = saturation_flag[valid_mask]
    blast_flag = blast_flag[valid_mask]
    deadtime_flag = deadtime_flag[valid_mask]
    low_channel_rms_flag = low_channel_rms_flag[valid_mask]
    avg_lin_vpol = avg_lin_vpol[valid_mask]
    avg_circ_vpol = avg_circ_vpol[valid_mask]
    avg_pol_vpol = avg_pol_vpol[valid_mask]
    avg_angle_vpol = avg_angle_vpol[valid_mask]
    isolate_muos_avg_lin_vpol = isolate_muos_avg_lin_vpol[valid_mask]
    isolate_muos_avg_circ_vpol = isolate_muos_avg_circ_vpol[valid_mask]
    isolate_muos_avg_pol_vpol = isolate_muos_avg_pol_vpol[valid_mask]
    isolate_muos_avg_angle_vpol = isolate_muos_avg_angle_vpol[valid_mask]
    avg_lin_hpol = avg_lin_hpol[valid_mask]
    avg_circ_hpol = avg_circ_hpol[valid_mask]
    avg_pol_hpol = avg_pol_hpol[valid_mask]
    avg_angle_hpol = avg_angle_hpol[valid_mask]
    isolate_muos_avg_lin_hpol = isolate_muos_avg_lin_hpol[valid_mask]
    isolate_muos_avg_circ_hpol = isolate_muos_avg_circ_hpol[valid_mask]
    isolate_muos_avg_pol_hpol = isolate_muos_avg_pol_hpol[valid_mask]
    isolate_muos_avg_angle_hpol = isolate_muos_avg_angle_hpol[valid_mask]
    csw_snr_vpol = csw_snr_vpol[valid_mask]
    csw_snr_hpol = csw_snr_hpol[valid_mask]
    
    ### mask our events for now
    groundcal_vpol_mask = np.logical_and(flag == 'TD', groundcal_pol == 'V')
    rf_mask = flag == 'RF'
    nu_mask = flag == 'NuMC'
    print(f'have {np.sum(groundcal_vpol_mask)} ground cal v-pol events, {np.sum(rf_mask)} rf triggers, and {np.sum(nu_mask)} simulated neutrinos!')

    ### now let's grab some events in different populations and plot!
    all_data_sample_masks = [rf_mask, groundcal_vpol_mask, nu_mask]
    colors = ['black', '#d62728', '#1e90ff']
    labels = ['RF Triggers', 'TD Ground Cal', 'NuMC']
    
    variables = [avg_lin_vpol[:,1]]
    var_names = ['Avg Lin Pol']
    sub_pops = [[0.05, 0.3, 0.5, 0.9]]
    eps = [[0.05, 0.1, 0.1, 0.1]]

    ### loop over each variable
    for v in range(len(variables)):
        this_var = variables[v]
        this_pops = sub_pops[v]
        this_eps = eps[v]

        ### loop over each population we want to investigate
        for p in range(len(this_pops)):

            ### now loop over each dataset
            for d in range(len(all_data_sample_masks)):
                this_mask = all_data_sample_masks[d]
                this_data = this_var[this_mask]
                this_runs = run[this_mask]
                this_evs = entry[this_mask]

                ### pick out some runs, evs in our outlier populations
                outlier_mask = np.abs(this_data - this_pops[p]) < this_eps[p]
                n_outliers = np.sum(outlier_mask)

                print(f'for {labels[d]}, have {n_outliers} outliers in {var_names[v]} = {this_pops[p]} +- {this_eps[p]}')
                if n_outliers >= n_events:
                    #### now let's plot!
                    outlier_runs = this_runs[outlier_mask]
                    outlier_evs = this_evs[outlier_mask]
                    outlier_data = this_data[outlier_mask]

                    for r in range(n_events):
                        plot_wfs(int(outlier_runs[r]), int(outlier_evs[r]), colors[d], labels[d], outlier_data[r], var_names[v])

def plot_outliers_by_run_ev():
    ### let's read in our json data
    data = grab_data_json()
    
    ### unpack it all
    run = np.asarray(data['run'])
    entry = np.asarray(data['entry'])
    avg_lin_vpol = np.asarray(data['avg_lin_vpol'])
    avg_lin_vpol_trigger_window = avg_lin_vpol[:,1]
    csw_snr_vpol = np.asarray(data['csw_snr_vpol'])
    csw_snr_vpol = csw_snr_vpol.reshape(-1, 2)[:, 1]
  
    ### now select events based in our desired run, ev lists!
    #outlier_runs = [783, 783, 783, 783, 783, 783, 783, 783, 783, 783] ### big snr, small linear pol
    #outlier_evs = [155367, 156673, 158519, 158594, 158668, 160133, 160524, 160717, 162585, 163182]
    outlier_runs = [793, 793, 793, 793, 793, 793, 793, 793, 793, 793] ### big snr, big linear pol
    outlier_evs = [43362, 60754, 62360, 66009, 69210, 72109, 74045, 74503, 75931, 78540]

    for r in range(len(outlier_runs)):
        this_outlier_run = outlier_runs[r]
        this_outlier_ev = outlier_evs[r]
        idx = np.where((run == this_outlier_run) & (entry == this_outlier_ev))[0]
        this_outlier_val = avg_lin_vpol_trigger_window[idx[0]]
        
        ### plot!
        plot_wfs(int(this_outlier_run), int(this_outlier_ev), '#d62728', 'TD Ground Cal', this_outlier_val, 'Avg Lin Pol')

if __name__ == "__main__":
    #plot_outliers()
    plot_outliers_by_run_ev()

