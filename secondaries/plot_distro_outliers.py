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

def grab_data_json(base_path = '/scratch/midway3/darcynewmark/secondaries/multipulse_metrics/jsons'):

    data = {}
    path = f'{base_path}/secondary_metrics_*.json'
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

def plot_wfs(this_run, this_ev, color, title, outlier_value, outlier_time, outlier_name, poli = 'V'):
    no_spaces = title.replace(" ", "")
    
    ### check if this is a simulation set
    sim_flag = this_run > 180000

    ### make our correlator
    corr = correlator.Correlator(run = this_run, sim = sim_flag)
    geography_helper = geography.GeographyHelper(root_file=os.environ['PUEO_ANALYSIS_TOOLS'] + "/pueoTALON/lookup/abxtwo.root", root_source="ABX-Two")

    ### load this event
    corr.ChannelGeo.load_event(this_ev)

    ### update geography helper
    event_time = corr.ChannelGeo.triggerTime()
    geography_helper.updatePosition(event_time=event_time, max_time_delta = 3)
    expected_sun_el_az = geography_helper.get_sun_position(event_time)
    expected_td_el_az = geography_helper.get_expected_source_location('taylor_dome')
    expected_ldb_el_az = geography_helper.get_expected_source_location('ldb')

    ### set up our plot
    fig, (ax, ax_freq) = plt.subplots(2, 1, figsize=(10, 12))
    ax.minorticks_on()
    ax_freq.minorticks_on()

    ### now reconstruct the directions and coherent sum
    skymap, peak_zen, peak_az, corr_vals, zoomedskymap, zoomedpeak_zen, zoomedpeak_az, zoomedcorr_vals, all_zoomedzen, all_zoomedaz = corr.reconstruct(poli, zoom_angle_range=7, N_peaks=1)
    main_peak_zen = zoomedpeak_zen[0][0]
    main_peak_el = 90 - main_peak_zen
    main_peak_az = zoomedpeak_az[0][0]
    
    ### plot our skymap
    pltr.plot_skymap(skymap, [peak_zen, peak_az], corr.az, corr.el, title=f'Main Peak Skymap, Az = {peak_az[0]:0.2f}, El = {90-peak_zen[0]:0.2f}',
                     filename = f'/scratch/midway3/darcynewmark/secondaries/multipulse_metrics/outlierskymap_{no_spaces}_mainpeak_{this_run}_e{this_ev}',
                     locations=[(expected_td_el_az[1], expected_td_el_az[0], 'Taylor Dome'), (expected_ldb_el_az[1], expected_ldb_el_az[0], 'LDB'), (expected_sun_el_az[1], expected_sun_el_az[0], 'Sun')])
    
    ### let's do the same for our secondary peak!
    original_dt = 1/3.0
    padding_samples = 200
    secondary_peak_window = [150, int(outlier_time / original_dt) + padding_samples]
    secondary_skymap, secondary_peak_zen, secondary_peak_az, _, _, _, _, _, _, _ = corr.reconstruct(poli, window=secondary_peak_window, zoom_angle_range=7, N_peaks=1)
    pltr.plot_skymap(secondary_skymap, [secondary_peak_zen, secondary_peak_az], corr.az, corr.el, title=f'Secondary Peak Skymap, Az = {secondary_peak_az[0]:0.2f}, El = {90-secondary_peak_zen[0]:0.2f}',
                     filename = f'/scratch/midway3/darcynewmark/secondaries/multipulse_metrics/outlierskymap_{no_spaces}_secondarypeak_{this_run}_e{this_ev}',
                     locations=[(expected_td_el_az[1], expected_td_el_az[0], 'Taylor Dome'), (expected_ldb_el_az[1], expected_ldb_el_az[0], 'LDB'), (expected_sun_el_az[1], expected_sun_el_az[0], 'Sun')])

    ### get coherent sum
    cohsum, overlap_start, overlap_end = corr.coherent_sum(main_peak_zen, main_peak_az, polarization=poli, return_overlap_boundaries=True)
    cohsum_snr = utils.SNR(cohsum, dynamic_rms_window = True, valid_window = (overlap_start, overlap_end))
    cohsum = cohsum[overlap_start:overlap_end]
    times = np.arange(cohsum.size)*corr.dt
    
    ### let's also get the fft
    f, vf = fourierTransform(times, cohsum)
    dt = times[1] - times[0]
    power = volt_to_dB(np.abs(vf*dt))

    ### plot!
    ax.plot(times, cohsum, color = color, label = f'CSW {poli}-pol')
    ax.axvline(outlier_time, color = 'black', linestyle = '--', alpha = 0.3, label = f'{outlier_name} = {outlier_value:0.2f} at t = {outlier_time:0.2f} ns')
    ax_freq.plot(f, power, color = color, label = f'CSW FFT {poli}-pol')
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
    fig.savefig(f'/scratch/midway3/darcynewmark/secondaries/multipulse_metrics/outlierplotv3_{no_spaces}_{this_run}_e{this_ev}.png', dpi=300)
    plt.close(fig)
    
    print(f'plotted csw for run = {this_run} and ev = {this_ev}')

def plot_outliers(n_events = 15):
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
    trigger_window_max_he = np.asarray(data['trigger_window_max_he'])
    trigger_window_max_time = np.asarray(data['trigger_window_max_time'])
    dist_mcmurdo = np.asarray(data['dist_mcmurdo'])
    csw_snr_vpol = np.asarray(data['csw_snr_vpol'])
    csw_rms_restrictedrange_vpol = np.asarray(data['csw_rms_restrictedrange_vpol'])
    csw_rms_fullrange_vpol = np.asarray(data['csw_rms_fullrange_vpol'])
    reco_az_vpol = np.asarray(data['reco_az_vpol'])
    reco_el_vpol = np.asarray(data['reco_el_vpol'])
    csw_snr_hpol = np.asarray(data['csw_snr_hpol'])
    csw_rms_restrictedrange_hpol = np.asarray(data['csw_rms_restrictedrange_hpol'])
    csw_rms_fullrange_hpol = np.asarray(data['csw_rms_fullrange_hpol'])
    reco_az_hpol = np.asarray(data['reco_az_hpol'])
    reco_el_hpol = np.asarray(data['reco_el_hpol'])
    max_freq_fraction_vpol = np.asarray(data['max_freq_fraction_vpol'])
    max_freq_fraction_hpol = np.asarray(data['max_freq_fraction_hpol'])
    csw_max_corr_vpol = np.asarray(data['csw_max_corr_vpol'])
    csw_max_lag_ns_vpol = np.asarray(data['csw_max_lag_ns_vpol'])
    csw_max_corr_top3peaks_az_vpol = np.asarray(data['csw_max_corr_top3peaks_az_vpol'])
    csw_max_corr_top3peaks_el_vpol = np.asarray(data['csw_max_corr_top3peaks_el_vpol'])
    csw_max_corr_hpol = np.asarray(data['csw_max_corr_hpol'])
    csw_max_lag_ns_hpol = np.asarray(data['csw_max_lag_ns_hpol'])
    csw_max_corr_top3peaks_az_hpol = np.asarray(data['csw_max_corr_top3peaks_az_hpol'])
    csw_max_corr_top3peaks_el_hpol = np.asarray(data['csw_max_corr_top3peaks_el_hpol'])
    he_peak_vpol = np.asarray(data['he_peak_vpol'])
    he_peak_time_vpol = np.asarray(data['he_peak_time_vpol'])
    he_peak_top3peaks_az_vpol = np.asarray(data['he_peak_top3peaks_az_vpol'])
    he_peak_top3peaks_el_vpol = np.asarray(data['he_peak_top3peaks_el_vpol'])
    he_peak_hpol = np.asarray(data['he_peak_hpol'])
    he_peak_time_hpol = np.asarray(data['he_peak_time_hpol'])
    he_peak_top3peaks_az_hpol = np.asarray(data['he_peak_top3peaks_az_hpol'])
    he_peak_top3peaks_el_hpol = np.asarray(data['he_peak_top3peaks_el_hpol'])

    ### add basic saturation, deadtime, and blast mask
    #valid_mask = functools.reduce(np.logical_and, [saturation_flag == False, blast_flag == False, deadtime_flag == False, low_channel_rms_flag == False, csw_max_lag_ns_vpol > 120])
    valid_mask = functools.reduce(np.logical_and, [saturation_flag == False, blast_flag == False, deadtime_flag == False, low_channel_rms_flag == False, ])
    print(f'basic cuts on saturation, deadtime, blast, and low thermal remove {len(saturation_flag) - np.sum(valid_mask)} events!')
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
    trigger_window_max_he = trigger_window_max_he[valid_mask]
    trigger_window_max_time = trigger_window_max_time[valid_mask]
    dist_mcmurdo = dist_mcmurdo[valid_mask]
    csw_snr_vpol = csw_snr_vpol[valid_mask]
    csw_rms_restrictedrange_vpol = csw_rms_restrictedrange_vpol[valid_mask]
    csw_rms_fullrange_vpol = csw_rms_fullrange_vpol[valid_mask]
    reco_az_vpol = reco_az_vpol[valid_mask]
    reco_el_vpol = reco_el_vpol[valid_mask]
    csw_snr_hpol = csw_snr_hpol[valid_mask]
    csw_rms_restrictedrange_hpol = csw_rms_restrictedrange_hpol[valid_mask]
    csw_rms_fullrange_hpol = csw_rms_fullrange_hpol[valid_mask]
    reco_az_hpol = reco_az_hpol[valid_mask]
    reco_el_hpol = reco_el_hpol[valid_mask]
    max_freq_fraction_vpol = max_freq_fraction_vpol[valid_mask]
    max_freq_fraction_hpol = max_freq_fraction_hpol[valid_mask]
    csw_max_corr_vpol = csw_max_corr_vpol[valid_mask]
    csw_max_lag_ns_vpol = csw_max_lag_ns_vpol[valid_mask]
    #csw_max_corr_top3peaks_az_vpol = csw_max_corr_top3peaks_az_vpol[valid_mask]
    #csw_max_corr_top3peaks_el_vpol = csw_max_corr_top3peaks_el_vpol[valid_mask]
    csw_max_corr_hpol = csw_max_corr_hpol[valid_mask]
    csw_max_lag_ns_hpol = csw_max_lag_ns_hpol[valid_mask]
    #csw_max_corr_top3peaks_az_hpol = csw_max_corr_top3peaks_az_hpol[valid_mask]
    #csw_max_corr_top3peaks_el_hpol = csw_max_corr_top3peaks_el_hpol[valid_mask]
    he_peak_vpol = he_peak_vpol[valid_mask]
    he_peak_time_vpol = he_peak_time_vpol[valid_mask]
    #he_peak_top3peaks_az_vpol = he_peak_top3peaks_az_vpol[valid_mask]
    #he_peak_top3peaks_el_vpol = he_peak_top3peaks_el_vpol[valid_mask]
    he_peak_hpol = he_peak_hpol[valid_mask]
    he_peak_time_hpol = he_peak_time_hpol[valid_mask]
    #he_peak_top3peaks_az_hpol = he_peak_top3peaks_az_hpol[valid_mask]
    #he_peak_top3peaks_el_hpol = he_peak_top3peaks_el_hpol[valid_mask]
    
    ### mask our events for now
    groundcal_vpol_mask = np.logical_and(flag == 'TD', groundcal_pol == 'V')
    force_mask = flag == 'Force'
    rf_mask = flag == 'RF'
    numc_mask_nue = np.logical_and(flag == 'NuMC', numc_pdg == 12) 
    numc_mask_numu = np.logical_and(flag == 'NuMC', numc_pdg == 14) 
    numc_mask_nutau = np.logical_and(flag == 'NuMC', numc_pdg == 16) 
    print(f'have {np.sum(groundcal_vpol_mask)} ground cal v-pol events, {np.sum(force_mask)} force triggers, {np.sum(rf_mask)} rf triggers, {np.sum(numc_mask_nue)} nue events, {np.sum(numc_mask_numu)} numu events, and {np.sum(numc_mask_nutau)} nutau events')

    ### now let's grab some events in different populations and plot csw
    #all_data_sample_masks = [groundcal_vpol_mask, force_mask, numc_mask_nue, numc_mask_numu, numc_mask_nutau]
    #colors = ['black', '#1e90ff', '#21b2aa', '#c71585', 'orange']
    #labels = ['Taylor Dome', 'Force Triggers', 'NuE', 'NuMu', 'NuTau']
    all_data_sample_masks = [rf_mask]
    colors = ['#d62728']
    labels = ['RF Triggers', ]
    
    variables = [csw_max_corr_vpol]
    variable_times = [csw_max_lag_ns_vpol]
    var_names = ['CSW Max Corr']
    #sub_pops = [[2e-2, 1e-1, 5e1, 1e2, 3e2, 4e3]]
    #eps = [[1e-3, 1e-3, 1, 10, 10, 100]]
    #sub_pops = [[3.5e4, 1e5]]
    #eps = [[300, 300]]
    #sub_pops = [[0.35]]
    #eps = [[0.05]]
    sub_pops = [[0.35, 0.8]]
    eps = [[0.05, 0.2]]

    ### loop over each variable
    for v in range(len(variables)):
        this_var = variables[v]
        this_var_times = variable_times[v]
        this_pops = sub_pops[v]
        this_eps = eps[v]

        ### loop over each population we want to investigate
        for p in range(1, len(this_pops)):

            ### now loop over each dataset
            for d in range(len(all_data_sample_masks)):
                this_mask = all_data_sample_masks[d]
                this_data = this_var[this_mask]
                this_data_times = this_var_times[this_mask]
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
                    outlier_data_times = this_data_times[outlier_mask]

                    for r in range(n_events):
                        plot_wfs(int(outlier_runs[r]), int(outlier_evs[r]), colors[d], labels[d], outlier_data[r], outlier_data_times[r], var_names[v])
                    
if __name__ == "__main__":
    plot_outliers()

