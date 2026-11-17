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

def fourierTransform(time, voltage):
    freq=np.fft.rfftfreq(time.shape[-1],time[1]-time[0])
    amp_f= np.fft.rfft(voltage)
    voltage_ft= np.abs(amp_f)*np.exp(np.unwrap(np.angle(amp_f))*1j)
    return freq, voltage_ft

def volt_to_dB(voltage_ratio):
    dB = np.zeros_like(voltage_ratio)
    dB[np.abs(voltage_ratio) > 1e-10] = 20*np.log10(np.abs(voltage_ratio[np.abs(voltage_ratio) > 1e-10]))
    return dB

def filter_ground_cal(rf_runs, rf_evs):
    ground_cal_names = ['TD', 'LDB', 'S200']
    source_loc_names = ['taylor_dome', 'ldb', 'S+200']

    non_groundcal_masks = []
    for g in range(len(ground_cal_names)):
        ground_cal_name = ground_cal_names[g]
        source_loc_name = source_loc_names[g]

        groundcal_events = np.genfromtxt(os.environ['PUEO_ANALYSIS_TOOLS'] + f'/pueoTALON/lookup/my{ground_cal_name}_list.csv', delimiter=',', names=True, dtype=None)
        groundcal_runs = np.asarray(groundcal_events['run'])
        groundcal_evs = np.asarray(groundcal_events['entry_number'])

        groundcal_pairs = set(zip(groundcal_runs, groundcal_evs))
        non_groundcal_mask = [(r, e) not in groundcal_pairs for r, e in zip(rf_runs, rf_evs)]
        non_groundcal_masks.append(non_groundcal_mask)

    return functools.reduce(np.logical_and, non_groundcal_masks)

def save_secondary_metrics(array, ev_per_array = 1000, ground_cal_name = 'TD', delta_az_tol = 3, delta_el_tol = 1, trigger_window_length_ns = 40,):
    ### ok we want to process three types of data -- TD ground cal, force triggers, and neutrino sim
    if ground_cal_name == 'TD':
        source_loc_name = 'taylor_dome'
    if ground_cal_name == 'LDB':
        source_loc_name = 'ldb'
    if ground_cal_name == 'S200':
        source_loc_name = 'S+200'

    ### let's figure out based on the array which one we will process
    all_runs = []
    all_entries = []
    all_flags = []
    all_groundcal_pol = []
    all_numc_pdg = []
    all_numc_weight = []
    all_numc_interaction_current = []

    ### first, get number of TD events
    groundcal_events = np.genfromtxt(os.environ['PUEO_ANALYSIS_TOOLS'] + f'/pueoTALON/lookup/my{ground_cal_name}_list.csv', delimiter=',', names=True, dtype=None)
    groundcal_runs = np.asarray(groundcal_events['run'])
    groundcal_evs = np.asarray(groundcal_events['entry_number'])
    groundcal_pol = np.asarray(groundcal_events['pol'])
    total_groundcal_events = len(groundcal_runs)
    [all_runs.append(int(r)) for r in groundcal_runs]
    [all_entries.append(int(e)) for e in groundcal_evs]
    [all_flags.append(ground_cal_name) for x in range(len(groundcal_runs))]
    [all_groundcal_pol.append(p) for p in groundcal_pol]
    [all_numc_pdg.append(0) for x in range(len(groundcal_runs))]
    [all_numc_weight.append(0) for x in range(len(groundcal_runs))]
    [all_numc_interaction_current.append(0) for x in range(len(groundcal_runs))]

    ### now get number of force triggers
    ### already went through some runs and saved event numbers for non-MIRF triggers
    path = f'/scratch/midway3/darcynewmark/useful/nonrftriggers_run783_run808.json'
    with open(path) as f:
        force_dict = json.load(f)
    force_runs = np.asarray(force_dict['run'])
    force_evs = np.asarray(force_dict['entry'])
    total_forcetrigger_events = len(force_evs)
    [all_runs.append(int(r)) for r in force_runs]
    [all_entries.append(int(e)) for e in force_evs]
    [all_flags.append('Force') for x in range(len(force_runs))]
    [all_groundcal_pol.append(0) for x in range(len(force_runs))]
    [all_numc_pdg.append(0) for x in range(len(force_runs))]
    [all_numc_weight.append(0) for x in range(len(force_runs))]
    [all_numc_interaction_current.append(0) for x in range(len(force_runs))]

    ### let's also grab some RF triggers from when the detector was around mcmurdo!
    path = f'/scratch/midway3/darcynewmark/useful/rftriggers_run755_run900.json'
    with open(path) as f:
        rf_dict = json.load(f)
    rf_runs = np.asarray(rf_dict['run'])
    rf_evs = np.asarray(rf_dict['entry'])
    ### REMOVE ANY RUNS/EVENTS THAT ARE ALSO IN THE GROUNDCAL LIST
    non_groundcal_mask = filter_ground_cal(rf_runs, rf_evs)
    rf_runs = rf_runs[non_groundcal_mask]
    rf_evs = rf_evs[non_groundcal_mask]
    run_mask = np.logical_and(rf_runs > 782, rf_runs < 800) ### cut down on the data to not take forever..
    rf_runs = rf_runs[run_mask]
    rf_evs = rf_evs[run_mask]
    rf_runs = rf_runs[::5] ### even MORE downselect
    rf_evs = rf_evs[::5]
    total_rftrigger_events = len(rf_evs)
    [all_runs.append(int(r)) for r in rf_runs]
    [all_entries.append(int(e)) for e in rf_evs]
    [all_flags.append('RF') for x in range(len(rf_runs))]
    [all_groundcal_pol.append(0) for x in range(len(rf_runs))]
    [all_numc_pdg.append(0) for x in range(len(rf_runs))]
    [all_numc_weight.append(0) for x in range(len(rf_runs))]
    [all_numc_interaction_current.append(0) for x in range(len(rf_runs))]

    ### finally, let's get our simulated neutrino runs + entries
    path = f'/scratch/midway3/darcynewmark/useful/nonempty_neutrino_sim_oct2026_weightsandcurrrent.json'
    with open(path) as f:
        numc_dict = json.load(f)
    numc_runs = np.asarray(numc_dict['run'])
    numc_evs = np.asarray(numc_dict['entry'])
    numc_pdg = np.asarray(numc_dict['nu_pdg'])
    numc_weight = np.asarray(numc_dict['weight'])
    numc_interaction_current = np.asarray(numc_dict['interaction_current'])
    total_numc_events = len(numc_evs)
    [all_runs.append(int(r)) for r in numc_runs]
    [all_entries.append(int(e)) for e in numc_evs]
    [all_flags.append('NuMC') for x in range(len(numc_runs))]
    [all_groundcal_pol.append(0) for x in range(len(numc_runs))]
    [all_numc_pdg.append(p) for p in numc_pdg]
    [all_numc_weight.append(w) for w in numc_weight]
    [all_numc_interaction_current.append(c) for c in numc_interaction_current]

    print(f'total of {len(all_runs)} events to process!')
    print(f'total of {total_groundcal_events} ground cal, {total_forcetrigger_events} force triggers, {total_rftrigger_events} rf triggers, and {total_numc_events} nu mc events !')

    ### let's chunk up run, entry for this array!
    all_runs = np.asarray(all_runs)
    all_entries = np.asarray(all_entries)
    all_flags = np.asarray(all_flags)
    all_groundcal_pol = np.asarray(all_groundcal_pol)
    all_numc_pdg = np.asarray(all_numc_pdg)
    all_numc_weight = np.asarray(all_numc_weight)
    all_numc_interaction_current = np.asarray(all_numc_interaction_current)
    start_idx = array * ev_per_array
    end_idx = (array + 1) * ev_per_array
    print(f'for array = {array}, processing {start_idx} to {end_idx}')

    if start_idx > len(all_runs):
        print(f'oops! start idx is more than total number of events! bye bye!')
        return

    this_array_runs = all_runs[start_idx:end_idx]
    this_array_entries = all_entries[start_idx:end_idx]
    this_array_flags = all_flags[start_idx:end_idx]
    this_array_groundcal_pol = all_groundcal_pol[start_idx:end_idx]
    this_array_numc_pdg = all_numc_pdg[start_idx:end_idx]
    this_array_numc_weight = all_numc_weight[start_idx:end_idx]
    this_array_numc_interaction_current = all_numc_interaction_current[start_idx:end_idx]

    ### list of things to save
    ### some general things we might want
    run = []
    entry = []
    flag = []
    groundcal_pol = []
    numc_pdg = []
    numc_weight = []
    numc_interaction_current = []
    trigger_time = []
    trigger_subsecond = []
    saturation_flag = []
    blast_flag = []
    deadtime_flag = []
    low_channel_rms_flag = []
    trigger_window_max_he = []
    trigger_window_max_time = []
    dist_mcmurdo = []
    sun_el = []
    sun_az = []
    td_el = []
    td_az = []
    ldb_el = []
    ldb_az = []
    csw_snr_vpol = []
    csw_rms_restrictedrange_vpol = []
    csw_rms_fullrange_vpol = []
    reco_az_vpol = []
    reco_el_vpol = []
    csw_snr_hpol = []
    csw_rms_restrictedrange_hpol = []
    csw_rms_fullrange_hpol = []
    reco_az_hpol = []
    reco_el_hpol = []
    max_freq_fraction_vpol = []
    max_freq_fraction_hpol = []

    ### now some secondary specific things
    ### related to cross correlating main pulse with itself
    csw_max_corr_vpol = []
    csw_max_lag_ns_vpol = []
    csw_max_corr_secondarypeak_az_vpol = []
    csw_max_corr_secondarypeak_el_vpol = []
    csw_max_corr_hpol = []
    csw_max_lag_ns_hpol = []
    csw_max_corr_secondarypeak_az_hpol = []
    csw_max_corr_secondarypeak_el_hpol = []

    ### top hilbert envelope value [AFTER TRIGGER]
    he_peak_vpol = []
    he_peak_time_vpol = []
    he_peak_secondarypeak_az_vpol = []
    he_peak_secondarypeak_el_vpol = []
    he_peak_hpol = []
    he_peak_time_hpol = []
    he_peak_secondarypeak_az_hpol = []
    he_peak_secondarypeak_el_hpol = []

    prev_run = None
    polarizations = ['V', 'H']
    for r in range(len(this_array_runs)):
        this_run = int(this_array_runs[r])
        this_ev = int(this_array_entries[r])
        this_flag = this_array_flags[r]
        this_groundcal_pol = this_array_groundcal_pol[r]
        this_numc_pdg = this_array_numc_pdg[r]
        this_numc_weight = this_array_numc_weight[r]
        this_numc_interaction_current = this_array_numc_interaction_current[r]

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
        geography_helper.updatePosition(event_time = event_time, max_time_delta = 3)

        ### get distance to mcmurdo
        expected_mcm_el_az = geography_helper.get_expected_source_location('McMurdo')
        expected_mcm_el = expected_mcm_el_az[0]
        expected_mcm_az = expected_mcm_el_az[1]
        dist_mcm = geography_helper.get_distance(expected_mcm_el, expected_mcm_az)

        ### get source locations
        this_sun_el, this_sun_az = geography_helper.get_sun_position(event_time = event_time, max_time_delta = 3)
        this_td_el, this_td_az = geography_helper.get_expected_source_location('taylor_dome')
        this_ldb_el, this_ldb_az = geography_helper.get_expected_source_location('ldb')

        ### get some flags
        this_event_summary = ROOT.PueoEventSummary()
        qual = corr.ChannelGeo.qualityCheck(this_event_summary)
        this_blast_flag = this_event_summary.flags.isPayloadBlast
        this_sat_flag = this_event_summary.flags.isSaturated
        this_deadtime_flag = this_event_summary.flags.isInDeadtime
        this_low_channel_rms_flag = this_event_summary.flags.lowChannelThermal

        ### grab trigger time + subsec
        this_trigger_time = corr.ChannelGeo.triggerTime()
        this_trigger_subsecond = corr.ChannelGeo.subsecond()

        ### quality check on ground cal
        if this_flag == ground_cal_name:
            ### ok this is a ground cal event, let's make sure it's pointing in the right direction
            ### reco direction for this polarization
            skymap, peak_zen, peak_az, corr_vals, zoomedskymap, zoomedpeak_zen, zoomedpeak_az, zoomedcorr_vals, all_zoomedzen, all_zoomedaz = corr.reconstruct(this_groundcal_pol, zoom_angle_range=7, N_peaks=1)
            main_peak_zen = zoomedpeak_zen[0][0]
            main_peak_el = 90 - main_peak_zen
            main_peak_az = zoomedpeak_az[0][0]

            ### get the expected source loc for cuts
            expected_source_el_az = geography_helper.get_expected_source_location(source_loc_name)
            expected_source_el = expected_source_el_az[0]
            expected_source_az = expected_source_el_az[1]

            ### check the dir
            delta_az = np.abs(main_peak_az - expected_source_az)
            delta_el = np.abs(main_peak_el - expected_source_el)
            if (delta_az > delta_az_tol) or (delta_el > delta_el_tol):
                print(f'run = {this_run}, ev = {this_ev}, {this_flag} event removed by source pointing check!')
                continue

        ### loop over each polarization
        for poli in polarizations:
            ### now reconstruct the directions and coherent sum
            skymap, peak_zen, peak_az, corr_vals, zoomedskymap, zoomedpeak_zen, zoomedpeak_az, zoomedcorr_vals, all_zoomedzen, all_zoomedaz = corr.reconstruct(poli, zoom_angle_range=7, N_peaks=1)
            main_peak_zen = zoomedpeak_zen[0][0]
            main_peak_el = 90 - main_peak_zen
            main_peak_az = zoomedpeak_az[0][0]

            ### get coherent sum
            cohsum, overlap_start, overlap_end = corr.coherent_sum(main_peak_zen, main_peak_az, polarization=poli, return_overlap_boundaries=True)
            cohsum_snr, cohsum_rms_restrictedrange = utils.SNR(cohsum, dynamic_rms_window = True, valid_window = (overlap_start, overlap_end), return_rms = True)
            cohsum = cohsum[overlap_start:overlap_end]
            cohsum_rms_fullrange = utils.RMS(cohsum)
            cswhil = utils.hilbert_envelope(cohsum)
            times = np.arange(cohsum.size)*corr.dt

            ### let's start by finding our trigger window (we have a rough guess but can hone in on it) --> then window!
            ### assuming trigger is sometime between 30 and 90 ns (relatively wide bc simuluation seems to be a bit offset from 60...)
            rought_trigger_window_start_time = 30
            rought_trigger_window_end_time = 90
            rough_trigger_window_start_idx = int(rought_trigger_window_start_time / corr.dt)
            rough_trigger_window_end_idx = int(rought_trigger_window_end_time / corr.dt)
            rough_trigger_window_cswhil = cswhil[rough_trigger_window_start_idx:rough_trigger_window_end_idx]
            max_he_idx = np.argmax(rough_trigger_window_cswhil)
            max_he_val = rough_trigger_window_cswhil[max_he_idx]
            max_he_global_idx = rough_trigger_window_start_idx + max_he_idx
            max_he_time = times[max_he_global_idx]

            ### now we can window around max_he_global_idx
            trigger_window_start_time = max_he_time - (trigger_window_length_ns/2)
            trigger_window_end_time = max_he_time + (trigger_window_length_ns/2)
            if this_flag == 'NuMC':
                trigger_window_start_time = 60
                trigger_window_end_time = 100
            else:
                trigger_window_start_time = 30
                trigger_window_end_time = 70
            trigger_window_start_idx = int(trigger_window_start_time / corr.dt)
            trigger_window_end_idx = int(trigger_window_end_time / corr.dt)
            trigger_window_csw = cohsum[trigger_window_start_idx:trigger_window_end_idx].copy()
            trigger_peak_idx = np.argmax(trigger_window_csw)

            ### let's also clip the wf so we dont get confused by the actual main pulse
            clipped_csw = cohsum[trigger_window_end_idx:].copy()

            ### apply a tukey window to both trigger_window_csw and clipped_csw for corss correlation
            trigger_window_csw *= scipy.signal.windows.tukey(len(trigger_window_csw), alpha=0.05)
            clipped_csw *= scipy.signal.windows.tukey(len(clipped_csw), alpha=0.05)

            ### now cross correlate!!!
            normalized_trigger_window_csw = (trigger_window_csw - np.mean(trigger_window_csw)) / np.sqrt(np.sum(np.abs(trigger_window_csw - np.mean(trigger_window_csw))**2))
            normalized_clipped_csw = (clipped_csw - np.mean(clipped_csw)) / np.sqrt(np.sum(np.abs(clipped_csw - np.mean(clipped_csw))**2))
            normalized_correlation = scipy.signal.correlate(normalized_trigger_window_csw, normalized_clipped_csw, mode='full', method='fft')
            lags = scipy.signal.correlation_lags(len(normalized_trigger_window_csw), len(normalized_clipped_csw), mode='full')
            max_lag = lags[np.argmax(normalized_correlation)]
            max_lag_time_ns = (trigger_window_end_idx + trigger_peak_idx - max_lag) * corr.dt

            ### let's also get the biggest peak in the hilbert envelope AFTER THE TRIGGER
            clipped_cswhil = cswhil[trigger_window_end_idx:].copy()
            cswhil_dt = times[1] - times[0]
            peak_idx = np.argmax(clipped_cswhil)
            peak_val = clipped_cswhil[peak_idx]
            peak_time_ns = (trigger_window_end_idx + peak_idx) * cswhil_dt

            ### now save our cross corr + max he + general reco info!!!
            if poli == 'V':
                reco_az_vpol.append(main_peak_az)
                reco_el_vpol.append(main_peak_el)
                csw_max_corr_vpol.append(np.amax(normalized_correlation))
                csw_max_lag_ns_vpol.append(max_lag_time_ns)
                csw_snr_vpol.append(cohsum_snr)
                csw_rms_restrictedrange_vpol.append(cohsum_rms_restrictedrange)
                csw_rms_fullrange_vpol.append(cohsum_rms_fullrange)
                he_peak_vpol.append(peak_val)
                he_peak_time_vpol.append(peak_time_ns)
            else:
                reco_az_hpol.append(main_peak_az)
                reco_el_hpol.append(main_peak_el)
                csw_max_corr_hpol.append(np.amax(normalized_correlation))
                csw_max_lag_ns_hpol.append(max_lag_time_ns)
                csw_snr_hpol.append(cohsum_snr)
                csw_rms_restrictedrange_hpol.append(cohsum_rms_restrictedrange)
                csw_rms_fullrange_hpol.append(cohsum_rms_fullrange)
                he_peak_hpol.append(peak_val)
                he_peak_time_hpol.append(peak_time_ns)

            ### now we have two potential secondaries -- max cross corr and max he, let's reconstruct the position!
            ### the tricky part is converting time windows into samples for reconstruct function in correlator.py
            ### we want our window as [length, center] in units of samples
            secondary_pulse_times = [max_lag_time_ns, peak_time_ns]
            original_dt = 1/3.0
            padding_samples = 200 ### hardcoding for now, maybe make general one day
            cohsum_upsample_factor = original_dt / corr.dt
            secondary_window_length_ns = 50.0
            secondary_window_length_samples = int(secondary_window_length_ns / original_dt)

            for p in range(len(secondary_pulse_times)):
                this_peak_time = secondary_pulse_times[p]
                this_peak_window_center_samples = int(this_peak_time / original_dt) + padding_samples

                ### make sure our window isnt too long!
                if this_peak_window_center_samples > (1024 + padding_samples + padding_samples - secondary_window_length_samples):
                    print(f'!!!!!!!!!!!! oops! this_peak_window_center_samples = {this_peak_window_center_samples}, cutting down to {1024 + padding_samples + padding_samples - secondary_window_length_samples}')
                    this_peak_window_center_samples = 1024 + padding_samples + padding_samples - secondary_window_length_samples

                this_peak_window = [secondary_window_length_samples, this_peak_window_center_samples]

                ### now let's reco this peak!
                N_peaks = 1
                _, _, _, _, _, zoomedpeak_zen, zoomedpeak_az, _, _, _ = corr.reconstruct(poli, window=this_peak_window,  zoom_angle_range=7, N_peaks=N_peaks)
                secondary_peak_zen = [zoomedpeak_zen[z][0] for z in range(len(zoomedpeak_zen))]
                secondary_peak_el = [90 - zen for zen in secondary_peak_zen]
                secondary_peak_az = [zoomedpeak_az[z][0] for z in range(len(zoomedpeak_az))]

                ### make sure our arrays are the right length
                secondary_peak_az = secondary_peak_az[:N_peaks]
                secondary_peak_el = secondary_peak_el[:N_peaks]
                while len(secondary_peak_az) < N_peaks:
                    secondary_peak_az.append(np.nan)
                while len(secondary_peak_el) < N_peaks:
                    secondary_peak_el.append(np.nan)

                ### convert to array
                secondary_peak_az = np.asarray(secondary_peak_az)
                secondary_peak_el = np.asarray(secondary_peak_el)

                ### save our secondary peaks!
                if poli == 'V' and p == 0:
                    csw_max_corr_secondarypeak_az_vpol.append(secondary_peak_az)
                    csw_max_corr_secondarypeak_el_vpol.append(secondary_peak_el)
                if poli == 'V' and p > 0:
                    he_peak_secondarypeak_az_vpol.append(secondary_peak_az)
                    he_peak_secondarypeak_el_vpol.append(secondary_peak_el)
                if poli == 'H' and p == 0:
                    csw_max_corr_secondarypeak_az_hpol.append(secondary_peak_az)
                    csw_max_corr_secondarypeak_el_hpol.append(secondary_peak_el)
                if poli == 'H' and p > 0:
                    he_peak_secondarypeak_az_hpol.append(secondary_peak_az)
                    he_peak_secondarypeak_el_hpol.append(secondary_peak_el)

            ### before we're done a quick foray into fourier space ...
            ### want to help tag big CW that are getting picked out for secondaries
            f, vf = fourierTransform(times, cohsum)
            dt = times[1] - times[0]
            power = volt_to_dB(np.abs(vf*dt))
            max_power = np.amax(power)
            median_power = np.median(power[power>0])
            max_freq_fraction = max_power/median_power

            if poli == 'V':
                max_freq_fraction_vpol.append(max_freq_fraction)
            else:
                max_freq_fraction_hpol.append(max_freq_fraction)

        ### finally, save our general info
        run.append(this_run)
        entry.append(this_ev)
        flag.append(this_flag)
        groundcal_pol.append(this_groundcal_pol)
        numc_pdg.append(this_numc_pdg)
        numc_weight.append(this_numc_weight)
        numc_interaction_current.append(this_numc_interaction_current)
        trigger_time.append(this_trigger_time)
        trigger_subsecond.append(this_trigger_subsecond)
        saturation_flag.append(this_sat_flag)
        blast_flag.append(this_blast_flag)
        deadtime_flag.append(this_deadtime_flag)
        low_channel_rms_flag.append(this_low_channel_rms_flag)
        trigger_window_max_he.append(max_he_val)
        trigger_window_max_time.append(max_he_time)
        dist_mcmurdo.append(dist_mcm)
        sun_el.append(this_sun_el)
        sun_az.append(this_sun_az)
        td_el.append(this_td_el)
        td_az.append(this_td_az)
        ldb_el.append(this_ldb_el)
        ldb_az.append(this_ldb_az)

    ### now save this all to a json!
    saving_path = f'/scratch/midway3/darcynewmark/secondaries/multipulse_metrics/jsons/secondary_metrics_{array}.json'
    data = {
            'run' : np.asarray(run).tolist(),
            'entry' : np.asarray(entry).tolist(),
            'flag' : np.asarray(flag).tolist(),
            'groundcal_pol' : np.asarray(groundcal_pol).tolist(),
            'numc_pdg' : np.asarray(numc_pdg).tolist(),
            'numc_weight' : np.asarray(numc_weight).tolist(),
            'numc_interaction_current' : np.asarray(numc_interaction_current).tolist(),
            'trigger_time' : np.asarray(trigger_time).tolist(),
            'trigger_subsecond' : np.asarray(trigger_subsecond).tolist(),
            'saturation_flag' : np.asarray(saturation_flag).tolist(),
            'blast_flag' : np.asarray(blast_flag).tolist(),
            'deadtime_flag' : np.asarray(deadtime_flag).tolist(),
            'low_channel_rms_flag' : np.asarray(low_channel_rms_flag).tolist(),
            'trigger_window_max_he' :  np.asarray(trigger_window_max_he).tolist(),
            'trigger_window_max_time' :  np.asarray(trigger_window_max_time).tolist(),
            'dist_mcmurdo' : np.asarray(dist_mcmurdo).tolist(),
            'sun_el' : np.asarray(sun_el).tolist(),
            'sun_az' : np.asarray(sun_az).tolist(),
            'td_el' : np.asarray(td_el).tolist(),
            'td_az' : np.asarray(td_az).tolist(),
            'ldb_el' : np.asarray(ldb_el).tolist(),
            'ldb_az' : np.asarray(ldb_az).tolist(),
            'csw_snr_vpol' : np.asarray(csw_snr_vpol).tolist(),
            'csw_rms_restrictedrange_vpol' : np.asarray(csw_rms_restrictedrange_vpol).tolist(),
            'csw_rms_fullrange_vpol' : np.asarray(csw_rms_fullrange_vpol).tolist(),
            'reco_az_vpol' : np.asarray(reco_az_vpol).tolist(),
            'reco_el_vpol' : np.asarray(reco_el_vpol).tolist(),
            'csw_snr_hpol' : np.asarray(csw_snr_hpol).tolist(),
            'csw_rms_restrictedrange_hpol' : np.asarray(csw_rms_restrictedrange_hpol).tolist(),
            'csw_rms_fullrange_hpol' : np.asarray(csw_rms_fullrange_hpol).tolist(),
            'reco_az_hpol' : np.asarray(reco_az_hpol).tolist(),
            'reco_el_hpol' : np.asarray(reco_el_hpol).tolist(),
            'max_freq_fraction_vpol' : np.asarray(max_freq_fraction_vpol).tolist(),
            'max_freq_fraction_hpol' : np.asarray(max_freq_fraction_hpol).tolist(),
            'csw_max_corr_vpol' : np.asarray(csw_max_corr_vpol).tolist(),
            'csw_max_lag_ns_vpol' : np.asarray(csw_max_lag_ns_vpol).tolist(),
            'csw_max_corr_secondarypeak_az_vpol' : np.asarray(csw_max_corr_secondarypeak_az_vpol).tolist(),
            'csw_max_corr_secondarypeak_el_vpol' : np.asarray(csw_max_corr_secondarypeak_el_vpol).tolist(),
            'csw_max_corr_hpol' : np.asarray(csw_max_corr_hpol).tolist(),
            'csw_max_lag_ns_hpol' : np.asarray(csw_max_lag_ns_hpol).tolist(),
            'csw_max_corr_secondarypeak_az_hpol' : np.asarray(csw_max_corr_secondarypeak_az_hpol).tolist(),
            'csw_max_corr_secondarypeak_el_hpol' : np.asarray(csw_max_corr_secondarypeak_el_hpol).tolist(),
            'he_peak_vpol' : np.asarray(he_peak_vpol).tolist(),
            'he_peak_time_vpol' : np.asarray(he_peak_time_vpol).tolist(),
            'he_peak_secondarypeak_az_vpol' : np.asarray(he_peak_secondarypeak_az_vpol).tolist(),
            'he_peak_secondarypeak_el_vpol' : np.asarray(he_peak_secondarypeak_el_vpol).tolist(),
            'he_peak_hpol' : np.asarray(he_peak_hpol).tolist(),
            'he_peak_time_hpol' : np.asarray(he_peak_time_hpol).tolist(),
            'he_peak_secondarypeak_az_hpol' : np.asarray(he_peak_secondarypeak_az_hpol).tolist(),
            'he_peak_secondarypeak_el_hpol' : np.asarray(he_peak_secondarypeak_el_hpol).tolist(),
    }
    with open(saving_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--array", type = int, default = 0)
    args = parser.parse_args()

    save_secondary_metrics(args.array)
