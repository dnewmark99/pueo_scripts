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

def save_stokes(array, ev_per_array = 500, ground_cal_name = 'TD', delta_az_tol = 3, delta_el_tol = 1, trigger_window_start_time = 30, trigger_window_end_time = 80):
    ### ok we want to process run 813 (bc lots of known muos contamination), ground cal, and MC!
    ### let's figure out based on the array which one we will process
    if ground_cal_name == 'TD':
        source_loc_name = 'taylor_dome'
    if ground_cal_name == 'LDB':
        source_loc_name = 'ldb'
    if ground_cal_name == 'S200':
        source_loc_name = 'S+200'

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

    ### let's also grab some RF triggers from run 813 (when muos was particuarly loud)
    path = f'/scratch/midway3/darcynewmark/useful/rftriggers_run813_run813.json'
    with open(path) as f:
        rf_dict = json.load(f)
    rf_runs = np.asarray(rf_dict['run'])
    rf_evs = np.asarray(rf_dict['entry'])
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
    print(f'total of {total_groundcal_events} ground cal, {total_rftrigger_events} rf triggers, and {total_numc_events} nu mc events !')

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

    ### stokes-specific things
    ### we're going to calculate over different time windows -- whole, trigger window, peak-50+100, peak-100+200, and save to our lists below
    ### also calculate stokes summing using v and h pol separately for delays
    avg_lin_vpol = []
    avg_circ_vpol = []
    avg_pol_vpol = []
    avg_angle_vpol = []
    isolate_muos_avg_lin_vpol = []
    isolate_muos_avg_circ_vpol = []
    isolate_muos_avg_pol_vpol = []
    isolate_muos_avg_angle_vpol = []
    avg_lin_hpol = []
    avg_circ_hpol = []
    avg_pol_hpol = []
    avg_angle_hpol = []
    isolate_muos_avg_lin_hpol = []
    isolate_muos_avg_circ_hpol = []
    isolate_muos_avg_pol_hpol = []
    isolate_muos_avg_angle_hpol = []

    ### and snr bc why not
    csw_snr_vpol = []
    csw_snr_hpol = []

    prev_run = None
    muos_configs = [True, False]
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

        ### update trigger window for simulation
        if sim_flag:
            trigger_window_start_time = 60
            trigger_window_end_time = 110

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
            _, _, _, _, _, zoomedpeak_zen, zoomedpeak_az, _, _, _ = corr.reconstruct(this_groundcal_pol, zoom_angle_range=7, N_peaks=1)
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
            opp_poli = 'H' if poli == 'V' else 'V'
            ### loop over each muos configuration
            for m in muos_configs:
                ### now reconstruct the directions and coherent sum
                _, _, _, _, _, zoomedpeak_zen, zoomedpeak_az, _, _, _ = corr.reconstruct(poli, zoom_angle_range=7, N_peaks=1, isolate_MUOS=m)
                main_peak_zen = zoomedpeak_zen[0][0]
                main_peak_el = 90 - main_peak_zen
                main_peak_az = zoomedpeak_az[0][0]

                ### get coherent sum
                cohsum, overlap_start, overlap_end = corr.coherent_sum(main_peak_zen, main_peak_az, polarization=poli, return_overlap_boundaries=True, isolate_MUOS=m)
                cohsum_snr = utils.SNR(cohsum, dynamic_rms_window = True, valid_window = (overlap_start, overlap_end))
                opp_cohsum, opp_overlap_start, opp_overlap_end = corr.coherent_sum(main_peak_zen, main_peak_az, polarization=opp_poli, return_overlap_boundaries=True, isolate_MUOS=m)
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

                ### get average linear and circular polarization fraction over various time windows
                time_windows = [(0, len(cohsum)-1), ### whole wf
                                (int(trigger_window_start_time/corr.dt), int(trigger_window_end_time/corr.dt)), ### around trigger window
                                (np.argmax(I)-50, np.argmax(I)+100), ### close to peak
                                (np.argmax(I)-100, np.argmax(I)+200) ### close to peak
                               ]
                all_time_p_lin = []
                all_time_p_circ = []
                all_time_p_pol = []
                all_time_angle = []

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
                    p_p = np.sqrt(avg_Q**2 + avg_U**2 + avg_V**2) / avg_I
                    a = np.degrees(0.5 * np.arctan2(avg_U, avg_Q))

                    all_time_p_lin.append(p_l)
                    all_time_p_circ.append(p_c)
                    all_time_p_pol.append(p_p)
                    all_time_angle.append(a)

                ### now save!
                if poli == 'V':
                    csw_snr_vpol.append(cohsum_snr)
                    if m:
                        isolate_muos_avg_lin_vpol.append(np.asarray(all_time_p_lin))
                        isolate_muos_avg_circ_vpol.append(np.asarray(all_time_p_circ))
                        isolate_muos_avg_pol_vpol.append(np.asarray(all_time_p_pol))
                        isolate_muos_avg_angle_vpol.append(np.asarray(all_time_angle))
                    else:
                        avg_lin_vpol.append(np.asarray(all_time_p_lin))
                        avg_circ_vpol.append(np.asarray(all_time_p_circ))
                        avg_pol_vpol.append(np.asarray(all_time_p_pol))
                        avg_angle_vpol.append(np.asarray(all_time_angle))
                else:
                    csw_snr_hpol.append(cohsum_snr)
                    if m:
                        isolate_muos_avg_lin_hpol.append(np.asarray(all_time_p_lin))
                        isolate_muos_avg_circ_hpol.append(np.asarray(all_time_p_circ))
                        isolate_muos_avg_pol_hpol.append(np.asarray(all_time_p_pol))
                        isolate_muos_avg_angle_hpol.append(np.asarray(all_time_angle))
                    else:
                        avg_lin_hpol.append(np.asarray(all_time_p_lin))
                        avg_circ_hpol.append(np.asarray(all_time_p_circ))
                        avg_pol_hpol.append(np.asarray(all_time_p_pol))
                        avg_angle_hpol.append(np.asarray(all_time_angle))

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

    ### now save this all to a json!
    saving_path = f'/scratch/midway3/darcynewmark/satellites/jsons/v3stokes_metrics_{array}.json'
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
            'avg_lin_vpol' : np.asarray(avg_lin_vpol).tolist(),
            'avg_circ_vpol' : np.asarray(avg_circ_vpol).tolist(),
            'avg_pol_vpol' : np.asarray(avg_pol_vpol).tolist(),
            'avg_angle_vpol' : np.asarray(avg_angle_vpol).tolist(),
            'isolate_muos_avg_lin_vpol' : np.asarray(isolate_muos_avg_lin_vpol).tolist(),
            'isolate_muos_avg_circ_vpol' : np.asarray(isolate_muos_avg_circ_vpol).tolist(),
            'isolate_muos_avg_pol_vpol' : np.asarray(isolate_muos_avg_pol_vpol).tolist(),
            'isolate_muos_avg_angle_vpol' : np.asarray(isolate_muos_avg_angle_vpol).tolist(),
            'avg_lin_hpol' : np.asarray(avg_lin_hpol).tolist(),
            'avg_circ_hpol' : np.asarray(avg_circ_hpol).tolist(),
            'avg_pol_hpol' : np.asarray(avg_pol_hpol).tolist(),
            'avg_angle_hpol' : np.asarray(avg_angle_hpol).tolist(),
            'isolate_muos_avg_lin_hpol' : np.asarray(isolate_muos_avg_lin_hpol).tolist(),
            'isolate_muos_avg_circ_hpol' : np.asarray(isolate_muos_avg_circ_hpol).tolist(),
            'isolate_muos_avg_pol_hpol' : np.asarray(isolate_muos_avg_pol_hpol).tolist(),
            'isolate_muos_avg_angle_hpol' : np.asarray(isolate_muos_avg_angle_hpol).tolist(),
            'csw_snr_vpol': np.asarray(csw_snr_vpol).tolist(),
            'csw_snr_hpol': np.asarray(csw_snr_hpol).tolist(),
    }
    with open(saving_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--array", type = int, default = 0)
    args = parser.parse_args()

    save_stokes(args.array)
