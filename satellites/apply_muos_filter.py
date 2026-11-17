import os
import ROOT
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time
import scipy.signal as signal

import pueoTALON.pueoPythonAnalyzer.eventSummary as summary
import pueoTALON.pueoPythonAnalyzer.waveform as waveform
import pueoTALON.pueoPythonAnalyzer.plotter as pltr
import pueoTALON.pueoPythonAnalyzer.antenna as antenna
import pueoTALON.pueoPythonAnalyzer.correlator as correlator
import antennaUtils.response as antResponse
import pueoTALON.pueoPythonAnalyzer.utils as utils
import pueoTALON.pueoPythonAnalyzer.geography as geography

ROOT.gSystem.Load("libpueoEvent.so")

import argparse
import json

def remove_muos(array, ev_per_array = 2000,):
    ### ok let's remove muos from some rf triggers
    ### let's figure out based on the array which one we will process
    all_runs = []
    all_entries = []

    ### now get number of RF triggers
    ### already went through some runs and saved event numbers for non-MIRF/RF triggers
    path = f'/scratch/midway3/darcynewmark/useful/rftriggers_run783_run808.json'
    with open(path) as f:
        rf_dict = json.load(f)
    rf_runs = np.asarray(rf_dict['run'])
    rf_evs = np.asarray(rf_dict['entry'])
    [all_runs.append(int(r)) for r in rf_runs]
    [all_entries.append(int(e)) for e in rf_evs]

    print(f'total of {len(all_runs)} events to process!')

    ### let's chunk up run, entry for this array!
    all_runs = np.asarray(all_runs)
    all_entries = np.asarray(all_entries)
    start_idx = array * ev_per_array
    end_idx = (array + 1) * ev_per_array
    print(f'for array = {array}, processing {start_idx} to {end_idx}')

    if start_idx > len(all_runs):
        print(f'oops! start idx is more than total number of events! bye bye!')
        return

    this_array_runs = all_runs[start_idx:end_idx]
    this_array_entries = all_entries[start_idx:end_idx]

    prev_run = None
    for r in range(len(this_array_runs)):
        this_run = int(this_array_runs[r])
        this_ev = int(this_array_entries[r])

        ### check if this is a simulation set
        sim_flag = this_run > 180000

        ### check if we have a correlator object for this run
        if prev_run == None or prev_run != this_run:
            ### set up correlator + geography helper per run (to allow for handling of sim)
            corr = correlator.Correlator(run = this_run, sim = sim_flag)
            geography_helper = geography.GeographyHelper(muosRootTree=ROOT.TFile.Open(os.environ['PUEO_ANALYSIS_TOOLS'] + "/pueoTALON/lookup/muos_adjusted.root"))
            prev_run = this_run

        ### load this event
        corr.ChannelGeo.load_event(this_ev)

        ### update geography helper
        event_time = corr.ChannelGeo.triggerTime()
        geography_helper.updatePosition(corr.ChannelGeo.dataset.gps())

        ### get some flags
        this_event_summary = ROOT.PueoEventSummary()
        qual = corr.ChannelGeo.qualityCheck(this_event_summary)
        this_blast_flag = this_event_summary.flags.isPayloadBlast
        this_sat_flag = this_event_summary.flags.isSaturated
        this_deadtime_flag = this_event_summary.flags.isInDeadtime
        this_low_channel_rms_flag = this_event_summary.flags.lowChannelThermal

        all_flags_false = not (this_blast_flag or this_sat_flag or this_deadtime_flag or this_low_channel_rms_flag)

        if all_flags_false:
            ### ok let's remove our muos satellites !    
            satellite_info = sat.Satellite(event_time, corr, geography_helper, this_run, this_event, triggeredOnly=False)
            result = satellite_info.muosBEGONE()
            
            
        else:
            print(f'!!!!!!!!!!!!!!!!!!!! run = {this_run}, ev = {this_ev} removed by quality cuts!!! this_blast_flag = {this_blast_flag}, this_sat_flag = {this_sat_flag}, this_deadtime_flag = {this_deadtime_flag}, this_low_channel_rms_flag = {this_low_channel_rms_flag}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--array", type = int, default = 0)
    args = parser.parse_args()

    remove_muos(args.array)
