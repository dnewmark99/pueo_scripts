import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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
os.environ["PUEO_ROOT_DATA"] = ("/project/cozzyd/postflight/rootified/")

def subsec_plot():
    ### plot distribution of event trigger time subseconds to pick out beamformer events !
    run_start = 1407
    run_end = 1418
    channels = waveform.Channels(run_start, 0)

    all_subsecond_fmod = []
    all_trigger_time = []
    for r in range(run_start, run_end+1):
        channels.load_run_event(int(r), 0)
        ev_N  = channels.dataset.N()

        subsecond = []
        trigger_time = []

        for ev in range(ev_N):
            if ev == 0:
                channels.dataset.firstEvent()
            else:
                channels.dataset.nextEvent()
            if channels.dataset.header().trigType == 1:
                subsec = channels.subsecond()
                event_time = channels.triggerTime()
                if subsec > 0:
                    subsecond.append(subsec)
                    trigger_time.append(event_time)
                    all_subsecond_fmod.append(subsec % 0.1)
                    all_trigger_time.append(event_time)

        subsecond = np.array(subsecond)
        trigger_time = np.asarray(trigger_time)
        trigger_time -= np.amin(trigger_time)

        ### plot!
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.minorticks_on()
        ax.hist(subsecond, range = (0,1), bins = 500, linewidth = 1.5, histtype='step', linestyle = '-', color = 'black')
        ax.set_title(f'Beamformer Test -- Run {r}', fontsize = 20)
        ax.set_xlabel(f'Event Subsecond [s]', fontsize = 20)
        ax.set_ylabel(f'Number of Events', fontsize = 20)
        ax.tick_params(axis='both', which='both', labelsize=20)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/subsecond_hist_r{r}.png', dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.minorticks_on()
        ax.scatter(trigger_time, subsecond, s = 1, color = 'black', alpha = 0.5)
        ax.set_title(f'Beamformer Test -- Run {r}', fontsize = 20)
        ax.set_xlabel(f'Trigger Time [s]', fontsize = 20)
        ax.set_ylabel(f'Event Subsecond [s]', fontsize = 20)
        ax.tick_params(axis='both', which='both', labelsize=20)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/subsecond_vs_triggertime_r{r}.png', dpi=300)
        plt.close(fig)


        print(f'plotted subsecond hist for run {r}!')

    ### plot all runs
    all_trigger_time -= np.amin(all_trigger_time)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    ax.scatter(all_trigger_time, all_subsecond_fmod, s = 1, color = 'black', alpha = 0.5)
    ax.set_title(f'Beamformer Test -- Run {run_start} to {run_end}', fontsize = 20)
    ax.set_xlabel(f'Trigger Time [s]', fontsize = 20)
    ax.set_ylabel(f'Event Subsecond (% 0.1) [s]', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/subsecond_vs_triggertime_allruns.png', dpi=300)
    plt.close(fig)


def plot_wfs(run = 1420, n_events = 50):
    ### plot some of the raw waveform data for the beamformer testing
    if run < 800:
        os.environ["PUEO_ROOT_DATA"] = ("/project/cozzyd/LDB/rootified/")
    else:
        os.environ["PUEO_ROOT_DATA"] = ("/project/cozzyd/postflight/rootified/")

    channels = waveform.Channels(run, 0)

    ### and let's just plot the raw data
    for n in range(0, n_events):
        this_entry = int(channels.dataset.nextEvent())
        this_run = int(channels.run)

        this_entry = int(n)
        channels.load_run_event(this_run,  this_entry)
        pltr.plot_all_waveforms(channels, f'/scratch/midway3/darcynewmark/beamformer_testing/BTSallwfms_r{this_run}_e{this_entry}')

def plot_outlier_cluster(n_events = 10):
    ### in some runs, there is an explosion in the event rate, let's plot some of those events
    run = 1409
    channels = waveform.Channels(run, 0)
    ev_N  = channels.dataset.N()

    cluster_start_time = 235
    cluster_end_time = 240
    min_trigger_time = None
    n_plotted = 0

    for ev in range(ev_N):
        if ev == 0:
            this_entry = int(channels.dataset.firstEvent())
        else:
            this_entry = int(channels.dataset.nextEvent())
        if channels.dataset.header().trigType == 1:
            subsec = channels.subsecond()
            event_time = channels.triggerTime()
            if min_trigger_time == None:
                min_trigger_time = event_time
            event_time -= min_trigger_time

            if (event_time > cluster_start_time and event_time < cluster_end_time and n_plotted < n_events):
                ### plot event in our cluster!
                channels.load_run_event(run, this_entry)
                pltr.plot_all_waveforms(channels, f'/scratch/midway3/darcynewmark/beamformer_testing/BTSoutlierwfms_r{run}_e{this_entry}_time', freq=False)
                pltr.plot_all_waveforms(channels, f'/scratch/midway3/darcynewmark/beamformer_testing/BTSoutlierwfms_r{run}_e{this_entry}_freq', freq=True)
                print(f'plotted run = {run}, entry = {this_entry}, event_time = {event_time}')
                n_plotted += 1

if __name__ == "__main__":
    #plot_wfs(run = 1420)
    #plot_wfs(run = 719)
    #plot_wfs(run = 1433)
    plot_wfs(run = 1434)
    #subsec_plot()
    #plot_outlier_cluster()
