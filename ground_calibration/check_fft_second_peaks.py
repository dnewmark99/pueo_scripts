import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ROOT
import scipy
import os
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import functools
from functools import reduce

import pueoTALON.pueoPythonAnalyzer.waveform as waveform
import pueoTALON.pueoPythonAnalyzer.plotter as pltr
import pueoTALON.pueoPythonAnalyzer.antenna as antenna
import pueoTALON.pueoPythonAnalyzer.correlator as correlator
import antennaUtils.response as antResponse
import pueoTALON.pueoPythonAnalyzer.utils as utils
import pueoTALON.pueoPythonAnalyzer.geography as geography

ROOT.gSystem.Load("libpueoEvent.so")

### testing snr calculating using dynamic window for computing the rms
### we already have a run and event that we know has some problems
N_peaks = 2 
run = 793
entries = [3717, 4683, 4716, 4910]
for entry in entries:
    ### grab the data
    corr = correlator.Correlator()
    geography_helper = geography.GeographyHelper(root_file=os.environ['PUEO_ANALYSIS_TOOLS'] + "/pueoTALON/lookup/abxtwo.root", root_source="ABX-Two")
    corr.ChannelGeo.load_run_event(int(run),  int(entry))
    event_time = corr.ChannelGeo.triggerTime()
    geography_helper.updatePosition(event_time, max_time_delta = 3)
                    
    ### get expected taylor dome location
    expected_td_el_az = geography_helper.get_expected_source_location('taylor_dome')
    expected_sun_el_az = geography_helper.get_sun_position(event_time)

    #make skymap and find peak location
    poli = 'V'
    skymap, peak_el, peak_az, _, zskymap, peak_zel, peak_zaz, _, all_zel, all_zaz  = corr.reconstruct(poli, zoom_angle_range=7, N_peaks=N_peaks)

    ### get the coherent summed wf around the second peak
    cohsum = corr.coherent_sum(peak_el[1], peak_az[1],polarization=poli)
    cohsum_times = np.arange(cohsum.size)*corr.dt

    ### plot fft of differnet time windows
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    ax[0].minorticks_on()
    ax[1].minorticks_on()
    ax[0].plot(cohsum_times, cohsum, color = 'black')
    
    time_windows = [(-1e6, 1e6), (50,200)]
    colors = ['black', '#c71585']

    for t in range(len(time_windows)):
        t_start = time_windows[t][0]
        t_end = time_windows[t][1]

        t_mask = np.logical_and(cohsum_times > t_start, cohsum_times < t_end)
        f, vf = utils.fourierTransform(cohsum_times[t_mask], cohsum[t_mask])
        p = utils.volt_to_dB(np.abs(vf))

        ax[0].plot(cohsum_times[t_mask], cohsum[t_mask], color = colors[t])
        ax[1].plot(f, p, color = colors[t])

    fig.suptitle(f'Taylor Dome Coherent Summed Wf (2nd Peak), Run = {run}, Entry = {entry}')
    ax[0].set_xlabel(f'Time [ns]', fontsize = 20)
    ax[0].set_ylabel(f'Voltage [mV]', fontsize = 20)
    ax[0].set_xlim(0, cohsum.size*corr.dt)
    ax[0].tick_params(axis='both', which='both', labelsize=16)
    ax[1].set_xlabel(f'Frequency [GHz]', fontsize = 20)
    ax[1].set_ylabel(f'Gain [dB]', fontsize = 20)
    ax[1].tick_params(axis='both', which='both', labelsize=16)
    ax[1].set_ylim(10,80)
    ax[1].set_xlim(0,1.5)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/ground_cal/secondpeak_fft_test_run={run}_entry={entry}.png', dpi = 300)
    plt.close(fig)

    print(f'plotted csw for entry = {entry}')
            

