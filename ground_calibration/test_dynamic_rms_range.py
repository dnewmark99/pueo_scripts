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
N_peaks = 1 
run = 793
entries = [3717, 4683, 4716, 4910,
           5067, 16883, 22129, 37323]
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

    ### get the coherent summed wf around the max peak
    cohsum = corr.coherent_sum(peak_el[0], peak_az[0],polarization=poli)

    ### get the snr using second half of wf
    fixed_snr, fixed_window_start, fixed_window_end = utils.SNR(cohsum, dynamic_rms_window = False, return_window = True)

    ### now get the snr using a dynamic window
    dynamic_snr, dynamic_window_start, dynamic_window_end = utils.SNR(cohsum, dynamic_rms_window = True, return_window = True)

    print(f'fixed snr = {fixed_snr} and dynamic snr = {dynamic_snr}')

    ### finally, plot!
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.minorticks_on()
    ax.plot(np.arange(cohsum.size)*corr.dt, cohsum, color = 'black', label = 'Coherent Summed Wf')
    ax.axvspan(fixed_window_start*corr.dt, fixed_window_end*corr.dt,  color = '#21b2aa', alpha = 0.3, label = f'Fixed Window SNR = {fixed_snr:0.2f}')
    ax.axvspan(dynamic_window_start*corr.dt, dynamic_window_end*corr.dt,  color = '#c71585', alpha = 0.3, label = f'Dynamic Window SNR = {dynamic_snr:0.2f}')
    ax.set_title(f'Taylor Dome Coherent Summed Wf, Run = {run}, Entry = {entry}')
    ax.set_xlabel(f'Time [ns]', fontsize = 20)
    ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
    ax.set_xlim(0, cohsum.size*corr.dt)
    ax.tick_params(axis='both', which='both', labelsize=16)
    ax.legend(fontsize=16, frameon=False)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/ground_cal/dynamic_snr_window_test_run={run}_entry={entry}.png', dpi = 300)
    plt.close(fig)

    print(f'plotted csw for entry = {entry}')
            

