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
#run = 793
#entries = [3717, 4683, 4716, 4910,
#           5067, 16883, 22129, 37323]
#run = 786
#entries = [108302, 108380, 108436]
run = 790
entries = [130433, 130548, 130623]
for entry in entries:
    ### grab the data
    corr = correlator.Correlator()
    geography_helper = geography.GeographyHelper(root_file=os.environ['PUEO_ANALYSIS_TOOLS'] + "/pueoTALON/lookup/abxtwo.root", root_source="ABX-Two")
    corr.ChannelGeo.load_run_event(int(run),  int(entry))
    event_time = corr.ChannelGeo.triggerTime()
    geography_helper.updatePosition(event_time, max_time_delta = 3)
                    

    ### finally, plot!
    pltr.plot_all_waveforms(corr.ChannelGeo, "/scratch/midway3/darcynewmark/ground_cal/allwfms_r%d_e%d" % (run, entry))

    print(f'plotted wfs for entry = {entry}')
            

