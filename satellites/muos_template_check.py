import numpy as np
import ROOT
import os
import json
import glob
import argparse
from scipy import stats
import scipy.signal

import pueoTALON.pueoPythonAnalyzer.waveform as waveform
import pueoTALON.pueoPythonAnalyzer.plotter as pltr
import pueoTALON.pueoPythonAnalyzer.antenna as antenna
import pueoTALON.pueoPythonAnalyzer.correlator as correlator
import antennaUtils.response as antResponse
import pueoTALON.pueoPythonAnalyzer.utils as utils
import pueoTALON.pueoPythonAnalyzer.geography as geography

ROOT.gSystem.Load("libpueoEvent.so")

if __name__ == "__main__":
    ### let's go through just a few run, evs to really understand what muos template removal is doing
    runs = [813]
    evs = [2435]

    prev_run = None
    for r in range(len(runs)):
        this_run = int(runs[r])
        this_ev = int(evs[r])

        ### check if we have a correlator object for this run
        if prev_run == None or prev_run != this_run:
            ### set up correlator per run
            corr = correlator.Correlator(run = this_run, entry = this_ev)
            prev_run = this_run

        ### load this event
        corr.ChannelGeo.load_event(this_ev)
        corr.ChannelGeo.load_waveforms()

