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

import pueoTALON.pueoPythonAnalyzer.waveform as waveform
import pueoTALON.pueoPythonAnalyzer.plotter as pltr
import pueoTALON.pueoPythonAnalyzer.antenna as antenna
import pueoTALON.pueoPythonAnalyzer.correlator as correlator
import antennaUtils.response as antResponse
import pueoTALON.pueoPythonAnalyzer.utils as utils
import pueoTALON.pueoPythonAnalyzer.geography as geography

ROOT.gSystem.Load("libpueoEvent.so")
PUEO_MC_DATA = os.environ['PUEO_MC_DATA']
BAD_SIM_RUNS = [196209]

def build_lookup(icefinal_tree):
    ### make look up dict between event number and interaction current
    icefinal_lookup = {}
    event_number_leaf = icefinal_tree.GetLeaf('eventNumber')
    current_leaf = icefinal_tree.GetLeaf('interaction.current')

    for ice_entry in range(icefinal_tree.GetEntries()):
        icefinal_tree.GetEntry(ice_entry)

        event_number = int(event_number_leaf.GetValue())
        current = int(current_leaf.GetValue())

        icefinal_lookup[event_number] = current

    return icefinal_lookup

def sim_stats():
    path = f'/project/avieregg/welling/pueo/simulation_dataset/run*'
    fnames = [name for name in glob.glob(path)]
    fnames = sorted(fnames)

    all_runs = []
    for f in fnames:
        this_run = f.split('/')[-1].split('run')[-1]
        all_runs.append(int(this_run))

    non_empty_runs = []
    sim_entries = []
    sim_pdg_code = []
    sim_weight = []
    sim_interaction_current = []
    empty_runs = []
    for run in all_runs:
        if run in BAD_SIM_RUNS:
            continue

        ### check to see if this run is empty or not
        event_file = f'{PUEO_MC_DATA}/run{run}/SimulatedPueoEventFile{run}.root'

        ### check if there are any events
        root_file = ROOT.TFile.Open(event_file)
        event_tree = root_file.Get('eventTree')
        if not event_tree or event_tree.GetEntries() == 0:
            empty_runs.append(run)
        else:
            channels = waveform.Channels(run, 0, sim=True) ### make channels object
            N_entry = channels.dataset.N() ### get total number of entries per run

            ### also get pdg code for neutrino
            truth_file = f'{PUEO_MC_DATA}/run{run}/SimulatedPueoTruthFile{run}.root'
            truth_root_file = ROOT.TFile.Open(truth_file)
            truth_tree = truth_root_file.Get('truthPueoTree')

            ### and interaction current from the icefinal file
            icefinal_file = f'{PUEO_MC_DATA}/run{run}/IceFinal_{run}_allTree.root'
            icefinal_root_file = ROOT.TFile.Open(icefinal_file)
            icefinal_tree = icefinal_root_file.Get('allTree')

            ### build a look up dict
            icefinal_lookup = build_lookup(icefinal_tree)

            for entry in range(truth_tree.GetEntries()):
                truth_tree.GetEntry(entry)
                event_number = int(truth_tree.GetLeaf('eventNumber').GetValue())
                pdg = truth_tree.truth.nu_pdg
                sim_pdg_code.append(pdg)
                weight = truth_tree.truth.weight
                sim_weight.append(weight)

                if event_number in icefinal_lookup:
                    sim_interaction_current.append(icefinal_lookup[event_number])
                else:
                    print(f'!!!!!!!!!!!!!! WARNING: No matching eventNumber {event_number} in IceFinal for run {run}')
                    sim_interaction_current.append(-1)

            ### now save run and entry
            [non_empty_runs.append(run) for n in range(N_entry)]
            [sim_entries.append(n) for n in range(N_entry)]
    print(f'total of {len(sim_entries)} simulated events!')

    ### now let's save our list of non empty neutrino sim!!
    saving_path = f'/scratch/midway3/darcynewmark/useful/nonempty_neutrino_sim_oct2026_weightsandcurrrent.json'
    data = {
        'run': non_empty_runs,
        'entry': sim_entries,
        'nu_pdg': sim_pdg_code,
        'weight': sim_weight,
        'interaction_current': sim_interaction_current,
    }
    with open(saving_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    sim_stats()
