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

def grab_data_json(source_name = 'TD', base_path = os.environ['PUEO_ANALYSIS_TOOLS'] + '/pueoTALON/lookup/scratch/', descrip = 'wraparoundandsnrfixes'):

    path = f'{base_path}/{source_name}_{descrip}_*.json'
    fnames = [name for name in glob.glob(path)]
    fnames = sorted(fnames)
    print(f'found {len(fnames)} files for source_name = {source_name} from descrip = {descrip}!')

    ### read in our data
    data = {}
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

def compare_pointing(source_name = 'TD'):
    ### first, grab our data
    ch0_sum_data = grab_data_json(source_name, descrip = 'wraparoundandsnrfixesREMOVESATURATEDEVENTS')
    origin_sum_data = grab_data_json(source_name, descrip = 'wraparoundandsnrfixesREMOVESATURATEDEVENTScohsumrelorigin')
    
    ### grab some info
    ch0_pol = np.asarray(ch0_sum_data["pol"])
    ch0_reco_az = np.asarray(ch0_sum_data["reco_az_samepol"])
    ch0_reco_el = np.asarray(ch0_sum_data["reco_el_samepol"])
    ch0_true_az = np.asarray(ch0_sum_data["source_az"])
    ch0_true_el = np.asarray(ch0_sum_data["source_el"])
    ch0_delta_az = ch0_true_az - ch0_reco_az
    ch0_delta_el = ch0_true_el - ch0_reco_el

    origin_pol = np.asarray(origin_sum_data["pol"])
    origin_reco_az = np.asarray(origin_sum_data["reco_az_samepol"])
    origin_reco_el = np.asarray(origin_sum_data["reco_el_samepol"])
    origin_true_az = np.asarray(origin_sum_data["source_az"])
    origin_true_el = np.asarray(origin_sum_data["source_el"])
    origin_delta_az = origin_true_az - origin_reco_az
    origin_delta_el = origin_true_el - origin_reco_el

    ### mask based on polarity
    ch0_vpol_mask = ch0_pol == 1
    origin_vpol_mask = origin_pol == 1

    ### now make 2D histograms!
    all_x_data = [ch0_delta_el[ch0_vpol_mask], origin_delta_el[origin_vpol_mask]]

    all_y_data = [ch0_delta_az[ch0_vpol_mask], origin_delta_az[origin_vpol_mask]]
            
    x_axes_labels = ['Delta El', 'Delta El']
    
    y_axes_labels = ['Delta Az', 'Delta Az']
            
    labels = ['V-Pol (CohSum Rel Ch0)' , 'V-Pol (CohSum Rel Origin)']
    
    saving_name = ['cohsumrelch0_deltael_deltaaz_vpol', 'cohsumrelorigin_deltael_deltaaz_vpol']
    
    for x in range(len(all_x_data)):
        x_data = np.asarray(all_x_data[x])
        y_data = np.asarray(all_y_data[x])

        ### set up bin edges -- hardcoding bin range fyi
        nbins = 100
        x_data_bin_edges = np.linspace(-2, 2, nbins + 1)
        y_data_bin_edges = np.linspace(-5, 5, nbins + 1)

        ### make our matrix
        M, _, _ = np.histogram2d(x_data, y_data, bins=[x_data_bin_edges, y_data_bin_edges])

        ### plot!
        cm = plt.get_cmap("Spectral_r")
        lin_norm = matplotlib.colors.LogNorm(vmin=1, vmax=np.amax(M.T))

        fig, ax = plt.subplots(figsize=(10, 7))
        mesh = ax.pcolormesh(x_data_bin_edges, y_data_bin_edges, M.T, cmap=cm, norm = lin_norm)
        cb = fig.colorbar(mesh, ax=ax)
        cb.ax.set_ylabel("Number of Events", fontsize = 16)
        cb.ax.minorticks_off()
        ax.set_title(f'{source_name} Ground Cal, {labels[x]} Trigger', fontsize = 16)
        ax.set_xlabel(f'{x_axes_labels[x]}', fontsize = 16)
        ax.set_ylabel(f'{y_axes_labels[x]}', fontsize = 16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        fig.tight_layout()
        plt.savefig(f'/scratch/midway3/darcynewmark/ground_cal/CheckDirReco_{source_name}_{saving_name[x]}.png', dpi=300)
        plt.close(fig)
            
if __name__ == "__main__":
    compare_pointing() 

