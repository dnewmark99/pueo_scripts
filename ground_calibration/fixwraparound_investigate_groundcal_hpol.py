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

def grab_data_json(source_name = 'TD', base_path = os.environ['PUEO_ANALYSIS_TOOLS'] + '/pueoTALON/lookup/scratch/'):

    path = f'{base_path}/{source_name}_wraparoundandsnrfixes_*.json'
    fnames = [name for name in glob.glob(path)]
    fnames = sorted(fnames)
    print(f'found {len(fnames)} files for source_name = {source_name}!')

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

def plot_distance_vs_snr(source_name = 'TD'):
    ### first, grab our data
    data = grab_data_json(source_name)
    
    ### grab some info
    pol = np.asarray(data["pol"])
    trigger_time = np.asarray(data["trigger_time"])
    distance_to_pueo = np.asarray(data["distance_source"]) 
    cswSNR = np.asarray(data["cswSNR_samepol"])
    cswmaxHil = np.asarray(data["cswmax_hilbert_samepol"])
    maxwfSNR = np.asarray(data["maxwfSNR_samepol"])
    maxwfmaxHil = np.asarray(data["maxwfmax_hilbert_samepol"])
    maxwfmaxHil_opppol = np.asarray(data["maxwfmax_hilbert_opppol"])
    reco_az = np.asarray(data["reco_az_samepol"])
    reco_el = np.asarray(data["reco_el_samepol"])
    true_az = np.asarray(data["source_az"])
    true_el = np.asarray(data["source_el"])
    earthcoords_true_az = np.asarray(data["direction_source_az"])

    ### mask based on reco direction
    reco_dir_mask = np.logical_and(np.abs(reco_az - true_az) < 3, np.abs(reco_el - true_el) < 1)
    pol = pol[reco_dir_mask]
    trigger_time = trigger_time[reco_dir_mask]
    distance_to_pueo = distance_to_pueo[reco_dir_mask]
    cswSNR = cswSNR[reco_dir_mask]
    cswmaxHil = cswmaxHil[reco_dir_mask]
    maxwfSNR = maxwfSNR[reco_dir_mask]
    maxwfmaxHil = maxwfmaxHil[reco_dir_mask]
    maxwfmaxHil_opppol = maxwfmaxHil_opppol[reco_dir_mask]
    true_az = true_az[reco_dir_mask]
    earthcoords_true_az = earthcoords_true_az[reco_dir_mask]

    ### mask based on polarity
    hpol_mask = pol == 0
    vpol_mask = pol == 1

    ### now make 2D histograms!
    all_x_data = [trigger_time[hpol_mask], trigger_time[vpol_mask], distance_to_pueo[hpol_mask], distance_to_pueo[vpol_mask],
                  trigger_time[hpol_mask], trigger_time[vpol_mask], distance_to_pueo[hpol_mask], distance_to_pueo[vpol_mask],
                  trigger_time[hpol_mask], trigger_time[vpol_mask],
                  distance_to_pueo[hpol_mask], distance_to_pueo[vpol_mask],
                  maxwfmaxHil[hpol_mask], maxwfmaxHil[vpol_mask],
                  true_az[hpol_mask], true_az[vpol_mask],
                  earthcoords_true_az[hpol_mask], earthcoords_true_az[vpol_mask]]
    all_y_data = [cswmaxHil[hpol_mask], cswmaxHil[vpol_mask], cswmaxHil[hpol_mask], cswmaxHil[vpol_mask],
                  maxwfmaxHil[hpol_mask], maxwfmaxHil[vpol_mask], maxwfmaxHil[hpol_mask], maxwfmaxHil[vpol_mask],
                  cswSNR[hpol_mask], cswSNR[vpol_mask],
                  cswSNR[hpol_mask], cswSNR[vpol_mask],
                  maxwfmaxHil_opppol[hpol_mask], maxwfmaxHil_opppol[vpol_mask],
                  maxwfmaxHil[hpol_mask], maxwfmaxHil[vpol_mask],
                  maxwfmaxHil[hpol_mask], maxwfmaxHil[vpol_mask]]
    x_axes_labels = ['Trigger Time [s]', 'Trigger Time [s]', 'Distance to PUEO [m]', 'Distance to PUEO [m]',
                     'Trigger Time [s]', 'Trigger Time [s]', 'Distance to PUEO [m]', 'Distance to PUEO [m]',
                     'Trigger Time [s]', 'Trigger Time [s]',
                     'Distance to PUEO [m]', 'Distance to PUEO [m]',
                     'Conditioned Wf Max Hilbert Envelope (H-Pol)', 'Conditioned Wf Max Hilbert Envelope (V-Pol)',
                     f'{source_name} Azimuth (Payload Coords) [deg]', f'{source_name} Azimuth (Payload Coords) [deg]',
                     f'{source_name} Azimuth (Earth Coords) [deg]', f'{source_name} Azimuth (Earth Coords) [deg]']
    y_axes_labels = ['CSW Max Hilbert Envelope', 'CSW Max Hilbert Envelope', 'CSW Max Hilbert Envelope', 'CSW Max Hilbert Envelope', 
                     'Conditioned Wf Max Hilbert Envelope', 'Conditioned Wf Max Hilbert Envelope', 'Conditioned Wf Max Hilbert Envelope', 'Conditioned Wf Max Hilbert Envelope', 
                     'CSW SNR', 'CSW SNR',
                     'CSW SNR', 'CSW SNR',
                     'Conditioned Wf Max Hilbert Envelope (V-Pol)', 'Conditioned Wf Max Hilbert Envelope (H-Pol)',
                     'Conditioned Wf Max Hilbert Envelope', 'Conditioned Wf Max Hilbert Envelope',
                     'Conditioned Wf Max Hilbert Envelope', 'Conditioned Wf Max Hilbert Envelope',]
    labels = ['H-Pol' , 'V-Pol', 'H-Pol' , 'V-Pol', 
              'H-Pol' , 'V-Pol', 'H-Pol' , 'V-Pol', 
              'H-Pol', 'V-Pol',
              'H-Pol', 'V-Pol',
              'H-Pol', 'V-Pol',
              'H-Pol', 'V-Pol',
              'H-Pol', 'V-Pol',]
    saving_name = ['trigger_time_vs_cswmaxhil_hpol', 'trigger_time_vs_cswmaxhil_vpol', 'dist_vs_cswmaxhil_hpol', 'dist_vs_cswmaxhil_vpol', 
                    'trigger_time_vs_maxwfmaxhil_hpol', 'trigger_time_vs_maxwfmaxhil_vpol', 'dist_vs_maxwfmaxhil_hpol', 'dist_vs_maxwfmaxhil_vpol', 
                    'trigger_time_vs_cswsnr_hpol', 'trigger_time_vs_cswsnr_vpol',
                    'dist_vs_cswsnr_hpol', 'dist_vs_cswsnr_vpol',
                    'maxwfmaxhilopppol_vs_maxwfmaxhilsamepol_hpol', 'maxwfmaxhilopppol_vs_maxwfmaxhilsamepol_vpol',
                    'trueazdetectorcoords_vs_maxwfmaxhil_hpol', 'trueeazdetectorcoords_vs_maxwfmaxhil_vpol',
                    'trueazearthcoords_vs_maxwfmaxhil_hpol', 'trueeazearthcoords_vs_maxwfmaxhil_vpol',]

    ### also calculate median and sttdev for 1D plots afterwards
    all_y_data_median = []
    all_y_data_stddev = []
    all_x_data_centers = []
    found_discon_time = False

    for x in range(len(all_x_data)):
        x_data = np.asarray(all_x_data[x])
        y_data = np.asarray(all_y_data[x])

        if found_discon_time == False:
            ### find hpol discontinuity trigger time
            if x_axes_labels[x] == 'Trigger Time [s]' and y_axes_labels[x] == 'Conditioned Wf Max Hilbert Envelope':
                y_mask = y_data > 25
                first_idx = np.argmax(y_mask)
                discon_time = x_data[first_idx]
                print(f'TD H-pol discontinuity occurs at trigger time = {discon_time} (relative trigger time = {discon_time - np.amin(x_data)})')
                found_discon_time = True
        

        if x_axes_labels[x] == 'Trigger Time [s]':
            ### subtract off min trigger time to make plot more readable
            min_trigger_time = np.amin(x_data)
            x_data -= min_trigger_time
        if "Distance" in x_axes_labels[x]:
            ### quick mask on dist = 0
            quality_mask = x_data > 0
            x_data = x_data[quality_mask]
            y_data = y_data[quality_mask]

        ### set up bin edges
        nbins = 100
        if 'Azimuth' in x_axes_labels[x]:
            nbins = 50

        x_min, x_max = np.amin(x_data), np.amax(x_data)
        x_bin_width = (x_max - x_min) / nbins

        x_data_bin_edges = np.linspace(x_min - x_bin_width, x_max + x_bin_width, nbins + 3)

        y_min, y_max = np.amin(y_data), np.amax(y_data)
        y_bin_width = (y_max - y_min) / nbins

        y_data_bin_edges = np.linspace(y_min - y_bin_width, y_max + y_bin_width, nbins + 3)

        ### make our matrix
        M, _, _ = np.histogram2d(x_data, y_data, bins=[x_data_bin_edges, y_data_bin_edges])

        ### grab the median and sttdev for 1D projection
        x_data_bin_centers = (x_data_bin_edges[:-1] + x_data_bin_edges[1:]) / 2
        x_bin_idx = np.digitize(x_data, x_data_bin_edges) - 1

        median_y = np.full(len(x_data_bin_edges) - 1, 0.0)
        std_y = np.full(len(x_data_bin_edges) - 1, 0.0)

        for i in range(len(median_y)):
            y_slice = y_data[x_bin_idx == i]
            if len(y_slice) > 0:
                median_y[i] = np.median(y_slice)
                std_y[i] = np.std(y_slice)

        ### remove the padded edge bins for 1D plots
        all_x_data_centers.append(x_data_bin_centers[1:-1])
        all_y_data_median.append(median_y[1:-1])
        all_y_data_stddev.append(std_y[1:-1])

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
        plt.savefig(f'/scratch/midway3/darcynewmark/ground_cal/FixWrap_{source_name}_{saving_name[x]}.png', dpi=300)
        plt.close(fig)

    ### let's also do some 1D plots -- median w stddev
    for x in range(len(all_x_data_centers)):
        if 'Azimuth' in x_axes_labels[x]:
            x_data = np.asarray(all_x_data_centers[x])
            y_data = np.asarray(all_y_data_median[x])
            y_data_err = np.asarray(all_y_data_stddev[x])

            valid_mask = y_data > 0
            fitting_mask = functools.reduce(np.logical_and, [valid_mask, x_data > 0, y_data_err > 0])

            ### plot!
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.scatter(x_data[valid_mask], y_data[valid_mask], s = 20, c = 'black', label = r'Median $\pm$ $\sigma$')
            ax.errorbar(x_data[valid_mask], y_data[valid_mask], yerr=y_data_err[valid_mask], fmt='none', capsize=3, color = 'black')
            ax.set_title(f'{source_name} Ground Cal, {labels[x]} Trigger', fontsize = 16)
            ax.set_xlabel(f'{x_axes_labels[x]}', fontsize = 16)
            ax.set_ylabel(f'{y_axes_labels[x]}', fontsize = 16)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=14)
            ax.legend(fontsize=16, frameon=False)
            fig.tight_layout()
            plt.savefig(f'/scratch/midway3/darcynewmark/ground_cal/FixWrap_{source_name}_{saving_name[x]}_onedmedians.png', dpi=300)
            plt.close(fig)


if __name__ == "__main__":
    plot_distance_vs_snr(source_name = 'TD')
    #plot_distance_vs_snr(source_name = 'LDB')
    #plot_distance_vs_snr(source_name = 'S200')


