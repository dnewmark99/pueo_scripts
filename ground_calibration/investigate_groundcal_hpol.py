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

def grab_data_csv(source_name = 'TD', base_path = os.environ['PUEO_ANALYSIS_TOOLS'] + '/pueoTALON/lookup/scratch/'):

    runs = []
    entries = []
    pol = []
    SNR = []
    trigger_time = []
    distance_to_pueo = []
    source_azimuth = []

    path = f'{base_path}/{source_name}*.csv'
    fnames = [name for name in glob.glob(path)]
    fnames = sorted(fnames)
    print(f'found {len(fnames)} files for source_name = {source_name}!')

    ### read in our data
    for f in fnames:
        data = np.loadtxt(f, delimiter = ',', dtype=float)
        for val in data:
            #val[4] contains the az difference in degrees between the events best reconstruction az and the true TD az
            #First correct this difference for wrap around effects
            if val[4] > 180:
                val[4] -= 360
            if val[4] < -180:
                val[4] += 360
            #Remove any accidentals in list by requiring that best reconstruction direction is sufficiently close in degrees to the true direction
            #val[3] is the el difference in degrees between the best reconstruction direction and the TD direction
            if np.abs(val[4]) < 3 and np.abs(val[3]) < 1:
                runs.append(val[0])
                entries.append(val[1])
                pol.append(val[2])
                SNR.append(val[5]) #SNR of CSW in best guess direction
                trigger_time.append(val[8])
                distance_to_pueo.append(val[11])
                ### get td az for this event
                az_diff = val[4] ### az_diff == td_az - reco_az
                reco_az = val[6] 
                td_az = az_diff + reco_az ### td_az - reco_az + reco_az = td_az !
                source_azimuth.append(td_az)

    runs = np.asarray(runs)
    entries = np.asarray(entries)
    pol = np.asarray(pol)
    SNR = np.asarray(SNR)
    trigger_time = np.asarray(trigger_time)
    distance_to_pueo = np.asarray(distance_to_pueo)
    source_azimuth = np.asarray(source_azimuth)

    return runs, entries, pol, SNR, trigger_time, distance_to_pueo, source_azimuth

def fit_func(x, a, n, c):
    return (a / (x**n)) + c

weird_hpol_dist = [100000, 200000]
weird_hpol_snr = [15, 25]

def plot_distance_vs_snr(source_name = 'TD'):
    ### first, grab our data
    runs, entries, pol, SNR, trigger_time, distance_to_pueo, source_azimuth = grab_data_csv(source_name)

    ### mask based on polarity
    hpol_mask = pol == 0
    vpol_mask = pol == 1

    #### print our weird hpol runs!
    #weird_hpol_dist_mask = np.logical_and(distance_to_pueo > weird_hpol_dist[0], distance_to_pueo < weird_hpol_dist[1])
    #weird_hpol_snr_mask = np.logical_and(SNR > weird_hpol_snr[0], SNR < weird_hpol_snr[1])
    #weird_hpol_mask = functools.reduce(np.logical_and, [hpol_mask, weird_hpol_dist_mask, weird_hpol_snr_mask])
    #weird_hpol_runs = runs[weird_hpol_mask]
    #weird_hpol_entries = entries[weird_hpol_mask]
    #print(f'total of {np.sum(weird_hpol_mask)} weird hpol events!')
    #start_idx = 1100
    #end_idx = 1200
    #print(f'weird_hpol_runs = {weird_hpol_runs[start_idx:end_idx]}')
    #print(f'weird_hpol_entries = {weird_hpol_entries[start_idx:end_idx]}')

    ### now make 2D histograms!
    all_x_data = [trigger_time[hpol_mask], trigger_time[vpol_mask], distance_to_pueo[hpol_mask], distance_to_pueo[vpol_mask], source_azimuth[hpol_mask], source_azimuth[vpol_mask]]
    all_y_data = [SNR[hpol_mask], SNR[vpol_mask], SNR[hpol_mask], SNR[vpol_mask], SNR[hpol_mask], SNR[vpol_mask]]
    x_axes_labels = ['Trigger Time [s]', 'Trigger Time [s]', 'Distance to PUEO [m]', 'Distance to PUEO [m]', 'Source Azimuth [deg]', 'Source Azimuth [deg]']
    y_axes_labels = ['CSW SNR', 'CSW SNR', 'CSW SNR', 'CSW SNR', 'CSW SNR', 'CSW SNR']
    labels = ['HPol' , 'VPol', 'HPol' , 'VPol', 'HPol' , 'VPol']
    saving_name = ['trigger_time_vs_cswsnr_hpol', 'trigger_time_vs_cswsnr_vpol', 'dist_vs_cswsnr_hpol', 'dist_vs_cswsnr_vpol', 'sourceaz_vs_cswsnr_hpol', 'sourceaz_vs_cswsnr_vpol']

    ### also calculate median and sttdev for 1D plots afterwards
    all_y_data_median = []
    all_y_data_stddev = []
    all_x_data_centers = []

    for x in range(len(all_x_data)):
        x_data = np.asarray(all_x_data[x])
        y_data = np.asarray(all_y_data[x])

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
        all_x_data_centers.append(x_data_bin_centers)

        x_bin_idx = np.digitize(x_data, x_data_bin_edges) - 1
        median_y = np.full(len(x_data_bin_edges) - 1, 0)
        std_y = np.full(len(x_data_bin_edges) - 1, 0)
        for i in range(len(median_y)):
            y_slice = y_data[x_bin_idx == i]

            if len(y_slice) > 0:
                median_y[i] = np.median(y_slice)
                std_y[i] = np.std(y_slice)

        all_y_data_median.append(median_y)
        all_y_data_stddev.append(std_y)

        ### plot!
        cm = plt.get_cmap("Spectral_r")
        lin_norm = matplotlib.colors.LogNorm(vmin=1, vmax=np.amax(M.T))

        fig, ax = plt.subplots(figsize=(10, 7))
        mesh = ax.pcolormesh(x_data_bin_edges, y_data_bin_edges, M.T, cmap=cm, norm = lin_norm)
        cb = fig.colorbar(mesh, ax=ax)
        cb.ax.set_ylabel("Number of Events", fontsize = 16)
        cb.ax.minorticks_off()
        ax.set_title(f'{source_name} Ground Cal, {labels[x]}', fontsize = 16)
        ax.set_xlabel(f'{x_axes_labels[x]}', fontsize = 16)
        ax.set_ylabel(f'{y_axes_labels[x]}', fontsize = 16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        fig.tight_layout()
        plt.savefig(f'/scratch/midway3/darcynewmark/ground_cal/OldProcessing_{source_name}_{saving_name[x]}.png', dpi=300)
        plt.close(fig)

    ### let's also do some 1D plots -- median w stddev
    for x in range(len(all_x_data_centers)):
        x_data = np.asarray(all_x_data_centers[x])
        y_data = np.asarray(all_y_data_median[x])
        y_data_err = np.asarray(all_y_data_stddev[x])

        valid_mask = y_data > 0
        fitting_mask = functools.reduce(np.logical_and, [valid_mask, x_data > 0, y_data_err > 0])

        fig, ax = plt.subplots(figsize=(10, 7))

        #if x_axes_labels[x] == 'Distance to PUEO [m]':
        #    ### let's also fit our data with 1/r^n
        #    popt, pcov = curve_fit(fit_func, x_data[fitting_mask], y_data[fitting_mask], sigma=y_data_err[fitting_mask], absolute_sigma=True, maxfev = int(1e4),
        #                            p0=[1e12, 2, 0], bounds=([0, 1, 0], [1e20, 3, 100] ))
        #    perr = np.sqrt(np.diag(pcov))
        #    best_fit_a = popt[0]
        #    best_fit_n = popt[1]
        #    best_fit_c = popt[2]
        #
        #    fine_grained_x = np.linspace(np.amin(x_data[fitting_mask]), np.amax(x_data[fitting_mask]), 1000)
        #    fine_grained_pred = fit_func(fine_grained_x, best_fit_a, best_fit_n, best_fit_c)
        #    ax.plot(fine_grained_x, fine_grained_pred,  color = '#c71585')
        #    print(f"a = {popt[0]:.3g} +/- {perr[0]:.3g}")
        #    print(f"n = {popt[1]:.3g} +/- {perr[1]:.3g}")
        #    print(f"c = {popt[2]:.3g} +/- {perr[2]:.3g}")
        #if x_axes_labels[x] == 'Distance to PUEO [m]':
        #    temp_mask = np.logical_and(x_data > 100000, x_data < 300000)
        #    temp_combined_mask = functools.reduce(np.logical_and, [valid_mask, temp_mask])
        #    temp_x = x_data[temp_combined_mask]
        #    temp_y = y_data[temp_combined_mask]

        #    max_snr_distance = temp_x[np.argmax(temp_y)]
        #    min_snr_distance = temp_x[np.argmin(temp_y)]
        #    ax.axvline(max_snr_distance, color = '#c71585', label = f'SNR Peak at Dist = {max_snr_distance:.3e}m')
        #    ax.axvline(min_snr_distance, color = '#21b2aa', label = f'SNR Trough at Dist = {min_snr_distance:.3e}m')

        ### plot!
        ax.scatter(x_data[valid_mask], y_data[valid_mask], s = 20, c = 'black', label = r'Median $\pm$ $\sigma$')
        ax.errorbar(x_data[valid_mask], y_data[valid_mask], yerr=y_data_err[valid_mask], fmt='none', capsize=3, color = 'black')
        ax.set_title(f'{source_name} Ground Cal, {labels[x]}', fontsize = 16)
        ax.set_xlabel(f'{x_axes_labels[x]}', fontsize = 16)
        ax.set_ylabel(f'{y_axes_labels[x]}', fontsize = 16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        ax.legend(fontsize=16, frameon=False)
        fig.tight_layout()
        plt.savefig(f'/scratch/midway3/darcynewmark/ground_cal/OldProcessing_{source_name}_{saving_name[x]}_onedmedians.png', dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    plot_distance_vs_snr(source_name = 'TD')
    #plot_distance_vs_snr(source_name = 'LDB')


