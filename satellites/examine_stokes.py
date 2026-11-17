import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import os
import json
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import glob
import functools
from functools import reduce

def grab_data_json(base_path = '/scratch/midway3/darcynewmark/satellites/jsons'):

    data = {}
    path = f'{base_path}/v3stokes_metrics_*.json'
    fnames = [name for name in glob.glob(path)]
    fnames = sorted(fnames)

    ### read in our data
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

def one_d_plots(muos_x_data, nonmuos_x_data, groundcal_mask, rf_mask, numc_mask, numc_weight, title, x_label, saving_name, x_scale, eff = False):
    colors = ['black', '#d62728', '#1e90ff']
    labels = ['RF Triggers', 'TD Ground Cal', 'NuMC']
    
    muos_x_dist = [muos_x_data[rf_mask], muos_x_data[groundcal_mask], muos_x_data[numc_mask]]
    nonmuos_x_dist = [nonmuos_x_data[rf_mask], nonmuos_x_data[groundcal_mask], nonmuos_x_data[numc_mask]]
    weights = [np.ones(len(muos_x_data[rf_mask])), np.ones(len(muos_x_data[groundcal_mask])), numc_weight[numc_mask],]

    bin_start = np.amin([np.amin(muos_x_data[rf_mask]), np.amin(muos_x_data[groundcal_mask]), np.amin(muos_x_data[numc_mask]),
                         np.amin(nonmuos_x_data[rf_mask]), np.amin(nonmuos_x_data[groundcal_mask]), np.amin(nonmuos_x_data[numc_mask])]) 
    bin_end = np.amax([np.amax(muos_x_data[rf_mask]), np.amax(muos_x_data[groundcal_mask]), np.amax(muos_x_data[numc_mask]),
                       np.amax(nonmuos_x_data[rf_mask]), np.amax(nonmuos_x_data[groundcal_mask]), np.amax(nonmuos_x_data[numc_mask])]) 
    
    nbins = 500

    if x_scale == 'log':
        bins = np.logspace(np.log10(bin_start), np.log10(bin_end), nbins + 1)
    else:
        bins = np.linspace(bin_start, bin_end, nbins + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    for d in range(len(muos_x_dist)):
        ax.hist(nonmuos_x_dist[d], bins = bins, weights = weights[d], density=True, linewidth = 1.5, histtype='step', linestyle = '-', color = colors[d], label = f'{labels[d]} (MUOS Notched)')
        #ax.hist(muos_x_dist[d], bins = bins, density=True, linewidth = 1.5, histtype='step', linestyle = '--', color = colors[d], label = f'{labels[d]} (MUOS Only)')

    ax.set_title(title, fontsize = 20)
    ax.set_xlabel(x_label, fontsize = 20)
    ax.set_ylabel(f'Probability Density', fontsize = 20)
    ax.legend(fontsize=14, frameon=False)
    ax.set_yscale('log')
    if x_scale == 'log':
        ax.set_xscale('log')
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', labelsize=20)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/satellites/nomuos{saving_name}.png', dpi=300)
    plt.close(fig)

def two_d_plots(run, entry, this_x, this_y, groundcal_vpol_mask, rf_mask, numc_mask, numc_weight, x_label, y_label, saving_name):
    labels = ['TD','RFTriggers', 'NuMC']
    titles = ['Taylor Dome (V-Pol)', 'RF Triggers', 'NuMC',]
    
    x_dist = [this_x[groundcal_vpol_mask], this_x[rf_mask], this_x[numc_mask], ]
    y_dist = [this_y[groundcal_vpol_mask], this_y[rf_mask], this_y[numc_mask], ]
    run_dist = [run[groundcal_vpol_mask], run[rf_mask], run[numc_mask]]
    entry_dist = [entry[groundcal_vpol_mask], entry[rf_mask], entry[numc_mask]]
    weight_dist = [None, None, numc_weight[numc_mask],]

    nbins = 100
    
    ### one plot per variable
    for d in range(len(x_dist)):
        if d == 2:
            log_space = True
        else:
            log_space = True 

        this_x_data = x_dist[d]
        this_y_data = y_dist[d]
        this_weights = weight_dist[d]
        this_run = run_dist[d]
        this_entry = entry_dist[d]

        ### let's print a few outlier runs, evs
        if d == 0:
            outlier_snr = (70, 110)
            outlier_pol = (0.4, 0.6)
            #outlier_pol = (0.7, 1)
            outlier_mask = functools.reduce(np.logical_and, [this_x_data > outlier_snr[0], this_x_data < outlier_snr[1], this_y_data > outlier_pol[0], this_y_data < outlier_pol[1]])
            print(f'for {titles[d]}, have {np.sum(outlier_mask)} outlier events!')
            outlier_runs = this_run[outlier_mask]
            outlier_entries = this_entry[outlier_mask]
            print(f'first 10 outlier runs = {outlier_runs[:10].tolist()}')
            print(f'first 10 outlier entries = {outlier_entries[:10].tolist()}')

        ### set up bin edges
        x_min, x_max = np.amin(this_x_data), np.amax(this_x_data)
        y_min, y_max = np.amin(this_y_data), np.amax(this_y_data)

        if log_space:
            x_data_bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), nbins + 1)
        else:
            x_data_bin_edges = np.linspace(x_min, x_max, nbins + 1)
        
        y_data_bin_edges = np.linspace(y_min, y_max, nbins + 1)

        ### make our matrix
        M, _, _ = np.histogram2d(this_x_data, this_y_data, bins=[x_data_bin_edges, y_data_bin_edges], weights=this_weights)

        ### plot!
        cm = plt.get_cmap("Spectral_r")
        lin_norm = matplotlib.colors.LogNorm(vmin=np.amin(M.T[M.T>0]), vmax=np.amax(M.T))

        fig, ax = plt.subplots(figsize=(10, 7))
        mesh = ax.pcolormesh(x_data_bin_edges, y_data_bin_edges, M.T, cmap=cm, norm = lin_norm)
        cb = fig.colorbar(mesh, ax=ax)
        cb.ax.set_ylabel("Sum of MC Weights" if d == 2 else "Number of Events", fontsize=20)
        cb.ax.minorticks_off()
        if log_space:
            ax.set_xscale('log')
        ax.set_title(f'{titles[d]}', fontsize = 20)
        ax.set_xlabel(x_label, fontsize = 20)
        ax.set_ylabel(y_label, fontsize = 20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        fig.tight_layout()
        plt.savefig(f'/scratch/midway3/darcynewmark/satellites/twod_{labels[d]}_{saving_name}.png', dpi=300)
        plt.close(fig)

def plot_vars():
    ### let's read in our json data
    data = grab_data_json()
    
    ### unpack it all
    run = np.asarray(data['run'])
    entry = np.asarray(data['entry'])
    flag = np.asarray(data['flag'])
    groundcal_pol = np.asarray(data['groundcal_pol'])
    numc_pdg = np.asarray(data['numc_pdg'])
    numc_weight = np.asarray(data['numc_weight'])
    numc_interaction_current = np.asarray(data['numc_interaction_current'])
    trigger_time = np.asarray(data['trigger_time'])
    trigger_subsecond = np.asarray(data['trigger_subsecond'])
    saturation_flag = np.asarray(data['saturation_flag'])
    blast_flag = np.asarray(data['blast_flag'])
    deadtime_flag = np.asarray(data['deadtime_flag'])
    low_channel_rms_flag = np.asarray(data['low_channel_rms_flag'])
    avg_lin_vpol = np.asarray(data['avg_lin_vpol'])
    avg_circ_vpol = np.asarray(data['avg_circ_vpol'])
    avg_pol_vpol = np.asarray(data['avg_pol_vpol'])
    avg_angle_vpol = np.asarray(data['avg_angle_vpol'])
    isolate_muos_avg_lin_vpol = np.asarray(data['isolate_muos_avg_lin_vpol'])
    isolate_muos_avg_circ_vpol = np.asarray(data['isolate_muos_avg_circ_vpol'])
    isolate_muos_avg_pol_vpol = np.asarray(data['isolate_muos_avg_pol_vpol'])
    isolate_muos_avg_angle_vpol = np.asarray(data['isolate_muos_avg_angle_vpol'])
    avg_lin_hpol = np.asarray(data['avg_lin_hpol'])
    avg_circ_hpol = np.asarray(data['avg_circ_hpol'])
    avg_pol_hpol = np.asarray(data['avg_pol_hpol'])
    avg_angle_hpol = np.asarray(data['avg_angle_hpol'])
    isolate_muos_avg_lin_hpol = np.asarray(data['isolate_muos_avg_lin_hpol'])
    isolate_muos_avg_circ_hpol = np.asarray(data['isolate_muos_avg_circ_hpol'])
    isolate_muos_avg_pol_hpol = np.asarray(data['isolate_muos_avg_pol_hpol'])
    isolate_muos_avg_angle_hpol = np.asarray(data['isolate_muos_avg_angle_hpol'])
    csw_snr_vpol = np.asarray(data['csw_snr_vpol'])
    csw_snr_vpol = csw_snr_vpol.reshape(-1, 2)[:, 1]
    csw_snr_hpol = np.asarray(data['csw_snr_hpol'])
    csw_snr_hpol = csw_snr_hpol.reshape(-1, 2)[:, 1]

    ### add basic saturation, deadtime, and blast mask
    valid_mask = functools.reduce(np.logical_and, [saturation_flag == False, blast_flag == False, deadtime_flag == False, low_channel_rms_flag == False])
    print(f'basic cuts on saturation, deadtime, blast, and low thermal remove {len(saturation_flag) - np.sum(valid_mask)} events!')
    print(f'left with {np.sum(valid_mask)} events after basic cuts!')
    run = run[valid_mask]
    entry = entry[valid_mask]
    flag = flag[valid_mask]
    groundcal_pol = groundcal_pol[valid_mask]
    numc_pdg = numc_pdg[valid_mask]
    numc_weight = numc_weight[valid_mask]
    numc_interaction_current = numc_interaction_current[valid_mask]
    trigger_time = trigger_time[valid_mask]
    trigger_subsecond = trigger_subsecond[valid_mask]
    saturation_flag = saturation_flag[valid_mask]
    blast_flag = blast_flag[valid_mask]
    deadtime_flag = deadtime_flag[valid_mask]
    low_channel_rms_flag = low_channel_rms_flag[valid_mask]
    avg_lin_vpol = avg_lin_vpol[valid_mask]
    avg_circ_vpol = avg_circ_vpol[valid_mask]
    avg_pol_vpol = avg_pol_vpol[valid_mask]
    avg_angle_vpol = avg_angle_vpol[valid_mask]
    isolate_muos_avg_lin_vpol = isolate_muos_avg_lin_vpol[valid_mask]
    isolate_muos_avg_circ_vpol = isolate_muos_avg_circ_vpol[valid_mask]
    isolate_muos_avg_pol_vpol = isolate_muos_avg_pol_vpol[valid_mask]
    isolate_muos_avg_angle_vpol = isolate_muos_avg_angle_vpol[valid_mask]
    avg_lin_hpol = avg_lin_hpol[valid_mask]
    avg_circ_hpol = avg_circ_hpol[valid_mask]
    avg_pol_hpol = avg_pol_hpol[valid_mask]
    avg_angle_hpol = avg_angle_hpol[valid_mask]
    isolate_muos_avg_lin_hpol = isolate_muos_avg_lin_hpol[valid_mask]
    isolate_muos_avg_circ_hpol = isolate_muos_avg_circ_hpol[valid_mask]
    isolate_muos_avg_pol_hpol = isolate_muos_avg_pol_hpol[valid_mask]
    isolate_muos_avg_angle_hpol = isolate_muos_avg_angle_hpol[valid_mask]
    csw_snr_vpol = csw_snr_vpol[valid_mask]
    csw_snr_hpol = csw_snr_hpol[valid_mask]

    ### mask our events for now
    groundcal_vpol_mask = np.logical_and(flag == 'TD', groundcal_pol == 'V')
    rf_mask = flag == 'RF'
    nu_mask = flag == 'NuMC'
    print(f'have {np.sum(groundcal_vpol_mask)} ground cal v-pol events, {np.sum(rf_mask)} rf triggers, and {np.sum(nu_mask)} simulated neutrinos!')
    
    ### now let's make some  plots!!
    plotting_muos = [isolate_muos_avg_circ_vpol[:,0], isolate_muos_avg_lin_vpol[:,0],
                     isolate_muos_avg_circ_vpol[:,1], isolate_muos_avg_lin_vpol[:,1], 
                     isolate_muos_avg_circ_vpol[:,2], isolate_muos_avg_lin_vpol[:,2], 
                     isolate_muos_avg_circ_vpol[:,3], isolate_muos_avg_lin_vpol[:,3], 
                    ]
    plotting_nonmuos = [avg_circ_vpol[:,0], avg_lin_vpol[:,0],
                        avg_circ_vpol[:,1], avg_lin_vpol[:,1],
                        avg_circ_vpol[:,2], avg_lin_vpol[:,2],
                        avg_circ_vpol[:,3], avg_lin_vpol[:,3],
                       ]
    plotting_x_saving_name = ['avg_circ_entiretimewindow', 'avg_lin_entiretimewindow',
                              'avg_circ_triggertimewindow', 'avg_lin_triggertimewindow',
                              'avg_circ_smallesttimewindow', 'avg_lin_smallesttimewindow',
                              'avg_circ_secondsmallesttimewindow', 'avg_lin_secondsmallesttimewindow',
                             ]
    plotting_x_label = ['Avg Circular Polarization', 'Avg Linear Polarization',
                        'Avg Circular Polarization', 'Avg Linear Polarization',
                        'Avg Circular Polarization', 'Avg Linear Polarization',
                        'Avg Circular Polarization', 'Avg Linear Polarization']
    plotting_x_scale = ['linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear',]
    plotting_titles = ['Entire Time Window', 'Entire Time Window',
                       'Trigger Time Window', 'Trigger Time Window',
                       'Smallest Time Window', 'Smallest Time Window',
                       'Second Smallest Time Window', 'Second Smallest Time Window',]

    for p in range(len(plotting_muos)):
        this_muos = plotting_muos[p]
        this_nonmuos = plotting_nonmuos[p]

        ### let's first make 1D plots for each variable
        one_d_plots(this_muos, this_nonmuos, groundcal_vpol_mask, rf_mask, nu_mask, numc_weight, f'{plotting_x_label[p]} ({plotting_titles[p]})', plotting_x_label[p], plotting_x_saving_name[p], plotting_x_scale[p])

    ### let's make a special 2d snr vs trigger time avg lin pol
    plotting_x = [csw_snr_vpol]
    plotting_y = [avg_lin_vpol[:,1]]
    plotting_x_saving_name = ['csw_snr_vpol']
    plotting_y_saving_name = ['avg_lin_triggertimewindow']
    plotting_x_label = ['CSW SNR (V-Pol)',] 
    plotting_y_label = ['Avg Linear Polarization (Trigger Window)',] 

    for p in range(len(plotting_x)):
        this_x = plotting_x[p]
        this_y = plotting_y[p]

        ### now our 2D plots!!!
        two_d_plots(run, entry, this_x, this_y, groundcal_vpol_mask, rf_mask, nu_mask, numc_weight, plotting_x_label[p], plotting_y_label[p], f'{plotting_x_saving_name[p]}_{plotting_y_saving_name[p]}')
        
if __name__ == "__main__":
    plot_vars()
