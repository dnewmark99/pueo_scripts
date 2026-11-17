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

def grab_data_json(base_path = '/scratch/midway3/darcynewmark/secondaries/multipulse_metrics/jsons'):

    data = {}
    path = f'{base_path}/secondary_metrics_*.json'
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

def one_d_plots(this_x, numc_weight, groundcal_vpol_mask, force_mask, rf_mask, numc_mask_nue, numc_mask_numu, numc_mask_nutau, title, x_label, saving_name, x_scale, eff = False):
    colors = ['black', '#1e90ff', '#d62728', '#21b2aa', '#c71585', 'orange']
    labels = ['Taylor Dome', 'Force Triggers', 'RF Triggers', 'NuE (CC+NC)', 'NuMu (CC)', 'NuTau (CC)']
    
    x_dist = [this_x[groundcal_vpol_mask], this_x[force_mask], this_x[rf_mask], this_x[numc_mask_nue], this_x[numc_mask_numu], this_x[numc_mask_nutau]]
    weights = [np.ones(len(this_x[groundcal_vpol_mask])), np.ones(len(this_x[force_mask])), np.ones(len(this_x[rf_mask])), numc_weight[numc_mask_nue], numc_weight[numc_mask_numu], numc_weight[numc_mask_nutau]]

    bin_start = np.amin([np.amin(this_x[groundcal_vpol_mask]), np.amin(this_x[force_mask]), np.amin(this_x[rf_mask]),
                         np.amin(this_x[numc_mask_nue]), np.amin(this_x[numc_mask_numu]), np.amin(this_x[numc_mask_nutau])])
    bin_end = np.amax([np.amax(this_x[groundcal_vpol_mask]), np.amax(this_x[force_mask]), np.amax(this_x[rf_mask]),
                         np.amax(this_x[numc_mask_nue]), np.amax(this_x[numc_mask_numu]), np.amax(this_x[numc_mask_nutau])])
    nbins = 500

    if x_scale == 'log':
        bins = np.logspace(np.log10(bin_start), np.log10(bin_end), nbins + 1)
    else:
        bins = np.linspace(bin_start, bin_end, nbins + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    if eff:
        fig_eff, ax_eff = plt.subplots(figsize=(10, 6))
    
    for d in range(len(x_dist)):
        if labels[d] == 'RF Triggers':
            alpha = 1
            zorder = 10
        else:
            alpha = 0.5
            zorder = 1
        ax.hist(x_dist[d], bins = bins, weights = weights[d], density = True, alpha = alpha, zorder=zorder, linewidth = 1.5, histtype='step', linestyle = '-', color = colors[d], label = labels[d])
        if eff:
            ### let's calculate the efficiency
            thresholds = bins[:-1]
            efficiency = []
            total_weight = np.sum(weights[d])

            for threshold in thresholds:
                passed = x_dist[d] > threshold
                efficiency.append(np.sum(weights[d][passed]) / total_weight)
                #print(f'{labels[d]}: threshold = {threshold}, eff = {(np.sum(weights[d][passed]) / total_weight)*100:0.2f}%')
            ax_eff.plot(thresholds, efficiency, linewidth = 1.5, color = colors[d], label = labels[d])

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
    fig.savefig(f'/scratch/midway3/darcynewmark/secondaries/multipulse_metrics/v3{saving_name}.png', dpi=300)
    plt.close(fig)
    
    if eff:
        ax_eff.set_title(title, fontsize = 20)
        ax_eff.set_xlabel(x_label, fontsize = 20)
        ax_eff.set_ylabel(f'Efficiency', fontsize = 20)
        ax_eff.legend(fontsize=14, frameon=False)
        ax_eff.minorticks_on()
        ax_eff.tick_params(axis='both', which='both', labelsize=20)
        fig_eff.tight_layout()
        fig_eff.savefig(f'/scratch/midway3/darcynewmark/secondaries/multipulse_metrics/v3{saving_name}_efficiency.png', dpi=300)
        plt.close(fig_eff)

def two_d_plots(this_x, this_y, groundcal_vpol_mask, force_mask, rf_mask, numc_mask_nue, numc_mask_numu, numc_mask_nutau, x_label, y_label, saving_name):
    colors = ['black', '#1e90ff', '#d62728', '#21b2aa', '#c71585', 'orange']
    labels = ['TD', 'ForceTriggers', 'RFTriggers', 'NuE', 'NuMu', 'NuTau']
    titles = ['Taylor Dome', 'Force Triggers', 'RF Triggers', 'NuE', 'NuMu', 'NuTau']
    
    x_dist = [this_x[groundcal_vpol_mask], this_x[force_mask], this_x[rf_mask], this_x[numc_mask_nue], this_x[numc_mask_numu], this_x[numc_mask_nutau]]
    y_dist = [this_y[groundcal_vpol_mask], this_y[force_mask], this_y[rf_mask], this_y[numc_mask_nue], this_y[numc_mask_numu], this_y[numc_mask_nutau]]
    
    nbins = 100
    log_space = True
    if y_label == 'Dist to McMurdo [m]':
        log_space = False
        fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
        fig_dist_rf, ax_dist_rf = plt.subplots(figsize=(10, 6))
        fig_dist_td, ax_dist_td = plt.subplots(figsize=(10, 6))
        fig_dist_force, ax_dist_force = plt.subplots(figsize=(10, 6))
    
    ### one plot per variable
    for d in range(len(x_dist)):
        this_x_data = x_dist[d]
        this_y_data = y_dist[d]

        if y_label == 'Dist to McMurdo [m]':
            mask = this_y_data >= 0 ### sometimes returns -999 for distance to source...
            this_x_data = this_x_data[mask]
            this_y_data = this_y_data[mask]

        ### set up bin edges
        x_min, x_max = np.amin(this_x_data), np.amax(this_x_data)
        y_min, y_max = np.amin(this_y_data), np.amax(this_y_data)

        if log_space:
            x_data_bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), nbins + 1)
            y_data_bin_edges = np.logspace(np.log10(y_min), np.log10(y_max), nbins + 1)
        else:
            x_data_bin_edges = np.linspace(x_min, x_max, nbins + 1)
            y_data_bin_edges = np.linspace(y_min, y_max, nbins + 1)

        ### make our matrix
        M, _, _ = np.histogram2d(this_x_data, this_y_data, bins=[x_data_bin_edges, y_data_bin_edges])

        ### plot!
        cm = plt.get_cmap("Spectral_r")
        lin_norm = matplotlib.colors.LogNorm(vmin=np.amin(M.T[M.T>0]), vmax=np.amax(M.T))

        fig, ax = plt.subplots(figsize=(10, 7))
        mesh = ax.pcolormesh(x_data_bin_edges, y_data_bin_edges, M.T, cmap=cm, norm = lin_norm)
        cb = fig.colorbar(mesh, ax=ax)
        cb.ax.set_ylabel("Number of Events", fontsize = 20)
        cb.ax.minorticks_off()
        if log_space:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_title(f'{titles[d]}', fontsize = 20)
        ax.set_xlabel(x_label, fontsize = 20)
        ax.set_ylabel(y_label, fontsize = 20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        fig.tight_layout()
        plt.savefig(f'/scratch/midway3/darcynewmark/secondaries/multipulse_metrics/v3{labels[d]}_{saving_name}.png', dpi=300)
        plt.close(fig)

        ### let's make a 1d plot of dist to mcmurdo vs avg csw max corr to see if a depedence..
        if y_label == 'Dist to McMurdo [m]' and 'Nu' not in labels[d]:
            distance_bins = np.linspace(np.min(this_y_data), np.max(this_y_data), nbins + 1)
            distance_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
            bin_indices = np.digitize(this_y_data, distance_bins) - 1
            means = np.full(nbins, np.nan)
            stds = np.full(nbins, np.nan)
            
            n_distros = 10
            cmap = plt.get_cmap('Spectral_r')
            norm = mpl.colors.Normalize(vmin=0, vmax=n_distros - 1)
            distro_counter = 0
            for i in range(nbins):
                bin_mask = bin_indices == i
                if np.any(bin_mask):
                    means[i] = np.mean(this_x_data[bin_mask])
                    stds[i] = np.std(this_x_data[bin_mask])
            
                    ### let's also plot the distribution for a few bins
                    if i < (n_distros/2) or i > (nbins-(n_distros/2)-1):
                        colormap = cmap(norm(distro_counter))
                        ls = '-'
                        if i < (n_distros/2):
                            ls = '--'
                        if titles[d] == 'RF Triggers':
                            ax_dist_rf.hist(this_x_data[bin_mask], bins = x_data_bin_edges, density = True, linewidth = 1.5, histtype='step', linestyle = ls, color = colormap, label = f'Dist = {distance_centers[i]:0.2e}')
                        if titles[d] == 'Taylor Dome':
                            ax_dist_td.hist(this_x_data[bin_mask], bins = x_data_bin_edges, density = True, linewidth = 1.5, histtype='step', linestyle = ls, color = colormap, label = f'Dist = {distance_centers[i]:0.2e}')
                        if titles[d] == 'Force Triggers':
                            ax_dist_force.hist(this_x_data[bin_mask], bins = x_data_bin_edges, density = True, linewidth = 1.5, histtype='step', linestyle = ls, color = colormap, label = f'Dist = {distance_centers[i]:0.2e}')
                        distro_counter += 1
    
            nonempty = ~np.isnan(means)

            ### now plot!!!
            ax_dist.errorbar(distance_centers[nonempty], means[nonempty], yerr=stds[nonempty], fmt='o', markersize=5, capsize=3, linestyle='none', color = colors[d], label = titles[d])
    
    if y_label == 'Dist to McMurdo [m]':
        ax_dist.set_xlabel(y_label, fontsize = 20)
        ax_dist.set_ylabel(f'Avg {x_label}', fontsize = 20)
        ax_dist.legend(fontsize=14, frameon=False)
        ax_dist.minorticks_on()
        ax_dist.tick_params(axis='both', which='both', labelsize=20)
        ax_dist.legend(fontsize=18, frameon=False)
        fig_dist.tight_layout()
        fig_dist.savefig(f'/scratch/midway3/darcynewmark/secondaries/multipulse_metrics/v3{saving_name}.png', dpi=300)
        plt.close(fig_dist)
        
        ax_dist_rf.set_title('RF Triggers', fontsize = 20)
        ax_dist_rf.set_xlabel(x_label, fontsize = 20)
        ax_dist_rf.set_ylabel('Probability Density', fontsize = 20)
        ax_dist_rf.legend(fontsize=14, frameon=False)
        ax_dist_rf.minorticks_on()
        ax_dist_rf.tick_params(axis='both', which='both', labelsize=20)
        ax_dist_rf.legend(fontsize=18, frameon=False)
        ax_dist_rf.set_yscale('log')
        fig_dist_rf.tight_layout()
        fig_dist_rf.savefig(f'/scratch/midway3/darcynewmark/secondaries/multipulse_metrics/v3{saving_name}_RFTriggersDistro.png', dpi=300)
        plt.close(fig_dist_rf)
        
        ax_dist_td.set_title('Taylor Dome', fontsize = 20)
        ax_dist_td.set_xlabel(x_label, fontsize = 20)
        ax_dist_td.set_ylabel('Probability Density', fontsize = 20)
        ax_dist_td.legend(fontsize=14, frameon=False)
        ax_dist_td.minorticks_on()
        ax_dist_td.tick_params(axis='both', which='both', labelsize=20)
        ax_dist_td.legend(fontsize=18, frameon=False)
        ax_dist_td.set_yscale('log')
        fig_dist_td.tight_layout()
        fig_dist_td.savefig(f'/scratch/midway3/darcynewmark/secondaries/multipulse_metrics/v3{saving_name}_TaylorDomeDistro.png', dpi=300)
        plt.close(fig_dist_td)
        
        ax_dist_force.set_title('Force Triggers', fontsize = 20)
        ax_dist_force.set_xlabel(x_label, fontsize = 20)
        ax_dist_force.set_ylabel('Probability Density', fontsize = 20)
        ax_dist_force.legend(fontsize=14, frameon=False)
        ax_dist_force.minorticks_on()
        ax_dist_force.tick_params(axis='both', which='both', labelsize=20)
        ax_dist_force.legend(fontsize=18, frameon=False)
        ax_dist_force.set_yscale('log')
        fig_dist_force.tight_layout()
        fig_dist_force.savefig(f'/scratch/midway3/darcynewmark/secondaries/multipulse_metrics/v3{saving_name}_ForceTriggersDistro.png', dpi=300)
        plt.close(fig_dist_force)

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
    trigger_window_max_he = np.asarray(data['trigger_window_max_he'])
    trigger_window_max_time = np.asarray(data['trigger_window_max_time'])
    dist_mcmurdo = np.asarray(data['dist_mcmurdo'])
    csw_snr_vpol = np.asarray(data['csw_snr_vpol'])
    csw_rms_restrictedrange_vpol = np.asarray(data['csw_rms_restrictedrange_vpol'])
    csw_rms_fullrange_vpol = np.asarray(data['csw_rms_fullrange_vpol'])
    reco_az_vpol = np.asarray(data['reco_az_vpol'])
    reco_el_vpol = np.asarray(data['reco_el_vpol'])
    csw_snr_hpol = np.asarray(data['csw_snr_hpol'])
    csw_rms_restrictedrange_hpol = np.asarray(data['csw_rms_restrictedrange_hpol'])
    csw_rms_fullrange_hpol = np.asarray(data['csw_rms_fullrange_hpol'])
    reco_az_hpol = np.asarray(data['reco_az_hpol'])
    reco_el_hpol = np.asarray(data['reco_el_hpol'])
    max_freq_fraction_vpol = np.asarray(data['max_freq_fraction_vpol'])
    max_freq_fraction_hpol = np.asarray(data['max_freq_fraction_hpol'])
    csw_max_corr_vpol = np.asarray(data['csw_max_corr_vpol'])
    csw_max_lag_ns_vpol = np.asarray(data['csw_max_lag_ns_vpol'])
    csw_max_corr_top3peaks_az_vpol = np.asarray(data['csw_max_corr_top3peaks_az_vpol'])
    csw_max_corr_top3peaks_el_vpol = np.asarray(data['csw_max_corr_top3peaks_el_vpol'])
    csw_max_corr_hpol = np.asarray(data['csw_max_corr_hpol'])
    csw_max_lag_ns_hpol = np.asarray(data['csw_max_lag_ns_hpol'])
    csw_max_corr_top3peaks_az_hpol = np.asarray(data['csw_max_corr_top3peaks_az_hpol'])
    csw_max_corr_top3peaks_el_hpol = np.asarray(data['csw_max_corr_top3peaks_el_hpol'])
    he_peak_vpol = np.asarray(data['he_peak_vpol'])
    he_peak_time_vpol = np.asarray(data['he_peak_time_vpol'])
    he_peak_top3peaks_az_vpol = np.asarray(data['he_peak_top3peaks_az_vpol'])
    he_peak_top3peaks_el_vpol = np.asarray(data['he_peak_top3peaks_el_vpol'])
    he_peak_hpol = np.asarray(data['he_peak_hpol'])
    he_peak_time_hpol = np.asarray(data['he_peak_time_hpol'])
    he_peak_top3peaks_az_hpol = np.asarray(data['he_peak_top3peaks_az_hpol'])
    he_peak_top3peaks_el_hpol = np.asarray(data['he_peak_top3peaks_el_hpol'])

    ### add basic saturation, deadtime, and blast mask
    valid_mask = functools.reduce(np.logical_and, [saturation_flag == False, blast_flag == False, deadtime_flag == False, low_channel_rms_flag == False])
    print(f'basic cuts on saturation, deadtime, blast, and low thermal remove {len(saturation_flag) - np.sum(valid_mask)} events!')
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
    trigger_window_max_he = trigger_window_max_he[valid_mask]
    trigger_window_max_time = trigger_window_max_time[valid_mask]
    dist_mcmurdo = dist_mcmurdo[valid_mask]
    csw_snr_vpol = csw_snr_vpol[valid_mask]
    csw_rms_restrictedrange_vpol = csw_rms_restrictedrange_vpol[valid_mask]
    csw_rms_fullrange_vpol = csw_rms_fullrange_vpol[valid_mask]
    reco_az_vpol = reco_az_vpol[valid_mask]
    reco_el_vpol = reco_el_vpol[valid_mask]
    csw_snr_hpol = csw_snr_hpol[valid_mask]
    csw_rms_restrictedrange_hpol = csw_rms_restrictedrange_hpol[valid_mask]
    csw_rms_fullrange_hpol = csw_rms_fullrange_hpol[valid_mask]
    reco_az_hpol = reco_az_hpol[valid_mask]
    reco_el_hpol = reco_el_hpol[valid_mask]
    max_freq_fraction_vpol = max_freq_fraction_vpol[valid_mask]
    max_freq_fraction_hpol = max_freq_fraction_hpol[valid_mask]
    csw_max_corr_vpol = csw_max_corr_vpol[valid_mask]
    csw_max_lag_ns_vpol = csw_max_lag_ns_vpol[valid_mask]
    #csw_max_corr_top3peaks_az_vpol = csw_max_corr_top3peaks_az_vpol[valid_mask]
    #csw_max_corr_top3peaks_el_vpol = csw_max_corr_top3peaks_el_vpol[valid_mask]
    csw_max_corr_hpol = csw_max_corr_hpol[valid_mask]
    csw_max_lag_ns_hpol = csw_max_lag_ns_hpol[valid_mask]
    #csw_max_corr_top3peaks_az_hpol = csw_max_corr_top3peaks_az_hpol[valid_mask]
    #csw_max_corr_top3peaks_el_hpol = csw_max_corr_top3peaks_el_hpol[valid_mask]
    he_peak_vpol = he_peak_vpol[valid_mask]
    he_peak_time_vpol = he_peak_time_vpol[valid_mask]
    #he_peak_top3peaks_az_vpol = he_peak_top3peaks_az_vpol[valid_mask]
    #he_peak_top3peaks_el_vpol = he_peak_top3peaks_el_vpol[valid_mask]
    he_peak_hpol = he_peak_hpol[valid_mask]
    he_peak_time_hpol = he_peak_time_hpol[valid_mask]
    #he_peak_top3peaks_az_hpol = he_peak_top3peaks_az_hpol[valid_mask]
    #he_peak_top3peaks_el_hpol = he_peak_top3peaks_el_hpol[valid_mask]
    
    ### mask our events for now
    groundcal_vpol_mask = np.logical_and(flag == 'TD', groundcal_pol == 'V')
    force_mask = flag == 'Force'
    rf_mask = flag == 'RF'
    numc_mask_nue = np.logical_and(flag == 'NuMC', numc_pdg == 12) 
    numc_mask_numu = functools.reduce(np.logical_and, [flag == 'NuMC', numc_pdg == 14, numc_interaction_current == 0])
    numc_mask_nutau = functools.reduce(np.logical_and, [flag == 'NuMC', numc_pdg == 16, numc_interaction_current == 0])
    csw_max_lag_mask = csw_max_lag_ns_vpol > 120
    print(f'have {np.sum(groundcal_vpol_mask)} ground cal v-pol events, {np.sum(force_mask)} force triggers, {np.sum(rf_mask)} rf triggers, {np.sum(numc_mask_nue)} nue events, {np.sum(numc_mask_numu)} numu events, and {np.sum(numc_mask_nutau)} nutau events')

    ### now let's make some one and two-d plots!!
    plotting_x = [csw_max_corr_vpol, he_peak_vpol, trigger_window_max_time, csw_rms_restrictedrange_vpol, csw_max_lag_ns_vpol, csw_snr_vpol, csw_max_corr_vpol, csw_max_corr_vpol]
    plotting_y = [csw_snr_vpol, csw_snr_vpol, None, None, None, None, csw_max_lag_ns_vpol, dist_mcmurdo]
    plotting_x_saving_name = ['csw_max_corr_vpol', 'he_peak_vpol', 'trigger_window_max_time', 'csw_rms_fullrange_vpol', 'csw_max_lag_ns_vpol', 'csw_snr_vpol', 'csw_max_corr_vpol', 'csw_max_corr_vpol']
    plotting_y_saving_name = ['csw_snr_vpol', 'csw_snr_vpol', None, None, None, None, 'csw_max_lag_ns_vpol', 'dist_mcmurdo']
    plotting_x_label = ['CSW Max Correlation', 'Max HE Peak After Trigger', 'Trigger Peak Time [ns]', 'CSW RMS [mV]', 'CSW Max Lag [ns]', 'CSW SNR', 'CSW Max Correlation', 'CSW Max Correlation']
    plotting_y_label = ['CSW SNR', 'CSW SNR', None, None, None, None, 'CSW Max Lag [ns]', 'Dist to McMurdo [m]']
    plotting_x_scale = ['linear', 'log', 'linear', 'log', 'linear', 'log', 'linear', 'linear']
    plotting_y_scale = ['log', 'log', None, None, None, None, 'liner', 'linear']

    for p in range(len(plotting_x)):
        this_x = plotting_x[p]
        this_y = plotting_y[p]

        ### let's first make 1D plots for each variable
        if plotting_x_saving_name[p] == 'csw_max_corr_vpol':
            one_d_plots(this_x[csw_max_lag_mask], numc_weight[csw_max_lag_mask], groundcal_vpol_mask[csw_max_lag_mask], force_mask[csw_max_lag_mask], rf_mask[csw_max_lag_mask],
                        numc_mask_nue[csw_max_lag_mask], numc_mask_numu[csw_max_lag_mask], numc_mask_nutau[csw_max_lag_mask],
                        f'{plotting_x_label[p]} (V-pol)', plotting_x_label[p], plotting_x_saving_name[p], plotting_x_scale[p], eff = True)
        else:
            one_d_plots(this_x, numc_weight, groundcal_vpol_mask, force_mask, rf_mask, numc_mask_nue, numc_mask_numu, numc_mask_nutau,
                        f'{plotting_x_label[p]} (V-pol)', plotting_x_label[p], plotting_x_saving_name[p], plotting_x_scale[p])
        
        if this_y is not None:
            one_d_plots(this_y, numc_weight, groundcal_vpol_mask, force_mask, rf_mask, numc_mask_nue, numc_mask_numu, numc_mask_nutau,
                        f'{plotting_y_label[p]} (V-pol)', plotting_y_label[p], plotting_y_saving_name[p], plotting_y_scale[p])

            ### now our 2D plots!!!
            two_d_plots(this_x, this_y, groundcal_vpol_mask, force_mask, rf_mask, numc_mask_nue, numc_mask_numu, numc_mask_nutau, 
                        plotting_x_label[p], plotting_y_label[p], f'{plotting_x_saving_name[p]}_{plotting_y_saving_name[p]}')
        

if __name__ == "__main__":
    plot_vars()
