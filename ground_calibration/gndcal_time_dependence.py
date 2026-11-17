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

SNRarr = []
runs = []
entries = []
pol = []
true_az = []
trigger_time = []
max_hilbert = []
snr_hilbert = []

#Fastest way to do this is to already have pointing and coherent sum SNR information in a lookup file.
#This code is using the results from the batch_pointjob.slurm script in the batch/ folder

### TODO: do for S+200
for i in range(0,90):
    try:
        data = np.loadtxt(os.environ['PUEO_ANALYSIS_TOOLS'] + '/pueoTALON/lookup/scratch/'+'TD_fulllist_withtriggertime_%d.csv' % i, delimiter = ',', dtype=float)
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
                SNRarr.append(val[5]) #SNR of CSW in best guess direction
                trigger_time.append(val[8])
                max_hilbert.append(val[9])
                snr_hilbert.append(val[10])

    except:
        pass

channels = waveform.Channels(784, 0)
unique_runs = np.unique(runs)
print(f'found {len(unique_runs)} unique runs!')

### set up lists to save all the data
hpol_all_trigger_times = []
hpol_all_cswsnr = []
hpol_all_maxhil = []
hpol_all_hilsnr = []
vpol_all_trigger_times = []
vpol_all_cswsnr = []
vpol_all_maxhil = []
vpol_all_hilsnr = []

for run in unique_runs:
    #Restrict arrays to only the current run
    en = np.array(entries)[np.array(runs)==run]
    p = np.array(pol)[np.array(runs)==run]
    cswsnr = np.array(SNRarr)[np.array(runs)==run]
    tt = np.array(trigger_time)[np.array(runs)==run]
    mh = np.array(max_hilbert)[np.array(runs)==run]
    hilsnr = np.array(snr_hilbert)[np.array(runs)==run]

    ### for right now, just save the time and snr
    hpol_mask = p == 0
    vpol_mask = p == 1
    [hpol_all_trigger_times.append(t) for t in tt[hpol_mask]]
    [hpol_all_cswsnr.append(s) for s in cswsnr[hpol_mask]]
    [hpol_all_maxhil.append(m) for m in mh[hpol_mask]]
    [hpol_all_hilsnr.append(h) for h in hilsnr[hpol_mask]]
    [vpol_all_trigger_times.append(t) for t in tt[vpol_mask]]
    [vpol_all_cswsnr.append(s) for s in cswsnr[vpol_mask]]
    [vpol_all_maxhil.append(m) for m in mh[vpol_mask]]
    [vpol_all_hilsnr.append(h) for h in hilsnr[vpol_mask]]

    #### TODO: run through all events to check for "missed" triggers due to phi sector masking

### now let's make a 2D histogram of trigger time vs SNR
att = [hpol_all_trigger_times, vpol_all_trigger_times, hpol_all_trigger_times, vpol_all_trigger_times, hpol_all_trigger_times, vpol_all_trigger_times]
asnr = [hpol_all_cswsnr, vpol_all_cswsnr, hpol_all_maxhil, vpol_all_maxhil, hpol_all_hilsnr, vpol_all_hilsnr]
y_axes_labels = ['Coherent Summed Wf SNR', 'Coherent Summed Wf SNR', 'Max Hilbert Envelope', 'Max Hilbert Envelope', 'Hilbert Envelope SNR', 'Hilbert Envelope SNR']
labels = ['HPol' , 'VPol', 'HPol' , 'VPol', 'HPol' , 'VPol']
saving_name = ['hpol', 'vpol', 'hpol', 'vpol', 'hpol', 'vpol']
saving_snr_name = ['cswsnr', 'cswsnr', 'maxhil', 'maxhil', 'hilsnr', 'hilsnr']

for x in range(len(att)):
    all_trigger_times = np.asarray(att[x])
    all_snr = np.asarray(asnr[x])

    min_trigger_time = np.amin(all_trigger_times)
    all_trigger_times -= min_trigger_time

    ### set up bin edges
    trigger_time_bin_edges = np.linspace(np.amin(all_trigger_times), np.amax(all_trigger_times), 100)
    SNR_bin_edges = np.linspace(0, np.amax(all_snr), 100)

    ### make our matrix
    M, _, _ = np.histogram2d(all_trigger_times, all_snr, bins=[trigger_time_bin_edges, SNR_bin_edges])

    ### plot!
    cm = plt.get_cmap("Spectral_r")
    lin_norm = matplotlib.colors.LogNorm(vmin=1, vmax=np.amax(M.T))

    fig, ax = plt.subplots(figsize=(10, 7))
    mesh = ax.pcolormesh(trigger_time_bin_edges, SNR_bin_edges, M.T, cmap=cm, norm = lin_norm)
    cb = fig.colorbar(mesh, ax=ax)
    cb.ax.set_ylabel("Number of Events", fontsize = 16)
    cb.ax.minorticks_off()
    ax.set_title(f'Taylor Dome Ground Cal, {labels[x]}', fontsize = 16)
    ax.set_xlabel(f'Trigger Time [s]', fontsize = 16)
    ax.set_ylabel(f'{y_axes_labels[x]}', fontsize = 16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    fig.tight_layout()
    plt.savefig(f'/scratch/midway3/darcynewmark/ground_cal/taylordome_trigger_time_vs_{saving_snr_name[x]}_{saving_name[x]}.png', dpi=300)
    plt.close(fig)

### before we're done, let's plot a few outliers
corr = correlator.Correlator()
geography_helper = geography.GeographyHelper(root_file=os.environ['PUEO_ANALYSIS_TOOLS'] + "/pueoTALON/lookup/abxtwo.root", root_source="ABX-Two")
vpol_time_outlier_range = (30000, 60000)
vpol_cswsnr_outlier_range = (0, 20)
hpol_time_outlier_range = (25000, 35000)
hpol_cswsnr_outlier_range = (55,65)
#hpol_time_outlier_range = (10000, 20000)
#hpol_cswsnr_outlier_range = (15,25)
n_vpol_outliers_plotted = 0
max_vpol_outliers_to_plot = 4
n_hpol_outliers_plotted = 0
max_hpol_outliers_to_plot = 4
N_peaks = 2 #how many peaks per skymap you want to find

for run in unique_runs:
    #Restrict arrays to only the current run
    en = np.array(entries)[np.array(runs)==run]
    p = np.array(pol)[np.array(runs)==run]
    cswsnr = np.array(SNRarr)[np.array(runs)==run]
    tt = np.array(trigger_time)[np.array(runs)==run]
    mh = np.array(max_hilbert)[np.array(runs)==run]
    hilsnr = np.array(snr_hilbert)[np.array(runs)==run]

    ### adjust trigger time
    tt -= min_trigger_time

    #### get vpol outliers
    #vpol_mask = p == 1
    #vpol_time_mask = np.logical_and(tt > vpol_time_outlier_range[0], tt < vpol_time_outlier_range[1])
    #vpol_cswsnr_mask = np.logical_and(cswsnr > vpol_cswsnr_outlier_range[0], cswsnr < vpol_cswsnr_outlier_range[1])
    #vpol_outlier_mask = functools.reduce(np.logical_and, [vpol_mask, vpol_time_mask, vpol_cswsnr_mask])
 
    #if np.sum(vpol_outlier_mask) > 0 and n_vpol_outliers_plotted < max_vpol_outliers_to_plot: 
    #    ### ok let's make skymap plot and CSW 
    #    outlier_entries = en[vpol_outlier_mask]
    #    outlier_tt = tt[vpol_outlier_mask]
    #    outlier_cswsnr = cswsnr[vpol_outlier_mask]
    #    for i in range(len(outlier_entries)):
    #        entry = outlier_entries[i]

    #        if n_vpol_outliers_plotted < max_vpol_outliers_to_plot:
    #            ### first, let's update the correlator object
    #            corr.ChannelGeo.load_run_event(int(run),  int(entry))
    #            event_time = corr.ChannelGeo.triggerTime()
    #            geography_helper.updatePosition(event_time, max_time_delta = 3)
    #            
    #            ### get expected taylor dome location
    #            expected_td_el_az = geography_helper.get_expected_source_location('taylor_dome')
    #            expected_sun_el_az = geography_helper.get_sun_position(event_time)

    #            #make skymap and find peak location
    #            poli = 'V'
    #            skymap, peak_el, peak_az, _, zskymap, peak_zel, peak_zaz, _, all_zel, all_zaz  = corr.reconstruct(poli, zoom_angle_range=7, N_peaks=N_peaks)

    #            ### plot skymap
    #            pltr.plot_skymap(skymap, [peak_el, peak_az], corr.az, corr.el, title=f'Skymap,TT={outlier_tt[i]:0.2f},CSWSNR={outlier_cswsnr[i]:0.2f},Run={int(run)},Entry={int(entry)}',
    #                    filename="/scratch/midway3/darcynewmark/ground_cal/skymap_vpol_r%d_e%d_p%d" % (run, entry, i),
    #                    locations=[(expected_td_el_az[1], expected_td_el_az[0], 'Taylor Dome'),
    #                        (expected_sun_el_az[1], expected_sun_el_az[0], 'Sun')]) #mark the expected pulser location on the skymap with a yellow circle
    #            print("Plotted Skymap")
    #            
    #            for n in range(N_peaks):
    #                ### plot coherent summed wf
    #                cohsum = corr.coherent_sum(peak_el[n], peak_az[n],polarization=poli)
    #                pltr.plot_time_and_frequency_domain(np.arange(cohsum.size)*corr.dt, cohsum,
    #                        title=f'CSWofChannsatPeak{n},TT={outlier_tt[i]:0.2f},CSWSNR={outlier_cswsnr[i]:0.2f},Run={int(run)},Entry={int(entry)},N={int(n)}',
    #                        filename="/scratch/midway3/darcynewmark/ground_cal/coherent_sum_vpol_r%d_e%d_n%d_p%d" % (run, entry, n, i), ylimdB=(10,80), xlimfreq=(0,1.5))
    #            print("Plotted Coherent Sum")
    #            print(f'vpol outlier run = {run} and entry = {entry}')
    #            
    #            n_vpol_outliers_plotted += 1 
    
    ### get hpol outliers
    hpol_mask = p == 0
    hpol_time_mask = np.logical_and(tt > hpol_time_outlier_range[0], tt < hpol_time_outlier_range[1])
    hpol_cswsnr_mask = np.logical_and(cswsnr > hpol_cswsnr_outlier_range[0], cswsnr < hpol_cswsnr_outlier_range[1])
    hpol_outlier_mask = functools.reduce(np.logical_and, [hpol_mask, hpol_time_mask, hpol_cswsnr_mask])
 
    if np.sum(hpol_outlier_mask) > 0 and n_hpol_outliers_plotted < max_hpol_outliers_to_plot: 
        ### ok let's make skymap plot and CSW 
        outlier_entries = en[hpol_outlier_mask]
        outlier_tt = tt[hpol_outlier_mask]
        outlier_cswsnr = cswsnr[hpol_outlier_mask]
        for i in range(len(outlier_entries)):
            entry = outlier_entries[i]

            if n_hpol_outliers_plotted < max_hpol_outliers_to_plot:
                ### first, let's update the correlator object
                corr.ChannelGeo.load_run_event(int(run),  int(entry))
                event_time = corr.ChannelGeo.triggerTime()
                geography_helper.updatePosition(event_time, max_time_delta = 3)
                
                ### get expected taylor dome location
                expected_td_el_az = geography_helper.get_expected_source_location('taylor_dome')
                expected_sun_el_az = geography_helper.get_sun_position(event_time)

                #make skymap and find peak location
                poli = 'H'
                skymap, peak_el, peak_az, _, zskymap, peak_zel, peak_zaz, _, all_zel, all_zaz  = corr.reconstruct(poli, zoom_angle_range=7, N_peaks=N_peaks)

                ### plot skymap
                pltr.plot_skymap(skymap, [peak_el, peak_az], corr.az, corr.el, title=f'Skymap,TT={outlier_tt[i]:0.2f},CSWSNR={outlier_cswsnr[i]:0.2f},Run={int(run)},Entry={int(entry)}',
                        filename="/scratch/midway3/darcynewmark/ground_cal/skymap_hpol_r%d_e%d_p%d" % (run, entry, i),
                        locations=[(expected_td_el_az[1], expected_td_el_az[0], 'Taylor Dome'),
                            (expected_sun_el_az[1], expected_sun_el_az[0], 'Sun')]) #mark the expected pulser location on the skymap with a yellow circle
                print("Plotted Skymap")
                
                for n in range(N_peaks):
                    ### plot coherent summed wf
                    cohsum = corr.coherent_sum(peak_el[n], peak_az[n],polarization=poli)
                    pltr.plot_time_and_frequency_domain(np.arange(cohsum.size)*corr.dt, cohsum,
                            title=f'CSWofChannsatPeak{n},TT={outlier_tt[i]:0.2f},CSWSNR={outlier_cswsnr[i]:0.2f},Run={int(run)},Entry={int(entry)},N={int(n)}',
                            filename="/scratch/midway3/darcynewmark/ground_cal/coherent_sum_hpol_r%d_e%d_n%d_p%d" % (run, entry, n, i), ylimdB=(10,80), xlimfreq=(0,1.5))
                print("Plotted Coherent Sum")
               
                print(f'!!!!!!!!!!!!!!!! hpol outlier run = {run} and entry = {entry} !!!!!!!!!!!!!!')
                n_hpol_outliers_plotted += 1 

##Plot for average CSW SNR and efficiency per run
#sortV = np.argsort(avg_cohsum_snrV)
#sortH = np.argsort(avg_cohsum_snrH)
#fig, ax = plt.subplots(figsize=(10, 6))
#ax.minorticks_on()
#ax.plot(np.array(avg_cohsum_snrV)[sortV], np.array(run_effV)[sortV], color = '#21b2aa', label='Vpol')
#ax.plot(np.array(avg_cohsum_snrH)[sortH], np.array(run_effH)[sortH], color = '#c71585', label='Hpol')
#ax.set_title(f'TD Ground Cal Efficiency')
#ax.set_xlabel(f'Coherent Sum SNR', fontsize = 20)
#ax.set_ylabel(f'Trigger Efficiency', fontsize = 20)
#ax.tick_params(axis='both', which='both', labelsize=16)
#ax.legend(fontsize=20, frameon=False)
#fig.tight_layout()
#fig.savefig(f'./efficiency.png', dpi = 300)
#plt.close(fig)

