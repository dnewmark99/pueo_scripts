import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import ROOT
import os
import json
from tqdm import tqdm
import time

import pueoTALON.pueoPythonAnalyzer.waveform as waveform
import pueoTALON.pueoPythonAnalyzer.plotter as pltr
import pueoTALON.pueoPythonAnalyzer.antenna as antenna
import pueoTALON.pueoPythonAnalyzer.correlator as correlator
import antennaUtils.response as antResponse
import pueoTALON.pueoPythonAnalyzer.utils as utils
import pueoTALON.pueoPythonAnalyzer.geography as geography
import pueoTALON.pueoPythonAnalyzer.plotter as pltr

ROOT.gSystem.Load("libpueoEvent.so")

def plot_wfs(this_run, this_ev, poli = 'V', opp_poli = 'H', filtered = True):
    ### make our correlator
    corr = correlator.Correlator(run = this_run)

    ### load this event
    corr.ChannelGeo.load_event(this_ev)

    ### set up our plot
    fig, (ax, ax_freq) = plt.subplots(2, 1, figsize=(10, 12))
    ax.minorticks_on()
    ax_freq.minorticks_on()

    ### now reconstruct the directions and coherent sum
    skymap, peak_zen, peak_az, corr_vals, zoomedskymap, zoomedpeak_zen, zoomedpeak_az, zoomedcorr_vals, all_zoomedzen, all_zoomedaz = corr.reconstruct(poli, zoom_angle_range=7, N_peaks=1)
    main_peak_zen = zoomedpeak_zen[0][0]
    main_peak_el = 90 - main_peak_zen
    main_peak_az = zoomedpeak_az[0][0]

    ### get coherent sum
    filt_cohsum, _, _, unfilt_cohsum, overlap_start, overlap_end = corr.coherent_sum(main_peak_zen, main_peak_az, polarization=poli, return_all = True)
    filt_opp_cohsum, _, _, unfilt_opp_cohsum, opp_overlap_start, opp_overlap_end = corr.coherent_sum(main_peak_zen, main_peak_az, polarization=opp_poli, return_all = True)
    if filtered:
        cohsum = filt_cohsum
        opp_cohsum = filt_opp_cohsum
    else:
        cohsum = unfilt_cohsum
        opp_cohsum = unfilt_opp_cohsum

    cohsum_snr = utils.SNR(cohsum, dynamic_rms_window = True, valid_window = (overlap_start, overlap_end))
    opp_cohsum_snr = utils.SNR(opp_cohsum, dynamic_rms_window = True, valid_window = (opp_overlap_start, opp_overlap_end))
    cohsum_start = max(overlap_start, opp_overlap_start)
    cohsum_end = min(overlap_end, opp_overlap_end)
    cohsum = cohsum[cohsum_start:cohsum_end]
    opp_cohsum = opp_cohsum[cohsum_start:cohsum_end]
    times = np.arange(cohsum.size)*corr.dt

    ### let's also get the fft
    f, vf = utils.fourierTransform(times, cohsum)
    opp_f, opp_vf = utils.fourierTransform(times, opp_cohsum)
    dt = times[1] - times[0]
    power = utils.volt_to_dB(np.abs(vf*dt))
    opp_power = utils.volt_to_dB(np.abs(opp_vf*dt))

    ### plot!
    ax.plot(times, cohsum, color = 'black', label = f'CSW {poli}-pol, SNR = {cohsum_snr:0.2f}')
    ax.plot(times, opp_cohsum, color = 'orange', alpha = 0.5, zorder = 10, label = f'CSW {opp_poli}-pol, SNR = {opp_cohsum_snr:0.2f}')
    ax_freq.plot(f, power, color = 'black', label = f'CSW FFT {poli}-pol')
    ax_freq.plot(f, opp_power, color = 'orange', alpha = 0.5, zorder = 10, label = f'CSW FFT {opp_poli}-pol')
    ax.set_title(f'Run = {this_run}, Ev = {this_ev}, MUOS Removal + Filter = {filtered}')
    ax.set_xlabel(f'Times [ns]', fontsize = 20)
    ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
    ax_freq.set_xlabel(f'Frequency [GHz]', fontsize = 20)
    ax_freq.set_ylabel(f'Gain [dB]', fontsize = 20)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax_freq.tick_params(axis='both', which='both', labelsize=20)
    ax.legend(fontsize=16, frameon=False)
    ax_freq.legend(fontsize=16, frameon=False)
    ax_freq.set_xlim(0,1.5)
    ax_freq.set_ylim(0,np.amax(utils.volt_to_dB(np.abs(vf*dt)))*1.1)
    fig.tight_layout()
    fig.savefig(f'/scratch/midway3/darcynewmark/satellites/muosremoval_{this_run}_e{this_ev}_filter={filtered}.png', dpi=300)
    plt.close(fig)

    print(f'plotted csw for run = {this_run} and ev = {this_ev} (filtered  = {filtered})')

def plot_run_ev():
    evs = [2372, 2392, 2402, 2420, 2435, 2453, 2455, 2467, 2476, 2488, 2498, 2506, 2510, 2523, 2535, 2552, 2564, 2572, 2576, 2589]

    for r in range(len(evs)):
        this_run = 813
        this_ev = evs[r]

        ### plot!
        #plot_wfs(int(this_run), int(this_ev), filtered = True)
        plot_wfs(int(this_run), int(this_ev), filtered = False)

def waterfalls(dataset = 'ForceTriggers', n_evs = 1000):
    ### make waterfall plots to understand our filtering better
    all_runs = []
    all_entries = []

    if dataset == 'ForceTriggers':
        ### use some force triggers from run 813 for testing
        path = f'/scratch/midway3/darcynewmark/useful/nonrftriggers_run813_run813.json'
        with open(path) as f:
            rf_dict = json.load(f)
        rf_runs = np.asarray(rf_dict['run'])
        rf_evs = np.asarray(rf_dict['entry'])
        total_rftrigger_events = len(rf_evs)
        [all_runs.append(int(r)) for r in rf_runs]
        [all_entries.append(int(e)) for e in rf_evs]

    if dataset == 'TaylorDome':
        ground_cal_name = 'TD'
        groundcal_events = np.genfromtxt(os.environ['PUEO_ANALYSIS_TOOLS'] + f'/pueoTALON/lookup/my{ground_cal_name}_list.csv', delimiter=',', names=True, dtype=None)
        groundcal_runs = np.asarray(groundcal_events['run'])
        groundcal_evs = np.asarray(groundcal_events['entry_number'])
        groundcal_pol = np.asarray(groundcal_events['pol'])
        total_groundcal_events = len(groundcal_runs)
        [all_runs.append(int(r)) for r in groundcal_runs]
        [all_entries.append(int(e)) for e in groundcal_evs]
    print(f'total of {len(all_entries)} events!')

    ### now let's go through our runs and make waterfall plot
    prev_run = None
    power_filtered_vpol = []
    power_unfiltered_vpol = []
    power_filtered_hpol = []
    power_unfiltered_hpol = []
    passed_events = []
    pbar = tqdm(total=n_evs, desc="Good events", unit="event")
    total_time_per_event = []
    for r in range(len(all_runs)):
        if len(power_filtered_vpol) >= n_evs:
            break
        this_run = int(all_runs[r])
        this_ev = int(all_entries[r])

        ### check if we have a correlator object for this run
        if prev_run == None or prev_run != this_run:
            ### set up correlator per run
            corr = correlator.Correlator(run = this_run)
            prev_run = this_run

            ### make our masks to use later
            channel_names = np.asarray(corr.ChannelGeo.MI_ringphipol)
            mask_H = np.array([int(ch[:2]) <= 4 and int(ch[2:4]) <= 24 and ch[-1] == 'H' for ch in channel_names])
            mask_V = np.array([int(ch[:2]) <= 4 and int(ch[2:4]) <= 24 and ch[-1] == 'V' for ch in channel_names])

        ### load this event
        corr.ChannelGeo.load_event(this_ev)

        ### get some flags
        this_event_summary = ROOT.PueoEventSummary()
        qual = corr.ChannelGeo.qualityCheck(this_event_summary)
        this_blast_flag = this_event_summary.flags.isPayloadBlast
        this_sat_flag = this_event_summary.flags.isSaturated
        this_deadtime_flag = this_event_summary.flags.isInDeadtime
        this_low_channel_rms_flag = this_event_summary.flags.lowChannelThermal

        if not any([this_blast_flag, this_sat_flag, this_deadtime_flag, this_low_channel_rms_flag]):
            ### get our filtered and un-filtered wfs
            start = time.perf_counter()
            corr.ChannelGeo.load_waveforms()
            unfilt_wfs = np.asarray(corr.ChannelGeo.waveforms_unfilt)
            filt_wfs = np.asarray(corr.ChannelGeo.waveforms)
            times = np.arange(unfilt_wfs.shape[-1]) * (1/3)
            end = time.perf_counter()
            total_time = end - start

            ### now let's get the fft
            dt = times[1] - times[0]
            unfilt_f, unfilt_vf = utils.fourierTransform(times, unfilt_wfs)
            filt_f, filt_vf = utils.fourierTransform(times, filt_wfs)

            ### now let's save!!!
            power_filtered_vpol.append(np.real(utils.volt_to_dB(np.mean(filt_vf[mask_V], axis=0))))
            power_unfiltered_vpol.append(np.real(utils.volt_to_dB(np.mean(unfilt_vf[mask_V], axis=0))))
            power_filtered_hpol.append(np.real(utils.volt_to_dB(np.mean(filt_vf[mask_H], axis=0))))
            power_unfiltered_hpol.append(np.real(utils.volt_to_dB(np.mean(unfilt_vf[mask_H], axis=0))))

            pbar.update(1)
            passed_events.append(this_ev)
            total_time_per_event.append(total_time)

    pbar.close()
    print(f'average filtering time per event = {np.mean(total_time_per_event)}')

    ### now make our 2d waterfall plot!
    all_waterfalls = [power_filtered_vpol, power_unfiltered_vpol, power_filtered_hpol, power_unfiltered_hpol,]
    all_titles = ['Filtered (V-Pol)', 'Unfiltered (V-Pol)', 'Filtered (H-Pol)', 'Unfiltered (H-Pol)', ]
    all_saving_names = ['filt_vpol', 'unfilt_vpol', 'filt_hpol', 'unfilt_hpol']

    ### event bin edges
    n_events = len(power_filtered_vpol)
    evs_bin_edges = np.arange(n_events + 1) - 0.5

    ### frequency bin edges
    df = filt_f[1] - filt_f[0]
    freq_bin_edges = np.concatenate([[max(0, filt_f[0] - df/2)], (filt_f[:-1] + filt_f[1:]) / 2,[filt_f[-1] + df/2]])
    max_color_bar = 60.0
    for power, title, saving_name in zip(all_waterfalls, all_titles, all_saving_names):

        ### make our "matrix" .. really just the power weights
        M = np.asarray(power)

        ### plot!
        cm = plt.get_cmap("Spectral_r")
        lin_norm = matplotlib.colors.Normalize(vmin=0, vmax=max_color_bar)

        fig, ax = plt.subplots(figsize=(10, 7))
        mesh = ax.pcolormesh(evs_bin_edges, freq_bin_edges, M.T, cmap=cm, norm = lin_norm)
        cb = fig.colorbar(mesh, ax=ax)
        cb.ax.set_ylabel("Average Power [dB]", fontsize = 20)
        cb.set_ticks(np.arange(0, max_color_bar+1, 10))
        cb.ax.tick_params(labelsize=14)
        cb.ax.minorticks_off()
        ax.set_title(f'{dataset}, {title}', fontsize = 20)
        ax.set_xlabel(f'Event', fontsize = 20)
        ax.set_ylabel(f'Frequency [GHz]', fontsize = 20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        fig.tight_layout()
        plt.savefig(f'/scratch/midway3/darcynewmark/satellites/waterfall_{dataset}_{saving_name}.png', dpi=300)
        plt.close(fig)

    ### let's also make some one-d projections (averaging over events)
    one_d_filt = [power_filtered_vpol, power_filtered_hpol]
    one_d_unfilt = [power_unfiltered_vpol, power_unfiltered_hpol]
    freq_mask = np.logical_and(filt_f > 0, filt_f < 1.5)
    labels = ['V-Pol', 'H-Pol']
    saving_names = ['vpol', 'hpol']
    for p in range(len(one_d_filt)):
        avg_filt = np.mean(one_d_filt[p], axis = 0)
        std_filt = np.std(one_d_filt[p], axis=0)

        avg_unfilt = np.mean(one_d_unfilt[p], axis = 0)
        std_unfilt = np.std(one_d_unfilt[p], axis=0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(filt_f[freq_mask], avg_unfilt[freq_mask], color = 'black', label = 'Unfiltered')
        ax.fill_between(filt_f[freq_mask], (avg_unfilt - std_unfilt)[freq_mask], (avg_unfilt + std_unfilt)[freq_mask], color='black', alpha=0.2)
        ax.plot(filt_f[freq_mask], avg_filt[freq_mask], color = '#d62728', label = 'Filtered')
        ax.fill_between(filt_f[freq_mask], (avg_filt - std_filt)[freq_mask], (avg_filt + std_filt)[freq_mask], color='#d62728', alpha=0.2)
        ax.set_title(f'{dataset}, {labels[p]}, {len(one_d_filt[p])} Total Events', fontsize = 20)
        ax.set_xlabel(f'Frequency [GHz]', fontsize = 20)
        ax.set_ylabel(f'Average Power [dB]', fontsize = 20)
        ax.legend(fontsize=14, frameon=False)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.set_xlim(0,1.5)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/satellites/waterfall_oned_{dataset}_{saving_names[p]}.png', dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    #plot_run_ev()
    waterfalls(n_evs=1000)

