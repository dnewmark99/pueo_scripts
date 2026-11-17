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

def fourierTransform(time, voltage):
    freq=np.fft.rfftfreq(time.shape[-1],time[1]-time[0])
    amp_f= np.fft.rfft(voltage)
    voltage_ft= np.abs(amp_f)*np.exp(np.unwrap(np.angle(amp_f))*1j)
    return freq, voltage_ft

def volt_to_dB(voltage_ratio):
    dB = np.zeros_like(voltage_ratio)
    dB[np.abs(voltage_ratio) > 1e-10] = 20*np.log10(np.abs(voltage_ratio[np.abs(voltage_ratio) > 1e-10]))
    return dB

def plot_wfs(ground_cal_name = 'TD', n_events = 50, desired_pol = 'V', delta_az_tol = 3, delta_el_tol = 1):
    if ground_cal_name == 'TD':
        source_loc_name = 'taylor_dome'
    if ground_cal_name == 'LDB':
        source_loc_name = 'ldb'

    ### we want to just find some ground cal events with reflections
    ### so let's plot some cohsum wfs
    groundcal_events = np.genfromtxt(os.environ['PUEO_ANALYSIS_TOOLS'] + f'/pueoTALON/lookup/my{ground_cal_name}_list.csv', delimiter=',', names=True, dtype=None)

    try:
        groundcal_events = groundcal_events[groundcal_events['source'] == ground_cal_name]
    except:
        groundcal_events = groundcal_events

    ### set up correlator + geography helper
    corr = correlator.Correlator()
    geography_helper = geography.GeographyHelper(root_file=os.environ['PUEO_ANALYSIS_TOOLS'] + "/pueoTALON/lookup/abxtwo.root", root_source="ABX-Two")

    plotting_counter = 0
    for ev in range(len(groundcal_events)):
        if plotting_counter >= n_events:
            break

        ### check the polarization of this event
        poli = groundcal_events['pol'][ev]
        if poli != desired_pol:
            continue

        ### load this event
        run = int(groundcal_events['run'][ev])
        if run  == corr.ChannelGeo.run:
            corr.ChannelGeo.load_event(int(groundcal_events['entry_number'][ev]))
        else:
            corr.ChannelGeo.load_run_event(run, int(groundcal_events['entry_number'][ev]))

        ### update geography helper
        event_time = corr.ChannelGeo.triggerTime()
        geography_helper.updatePosition(event_time, max_time_delta = 3)

        ### check quality cuts! 
        qual = corr.ChannelGeo.qualityCheck()
        if not qual:
            continue

        ### also check saturation (might be redudant w quality idk)
        sat = corr.ChannelGeo.checkSaturation()
        if sat:
            continue

        ### get the expected source loc for cuts
        expected_source_el_az = geography_helper.get_expected_source_location(source_loc_name)
        expected_source_el = expected_source_el_az[0]
        expected_source_az = expected_source_el_az[1]

        ### now reconstruct the directions and coherent sum
        skymap, peak_zen, peak_az, corr_vals, zoomedskymap, zoomedpeak_zen, zoomedpeak_az, zoomedcorr_vals, all_zoomedzen, all_zoomedaz = corr.reconstruct(poli, zoom_angle_range=7, N_peaks=1)
        main_peak_zen = zoomedpeak_zen[0][0]
        main_peak_el = 90 - main_peak_zen
        main_peak_az = zoomedpeak_az[0][0]

        ### check the dir
        delta_az = np.abs(main_peak_az - expected_source_az)
        delta_el = np.abs(main_peak_el - expected_source_el)
        if (delta_az > delta_az_tol) or (delta_el > delta_el_tol):
            continue

        ### get coherent sum
        cohsum, overlap_start, overlap_end = corr.coherent_sum(main_peak_zen, main_peak_az, polarization=poli, return_overlap_boundaries=True)
        cohsum = cohsum[overlap_start:overlap_end]
        cswhil = utils.hilbert_envelope(cohsum)
        times = np.arange(cohsum.size)*corr.dt

        ### again but with raw wfs!
        _, _, _, cohsum_raw, overlap_start_raw, overlap_end_raw = corr.coherent_sum(main_peak_zen, main_peak_az, polarization=poli, return_all=True, return_overlap_boundaries=True)
        cohsum_raw = cohsum_raw[overlap_start_raw:overlap_end_raw]
        cswhil_raw = utils.hilbert_envelope(cohsum_raw)
        times_raw = np.arange(cohsum_raw.size)*corr.dt

        ### let's also get the fft
        f, vf = fourierTransform(times, cohsum)
        dt = times[1] - times[0]
        f_raw, vf_raw = fourierTransform(times_raw, cohsum_raw)
        dt_raw = times_raw[1] - times_raw[0]

        ### plot cohsum + csw hilbert envelope !!!
        fig, (ax, ax_freq) = plt.subplots(2, 1, figsize=(10, 12))
        ax.minorticks_on()
        ax_freq.minorticks_on()
        ax.plot(times, cohsum, color = 'black', label = 'CSW')
        ax.plot(times, cswhil, color = '#D55E00', label = 'CSW Hilbert Envelope')
        #ax.plot(times_raw, cohsum_raw, color = '#009E73', label = 'Raw CSW')
        #ax.plot(times_raw, cswhil_raw, color = '#009E73', linestyle = '--', label = 'Raw CSW Hilbert Envelope')
        ax_freq.plot(f, volt_to_dB(np.abs(vf*dt)), color = 'black', label = 'CSW FFT')
        #ax_freq.plot(f_raw, volt_to_dB(np.abs(vf_raw*dt_raw)), color = '#009E73', label = 'Raw CSW FFT')
        ax.set_title(f'{ground_cal_name} Ground Cal, Run {run}, Event {ev}', fontsize = 20)
        ax.set_xlabel(f'Times [ns]', fontsize = 20)
        ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
        ax_freq.set_xlabel(f'Frequency [GHz]', fontsize = 20)
        ax_freq.set_ylabel(f'Gain [dB]', fontsize = 20)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax_freq.tick_params(axis='both', which='both', labelsize=20)
        ax.legend(fontsize=16, frameon=False)
        ax_freq.legend(fontsize=16, frameon=False)
        ax_freq.set_xlim(0,1.5)
        ax_freq.set_ylim(0,np.amax(volt_to_dB(np.abs(vf*dt)))*1.1)
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/secondaries/ground_cal_csw/{ground_cal_name}_csw_run{run}_e{ev}.png', dpi=300)
        plt.close(fig)
        plotting_counter += 1

        print(f'plotted csw for run = {run} and ev = {ev}')

def explore_multipulsicity(ground_cal_name = 'TD', n_events = 50, desired_pol = 'V', delta_az_tol = 3, delta_el_tol = 1,
                           trigger_window = (40,80), n_biggest_peaks = 3, min_spacing_ns = 20,
                           n_box_filter = 3, min_peak_dhe = 1.5, min_peak_derivs = 30):
    ### let's grab some ground cal events with reflections and plot some "multipulsy" things about them
    ### thinking: 1) csw autocorrelation (window around trigger and correlate w rest of csw --> max corr val) [EXCLUDE TRIGGER PEAK]
    ###           2) n biggest he values [EXCLUDE TRIGGER PEAK]
    ###           3) he peak finding --> smooth + require some number of incr then decr derivs, cut on dhe
    ###           4) window secondary candidates and reco dir
    ###           5) bit of a shot in the dark, but split wf after trigger into thirds and make save top 3 peaks
    if ground_cal_name == 'TD':
        source_loc_name = 'taylor_dome'
    if ground_cal_name == 'LDB':
        source_loc_name = 'ldb'

    groundcal_events = np.genfromtxt(os.environ['PUEO_ANALYSIS_TOOLS'] + f'/pueoTALON/lookup/my{ground_cal_name}_list.csv', delimiter=',', names=True, dtype=None)

    try:
        groundcal_events = groundcal_events[groundcal_events['source'] == ground_cal_name]
    except:
        groundcal_events = groundcal_events

    ### set up correlator + geography helper
    corr = correlator.Correlator()
    geography_helper = geography.GeographyHelper(root_file=os.environ['PUEO_ANALYSIS_TOOLS'] + "/pueoTALON/lookup/abxtwo.root", root_source="ABX-Two")

    plotting_counter = 0
    for ev in range(len(groundcal_events)):
        if plotting_counter >= n_events:
            break

        ### check the polarization of this event
        poli = groundcal_events['pol'][ev]
        if poli != desired_pol:
            continue

        ### load this event
        run = int(groundcal_events['run'][ev])
        if run  == corr.ChannelGeo.run:
            corr.ChannelGeo.load_event(int(groundcal_events['entry_number'][ev]))
        else:
            corr.ChannelGeo.load_run_event(run, int(groundcal_events['entry_number'][ev]))

        ### update geography helper
        event_time = corr.ChannelGeo.triggerTime()
        geography_helper.updatePosition(event_time, max_time_delta = 3)

        ### check quality cuts! 
        qual = corr.ChannelGeo.qualityCheck()
        if not qual:
            continue

        ### also check saturation (might be redudant w quality idk)
        sat = corr.ChannelGeo.checkSaturation()
        if sat:
            continue

        ### get the expected source loc for cuts
        expected_source_el_az = geography_helper.get_expected_source_location(source_loc_name)
        expected_source_el = expected_source_el_az[0]
        expected_source_az = expected_source_el_az[1]

        ### now reconstruct the directions and coherent sum
        skymap, peak_zen, peak_az, corr_vals, zoomedskymap, zoomedpeak_zen, zoomedpeak_az, zoomedcorr_vals, all_zoomedzen, all_zoomedaz = corr.reconstruct(poli, zoom_angle_range=7, N_peaks=1)
        main_peak_zen = zoomedpeak_zen[0][0]
        main_peak_el = 90 - main_peak_zen
        main_peak_az = zoomedpeak_az[0][0]

        ### check the dir
        delta_az = np.abs(main_peak_az - expected_source_az)
        delta_el = np.abs(main_peak_el - expected_source_el)
        if (delta_az > delta_az_tol) or (delta_el > delta_el_tol):
            continue

        ### get coherent sum
        cohsum, overlap_start, overlap_end = corr.coherent_sum(main_peak_zen, main_peak_az, polarization=poli, return_overlap_boundaries=True)
        cohsum = cohsum[overlap_start:overlap_end]
        cswhil = utils.hilbert_envelope(cohsum)
        times = np.arange(cohsum.size)*corr.dt

        ### let's window around our trigger for the coherent sum for auto-correlation
        trigger_window_start_idx = int(trigger_window[0] / corr.dt)
        trigger_window_end_idx = int(trigger_window[1] / corr.dt)
        trigger_window_csw = cohsum[trigger_window_start_idx:trigger_window_end_idx].copy()
        trigger_peak_idx = np.argmax(trigger_window_csw)

        ### let's also clip the wf so we dont get confused by the actual main pulse
        clipped_csw = cohsum[trigger_window_end_idx:].copy()

        ### apply a tukey window to both trigger_window_csw and clipped_csw for corss correlation
        trigger_window_csw *= scipy.signal.windows.tukey(len(trigger_window_csw), alpha=0.05)
        clipped_csw *= scipy.signal.windows.tukey(len(clipped_csw), alpha=0.05)
        
        ### now cross correlate!!!
        correlation = scipy.signal.correlate(trigger_window_csw, clipped_csw, mode='full', method='fft')
        lags = scipy.signal.correlation_lags(len(trigger_window_csw), len(clipped_csw), mode='full')
        max_lag = lags[np.argmax(np.abs(correlation))]
        max_lag_time_ns = (trigger_window_end_idx + trigger_peak_idx - max_lag) * corr.dt

        ### plot cohsum + csw hilbert envelope !!!
        fig, (ax, ax_cc) = plt.subplots(2, 1, figsize=(10, 12))
        
        ### plot csw on ax
        ax.minorticks_on()
        ax.plot(times, cohsum, color = 'black', label = 'CSW')
        ax.plot(times, cswhil, color = '#D55E00', label = 'CSW Hilbert Envelope')
        ax.axvline(max_lag_time_ns, color = 'black', linestyle = '--', label = f'Max Corr = {np.amax(np.abs(correlation)):0.2e}')
        
        ### start saving time of potential secondaries
        cohsum_secondary_pulse_times = [max_lag_time_ns]
        
        ### let's also get n biggest he peak vals + dt (excluding trigger)
        clipped_cswhil = cswhil[trigger_window_end_idx:].copy()
        cswhil_dt = times[1] - times[0]
        min_spacing_samples = int(np.ceil(min_spacing_ns / cswhil_dt))
        remaining = clipped_cswhil.copy()
        peak_idxs = []
        
        for _ in range(n_biggest_peaks):
            peak_idx = np.argmax(remaining)
            peak_idxs.append(peak_idx)

            lo = max(0, peak_idx - min_spacing_samples)
            hi = min(len(remaining), peak_idx + min_spacing_samples + 1)
            remaining[lo:hi] = -np.inf
        
        peak_idxs = np.array(peak_idxs)
        peak_vals = clipped_cswhil[peak_idxs]
        peak_times_ns = (trigger_window_end_idx + peak_idxs) * cswhil_dt

        cmap = plt.get_cmap('Spectral_r')
        norm = mpl.colors.Normalize(vmin=0, vmax=n_biggest_peaks)
        for n in range(n_biggest_peaks):
            ax.axvline(peak_times_ns[n], color = cmap(norm(n)), linestyle = '--', label = f'#{n+1} HE Peak = {peak_vals[n]:0.2f}' )
            cohsum_secondary_pulse_times.append(peak_times_ns[n])

        ### before we're done, let's do some peak finding on the csw
        ### smooth csw he + then look for continuous rising then falling derivs, cut on change in he 
        clipped_start_time_ns = trigger_window_end_idx * cswhil_dt
        smooth_clipped_cswhil = sum(clipped_cswhil[i:len(clipped_cswhil) - n_box_filter + 1 + i] for i in range(n_box_filter)) / n_box_filter ### n_box_filter wide box filter
        smooth_cswhil_derivs = smooth_clipped_cswhil[1:] - smooth_clipped_cswhil[:-1]
        smooth_start_time_ns = clipped_start_time_ns + (n_box_filter - 1) / 2 * cswhil_dt
        signs = np.sign(smooth_cswhil_derivs) ### sort into rising and falling derivs
        turning_points = np.where((signs[:-1] > 0) & (signs[1:] < 0))[0] + 1 ### find where it goes pos to neg
        ### now look for valid peaks with at least min_peak_derivs rising + falling derivs in a row and min_peak_dt
        valid_peaks_ns = []

        for peak_idx in turning_points:
            rise_start = peak_idx - 1
            while rise_start >= 0 and smooth_cswhil_derivs[rise_start] > 0:
                rise_start -= 1
            n_rising = peak_idx - rise_start - 1

            fall_end = peak_idx
            while fall_end < len(smooth_cswhil_derivs) and smooth_cswhil_derivs[fall_end] < 0:
                fall_end += 1
            n_falling = fall_end - peak_idx
           
            peak_dhe = smooth_clipped_cswhil[peak_idx] - smooth_clipped_cswhil[rise_start + 1]
            if n_rising >= min_peak_derivs and n_falling >= min_peak_derivs:
                peak_start_time_ns = smooth_start_time_ns + (rise_start + 1) * cswhil_dt
                peak_end_time_ns = smooth_start_time_ns + fall_end * cswhil_dt
                if peak_dhe > min_peak_dhe:
                    valid_peaks_ns.append(peak_start_time_ns)

        ### add our peaks to the plot!
        cmap = plt.get_cmap('Spectral_r')
        norm = mpl.colors.Normalize(vmin=0, vmax=len(valid_peaks_ns)-1)
        for n in range(len(valid_peaks_ns)):
            ax.axvline(valid_peaks_ns[n], color = cmap(norm(n)), linestyle = ':', label = f'Peak at t = {valid_peaks_ns[n]:0.2f} ns' )
            cohsum_secondary_pulse_times.append(valid_peaks_ns[n])

        ax.set_title(f'{ground_cal_name} Ground Cal, Run {run}, Event {ev}', fontsize = 20)
        ax.set_xlabel(f'Times [ns]', fontsize = 20)
        ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.legend(fontsize=10, frameon=False)
        
        ### plot csw cross correlation on ax_cc
        ax_cc.minorticks_on()
        ax_cc.plot()
        ax_cc.plot(lags, correlation, color = 'black', label = f'CSW Autocorrelation')
        ax_cc.axvline(lags[np.argmax(np.abs(correlation))], color = 'black', linestyle = '--', label = f'Max Corr = {np.amax(np.abs(correlation)):0.2e}')
        ax_cc.set_xlabel(f'Lags', fontsize = 20)
        ax_cc.set_ylabel(f'Cross-Correlation', fontsize = 20)
        ax_cc.tick_params(axis='both', which='both', labelsize=20)
        ax_cc.legend(fontsize=16, frameon=False)
        
        ### save!
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/secondaries/multipulse_metrics/{ground_cal_name}_multipulsicity_run{run}_e{ev}.png', dpi=300)
        plt.close(fig)
        plotting_counter += 1

        ### before we're done, let's reconstruct some of our promising secondary candidates!
        ### the tricky part is converting time windows into samples for reconstruct function in correlator.py
        ### we want our window as [length, center] in units of samples
        original_dt = 1/3.0
        padding_samples = 200 ### hardcoding for now, maybe make general one day
        cohsum_upsample_factor = original_dt / corr.dt
        secondary_window_length_ns = 30.0 * 2
        secondary_window_length_samples = int(secondary_window_length_ns / original_dt)

        for p in range(len(cohsum_secondary_pulse_times)):
            this_peak_time = cohsum_secondary_pulse_times[p]
            this_peak_window = [secondary_window_length_samples, int(this_peak_time / original_dt) + padding_samples]

            ### now let's reco this peak!
            skymap, peak_zen, peak_az, corr_vals, zoomedskymap, zoomedpeak_zen, zoomedpeak_az, zoomedcorr_vals, all_zoomedzen, all_zoomedaz = corr.reconstruct(poli, window=this_peak_window,  zoom_angle_range=7, N_peaks=1)
            secondary_peak_zen = zoomedpeak_zen[0][0]
            secondary_peak_el = 90 - secondary_peak_zen
            secondary_peak_az = zoomedpeak_az[0][0]

            print(f'for potential secondary at t = {this_peak_time:0.2f} ns, reco az, el = {secondary_peak_az:0.2f}, {secondary_peak_el:0.2f} (main peak reco az, el = {main_peak_az:0.2f}, {main_peak_el:0.2f})')

        print(f'plotted csw for run = {run} and ev = {ev}')

def explore_multipulsicity_forcetriggers(n_events = 50, desired_pol = 'V', trigger_window = (40,80), n_biggest_peaks = 3, min_spacing_ns = 20,
                                         n_box_filter = 3, min_peak_dhe = 1.5, min_peak_derivs = 30, poli = 'V'):
    ### let's grab some ground cal events with reflections and plot some "multipulsy" things about them
    ### thinking: 1) csw autocorrelation (window around trigger and correlate w rest of csw --> max corr val) [EXCLUDE TRIGGER PEAK]
    ###           2) n biggest he values [EXCLUDE TRIGGER PEAK]
    ###           3) he peak finding --> smooth + require some number of incr then decr derivs, cut on dhe
    ###           4) window secondary candidates and reco dir

    ### get list of runs, evs that are force triggers
    path = f'/scratch/midway3/darcynewmark/useful/nonrftriggers_run783_run808.json'
    with open(path) as f:
        force_dict = json.load(f)
    force_runs = np.asarray(force_dict['run'])
    force_evs = np.asarray(force_dict['entry'])
    print(f'total of {len(force_evs)} force trigger entries')

    ### set up correlator + geography helper
    corr = correlator.Correlator()
    geography_helper = geography.GeographyHelper(root_file=os.environ['PUEO_ANALYSIS_TOOLS'] + "/pueoTALON/lookup/abxtwo.root", root_source="ABX-Two")

    plotting_counter = 0
    for e in range(len(force_runs)):
        if plotting_counter >= n_events:
            break
        
        run = int(force_runs[e])
        ev = int(force_evs[e])
    
        ### load this event
        if run  == corr.ChannelGeo.run:
            corr.ChannelGeo.load_event(ev)
        else:
            corr.ChannelGeo.load_run_event(run, ev)

        ### update geography helper
        event_time = corr.ChannelGeo.triggerTime()
        geography_helper.updatePosition(event_time, max_time_delta = 3)

        ### check quality cuts! 
        qual = corr.ChannelGeo.qualityCheck()
        if not qual:
            continue

        ### also check saturation (might be redudant w quality idk)
        sat = corr.ChannelGeo.checkSaturation()
        if sat:
            continue

        ### now reconstruct the directions and coherent sum
        skymap, peak_zen, peak_az, corr_vals, zoomedskymap, zoomedpeak_zen, zoomedpeak_az, zoomedcorr_vals, all_zoomedzen, all_zoomedaz = corr.reconstruct(poli, zoom_angle_range=7, N_peaks=1)
        main_peak_zen = zoomedpeak_zen[0][0]
        main_peak_el = 90 - main_peak_zen
        main_peak_az = zoomedpeak_az[0][0]

        ### get coherent sum
        cohsum, overlap_start, overlap_end = corr.coherent_sum(main_peak_zen, main_peak_az, polarization=poli, return_overlap_boundaries=True)
        cohsum = cohsum[overlap_start:overlap_end]
        cswhil = utils.hilbert_envelope(cohsum)
        times = np.arange(cohsum.size)*corr.dt

        ### let's window around our trigger for the coherent sum for auto-correlation
        trigger_window_start_idx = int(trigger_window[0] / corr.dt)
        trigger_window_end_idx = int(trigger_window[1] / corr.dt)
        trigger_window_csw = cohsum[trigger_window_start_idx:trigger_window_end_idx].copy()
        trigger_peak_idx = np.argmax(trigger_window_csw)

        ### let's also clip the wf so we dont get confused by the actual main pulse
        clipped_csw = cohsum[trigger_window_end_idx:].copy()

        ### apply a tukey window to both trigger_window_csw and clipped_csw for corss correlation
        trigger_window_csw *= scipy.signal.windows.tukey(len(trigger_window_csw), alpha=0.05)
        clipped_csw *= scipy.signal.windows.tukey(len(clipped_csw), alpha=0.05)
        
        ### now cross correlate!!!
        correlation = scipy.signal.correlate(trigger_window_csw, clipped_csw, mode='full', method='fft')
        lags = scipy.signal.correlation_lags(len(trigger_window_csw), len(clipped_csw), mode='full')
        max_lag = lags[np.argmax(np.abs(correlation))]
        max_lag_time_ns = (trigger_window_end_idx + trigger_peak_idx - max_lag) * corr.dt

        ### plot cohsum + csw hilbert envelope !!!
        fig, (ax, ax_cc) = plt.subplots(2, 1, figsize=(10, 12))
        
        ### plot csw on ax
        ax.minorticks_on()
        ax.plot(times, cohsum, color = 'black', label = 'CSW')
        ax.plot(times, cswhil, color = '#D55E00', label = 'CSW Hilbert Envelope')
        ax.axvline(max_lag_time_ns, color = 'black', linestyle = '--', label = f'Max Corr = {np.amax(np.abs(correlation)):0.2e}')
        
        ### start saving time of potential secondaries
        cohsum_secondary_pulse_times = [max_lag_time_ns]
        
        ### let's also get n biggest he peak vals + dt (excluding trigger)
        clipped_cswhil = cswhil[trigger_window_end_idx:].copy()
        cswhil_dt = times[1] - times[0]
        min_spacing_samples = int(np.ceil(min_spacing_ns / cswhil_dt))
        remaining = clipped_cswhil.copy()
        peak_idxs = []
        
        for _ in range(n_biggest_peaks):
            peak_idx = np.argmax(remaining)
            peak_idxs.append(peak_idx)

            lo = max(0, peak_idx - min_spacing_samples)
            hi = min(len(remaining), peak_idx + min_spacing_samples + 1)
            remaining[lo:hi] = -np.inf
        
        peak_idxs = np.array(peak_idxs)
        peak_vals = clipped_cswhil[peak_idxs]
        peak_times_ns = (trigger_window_end_idx + peak_idxs) * cswhil_dt

        cmap = plt.get_cmap('Spectral_r')
        norm = mpl.colors.Normalize(vmin=0, vmax=n_biggest_peaks)
        for n in range(n_biggest_peaks):
            ax.axvline(peak_times_ns[n], color = cmap(norm(n)), linestyle = '--', label = f'#{n+1} HE Peak = {peak_vals[n]:0.2f}' )
            cohsum_secondary_pulse_times.append(peak_times_ns[n])

        ### before we're done, let's do some peak finding on the csw
        ### smooth csw he + then look for continuous rising then falling derivs, cut on change in he 
        clipped_start_time_ns = trigger_window_end_idx * cswhil_dt
        smooth_clipped_cswhil = sum(clipped_cswhil[i:len(clipped_cswhil) - n_box_filter + 1 + i] for i in range(n_box_filter)) / n_box_filter ### n_box_filter wide box filter
        smooth_cswhil_derivs = smooth_clipped_cswhil[1:] - smooth_clipped_cswhil[:-1]
        smooth_start_time_ns = clipped_start_time_ns + (n_box_filter - 1) / 2 * cswhil_dt
        signs = np.sign(smooth_cswhil_derivs) ### sort into rising and falling derivs
        turning_points = np.where((signs[:-1] > 0) & (signs[1:] < 0))[0] + 1 ### find where it goes pos to neg
        ### now look for valid peaks with at least min_peak_derivs rising + falling derivs in a row and min_peak_dt
        valid_peaks_ns = []

        for peak_idx in turning_points:
            rise_start = peak_idx - 1
            while rise_start >= 0 and smooth_cswhil_derivs[rise_start] > 0:
                rise_start -= 1
            n_rising = peak_idx - rise_start - 1

            fall_end = peak_idx
            while fall_end < len(smooth_cswhil_derivs) and smooth_cswhil_derivs[fall_end] < 0:
                fall_end += 1
            n_falling = fall_end - peak_idx
           
            peak_dhe = smooth_clipped_cswhil[peak_idx] - smooth_clipped_cswhil[rise_start + 1]
            if n_rising >= min_peak_derivs and n_falling >= min_peak_derivs:
                peak_start_time_ns = smooth_start_time_ns + (rise_start + 1) * cswhil_dt
                peak_end_time_ns = smooth_start_time_ns + fall_end * cswhil_dt
                if peak_dhe > min_peak_dhe:
                    valid_peaks_ns.append(peak_start_time_ns)

        ### add our peaks to the plot!
        cmap = plt.get_cmap('Spectral_r')
        norm = mpl.colors.Normalize(vmin=0, vmax=len(valid_peaks_ns)-1)
        for n in range(len(valid_peaks_ns)):
            ax.axvline(valid_peaks_ns[n], color = cmap(norm(n)), linestyle = ':', label = f'Peak at t = {valid_peaks_ns[n]:0.2f} ns' )
            cohsum_secondary_pulse_times.append(valid_peaks_ns[n])

        ax.set_title(f'Neutrino MC, Run {run}, Event {ev}', fontsize = 20)
        ax.set_xlabel(f'Times [ns]', fontsize = 20)
        ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.legend(fontsize=10, frameon=False)
        
        ### plot csw cross correlation on ax_cc
        ax_cc.minorticks_on()
        ax_cc.plot()
        ax_cc.plot(lags, correlation, color = 'black', label = f'CSW Autocorrelation')
        ax_cc.axvline(lags[np.argmax(np.abs(correlation))], color = 'black', linestyle = '--', label = f'Max Corr = {np.amax(np.abs(correlation)):0.2e}')
        ax_cc.set_xlabel(f'Lags', fontsize = 20)
        ax_cc.set_ylabel(f'Cross-Correlation', fontsize = 20)
        ax_cc.tick_params(axis='both', which='both', labelsize=20)
        ax_cc.legend(fontsize=16, frameon=False)
        
        ### save!
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/secondaries/multipulse_metrics/forcetriggers_multipulsicity_run{run}_e{ev}.png', dpi=300)
        plt.close(fig)
        plotting_counter += 1

        ### before we're done, let's reconstruct some of our promising secondary candidates!
        ### the tricky part is converting time windows into samples for reconstruct function in correlator.py
        ### we want our window as [length, center] in units of samples
        original_dt = 1/3.0
        padding_samples = 200 ### hardcoding for now, maybe make general one day
        cohsum_upsample_factor = original_dt / corr.dt
        secondary_window_length_ns = 30.0 * 2
        secondary_window_length_samples = int(secondary_window_length_ns / original_dt)

        for p in range(len(cohsum_secondary_pulse_times)):
            this_peak_time = cohsum_secondary_pulse_times[p]
            this_peak_window = [secondary_window_length_samples, int(this_peak_time / original_dt) + padding_samples]

            ### now let's reco this peak!
            skymap, peak_zen, peak_az, corr_vals, zoomedskymap, zoomedpeak_zen, zoomedpeak_az, zoomedcorr_vals, all_zoomedzen, all_zoomedaz = corr.reconstruct(poli, window=this_peak_window,  zoom_angle_range=7, N_peaks=1)
            secondary_peak_zen = zoomedpeak_zen[0][0]
            secondary_peak_el = 90 - secondary_peak_zen
            secondary_peak_az = zoomedpeak_az[0][0]

            print(f'for potential secondary at t = {this_peak_time:0.2f} ns, reco az, el = {secondary_peak_az:0.2f}, {secondary_peak_el:0.2f} (main peak reco az, el = {main_peak_az:0.2f}, {main_peak_el:0.2f})')

        print(f'plotted csw for run = {run} and ev = {ev}')

def explore_multipulsicity_mc(n_events = 50, desired_pol = 'V', trigger_window = (40,80), n_biggest_peaks = 3, min_spacing_ns = 20,
                              n_box_filter = 3, min_peak_dhe = 1.5, min_peak_derivs = 30, poli = 'V'):

    run = 194000
    channels = waveform.Channels(run, 0, sim=True) ### make channels object
    N_entry = channels.dataset.N() ### get total number of entries per run
    print(f'for run = {run}, N_entry = {N_entry}')
    
    ### set up correlator + geography helper
    corr = correlator.Correlator(run = run, sim=True)
    geography_helper = geography.GeographyHelper(root_file=os.environ['PUEO_ANALYSIS_TOOLS'] + "/pueoTALON/lookup/abxtwo.root", root_source="ABX-Two")

    plotting_counter = 0
    for e in range(N_entry):
        if plotting_counter >= n_events:
            break
        
        run = int(run)
        ev = int(e)

        ### load this event
        if run  == corr.ChannelGeo.run:
            corr.ChannelGeo.load_event(ev)
        else:
            corr.ChannelGeo.load_run_event(run, ev)

        ### update geography helper
        event_time = corr.ChannelGeo.triggerTime()
        geography_helper.updatePosition(event_time, max_time_delta = 3)

        ### check quality cuts! 
        qual = corr.ChannelGeo.qualityCheck()
        if not qual:
            continue

        ### also check saturation (might be redudant w quality idk)
        sat = corr.ChannelGeo.checkSaturation()
        if sat:
            continue

        ### now reconstruct the directions and coherent sum
        skymap, peak_zen, peak_az, corr_vals, zoomedskymap, zoomedpeak_zen, zoomedpeak_az, zoomedcorr_vals, all_zoomedzen, all_zoomedaz = corr.reconstruct(poli, zoom_angle_range=7, N_peaks=1)
        main_peak_zen = zoomedpeak_zen[0][0]
        main_peak_el = 90 - main_peak_zen
        main_peak_az = zoomedpeak_az[0][0]

        ### get coherent sum
        cohsum, overlap_start, overlap_end = corr.coherent_sum(main_peak_zen, main_peak_az, polarization=poli, return_overlap_boundaries=True)
        cohsum = cohsum[overlap_start:overlap_end]
        cswhil = utils.hilbert_envelope(cohsum)
        times = np.arange(cohsum.size)*corr.dt

        ### let's window around our trigger for the coherent sum for auto-correlation
        trigger_window_start_idx = int(trigger_window[0] / corr.dt)
        trigger_window_end_idx = int(trigger_window[1] / corr.dt)
        trigger_window_csw = cohsum[trigger_window_start_idx:trigger_window_end_idx].copy()
        trigger_peak_idx = np.argmax(trigger_window_csw)

        ### let's also clip the wf so we dont get confused by the actual main pulse
        clipped_csw = cohsum[trigger_window_end_idx:].copy()

        ### apply a tukey window to both trigger_window_csw and clipped_csw for corss correlation
        trigger_window_csw *= scipy.signal.windows.tukey(len(trigger_window_csw), alpha=0.05)
        clipped_csw *= scipy.signal.windows.tukey(len(clipped_csw), alpha=0.05)
        
        ### now cross correlate!!!
        correlation = scipy.signal.correlate(trigger_window_csw, clipped_csw, mode='full', method='fft')
        lags = scipy.signal.correlation_lags(len(trigger_window_csw), len(clipped_csw), mode='full')
        max_lag = lags[np.argmax(np.abs(correlation))]
        max_lag_time_ns = (trigger_window_end_idx + trigger_peak_idx - max_lag) * corr.dt

        ### plot cohsum + csw hilbert envelope !!!
        fig, (ax, ax_cc) = plt.subplots(2, 1, figsize=(10, 12))
        
        ### plot csw on ax
        ax.minorticks_on()
        ax.plot(times, cohsum, color = 'black', label = 'CSW')
        ax.plot(times, cswhil, color = '#D55E00', label = 'CSW Hilbert Envelope')
        ax.axvline(max_lag_time_ns, color = 'black', linestyle = '--', label = f'Max Corr = {np.amax(np.abs(correlation)):0.2e}')
        
        ### start saving time of potential secondaries
        cohsum_secondary_pulse_times = [max_lag_time_ns]
        
        ### let's also get n biggest he peak vals + dt (excluding trigger)
        clipped_cswhil = cswhil[trigger_window_end_idx:].copy()
        cswhil_dt = times[1] - times[0]
        min_spacing_samples = int(np.ceil(min_spacing_ns / cswhil_dt))
        remaining = clipped_cswhil.copy()
        peak_idxs = []
        
        for _ in range(n_biggest_peaks):
            peak_idx = np.argmax(remaining)
            peak_idxs.append(peak_idx)

            lo = max(0, peak_idx - min_spacing_samples)
            hi = min(len(remaining), peak_idx + min_spacing_samples + 1)
            remaining[lo:hi] = -np.inf
        
        peak_idxs = np.array(peak_idxs)
        peak_vals = clipped_cswhil[peak_idxs]
        peak_times_ns = (trigger_window_end_idx + peak_idxs) * cswhil_dt

        cmap = plt.get_cmap('Spectral_r')
        norm = mpl.colors.Normalize(vmin=0, vmax=n_biggest_peaks)
        for n in range(n_biggest_peaks):
            ax.axvline(peak_times_ns[n], color = cmap(norm(n)), linestyle = '--', label = f'#{n+1} HE Peak = {peak_vals[n]:0.2f}' )
            cohsum_secondary_pulse_times.append(peak_times_ns[n])

        ### before we're done, let's do some peak finding on the csw
        ### smooth csw he + then look for continuous rising then falling derivs, cut on change in he 
        clipped_start_time_ns = trigger_window_end_idx * cswhil_dt
        smooth_clipped_cswhil = sum(clipped_cswhil[i:len(clipped_cswhil) - n_box_filter + 1 + i] for i in range(n_box_filter)) / n_box_filter ### n_box_filter wide box filter
        smooth_cswhil_derivs = smooth_clipped_cswhil[1:] - smooth_clipped_cswhil[:-1]
        smooth_start_time_ns = clipped_start_time_ns + (n_box_filter - 1) / 2 * cswhil_dt
        signs = np.sign(smooth_cswhil_derivs) ### sort into rising and falling derivs
        turning_points = np.where((signs[:-1] > 0) & (signs[1:] < 0))[0] + 1 ### find where it goes pos to neg
        ### now look for valid peaks with at least min_peak_derivs rising + falling derivs in a row and min_peak_dt
        valid_peaks_ns = []

        for peak_idx in turning_points:
            rise_start = peak_idx - 1
            while rise_start >= 0 and smooth_cswhil_derivs[rise_start] > 0:
                rise_start -= 1
            n_rising = peak_idx - rise_start - 1

            fall_end = peak_idx
            while fall_end < len(smooth_cswhil_derivs) and smooth_cswhil_derivs[fall_end] < 0:
                fall_end += 1
            n_falling = fall_end - peak_idx
           
            peak_dhe = smooth_clipped_cswhil[peak_idx] - smooth_clipped_cswhil[rise_start + 1]
            if n_rising >= min_peak_derivs and n_falling >= min_peak_derivs:
                peak_start_time_ns = smooth_start_time_ns + (rise_start + 1) * cswhil_dt
                peak_end_time_ns = smooth_start_time_ns + fall_end * cswhil_dt
                if peak_dhe > min_peak_dhe:
                    valid_peaks_ns.append(peak_start_time_ns)

        ### add our peaks to the plot!
        cmap = plt.get_cmap('Spectral_r')
        norm = mpl.colors.Normalize(vmin=0, vmax=len(valid_peaks_ns)-1)
        for n in range(len(valid_peaks_ns)):
            ax.axvline(valid_peaks_ns[n], color = cmap(norm(n)), linestyle = ':', label = f'Peak at t = {valid_peaks_ns[n]:0.2f} ns' )
            cohsum_secondary_pulse_times.append(valid_peaks_ns[n])

        ax.set_title(f'Nu MC, Run {run}, Event {ev}', fontsize = 20)
        ax.set_xlabel(f'Times [ns]', fontsize = 20)
        ax.set_ylabel(f'Voltage [mV]', fontsize = 20)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.legend(fontsize=10, frameon=False)
        
        ### plot csw cross correlation on ax_cc
        ax_cc.minorticks_on()
        ax_cc.plot()
        ax_cc.plot(lags, correlation, color = 'black', label = f'CSW Autocorrelation')
        ax_cc.axvline(lags[np.argmax(np.abs(correlation))], color = 'black', linestyle = '--', label = f'Max Corr = {np.amax(np.abs(correlation)):0.2e}')
        ax_cc.set_xlabel(f'Lags', fontsize = 20)
        ax_cc.set_ylabel(f'Cross-Correlation', fontsize = 20)
        ax_cc.tick_params(axis='both', which='both', labelsize=20)
        ax_cc.legend(fontsize=16, frameon=False)
        
        ### save!
        fig.tight_layout()
        fig.savefig(f'/scratch/midway3/darcynewmark/secondaries/multipulse_metrics/numc_multipulsicity_run{run}_e{ev}.png', dpi=300)
        plt.close(fig)
        plotting_counter += 1

        ### before we're done, let's reconstruct some of our promising secondary candidates!
        ### the tricky part is converting time windows into samples for reconstruct function in correlator.py
        ### we want our window as [length, center] in units of samples
        original_dt = 1/3.0
        padding_samples = 200 ### hardcoding for now, maybe make general one day
        cohsum_upsample_factor = original_dt / corr.dt
        secondary_window_length_ns = 30.0 * 2
        secondary_window_length_samples = int(secondary_window_length_ns / original_dt)

        for p in range(len(cohsum_secondary_pulse_times)):
            this_peak_time = cohsum_secondary_pulse_times[p]
            this_peak_window = [secondary_window_length_samples, int(this_peak_time / original_dt) + padding_samples]

            ### now let's reco this peak!
            skymap, peak_zen, peak_az, corr_vals, zoomedskymap, zoomedpeak_zen, zoomedpeak_az, zoomedcorr_vals, all_zoomedzen, all_zoomedaz = corr.reconstruct(poli, window=this_peak_window,  zoom_angle_range=7, N_peaks=1)
            secondary_peak_zen = zoomedpeak_zen[0][0]
            secondary_peak_el = 90 - secondary_peak_zen
            secondary_peak_az = zoomedpeak_az[0][0]

            print(f'for potential secondary at t = {this_peak_time:0.2f} ns, reco az, el = {secondary_peak_az:0.2f}, {secondary_peak_el:0.2f} (main peak reco az, el = {main_peak_az:0.2f}, {main_peak_el:0.2f})')

        print(f'plotted csw for run = {run} and ev = {ev}')

if __name__ == "__main__":
    #plot_wfs(ground_cal_name = 'LDB')
    #plot_wfs(ground_cal_name = 'TD')
    #explore_multipulsicity(ground_cal_name = 'TD', n_events = 5)
    #explore_multipulsicity_forcetriggers(n_events = 5)
    explore_multipulsicity_mc(n_events = 5)
