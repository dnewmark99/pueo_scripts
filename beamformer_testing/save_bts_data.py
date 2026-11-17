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

def mode(inputData, axis=None, dtype=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """

    if axis is not None:
        fnc = lambda x: mode(x, dtype=dtype)
        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        def _hsm(data):
            if data.size == 1:
                return data[0]
            elif data.size == 2:
                return data.mean()
            elif data.size == 3:
                i1 = data[1] - data[0]
                i2 = data[2] - data[1]
                if i1 < i2:
                    return data[:2].mean()
                elif i2 > i1:
                    return data[1:].mean()
                else:
                    return data[1]
            else:
                wMin = data[-1] - data[0]
                N = int(data.size / 2 + data.size % 2)
                j = None
                for i in range(0, N):
                    w = data[i+N-1] - data[i]
                    if w < wMin:
                        wMin = w
                        j = i
                if j is None:
                    return data.mean()
                return _hsm(data[j:j+N])

        data = inputData.ravel()
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if dtype is not None:
            data = data.astype(dtype)

        # The data need to be sorted for this to work
        data = np.sort(data)

        # Find the mode
        dataMode = _hsm(data)

    return dataMode

def save_trigger_eff_data(array):
    ### we need to run over the desired run for the desired chunk
    #run_ranges = [(719, 730), (1501, 1524)] ### ldb, uc data change bts pulse shapes
    #chunks = [20, 10]
    #run_ranges = [(1525, 1654), ] ### azimuth scan data 
    #chunks = [10]
    #run_ranges = [(1679, 1690), ] ### surf linearity data 
    #run_ranges = [(1720, 1732), ] ### different per channel attenuation data 
    #run_ranges = [(1745, 1756),]
    #run_ranges = [(1757, 1774),]
    #run_ranges = [(1501, 1524)]
    #chunks = [10]
    run_ranges = [(1775, 1789)]
    chunks = [5]

    ### figure out which run this array belongs to
    array_remaining = array

    for (run_start, run_end), total_chunks in zip(run_ranges, chunks):

        n_runs = run_end - run_start + 1
        n_arrays = n_runs * total_chunks

        if array_remaining < n_arrays:
            ### this array belongs to this run range
            run_index = array_remaining // total_chunks
            chunk = array_remaining % total_chunks

            run = run_start + run_index
            break

        ### skip this entire run range
        array_remaining -= n_arrays

    else:
        raise ValueError(f"Array index {array} is out of range")

    ### choose the correct data location based on the actual run
    if run < 800:
        os.environ["PUEO_ROOT_DATA"] = ("/project/cozzyd/LDB/rootified/")
    else:
        os.environ["PUEO_ROOT_DATA"] = ("/project/cozzyd/postflight/rootified/")

    ### get entries in our desired chunk
    channels = waveform.Channels(run, 0)
    N_entry = channels.dataset.N()

    chunk_size = N_entry // total_chunks

    entry_start = chunk * chunk_size
    entry_end = (chunk + 1) * chunk_size

    ### include any leftover events in final chunk
    if chunk == total_chunks - 1:
        entry_end = N_entry

    print(f'for array = {array}, processing run = {run} and chunk = {chunk}, processing entries {entry_start} to {entry_end} (out of {N_entry} total entries)')

    ### now we can loop over our events and calculate what we need!
    ### used phi sectors 1-4 V-pol for the beamformer tests
    all_16ch = np.asarray(['0101V', '0201V', '0301V', '0401V',
                           '0102V', '0202V', '0302V', '0402V',
                           '0103V', '0203V', '0303V', '0403V',
                           '0104V', '0204V', '0304V', '0404V'])
    all_16ch_mask = np.isin(channels.MI_ringphipol, all_16ch)
    only_12ch = np.asarray(['0201V', '0301V', '0401V',
                            '0202V', '0302V', '0402V',
                            '0203V', '0303V', '0403V',
                            '0204V', '0304V', '0404V'])
    only_12ch_mask = np.isin(channels.MI_ringphipol, only_12ch)
    run_num = []
    entry_num = []
    trigger_time = []
    trigger_subsec = []
    ### signal strength metrics
    wf_peak_to_peak = []
    wf_peak_to_peak_noupsample = []
    he_max = []
    ### noise metrics
    wf_rms = []
    wf_median = []
    wf_mode = []
    wf_mad = []
    he_rms = []
    he_median = []
    he_mode = []
    he_mad = []
    for ev in range(entry_start, entry_end):
        channels.load_event(ev) ### load our event
        if channels.dataset.header().trigType == 1: ### check trigger type
            subsec = channels.subsecond()
            event_time = channels.triggerTime()

            ### let's also get some values for our signal strength metric
            channels.load_waveforms(filter = False) ### load waveforms
            unfilt_wfs = channels.waveforms_unfilt
            ### upsample our unfiltered wfs!
            upsampled_unfilt_wfs = scipy.signal.resample(unfilt_wfs, len(unfilt_wfs[0])*10, axis=-1)  
            hilbert_env = utils.hilbert_envelope(upsampled_unfilt_wfs)
            
            ### get peak values
            wf_p2p = np.max(upsampled_unfilt_wfs[only_12ch_mask], axis=-1) - np.min(upsampled_unfilt_wfs[only_12ch_mask], axis=-1) ### only want peak to peak from 12 channels with signal
            wf_p2pnus = np.max(unfilt_wfs[only_12ch_mask], axis=-1) - np.min(unfilt_wfs[only_12ch_mask], axis=-1) ### only want peak to peak from 12 channels with signal
            hilenv_max = np.max(hilbert_env[only_12ch_mask], axis=-1)

            ### get noise metrics
            this_wf_rms = utils.RMS(upsampled_unfilt_wfs[all_16ch_mask])
            this_wf_median = np.median(upsampled_unfilt_wfs[all_16ch_mask], axis = -1) 
            this_wf_mode = mode(upsampled_unfilt_wfs[all_16ch_mask], axis = -1)
            this_wf_mad = stats.median_abs_deviation(upsampled_unfilt_wfs[all_16ch_mask], axis = -1)

            this_he_rms = utils.RMS(hilbert_env[all_16ch_mask])
            this_he_median = np.median(hilbert_env[all_16ch_mask], axis = -1) 
            this_he_mode = mode(hilbert_env[all_16ch_mask], axis = -1)
            this_he_mad = stats.median_abs_deviation(hilbert_env[all_16ch_mask], axis = -1)

            ### now save!
            run_num.append(run)
            entry_num.append(ev)
            trigger_time.append(event_time)
            trigger_subsec.append(subsec)
            wf_peak_to_peak.append(wf_p2p.tolist())
            wf_peak_to_peak_noupsample.append(wf_p2pnus.tolist())
            he_max.append(hilenv_max.tolist())
            wf_rms.append(this_wf_rms.tolist())
            wf_median.append(this_wf_median.tolist())
            wf_mode.append(this_wf_mode.tolist())
            wf_mad.append(this_wf_mad.tolist())
            he_rms.append(this_he_rms.tolist())
            he_median.append(this_he_median.tolist())
            he_mode.append(this_he_mode.tolist())
            he_mad.append(this_he_mad.tolist())

    ### now save!
    saving_path = f'/scratch/midway3/darcynewmark/beamformer_testing/beamformerdata_run{run}_chunk{chunk}.json'
    data = {
        'run': run_num,
        'entry': entry_num,
        'trigger_time': trigger_time,
        'trigger_subsec': trigger_subsec,
        'wf_peak_to_peak': wf_peak_to_peak,
        'wf_peak_to_peak_noupsample': wf_peak_to_peak_noupsample,
        'he_max': he_max,
        'wf_rms': wf_rms,
        'wf_median': wf_median,
        'wf_mode': wf_mode,
        'wf_mad': wf_mad,
        'he_rms': he_rms,
        'he_median': he_median,
        'he_mode': he_mode,
        'he_mad': he_mad,
    }
    with open(saving_path, 'w') as f:
        json.dump(data, f, indent=4)

def save_surf_linearity_data(array):
    ### we need to run over the desired run for the desired chunk
    #run_ranges = [(1679, 1690), (1691, 1707),]
    #chunks = [5, 5]
    #run_ranges = [(1708, 1719),]
    run_ranges = [(1745, 1756),]
    chunks = [5,]

    ### figure out which run this array belongs to
    array_remaining = array

    for (run_start, run_end), total_chunks in zip(run_ranges, chunks):

        n_runs = run_end - run_start + 1
        n_arrays = n_runs * total_chunks

        if array_remaining < n_arrays:
            ### this array belongs to this run range
            run_index = array_remaining // total_chunks
            chunk = array_remaining % total_chunks

            run = run_start + run_index
            break

        ### skip this entire run range
        array_remaining -= n_arrays

    else:
        raise ValueError(f"Array index {array} is out of range")

    ### load the data location
    os.environ["PUEO_ROOT_DATA"] = ("/project/cozzyd/postflight/rootified/")

    ### get entries in our desired chunk
    channels = waveform.Channels(run, 0)
    N_entry = channels.dataset.N()

    chunk_size = N_entry // total_chunks

    entry_start = chunk * chunk_size
    entry_end = (chunk + 1) * chunk_size

    ### include any leftover events in final chunk
    if chunk == total_chunks - 1:
        entry_end = N_entry

    print(f'for array = {array}, processing run = {run} and chunk = {chunk}, processing entries {entry_start} to {entry_end} (out of {N_entry} total entries)')

    ### now we can loop over our events and calculate what we need!
    ### only care about our special surf channel!!!
    special_surf_channel = '0101H'
    dt = 1/3.0
    upsample_factor = 10
    rms_region_min_time = 200
    run_num = []
    entry_num = []
    trigger_time = []
    trigger_subsec = []
    ### signal strength metrics
    wf_peak_to_peak = []
    he_max = []
    ### noise for sanity check
    wf_rms = []
    for ev in range(entry_start, entry_end):
        channels.load_event(ev) ### load our event
        if channels.dataset.header().trigType == 1: ### check trigger type
            subsec = channels.subsecond()
            event_time = channels.triggerTime()

            ### let's also get some values for our signal strength metric
            channels.load_waveforms(filter = False) ### load waveforms
            unfilt_wfs = channels.waveforms_unfilt
            upsampled_unfilt_wfs = scipy.signal.resample(unfilt_wfs, len(unfilt_wfs[0])*upsample_factor, axis=-1) ### upsample our unfiltered wfs!
            
            wf_mask = np.isin(channels.MI_ringphipol, np.asarray([special_surf_channel])) ### mask out our special channel
            this_wf = upsampled_unfilt_wfs[wf_mask].flatten() ### flatten to 1D
            this_times = np.arange(len(this_wf)) * dt / upsample_factor 

            ### get hilbert envelope and wf pk2pk
            hilbert_env = utils.hilbert_envelope(this_wf)
            hilenv_max = np.max(hilbert_env)
            wf_p2p = np.max(this_wf) - np.min(this_wf)

            ### get rms for times > 200 ns
            this_rms = utils.RMS(this_wf[this_times > rms_region_min_time]) 

            ### now save!
            run_num.append(run)
            entry_num.append(ev)
            trigger_time.append(event_time)
            trigger_subsec.append(subsec)
            wf_peak_to_peak.append(wf_p2p)
            he_max.append(hilenv_max)
            wf_rms.append(this_rms)

    ### now save!
    saving_path = f'/scratch/midway3/darcynewmark/beamformer_testing/surflinearitydata_run{run}_chunk{chunk}.json'
    data = {
        'run': run_num,
        'entry': entry_num,
        'trigger_time': trigger_time,
        'trigger_subsec': trigger_subsec,
        'wf_peak_to_peak': wf_peak_to_peak,
        'he_max': he_max,
        'wf_rms': wf_rms,
    }
    with open(saving_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--array", type = int, default = 0)
    args = parser.parse_args()
    
    save_trigger_eff_data(args.array)
    #save_surf_linearity_data(args.array)

