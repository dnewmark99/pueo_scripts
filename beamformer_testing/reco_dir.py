import numpy as np
import ROOT
import os
import json
import glob
import argparse
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm

import pueoTALON.pueoPythonAnalyzer.waveform as waveform
import pueoTALON.pueoPythonAnalyzer.plotter as pltr
import pueoTALON.pueoPythonAnalyzer.antenna as antenna
import pueoTALON.pueoPythonAnalyzer.correlator as correlator
import antennaUtils.response as antResponse
import pueoTALON.pueoPythonAnalyzer.utils as utils
import pueoTALON.pueoPythonAnalyzer.geography as geography

ROOT.gSystem.Load("libpueoEvent.so")

def save_data(array):
    ### we need to run over the desired run for the desired chunk
    runs = [719, 1433, 1446, 1464] ## just a few runs for now
    n_chunks = 10

    run_idx = array // n_chunks
    chunk = array % n_chunks

    run = runs[run_idx]

    if run < 800:
        ### ldb data
        os.environ["PUEO_ROOT_DATA"] = ("/project/cozzyd/LDB/rootified/")
    else:
        ### uchicago data
        os.environ["PUEO_ROOT_DATA"] = ("/project/cozzyd/postflight/rootified/")

    ### now let's chunk up entries for our desired run
    this_run_entry_path = f'/scratch/midway3/darcynewmark/beamformer_testing/btsselectedevents_run{run}.json'
    with open(this_run_entry_path) as f:
        all_entries = np.asarray(json.load(f))

    entry_chunks = np.array_split(all_entries, n_chunks)
    entries = entry_chunks[chunk]

    print(f'for array = {array}, run = {run}, chunk = {chunk}, running over {len(entries)} entries')

    ### now we can loop over our events and calculate what we need!
    corr = correlator.Correlator(run = run)
    all_az = []
    all_el = []
    for ev in entries:
        corr.ChannelGeo.load_run_event(int(run), int(ev))
        skymap, peak_zen, peak_az, corr_vals, zoomedskymap, zoomedpeak_zen, zoomedpeak_az, zoomedcorr_vals, all_zoomedzen, all_zoomedaz = corr.reconstruct('V', zoom_angle_range=7, N_peaks=1)
        main_peak_zen = zoomedpeak_zen[0][0]
        main_peak_el = 90 - main_peak_zen
        main_peak_az = zoomedpeak_az[0][0]

        all_az.append(main_peak_az)
        all_el.append(main_peak_el)

    ### now save!
    saving_path = f'/scratch/midway3/darcynewmark/beamformer_testing/recodirection_run{run}_chunk{chunk}.json'
    data = {
        'az': all_az,
        'el': all_el,
    }
    with open(saving_path, 'w') as f:
        json.dump(data, f, indent=4)

def read_data(run):

    data = {}
    path = f'/scratch/midway3/darcynewmark/beamformer_testing/recodirection_run{run}_chunk*.json'
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

def plot_reco_dir():
    ### read in our data and plot distributions!
    ldb_5_min15_data = read_data(run = 719)
    uc_5_min15_data = read_data(run = 1433)
    uc_5_min10_data = read_data(run = 1446)
    uc_5_min5_data = read_data(run = 1464)

    all_data = [ldb_5_min15_data, uc_5_min15_data, uc_5_min10_data, uc_5_min5_data]
    all_az = [5,5,5,5]
    all_el = [-15, -15, -10, -5]
    saving_name = ['LDB_r=719', 'UC_r=1433', 'UC_r=1446', 'UC_r=1464']
    title_descrip = ['LDB', 'UC', 'UC', 'UC']

    for a in range(len(all_data)):
        ### let's make two-dimensional plots of reco az vs reco el
        reco_az = all_data[a]["az"]
        reco_el = all_data[a]["el"]

        n_bins = 80
        az_bin_edges = np.linspace(22, 35, n_bins+1)
        el_bin_edges = np.linspace(all_el[a]-7, all_el[a]+5, n_bins+1)

        ### make our matrix
        M, _, _ = np.histogram2d(reco_az, reco_el, bins=[az_bin_edges, el_bin_edges])

        ### plot!
        cm = plt.get_cmap("Spectral_r")
        lin_norm = matplotlib.colors.LogNorm(vmin=1, vmax=np.amax(M.T))

        fig, ax = plt.subplots(figsize=(10, 7))
        mesh = ax.pcolormesh(az_bin_edges, el_bin_edges, M.T, cmap=cm, norm = lin_norm)
        cb = fig.colorbar(mesh, ax=ax)
        cb.ax.set_ylabel("Number of Events", fontsize = 16)
        cb.ax.minorticks_off()
        ax.scatter(all_az[a]+22.5, all_el[a], marker = '*', s = 250, color = 'black')
        ax.set_title(rf'BTS {title_descrip[a]} Test -- $\phi = {all_az[a]}^\circ,\ \theta = {all_el[a]}^\circ$', fontsize = 16)
        ax.set_xlabel(f'Reconstructed Azimuth [deg]', fontsize = 16)
        ax.set_ylabel(f'Reconstructed Elevation [deg]', fontsize = 16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        fig.tight_layout()
        plt.savefig(f'/scratch/midway3/darcynewmark/beamformer_testing/btsrecodir_{saving_name[a]}.png', dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--array", type = int, default = 0)
    args = parser.parse_args()

    #save_data(args.array)
    plot_reco_dir()
