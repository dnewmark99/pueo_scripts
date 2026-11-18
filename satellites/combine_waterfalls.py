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

filt_f = np.asarray([0.0, 0.0029296875, 0.005859375, 0.0087890625, 0.01171875, 0.0146484375, 0.017578125, 0.0205078125, 0.0234375, 0.0263671875, 0.029296875, 0.0322265625, 0.03515625, 0.0380859375, 0.041015625, 0.0439453125, 0.046875, 0.0498046875, 0.052734375, 0.0556640625, 0.05859375, 0.0615234375, 0.064453125, 0.0673828125, 0.0703125, 0.0732421875, 0.076171875, 0.0791015625, 0.08203125, 0.0849609375, 0.087890625, 0.0908203125, 0.09375, 0.0966796875, 0.099609375, 0.1025390625, 0.10546875, 0.1083984375, 0.111328125, 0.1142578125, 0.1171875, 0.1201171875, 0.123046875, 0.1259765625, 0.12890625, 0.1318359375, 0.134765625, 0.1376953125, 0.140625, 0.1435546875, 0.146484375, 0.1494140625, 0.15234375, 0.1552734375, 0.158203125, 0.1611328125, 0.1640625, 0.1669921875, 0.169921875, 0.1728515625, 0.17578125, 0.1787109375, 0.181640625, 0.1845703125, 0.1875, 0.1904296875, 0.193359375, 0.1962890625, 0.19921875, 0.2021484375, 0.205078125, 0.2080078125, 0.2109375, 0.2138671875, 0.216796875, 0.2197265625, 0.22265625, 0.2255859375, 0.228515625, 0.2314453125, 0.234375, 0.2373046875, 0.240234375, 0.2431640625, 0.24609375, 0.2490234375, 0.251953125, 0.2548828125, 0.2578125, 0.2607421875, 0.263671875, 0.2666015625, 0.26953125, 0.2724609375, 0.275390625, 0.2783203125, 0.28125, 0.2841796875, 0.287109375, 0.2900390625, 0.29296875, 0.2958984375, 0.298828125, 0.3017578125, 0.3046875, 0.3076171875, 0.310546875, 0.3134765625, 0.31640625, 0.3193359375, 0.322265625, 0.3251953125, 0.328125, 0.3310546875, 0.333984375, 0.3369140625, 0.33984375, 0.3427734375, 0.345703125, 0.3486328125, 0.3515625, 0.3544921875, 0.357421875, 0.3603515625, 0.36328125, 0.3662109375, 0.369140625, 0.3720703125, 0.375, 0.3779296875, 0.380859375, 0.3837890625, 0.38671875, 0.3896484375, 0.392578125, 0.3955078125, 0.3984375, 0.4013671875, 0.404296875, 0.4072265625, 0.41015625, 0.4130859375, 0.416015625, 0.4189453125, 0.421875, 0.4248046875, 0.427734375, 0.4306640625, 0.43359375, 0.4365234375, 0.439453125, 0.4423828125, 0.4453125, 0.4482421875, 0.451171875, 0.4541015625, 0.45703125, 0.4599609375, 0.462890625, 0.4658203125, 0.46875, 0.4716796875, 0.474609375, 0.4775390625, 0.48046875, 0.4833984375, 0.486328125, 0.4892578125, 0.4921875, 0.4951171875, 0.498046875, 0.5009765625, 0.50390625, 0.5068359375, 0.509765625, 0.5126953125, 0.515625, 0.5185546875, 0.521484375, 0.5244140625, 0.52734375, 0.5302734375, 0.533203125, 0.5361328125, 0.5390625, 0.5419921875, 0.544921875, 0.5478515625, 0.55078125, 0.5537109375, 0.556640625, 0.5595703125, 0.5625, 0.5654296875, 0.568359375, 0.5712890625, 0.57421875, 0.5771484375, 0.580078125, 0.5830078125, 0.5859375, 0.5888671875, 0.591796875, 0.5947265625, 0.59765625, 0.6005859375, 0.603515625, 0.6064453125, 0.609375, 0.6123046875, 0.615234375, 0.6181640625, 0.62109375, 0.6240234375, 0.626953125, 0.6298828125, 0.6328125, 0.6357421875, 0.638671875, 0.6416015625, 0.64453125, 0.6474609375, 0.650390625, 0.6533203125, 0.65625, 0.6591796875, 0.662109375, 0.6650390625, 0.66796875, 0.6708984375, 0.673828125, 0.6767578125, 0.6796875, 0.6826171875, 0.685546875, 0.6884765625, 0.69140625, 0.6943359375, 0.697265625, 0.7001953125, 0.703125, 0.7060546875, 0.708984375, 0.7119140625, 0.71484375, 0.7177734375, 0.720703125, 0.7236328125, 0.7265625, 0.7294921875, 0.732421875, 0.7353515625, 0.73828125, 0.7412109375, 0.744140625, 0.7470703125, 0.75, 0.7529296875, 0.755859375, 0.7587890625, 0.76171875, 0.7646484375, 0.767578125, 0.7705078125, 0.7734375, 0.7763671875, 0.779296875, 0.7822265625, 0.78515625, 0.7880859375, 0.791015625, 0.7939453125, 0.796875, 0.7998046875, 0.802734375, 0.8056640625, 0.80859375, 0.8115234375, 0.814453125, 0.8173828125, 0.8203125, 0.8232421875, 0.826171875, 0.8291015625, 0.83203125, 0.8349609375, 0.837890625, 0.8408203125, 0.84375, 0.8466796875, 0.849609375, 0.8525390625, 0.85546875, 0.8583984375, 0.861328125, 0.8642578125, 0.8671875, 0.8701171875, 0.873046875, 0.8759765625, 0.87890625, 0.8818359375, 0.884765625, 0.8876953125, 0.890625, 0.8935546875, 0.896484375, 0.8994140625, 0.90234375, 0.9052734375, 0.908203125, 0.9111328125, 0.9140625, 0.9169921875, 0.919921875, 0.9228515625, 0.92578125, 0.9287109375, 0.931640625, 0.9345703125, 0.9375, 0.9404296875, 0.943359375, 0.9462890625, 0.94921875, 0.9521484375, 0.955078125, 0.9580078125, 0.9609375, 0.9638671875, 0.966796875, 0.9697265625, 0.97265625, 0.9755859375, 0.978515625, 0.9814453125, 0.984375, 0.9873046875, 0.990234375, 0.9931640625, 0.99609375, 0.9990234375, 1.001953125, 1.0048828125, 1.0078125, 1.0107421875, 1.013671875, 1.0166015625, 1.01953125, 1.0224609375, 1.025390625, 1.0283203125, 1.03125, 1.0341796875, 1.037109375, 1.0400390625, 1.04296875, 1.0458984375, 1.048828125, 1.0517578125, 1.0546875, 1.0576171875, 1.060546875, 1.0634765625, 1.06640625, 1.0693359375, 1.072265625, 1.0751953125, 1.078125, 1.0810546875, 1.083984375, 1.0869140625, 1.08984375, 1.0927734375, 1.095703125, 1.0986328125, 1.1015625, 1.1044921875, 1.107421875, 1.1103515625, 1.11328125, 1.1162109375, 1.119140625, 1.1220703125, 1.125, 1.1279296875, 1.130859375, 1.1337890625, 1.13671875, 1.1396484375, 1.142578125, 1.1455078125, 1.1484375, 1.1513671875, 1.154296875, 1.1572265625, 1.16015625, 1.1630859375, 1.166015625, 1.1689453125, 1.171875, 1.1748046875, 1.177734375, 1.1806640625, 1.18359375, 1.1865234375, 1.189453125, 1.1923828125, 1.1953125, 1.1982421875, 1.201171875, 1.2041015625, 1.20703125, 1.2099609375, 1.212890625, 1.2158203125, 1.21875, 1.2216796875, 1.224609375, 1.2275390625, 1.23046875, 1.2333984375, 1.236328125, 1.2392578125, 1.2421875, 1.2451171875, 1.248046875, 1.2509765625, 1.25390625, 1.2568359375, 1.259765625, 1.2626953125, 1.265625, 1.2685546875, 1.271484375, 1.2744140625, 1.27734375, 1.2802734375, 1.283203125, 1.2861328125, 1.2890625, 1.2919921875, 1.294921875, 1.2978515625, 1.30078125, 1.3037109375, 1.306640625, 1.3095703125, 1.3125, 1.3154296875, 1.318359375, 1.3212890625, 1.32421875, 1.3271484375, 1.330078125, 1.3330078125, 1.3359375, 1.3388671875, 1.341796875, 1.3447265625, 1.34765625, 1.3505859375, 1.353515625, 1.3564453125, 1.359375, 1.3623046875, 1.365234375, 1.3681640625, 1.37109375, 1.3740234375, 1.376953125, 1.3798828125, 1.3828125, 1.3857421875, 1.388671875, 1.3916015625, 1.39453125, 1.3974609375, 1.400390625, 1.4033203125, 1.40625, 1.4091796875, 1.412109375, 1.4150390625, 1.41796875, 1.4208984375, 1.423828125, 1.4267578125, 1.4296875, 1.4326171875, 1.435546875, 1.4384765625, 1.44140625, 1.4443359375, 1.447265625, 1.4501953125, 1.453125, 1.4560546875, 1.458984375, 1.4619140625, 1.46484375, 1.4677734375, 1.470703125, 1.4736328125, 1.4765625, 1.4794921875, 1.482421875, 1.4853515625, 1.48828125, 1.4912109375, 1.494140625, 1.4970703125, 1.5])

def grab_data_json(base_path = '/scratch/midway3/darcynewmark/satellites/jsons'):

    data = {}
    path = f'{base_path}/waterfall_all_runs_*.json'
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

def plot_vars():
    ### let's read in our json data
    data = grab_data_json()
    
    ### unpack it all
    run = np.asarray(data['run'])
    entry = np.asarray(data['entry'])
    flag = np.asarray(data['flag'])
    groundcal_pol = np.asarray(data['groundcal_pol'])
    time_per_event = np.asarray(data['time_per_event'])
    power_filtered_vpol = np.asarray(data['power_filtered_vpol'])
    power_unfiltered_vpol = np.asarray(data['power_unfiltered_vpol'])
    power_filtered_hpol = np.asarray(data['power_filtered_hpol'])
    power_unfiltered_hpol = np.asarray(data['power_unfiltered_hpol'])

    unique_runs = np.unique(run)
    print(f'loaded data! total of {len(entry)} events across {len(unique_runs)} runs!')
    print(f'unique_runs = {unique_runs}')

    ### let's first plot our TD
    groundcal_vpol_mask = np.logical_and(flag == 'TD', groundcal_pol == 'V')
    groundcal_hpol_mask = np.logical_and(flag == 'TD', groundcal_pol == 'H')
  
    ### jk...havent gotten any ground cal yet :)
    print(f'processed {np.sum(groundcal_vpol_mask)} v-pol ground cal events and {np.sum(groundcal_hpol_mask)} h-pol ground cal events')
        
    ### ok let's make some basic plots
    for r in unique_runs:
        this_run_mask = run == r
        this_power_filtered_vpol = power_filtered_vpol[this_run_mask]
        this_power_unfiltered_vpol = power_unfiltered_vpol[this_run_mask]
        this_power_filtered_hpol = power_filtered_hpol[this_run_mask]
        this_power_unfiltered_hpol = power_unfiltered_hpol[this_run_mask]

        all_waterfalls = [this_power_filtered_vpol, this_power_unfiltered_vpol, this_power_filtered_hpol, this_power_unfiltered_hpol,]
        all_titles = ['Filtered (V-Pol)', 'Unfiltered (V-Pol)', 'Filtered (H-Pol)', 'Unfiltered (H-Pol)', ]
        all_saving_names = ['filt_vpol', 'unfilt_vpol', 'filt_hpol', 'unfilt_hpol']

        ### event bin edges
        n_events = len(this_power_filtered_vpol)
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
            ax.set_title(f'Run {r} Force Triggers, {title}', fontsize = 20)
            ax.set_xlabel(f'Event', fontsize = 20)
            ax.set_ylabel(f'Frequency [GHz]', fontsize = 20)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=14)
            fig.tight_layout()
            plt.savefig(f'/scratch/midway3/darcynewmark/satellites/waterfall_run{r}forcetriggers_{saving_name}.png', dpi=300)
            plt.close(fig)

        ### let's also make some one-d projections (averaging over events)
        one_d_filt = [this_power_filtered_vpol, this_power_filtered_hpol]
        one_d_unfilt = [this_power_unfiltered_vpol, this_power_unfiltered_hpol]
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
            ax.set_title(f'Run {r} Force Triggers, {labels[p]}, {len(one_d_filt[p])} Total Events', fontsize = 20)
            ax.set_xlabel(f'Frequency [GHz]', fontsize = 20)
            ax.set_ylabel(f'Average Power [dB]', fontsize = 20)
            ax.legend(fontsize=14, frameon=False)
            ax.minorticks_on()
            ax.tick_params(axis='both', which='both', labelsize=20)
            ax.set_xlim(0,1.5)
            fig.tight_layout()
            fig.savefig(f'/scratch/midway3/darcynewmark/satellites/waterfall_run{r}forcetriggers_oned_{saving_names[p]}.png', dpi=300)
            plt.close(fig)

if __name__ == "__main__":
    plot_vars()

