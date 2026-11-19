import numpy as np
import os
import json
import glob
from PIL import Image

def make_gif(fig_name, saving_name, base_path = '/scratch/midway3/darcynewmark/satellites'):
    ### get all our images
    path = f'{base_path}/{fig_name}'
    fnames = [name for name in glob.glob(path)]
    fnames = sorted(fnames)

    ### now load
    images = [Image.open(f).convert('RGB') for f in fnames]

    ### save as gif
    images[0].save(f'{base_path}/{saving_name}.gif', save_all=True, append_images=images[1:], duration=300, loop=0)

if __name__ == "__main__":
    ### we've already saved a bunch of waterfalls and spectorgram plots -- let's combine into gif!
    fig_names = ['waterfall_run*forcetriggers_unfilt_vpol.png', 'waterfall_run*forcetriggers_unfilt_hpol.png',
                 'waterfall_run*forcetriggers_filt_vpol.png', 'waterfall_run*forcetriggers_filt_hpol.png',
                 'waterfall_run*forcetriggers_oned_vpol.png', 'waterfall_run*forcetriggers_oned_hpol.png',
                ]
    saving_names = ['waterfall_unfilt_vpol', 'waterfall_unfilt_hpol',
                    'waterfall_filt_vpol', 'waterfall_filt_hpol',
                    'waterfall_oned_vpol', 'waterfall_oned_hpol',
                   ]
   
    for f in range(len(fig_names)):
        make_gif(fig_names[f], saving_names[f])

