import numpy as np
import os
import json
import glob
from PIL import Image

if __name__ == "__main__":
    ### we've already saved a bunch of waterfalls and spectorgram plots -- let's combine into gif!
    
    
    file_paths = sorted(glob.glob("path/to/images/*.png"))

# 2. Open all images and store them in a list
frames = [Image.open(image) for image in file_paths]

# 3. Save the first frame and append the rest
frames[0].save(
    "output.gif",
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=300,  # Time per frame in milliseconds (300ms = 0.3s)
    loop=0         # 0 = infinite loop; set to 1 or higher for a specific count
)
