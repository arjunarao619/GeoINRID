import rasterio
import os
from tqdm import tqdm

if __name__ == "__main__":

    fns = [
        f"/home/kklemmer/data/s2-300k/images/{fn}"
        for fn in os.listdir("/home/kklemmer/data/s2-300k/images/")
    ]

    # No compression
    for fn in tqdm(fns):
        with rasterio.open(fn) as f:
            profile = f.profile
            img = f.read()
        profile["blockxsize"] = 256
        profile["blockysize"] = 256
        profile["interleave"] = "band"
        profile["compress"] = None
        #profile["predictor"] = 2
        output_fn = fn.replace("/images/", "/no_compression/")
        #if not os.path.exists(output_fn):
        with rasterio.open(output_fn, "w", **profile) as f:
            f.write(img)