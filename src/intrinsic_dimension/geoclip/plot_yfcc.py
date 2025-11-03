#!/usr/bin/env python3
"""
plot_yfcc_samples.py

Randomly selects 4 images from your YFCC1k dataset, plots them in a single row,
and annotates each with its latitude and longitude. Saves the figure as a PNG.
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def main():
    parser = argparse.ArgumentParser(
        description="Plot 4 random YFCC1k images with geo-coordinates"
    )
    parser.add_argument(
        "--csv", type=str,
        default="/scratch/local/arra4944_images/yfcc/yfcc15m.csv",
        help="Path to YFCC15M metadata CSV"
    )
    parser.add_argument(
        "--img-dir", type=str,
        default="/scratch/local/arra4944_images/yfcc/images",
        help="Directory containing downloaded YFCC images"
    )
    parser.add_argument(
        "--output", type=str,
        default="yfcc_sample_4.png",
        help="Output filename for the saved figure"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling"
    )
    args = parser.parse_args()

    # 1) Load metadata
    df = pd.read_csv(args.csv, low_memory=False)
    df = df.dropna(subset=["latitude","longitude"])
    df = df[(df.latitude != 0) & (df.longitude != 0)]

    # 2) Build full image paths
    df["filename"] = df.photoid.astype(str) + "." + df.ext.str.lower().str.strip()
    df["path"] = df.filename.apply(lambda fn: os.path.join(args.img_dir, fn))

    # 3) Keep only files that exist
    df = df[df.path.apply(os.path.exists)].reset_index(drop=True)

    # 4) Sample 4 random entries
    sample = df.sample(n=4, random_state=args.seed)

    # 5) Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (_, row) in zip(axes, sample.iterrows()):
        img = Image.open(row.path).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Lat: {row.latitude:.4f}\nLon: {row.longitude:.4f}", fontsize=10)

    plt.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved sample figure to {args.output}")

if __name__ == "__main__":
    main()
