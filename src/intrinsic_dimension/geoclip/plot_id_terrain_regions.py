#!/usr/bin/env python3
"""
terrain_id_analysis_geoclip.py

Compute intrinsic‐dimension statistics for specified terrain regions
using GeoCLIP location‐encoder variants, and output a summary table.
"""
import os
import argparse
import numpy as np
import torch
import skdim.id
import pandas as pd
import wandb

from model.location_encoder import LocationEncoder

import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize
import wandb


# Define regions of interest with bounding boxes
TERRAIN_REGIONS = {
    'tokyo_metro':       {'name': 'Tokyo Metropolitan',   'bounds': (139.5, 35.5, 140.0, 35.9),  'type': 'urban'},
    'nyc_boston':        {'name': 'NYC–Boston Corridor',  'bounds': (-74.5, 40.4, -71.0, 42.5),  'type': 'urban'},
    'pearl_river_delta': {'name': 'Pearl River Delta',    'bounds': (112.5, 21.5, 114.5, 23.5),'type': 'urban'},
    'ruhr_area':         {'name': 'Ruhr Area',            'bounds': (6.5,   51.2, 7.8,   51.8),  'type': 'urban'},
    'mexico_city':       {'name': 'Mexico City Metro',     'bounds': (-99.35,19.10,-98.85,19.70),'type': 'urban'},
    'sahara_desert':     {'name': 'Sahara Desert',        'bounds': (-17.0,  15.0, 35.0,  35.0), 'type': 'desert'},
    'himalayas':         {'name': 'Himalayas',            'bounds': (70.0,   25.0, 95.0,  35.0), 'type': 'mountains'},
    'amazon':            {'name': 'Amazon Rainforest',    'bounds': (-75.0, -15.0, -45.0, 5.0),'type': 'rainforest'},
    'siberian_tundra':   {'name': 'Siberian Tundra',      'bounds': (90.0,   65.0, 120.0, 72.0),'type': 'tundra'},
    'florida_everglades':{'name': 'Florida Everglades',   'bounds': (-81.5,  25.0, -80.0, 26.5),'type': 'wetland'},
}

def plot_and_log_global_regions(results: dict,
                                     encoder_name: str,
                                     save_path: str = None,
                                     cmap_name: str = "turbo"):
    """
    Draw a high‐contrast world map on a dark background, show admin-1 state
    boundaries, and circle each terrain region in a color that reflects its
    global ID. Labels sit in white boxes with arrows.
    """
    if save_path is None:
        save_path = f"global_regions_{encoder_name}.png"

    # collect IDs and build shared colormap
    gids = [d["global_id"] for d in results.values()]
    cmap = get_cmap(cmap_name)
    norm = Normalize(vmin=min(gids), vmax=max(gids))
    sm   = ScalarMappable(norm=norm, cmap=cmap)

    # manual style: dark bg, light grid
    plt.rcParams.update({
        "figure.facecolor": "#1e1e1e",
        "axes.facecolor":   "#2b2b2b",
        "axes.edgecolor":   "white",
        "grid.color":       "#444444",
        "axes.grid":        True,
        "xtick.color":      "white",
        "ytick.color":      "white",
        "text.color":       "white",
        "font.size":        10
    })

    fig, ax = plt.subplots(1, 1, figsize=(14, 7),
                           subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_global()

    # dark land + white ocean
    ax.add_feature(cfeature.LAND,   facecolor="#2b2b2b", zorder=0)
    ax.add_feature(cfeature.OCEAN,  facecolor="#1e1e1e", zorder=0)
    ax.coastlines(resolution="110m", color="white", linewidth=0.8, zorder=1)

    # state/province lines
    admin1 = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces",
        scale="110m",
        facecolor="none",
        edgecolor="white",
        linewidth=0.5
    )
    ax.add_feature(admin1, zorder=2)

    # draw each region circle + label
    for data in results.values():
        info = data["info"]
        gid  = data["global_id"]
        w,s,e,n = info["bounds"]
        lon0, lat0 = 0.5*(w+e), 0.5*(s+n)
        r = 0.5 * ((e-w)**2 + (n-s)**2)**0.5

        color = cmap(norm(gid))
        circ = Circle((lon0, lat0), r,
                      transform=ccrs.PlateCarree(),
                      edgecolor=color,
                      facecolor="none",
                      linewidth=2,
                      zorder=3)
        ax.add_patch(circ)

        dx, dy = r*0.8, r*0.8
        label = f"{info['name']}\n{gid:.1f}"
        ax.annotate(label,
                    xy=(lon0, lat0),
                    xytext=(lon0+dx, lat0+dy),
                    textcoords="data",
                    ha="center", va="center",
                    fontsize=9,
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="white", ec="none", alpha=0.9),
                    arrowprops=dict(arrowstyle="->",
                                    color=color, lw=1.2),
                    zorder=4)

    # colorbar at bottom
    cb = fig.colorbar(sm, ax=ax,
                      orientation="horizontal",
                      fraction=0.05,
                      pad=0.08)
    cb.set_label("Global Intrinsic Dimension", color="white")
    cb.outline.set_edgecolor("white")
    cb.ax.xaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax.axes, 'xticklines'), color='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    wandb.log({f"{encoder_name}/global_region_map_dark": wandb.Image(save_path)})
    return save_path

def sample_region_points(bounds, n_points):
    """Uniformly sample n_points within lon/lat bounds."""
    west, south, east, north = bounds
    lons = np.random.uniform(west, east, n_points)
    u    = np.random.uniform(np.sin(np.radians(south)),
                             np.sin(np.radians(north)),
                             n_points)
    lats = np.degrees(np.arcsin(u))
    return np.stack([lons, lats], axis=1)

def analyze_terrain_id(encoder_cfg, device, regions, n_points, ks):
    """
    For each region, sample points, embed via GeoCLIP, compute global &
    local MLE intrinsic dimensions.
    """
    # build encoder
    sigmas = [2.0**e for e in encoder_cfg["sigma_exps"]]
    if encoder_cfg.get("ckpt") is None:
        loc_enc = LocationEncoder(sigma=sigmas, from_pretrained=True)
    else:
        loc_enc = LocationEncoder(sigma=sigmas, from_pretrained=False)
        state = torch.load(encoder_cfg["ckpt"], map_location=device)
        loc_enc.load_state_dict(state)
    loc_enc.to(device).eval()

    results = {}
    for key, info in regions.items():
        # sample and embed
        pts = sample_region_points(info['bounds'], n_points)
        model_in = torch.from_numpy(pts[:, [1,0]].astype(np.float32)).to(device)
        with torch.no_grad():
            emb = loc_enc(model_in).cpu().numpy()

        # global ID
        mle = skdim.id.MLE(neighborhood_based=True)
        k_glob = min(20, max(5, n_points//10))
        mle.fit(emb, n_neighbors=k_glob, n_jobs=-1)
        global_id = float(mle.dimension_)

        # local IDs
        local = {}
        for k in ks:
            if k < len(pts)//2:
                raw, smooth = mle.fit_transform_pw(emb, n_neighbors=k, n_jobs=-1, smooth=True)
                local[k] = {"raw": raw, "smooth": smooth}

        results[key] = {
            "info":       info,
            "coords":     pts,
            "global_id":  global_id,
            "local_ids":  local,
            "n_points":   len(pts)
        }
    return results

def create_id_statistics_report(results, k):
    """
    Build a pandas DataFrame summarizing global & local ID stats per region.
    """
    rows = []
    for key, data in results.items():
        info = data["info"]
        if k not in data["local_ids"]:
            continue
        smooth = data["local_ids"][k]["smooth"]
        rows.append({
            "Region":         info["name"],
            "Type":           info["type"],
            "Global_ID":      data["global_id"],
            "Local_ID_Mean":  float(np.mean(smooth)),
            "Local_ID_Std":   float(np.std(smooth)),
            "Local_ID_Min":   float(np.min(smooth)),
            "Local_ID_Max":   float(np.max(smooth)),
            "N_Points":       data["n_points"],
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["Type", "Global_ID"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_points", type=int, default=100000,
                        help="Samples per region")
    parser.add_argument("--ks",       type=int, nargs="+", default=[5,10,20],
                        help="Neighborhood sizes")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    wandb.init(
        project="EigenSpectrum Analysis",
        name="terrain_id_analysis_geoclip"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # encoder configurations
    encoders = {
        "hierarchy_3":       {"sigma_exps": np.linspace(0,8,3).tolist(),  "ckpt": None},
        "hierarchy_10":      {"sigma_exps": np.linspace(0,8,10).tolist(), "ckpt": "checkpoints/hierarchy_10/best_locenc.pt"},
        "extended_maxexp_12":{"sigma_exps": [0,4,8,12],                   "ckpt": "checkpoints/extended_maxexp_12/best_locenc.pt"},
        "extended_maxexp_16":{"sigma_exps": [0,4,8,12,16],                "ckpt": "checkpoints/extended_maxexp_16/best_locenc.pt"},
    }

    for name, cfg in encoders.items():
        print(f"Analyzing {name}…")
        results = analyze_terrain_id(
            cfg, device, TERRAIN_REGIONS,
            n_points=args.n_points,
            ks=args.ks
        )
        map_path = plot_and_log_global_regions(results, name)
        stats = create_id_statistics_report(results, k=args.ks[0])
        csv_path = f"terrain_id_stats_{name}.csv"
        stats.to_csv(csv_path, index=False)
        table = wandb.Table(dataframe=stats)
        wandb.log({f"{name}/id_statistics": table})
        print(stats.to_string())

if __name__ == "__main__":
    main()
