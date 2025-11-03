#!/usr/bin/env python
import os
import argparse
import numpy as np
import torch
import skdim.id
import matplotlib.pyplot as plt
import wandb
from scipy.stats import gaussian_kde
from pathlib import Path
from model.location_encoder import LocationEncoder


from cartopy.io import shapereader
from shapely.geometry import Point
from shapely.ops import unary_union
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm

# ─── Sampling functions ─────────────────────────────────────────────────────────
def sample_naive(N, **kwargs):
    lats = np.random.uniform(-90, 90, N)
    lons = np.random.uniform(-180, 180, N)
    return np.stack([lons, lats], axis=1)

def sample_sphere(N, **kwargs):
    u = np.random.uniform(-1, 1, N)
    lats = np.degrees(np.arcsin(u))
    lons = np.random.uniform(-180, 180, N)
    return np.stack([lons, lats], axis=1)

def sample_grid(N, lat_step=1.0, lon_step=1.0, **kwargs):
    lat_vals = np.arange(-90, 90+lat_step, lat_step)
    lon_vals = np.arange(-180, 180+lon_step, lon_step)
    grid = np.stack(np.meshgrid(lon_vals, lat_vals), axis=-1).reshape(-1,2)
    if len(grid) >= N:
        idx = np.random.choice(len(grid), N, replace=False)
        return grid[idx]
    extra = N - len(grid)
    return np.vstack([grid, sample_naive(extra)])

def sample_healpix(N, **kwargs):
    import healpy as hp
    nside = 1
    while hp.nside2npix(nside) < N: nside *= 2
    sel = np.random.choice(hp.nside2npix(nside), N, replace=False)
    theta, phi = hp.pix2ang(nside, sel)
    lats = 90 - np.degrees(theta)
    lons = np.degrees(phi) - 180
    return np.stack([lons, lats], axis=1)

def sample_fibonacci(N, **kwargs):
    i   = np.arange(N)
    phi = np.pi*(3.-np.sqrt(5.))
    lats = np.degrees(np.arcsin(2*(i/N)-1))
    lons = np.degrees((i*phi)%(2*np.pi) - np.pi)
    return np.stack([lons, lats], axis=1)

def sample_sobol(N, **kwargs):
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=2, scramble=True)
    m = int(np.ceil(np.log2(N)))
    u = sampler.random_base2(m)[:N]
    lons = u[:,0]*360.-180.
    lats = np.degrees(np.arcsin(2*u[:,1]-1))
    return np.stack([lons, lats], axis=1)

def sample_latin(N, **kwargs):
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=2)
    u = sampler.random(n=N)
    lons = u[:,0]*360.-180.
    lats = np.degrees(np.arcsin(2*u[:,1]-1))
    return np.stack([lons, lats], axis=1)

def sample_stratified(N, bands=10, **kwargs):
    per = N // bands
    lats_list, lons_list = [], []
    for i in range(bands):
        lat_min = -90 + i*(180/bands)
        lat_max = lat_min + 180/bands
        lats_list.append(np.random.uniform(lat_min, lat_max, per))
        lons_list.append(np.random.uniform(-180, 180, per))
    lats = np.concatenate(lats_list)
    lons = np.concatenate(lons_list)
    if len(lats) < N:
        extra = N - len(lats)
        return np.vstack([np.stack([lons, lats],axis=1), sample_naive(extra)])
    return np.stack([lons, lats], axis=1)

def sample_land(N, **kwargs):
    shp = shapereader.natural_earth('110m', 'physical', 'land')
    geoms = list(shapereader.Reader(shp).geometries())
    land_poly = unary_union(geoms)
    pts, batch = [], max(N, 10_000)
    while len(pts) < N:
        lons = np.random.uniform(-180, 180, batch)
        u    = np.random.uniform(-1, 1, batch)
        lats = np.degrees(np.arcsin(u))
        for lon, lat in zip(lons, lats):
            if land_poly.contains(Point(lon, lat)):
                pts.append((lon, lat))
                if len(pts) == N:
                    break
    return np.array(pts)

SAMPLERS = {
    'naive':     sample_naive,
    'sphere':    sample_sphere,
    'grid':      sample_grid,
    'healpix':   sample_healpix,
    'fibonacci': sample_fibonacci,
    'sobol':     sample_sobol,
    'latin':     sample_latin,
    'stratified':sample_stratified,
    'land': sample_land,
}

def get_sampler(name):
    return SAMPLERS[name]

# ─── Argument parsing ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Local MLE ID density analysis for GeoCLIP location encoders."
    )
    p.add_argument('--sampling', nargs='+', choices=list(SAMPLERS.keys()), required=True)
    p.add_argument('--n', type=int, default=100000,
                   help="Number of sample points per scheme")
    p.add_argument('--ks', type=int, nargs='+', default=[5,10,20],
                   help="Neighborhood sizes for local MLE")
    p.add_argument('--output-dir', type=str, default='density_data',
                   help="Directory to save density data")
    return p.parse_args()

# ─── Main ────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    np.random.seed(42)
    torch.manual_seed(42)
    
    os.environ.update({
        "OMP_NUM_THREADS":"72",
        "MKL_NUM_THREADS":"72",
        "OPENBLAS_NUM_THREADS":"72"
    })
    
    wandb.init(
        project="EigenSpectrum Analysis",
        name="local_mle_density_analysis_geoclip"
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # ─── Selected GeoCLIP variants & checkpoints ─────────────────────────────────
    encoders = {
        "geoclip_hierarchy_3":  {
            "sigma_exps": np.linspace(0,8,3).tolist(),
            "ckpt": None  # use from_pretrained
        },
        "geoclip_hierarchy_6":  {
            "sigma_exps": np.linspace(0,8,6).tolist(),
            "ckpt": "checkpoints/hierarchy_6/best_locenc.pt"
        },
        "geoclip_hierarchy_10": {
            "sigma_exps": np.linspace(0,8,10).tolist(),
            "ckpt": "checkpoints/hierarchy_10/best_locenc.pt"
        },
        "geoclip_extended_maxexp_12": {
            "sigma_exps": [0,4,8,12],
            "ckpt": "checkpoints/extended_maxexp_12/best_locenc.pt"
        },
        "geoclip_extended_maxexp_16": {
            "sigma_exps": [0,4,8,12,16],
            "ckpt": "checkpoints/extended_maxexp_16/best_locenc.pt"
        },
    }
    
    for sampling in args.sampling:
        sampler = get_sampler(sampling)
        orig_coords = sampler(args.n)  # (lon,lat)
        model_coords = orig_coords[:, [1,0]].astype(np.float32)  # swap to (lat,lon) for model
        coords_t = torch.from_numpy(model_coords).to(device)
        
        for name, meta in encoders.items():
            print(f"[{sampling}] → {name}")
            sigmas = [2.0**e for e in meta["sigma_exps"]]
            
            # Load model
            if name == "geoclip_hierarchy_3":
                # load standard pretrained GeoCLIP
                loc_enc = LocationEncoder(sigma=sigmas, from_pretrained=True).to(device)
            else:
                # load your fine-tuned checkpoint
                loc_enc = LocationEncoder(sigma=sigmas, from_pretrained=False).to(device)
                state = torch.load(meta["ckpt"], map_location=device)
                loc_enc.load_state_dict(state)
            
            loc_enc.eval()
            
            # Compute embeddings
            with torch.no_grad():
                emb = loc_enc(coords_t).cpu().numpy()
            
            # Compute & cache local IDs
            local_results = {}
            for k in args.ks:
                print(f"   Computing local MLE for k={k}")
                mle = skdim.id.MLE(neighborhood_based=True)
                raw, smooth = mle.fit_transform_pw(
                    emb,
                    n_neighbors=k,
                    n_jobs=-1,
                    smooth=True
                )
                local_results[k] = (raw, smooth)
            
            # Save individual file for this encoder/sampling combination
            # Use consistent naming with SatCLIP script
            individual_file = output_path / f"mle_values_{name}_{sampling}.npz"
            
            # Create a simple dictionary with just the values
            save_dict = {
                'encoder_name': name,
                'sampling_method': sampling,
            }
            
            # Add raw and smooth values for each k
            for k, (raw, smooth) in local_results.items():
                save_dict[f'raw_k{k}'] = raw
                save_dict[f'smooth_k{k}'] = smooth
            
            np.savez_compressed(individual_file, **save_dict)
            print(f"   Saved MLE values to {individual_file}")
            
            # ─── Histogram Density Plots ─────────────────────────────────────────
            print(f"   Creating histogram density plots...")
            
            fig, axes = plt.subplots(
                len(args.ks), 2, 
                figsize=(12, 4*len(args.ks)),
                squeeze=False
            )
            
            for idx, k in enumerate(args.ks):
                raw, smooth = local_results[k]
                
                # Raw density plot
                ax_raw = axes[idx, 0]
                ax_raw.hist(
                    raw, 
                    bins=50, 
                    density=True, 
                    alpha=0.7, 
                    color='steelblue',
                    edgecolor='black',
                    linewidth=0.5
                )
                ax_raw.axvline(
                    np.mean(raw), 
                    color='red', 
                    linestyle='--', 
                    linewidth=2,
                    label=f'Mean: {np.mean(raw):.2f}'
                )
                ax_raw.axvline(
                    np.median(raw), 
                    color='green', 
                    linestyle='--', 
                    linewidth=2,
                    label=f'Median: {np.median(raw):.2f}'
                )
                ax_raw.set_xlabel('Local MLE ID (raw)', fontsize=12)
                ax_raw.set_ylabel('Density', fontsize=12)
                ax_raw.set_title(f'k={k} - Raw Local MLE Distribution', fontsize=14)
                ax_raw.legend()
                ax_raw.grid(True, alpha=0.3)
                
                # Smoothed density plot
                ax_smooth = axes[idx, 1]
                ax_smooth.hist(
                    smooth, 
                    bins=50, 
                    density=True, 
                    alpha=0.7, 
                    color='coral',
                    edgecolor='black',
                    linewidth=0.5
                )
                ax_smooth.axvline(
                    np.mean(smooth), 
                    color='red', 
                    linestyle='--', 
                    linewidth=2,
                    label=f'Mean: {np.mean(smooth):.2f}'
                )
                ax_smooth.axvline(
                    np.median(smooth), 
                    color='green', 
                    linestyle='--', 
                    linewidth=2,
                    label=f'Median: {np.median(smooth):.2f}'
                )
                ax_smooth.set_xlabel('Local MLE ID (smoothed)', fontsize=12)
                ax_smooth.set_ylabel('Density', fontsize=12)
                ax_smooth.set_title(f'k={k} - Smoothed Local MLE Distribution', fontsize=14)
                ax_smooth.legend()
                ax_smooth.grid(True, alpha=0.3)
            
            fig.suptitle(
                f'{name} | {sampling} | Local MLE Density Distributions',
                fontsize=16
            )
            plt.tight_layout()
            
            density_fn = f"density_{name}_{sampling}.png"
            plt.savefig(density_fn, dpi=150, bbox_inches="tight")
            plt.close(fig)
            
            wandb.log({
                f"{name}/{sampling}/density_plots": wandb.Image(density_fn)
            })
            
            # ─── KDE Density Plots ─────────────────────────────────────────────
            print(f"   Creating KDE density plots...")
            
            fig_kde, axes_kde = plt.subplots(
                1, len(args.ks), 
                figsize=(6*len(args.ks), 5),
                squeeze=False
            )
            
            for idx, k in enumerate(args.ks):
                raw, smooth = local_results[k]
                ax = axes_kde[0, idx]
                
                # Calculate KDE
                kde_raw = gaussian_kde(raw)
                kde_smooth = gaussian_kde(smooth)
                
                # Create x-axis values
                x_min = min(raw.min(), smooth.min()) - 0.5
                x_max = max(raw.max(), smooth.max()) + 0.5
                x_vals = np.linspace(x_min, x_max, 200)
                
                # Plot KDEs
                ax.plot(x_vals, kde_raw(x_vals), 
                       label='Raw', color='steelblue', linewidth=2)
                ax.plot(x_vals, kde_smooth(x_vals), 
                       label='Smoothed', color='coral', linewidth=2)
                
                # Add vertical lines for means
                ax.axvline(np.mean(raw), color='steelblue', 
                          linestyle=':', linewidth=1.5, alpha=0.7)
                ax.axvline(np.mean(smooth), color='coral', 
                          linestyle=':', linewidth=1.5, alpha=0.7)
                
                ax.set_xlabel('Local MLE ID', fontsize=12)
                ax.set_ylabel('Density', fontsize=12)
                ax.set_title(f'k={k}', fontsize=14)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add statistics text
                stats_text = (
                    f'Raw: μ={np.mean(raw):.2f}, σ={np.std(raw):.2f}\n'
                    f'Smooth: μ={np.mean(smooth):.2f}, σ={np.std(smooth):.2f}'
                )
                ax.text(0.05, 0.95, stats_text, 
                       transform=ax.transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=10)
            
            fig_kde.suptitle(
                f'{name} | {sampling} | Local MLE KDE Distributions',
                fontsize=16
            )
            plt.tight_layout()
            
            kde_fn = f"kde_{name}_{sampling}.png"
            plt.savefig(kde_fn, dpi=150, bbox_inches="tight")
            plt.close(fig_kde)
            
            wandb.log({
                f"{name}/{sampling}/kde_plots": wandb.Image(kde_fn)
            })
            
            # ─── Log summary statistics ─────────────────────────────────────────
            print(f"   Logging summary statistics...")
            for k in args.ks:
                raw, smooth = local_results[k]
                wandb.log({
                    f"{name}/{sampling}/k{k}/raw_mean": np.mean(raw),
                    f"{name}/{sampling}/k{k}/raw_std": np.std(raw),
                    f"{name}/{sampling}/k{k}/raw_median": np.median(raw),
                    f"{name}/{sampling}/k{k}/raw_min": np.min(raw),
                    f"{name}/{sampling}/k{k}/raw_max": np.max(raw),
                    f"{name}/{sampling}/k{k}/smooth_mean": np.mean(smooth),
                    f"{name}/{sampling}/k{k}/smooth_std": np.std(smooth),
                    f"{name}/{sampling}/k{k}/smooth_median": np.median(smooth),
                    f"{name}/{sampling}/k{k}/smooth_min": np.min(smooth),
                    f"{name}/{sampling}/k{k}/smooth_max": np.max(smooth),
                })
    
    print("\nDone! All MLE values saved to density_data/")

if __name__ == "__main__":
    main()