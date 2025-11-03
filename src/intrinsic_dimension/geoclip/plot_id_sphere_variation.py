#!/usr/bin/env python
"""
mle_geoclip_heatmaps.py

Compute global & local MLE intrinsic‐dimension heatmaps for selected
GeoCLIP location‐encoder variants:
  • M=3 (standard pretrained via from_pretrained=True)
  • M=6 and M=10 hierarchies (load your checkpoints)
  • extended RFF branches at σ_max = 2**12 and 2**16 (load your checkpoints)
"""
import os
import argparse
import numpy as np
import torch
import skdim.id
import matplotlib.pyplot as plt
import wandb
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from model.location_encoder import LocationEncoder
from cartopy.io import shapereader
from shapely.geometry import Point
from shapely.ops import unary_union
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm


from scipy.stats import gaussian_kde
from scipy import stats
from scipy.spatial.distance import cdist


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
    'naive':  sample_naive,
    'sphere': sample_sphere,
    'land':   sample_land,
}

def get_sampler(name):
    return SAMPLERS[name]

# ─── Arg parsing ────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sampling', nargs='+', choices=list(SAMPLERS.keys()), required=True)
    p.add_argument('--n', type=int, default=100_000)
    p.add_argument('--ks', type=int, nargs='+', default=[5,10,20])
    return p.parse_args()

# ─── Main ────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    np.random.seed(42)
    torch.manual_seed(42)
    wandb.init(
        project="EigenSpectrum Analysis",
        name="mle_global_local_heatmaps_geoclip"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ─── Selected GeoCLIP variants & checkpoints ─────────────────────────────────
    encoders = {
        "hierarchy_3":  {"sigma_exps": np.linspace(0,8,3).tolist(),
                         "ckpt":       None},  # use from_pretrained
        "hierarchy_6":  {"sigma_exps": np.linspace(0,8,6).tolist(),
                         "ckpt":       "checkpoints/hierarchy_6/best_locenc.pt"},
        "hierarchy_10": {"sigma_exps": np.linspace(0,8,10).tolist(),
                         "ckpt":       "checkpoints/hierarchy_10/best_locenc.pt"},
        "extended_maxexp_12": {"sigma_exps": [0,4,8,12],
                               "ckpt":       "checkpoints/extended_maxexp_12/best_locenc.pt"},
        "extended_maxexp_16": {"sigma_exps": [0,4,8,12,16],
                               "ckpt":       "checkpoints/extended_maxexp_16/best_locenc.pt"},
    }
    
    for sampling in args.sampling:
        sampler     = get_sampler(sampling)
        orig_coords = sampler(args.n)                             # (lon,lat) for plotting
        model_coords = orig_coords[:, [1,0]].astype(np.float32)   # swap to (lat,lon)
        coords_t     = torch.from_numpy(model_coords).to(device)
        
        for name, meta in encoders.items():
            print(f"[{sampling}] → {name}")
            sigmas = [2.0**e for e in meta["sigma_exps"]]
            
            if name == "hierarchy_3":
                # load standard pretrained GeoCLIP
                loc_enc = LocationEncoder(sigma=sigmas, from_pretrained=True).to(device)
            else:
                # load your fine-tuned checkpoint
                loc_enc = LocationEncoder(sigma=sigmas, from_pretrained=False).to(device)
                state   = torch.load(meta["ckpt"], map_location=device)
                loc_enc.load_state_dict(state)
            
            loc_enc.eval()
            
            # compute embeddings
            with torch.no_grad():
                emb = loc_enc(coords_t).cpu().numpy()
            
            # global MLE
            mle_glob = skdim.id.MLE(neighborhood_based=True)
            mle_glob.fit(emb, n_neighbors=max(args.ks), n_jobs=-1)
            wandb.log({f"{name}/{sampling}/global_MLE": mle_glob.dimension_})
            
            # local & smoothed MLE
            local_results = {}
            for k in args.ks:
                mle = skdim.id.MLE(neighborhood_based=True)
                raw, smooth = mle.fit_transform_pw(
                    emb, n_neighbors=k, n_jobs=-1, smooth=True
                )
                local_results[k] = (raw, smooth)
            
            # robust color normalization
            all_vals = np.hstack([np.hstack((r, s)) for r, s in local_results.values()])
            vmin, vmax = np.percentile(all_vals, [2, 98])
            vcenter    = np.median(all_vals)
            norm       = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            
            # four globe views
            views = [
                (  0,   0, "Equatorial"   ),
                (  0,  90, "North Pole"   ),
                (  0, -90, "South Pole"   ),
                (120,  30, "Rotated 120°E")
            ]
            
            for k, (raw, smooth) in local_results.items():
                for vals, suffix, label in [(raw,    "",        "raw"),
                                            (smooth, "_smooth", "smoothed")]:
                    fig = plt.figure(figsize=(16,16))
                    for i,(lon0, lat0, title) in enumerate(views, start=1):
                        ax = fig.add_subplot(
                            2, 2, i,
                            projection=ccrs.Orthographic(
                                central_longitude=lon0,
                                central_latitude=lat0
                            )
                        )
                        ax.set_global()
                        ax.set_facecolor("#f0f0f0")
                        ax.add_patch(plt.Circle((0.5,0.5),0.495,
                            transform=ax.transAxes,
                            facecolor="#a6cee3", zorder=0))
                        ax.coastlines("110m", color="black", linewidth=1, zorder=2)
                        ax.add_feature(cfeature.BORDERS, zorder=3)
                        gl = ax.gridlines(draw_labels=False,
                                          color="gray", linestyle="--",
                                          linewidth=0.5, alpha=0.7)
                        gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
                        gl.ylocator = mticker.FixedLocator(np.arange(-90, 91,30))
                        
                        # plot using original lon/lat
                        sc = ax.scatter(
                            orig_coords[:,0], orig_coords[:,1],
                            c=vals, cmap="RdBu_r", norm=norm,
                            s=1, alpha=0.8,
                            transform=ccrs.PlateCarree(), zorder=1
                        )
                        ax.set_title(title, fontsize=14, pad=8)
                    
                    cbar = fig.colorbar(sc, orientation="horizontal",
                                        fraction=0.05, pad=0.03)
                    cbar.set_label("Intrinsic Dimensionality", fontsize=12)
                    fig.suptitle(
                        f"{name} | {sampling} | k={k} ({label})",
                        fontsize=16, y=0.96
                    )
                    out_fn = f"geo_mle_{name}_{sampling}_k{k}{suffix}.png"
                    plt.savefig(out_fn, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    wandb.log({
                        f"{name}/{sampling}/multi_view_k{k}{suffix}":
                        wandb.Image(out_fn)
                    })
            
            # ─── Density Plots for Local MLE ─────────────────────────────────────────
            print(f"   Creating density plots for {name}...")
            
            # Create histogram-based density plots
            fig_hist, axes_hist = plt.subplots(
                len(args.ks), 2, 
                figsize=(12, 4*len(args.ks)),
                squeeze=False
            )
            
            for idx, k in enumerate(args.ks):
                raw, smooth = local_results[k]
                
                # Raw density histogram
                ax_raw = axes_hist[idx, 0]
                n_raw, bins_raw, _ = ax_raw.hist(
                    raw, 
                    bins=50, 
                    density=True, 
                    alpha=0.7, 
                    color='#1f77b4',
                    edgecolor='black',
                    linewidth=0.5,
                    label='Histogram'
                )
                
                # Overlay normal distribution for comparison
                raw_mean, raw_std = np.mean(raw), np.std(raw)
                x_raw = np.linspace(raw.min()-1, raw.max()+1, 100)
                ax_raw.plot(x_raw, stats.norm.pdf(x_raw, raw_mean, raw_std), 
                           'r--', linewidth=2, label='Normal fit')
                
                ax_raw.axvline(raw_mean, color='red', linestyle='-', 
                              linewidth=2, label=f'Mean: {raw_mean:.2f}')
                ax_raw.axvline(np.median(raw), color='green', linestyle='-', 
                              linewidth=2, label=f'Median: {np.median(raw):.2f}')
                
                ax_raw.set_xlabel('Local MLE ID (raw)', fontsize=12)
                ax_raw.set_ylabel('Density', fontsize=12)
                ax_raw.set_title(f'k={k} - Raw Local MLE Distribution', fontsize=14)
                ax_raw.legend(loc='upper right')
                ax_raw.grid(True, alpha=0.3)
                
                # Add statistics box
                stats_text = f'μ={raw_mean:.2f}\nσ={raw_std:.2f}\nskew={stats.skew(raw):.2f}\nkurt={stats.kurtosis(raw):.2f}'
                ax_raw.text(0.02, 0.98, stats_text, transform=ax_raw.transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='white', alpha=0.8), fontsize=10)
                
                # Smoothed density histogram
                ax_smooth = axes_hist[idx, 1]
                n_smooth, bins_smooth, _ = ax_smooth.hist(
                    smooth, 
                    bins=50, 
                    density=True, 
                    alpha=0.7, 
                    color='#ff7f0e',
                    edgecolor='black',
                    linewidth=0.5,
                    label='Histogram'
                )
                
                smooth_mean, smooth_std = np.mean(smooth), np.std(smooth)
                x_smooth = np.linspace(smooth.min()-1, smooth.max()+1, 100)
                ax_smooth.plot(x_smooth, stats.norm.pdf(x_smooth, smooth_mean, smooth_std), 
                              'r--', linewidth=2, label='Normal fit')
                
                ax_smooth.axvline(smooth_mean, color='red', linestyle='-', 
                                 linewidth=2, label=f'Mean: {smooth_mean:.2f}')
                ax_smooth.axvline(np.median(smooth), color='green', linestyle='-', 
                                 linewidth=2, label=f'Median: {np.median(smooth):.2f}')
                
                ax_smooth.set_xlabel('Local MLE ID (smoothed)', fontsize=12)
                ax_smooth.set_ylabel('Density', fontsize=12)
                ax_smooth.set_title(f'k={k} - Smoothed Local MLE Distribution', fontsize=14)
                ax_smooth.legend(loc='upper right')
                ax_smooth.grid(True, alpha=0.3)
                
                # Add statistics box
                stats_text = f'μ={smooth_mean:.2f}\nσ={smooth_std:.2f}\nskew={stats.skew(smooth):.2f}\nkurt={stats.kurtosis(smooth):.2f}'
                ax_smooth.text(0.02, 0.98, stats_text, transform=ax_smooth.transAxes,
                              verticalalignment='top', bbox=dict(boxstyle='round', 
                              facecolor='white', alpha=0.8), fontsize=10)
            
            fig_hist.suptitle(
                f'GeoCLIP {name} | {sampling} | Local MLE Distributions',
                fontsize=16
            )
            plt.tight_layout()
            
            hist_fn = f"geo_density_hist_{name}_{sampling}.png"
            plt.savefig(hist_fn, dpi=150, bbox_inches="tight")
            plt.close(fig_hist)
            
            wandb.log({
                f"{name}/{sampling}/density_histograms": wandb.Image(hist_fn)
            })
            
            # ─── KDE Density Plots (smoother visualization) ──────────────────────────
            fig_kde = plt.figure(figsize=(15, 5))
            
            # Combined KDE plot for all k values
            ax_combined = fig_kde.add_subplot(1, 2, 1)
            colors = plt.cm.viridis(np.linspace(0, 1, len(args.ks)))
            
            for idx, k in enumerate(args.ks):
                raw, smooth = local_results[k]
                
                # Calculate KDE
                kde_raw = gaussian_kde(raw)
                kde_smooth = gaussian_kde(smooth)
                
                # Create x-axis values
                x_min = min([r.min() for r, _ in local_results.values()]) - 1
                x_max = max([r.max() for r, _ in local_results.values()]) + 1
                x_vals = np.linspace(x_min, x_max, 300)
                
                # Plot KDEs
                ax_combined.plot(x_vals, kde_raw(x_vals), 
                                color=colors[idx], linewidth=2, 
                                linestyle='-', label=f'k={k} (raw)')
                ax_combined.plot(x_vals, kde_smooth(x_vals), 
                                color=colors[idx], linewidth=2, 
                                linestyle='--', label=f'k={k} (smooth)')
            
            ax_combined.set_xlabel('Local MLE ID', fontsize=12)
            ax_combined.set_ylabel('Density', fontsize=12)
            ax_combined.set_title('All k-values Comparison', fontsize=14)
            ax_combined.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_combined.grid(True, alpha=0.3)
            
            # Box plots for distribution comparison
            ax_box = fig_kde.add_subplot(1, 2, 2)
            
            # Prepare data for box plots
            raw_data = [local_results[k][0] for k in args.ks]
            smooth_data = [local_results[k][1] for k in args.ks]
            
            positions = np.arange(len(args.ks))
            width = 0.35
            
            bp1 = ax_box.boxplot(raw_data, positions=positions - width/2, 
                                widths=width, patch_artist=True,
                                boxprops=dict(facecolor='lightblue'),
                                medianprops=dict(color='red', linewidth=2),
                                showfliers=False)
            
            bp2 = ax_box.boxplot(smooth_data, positions=positions + width/2, 
                                widths=width, patch_artist=True,
                                boxprops=dict(facecolor='lightcoral'),
                                medianprops=dict(color='red', linewidth=2),
                                showfliers=False)
            
            ax_box.set_xticks(positions)
            ax_box.set_xticklabels([f'k={k}' for k in args.ks])
            ax_box.set_xlabel('Neighborhood size (k)', fontsize=12)
            ax_box.set_ylabel('Local MLE ID', fontsize=12)
            ax_box.set_title('Distribution Comparison', fontsize=14)
            ax_box.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Raw', 'Smoothed'])
            ax_box.grid(True, alpha=0.3, axis='y')
            
            fig_kde.suptitle(
                f'GeoCLIP {name} | {sampling} | KDE Analysis',
                fontsize=16
            )
            plt.tight_layout()
            
            kde_fn = f"geo_kde_{name}_{sampling}.png"
            plt.savefig(kde_fn, dpi=150, bbox_inches="tight")
            plt.close(fig_kde)
            
            wandb.log({
                f"{name}/{sampling}/kde_analysis": wandb.Image(kde_fn)
            })
            
            # ─── Log detailed statistics ─────────────────────────────────────────────
            for k in args.ks:
                raw, smooth = local_results[k]
                
                # Calculate percentiles
                raw_percentiles = np.percentile(raw, [5, 25, 50, 75, 95])
                smooth_percentiles = np.percentile(smooth, [5, 25, 50, 75, 95])
                
                # Log comprehensive statistics
                wandb.log({
                    # Raw statistics
                    f"{name}/{sampling}/k{k}/raw_mean": np.mean(raw),
                    f"{name}/{sampling}/k{k}/raw_std": np.std(raw),
                    f"{name}/{sampling}/k{k}/raw_median": np.median(raw),
                    f"{name}/{sampling}/k{k}/raw_min": np.min(raw),
                    f"{name}/{sampling}/k{k}/raw_max": np.max(raw),
                    f"{name}/{sampling}/k{k}/raw_skewness": stats.skew(raw),
                    f"{name}/{sampling}/k{k}/raw_kurtosis": stats.kurtosis(raw),
                    f"{name}/{sampling}/k{k}/raw_p5": raw_percentiles[0],
                    f"{name}/{sampling}/k{k}/raw_p25": raw_percentiles[1],
                    f"{name}/{sampling}/k{k}/raw_p75": raw_percentiles[3],
                    f"{name}/{sampling}/k{k}/raw_p95": raw_percentiles[4],
                    
                    # Smoothed statistics
                    f"{name}/{sampling}/k{k}/smooth_mean": np.mean(smooth),
                    f"{name}/{sampling}/k{k}/smooth_std": np.std(smooth),
                    f"{name}/{sampling}/k{k}/smooth_median": np.median(smooth),
                    f"{name}/{sampling}/k{k}/smooth_min": np.min(smooth),
                    f"{name}/{sampling}/k{k}/smooth_max": np.max(smooth),
                    f"{name}/{sampling}/k{k}/smooth_skewness": stats.skew(smooth),
                    f"{name}/{sampling}/k{k}/smooth_kurtosis": stats.kurtosis(smooth),
                    f"{name}/{sampling}/k{k}/smooth_p5": smooth_percentiles[0],
                    f"{name}/{sampling}/k{k}/smooth_p25": smooth_percentiles[1],
                    f"{name}/{sampling}/k{k}/smooth_p75": smooth_percentiles[3],
                    f"{name}/{sampling}/k{k}/smooth_p95": smooth_percentiles[4],
                    
                    # Comparison metrics
                    f"{name}/{sampling}/k{k}/smoothing_effect": np.mean(smooth) - np.mean(raw),
                    f"{name}/{sampling}/k{k}/variance_reduction": (np.var(raw) - np.var(smooth)) / np.var(raw),
                })
            
            # ─── Optional: Spatial correlation analysis ──────────────────────────────
            # Analyze if nearby points have similar ID values
            print(f"   Computing spatial autocorrelation...")
            
            # Sample a subset for computational efficiency
            n_sample = min(5000, len(orig_coords))
            sample_idx = np.random.choice(len(orig_coords), n_sample, replace=False)
            sample_coords = orig_coords[sample_idx]
            
            # Compute pairwise distances (in degrees)
            dists = cdist(sample_coords, sample_coords, metric='euclidean')
            
            # Analyze correlation at different distance bins
            dist_bins = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 180)]
            
            for k in args.ks:
                raw, smooth = local_results[k]
                sample_raw = raw[sample_idx]
                
                correlations = []
                for d_min, d_max in dist_bins:
                    mask = (dists > d_min) & (dists <= d_max)
                    if mask.sum() > 100:  # Need enough pairs
                        # Calculate correlation for pairs in this distance bin
                        i_idx, j_idx = np.where(mask)
                        if len(i_idx) > 0:
                            corr = np.corrcoef(sample_raw[i_idx], sample_raw[j_idx])[0, 1]
                            correlations.append(corr)
                            wandb.log({
                                f"{name}/{sampling}/k{k}/spatial_corr_{d_min}_{d_max}deg": corr
                            })
            
            print(f"   Completed analysis for {name}")
    
    print("Done.")

if __name__ == "__main__":
    main()
