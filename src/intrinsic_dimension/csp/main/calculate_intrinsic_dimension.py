#!/usr/bin/env python
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from skdim.id import MLE
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from shapely.geometry import Point
from shapely.ops import unary_union
import wandb

from models import LocationEncoder, LocationImageEncoder
from SpatialRelationEncoder import GridCellSpatialRelationEncoder
from module import MultiLayerFeedForwardNN
from losses import rand_samples        # CSP’s built‑in sampler
from utils import get_spa_enc_list
from matplotlib.colors import LogNorm

from scipy.stats import gaussian_kde


def build_spa_enc(params, device):
    ffn = MultiLayerFeedForwardNN(
        input_dim=params['coord_dim'] * params['frequency_num'] * 2,
        output_dim=params['num_filts'],
        num_hidden_layers=params['num_hidden_layer'],
        dropout_rate=params['dropout'],
        hidden_dim=params['hidden_dim'],
        activation=params['spa_f_act'],
        use_layernormalize=params['use_layn'],
        skip_connection=params['skip_connection'],
        context_str='spa_enc'
    )
    if params['spa_enc_type'] == 'gridcell':
        return GridCellSpatialRelationEncoder(
            spa_embed_dim=params['num_filts'],
            coord_dim=params['coord_dim'],
            frequency_num=params['frequency_num'],
            max_radius=params['max_radius'],
            min_radius=params['min_radius'],
            freq_init=params['freq_init'],
            ffn=ffn,
            device=device
        )
    else:
        raise NotImplementedError(f"Unsupported spa_enc_type {params['spa_enc_type']}")

def build_random_location_encoder(params, device='cuda'):
    spa_enc = build_spa_enc(params, device)
    loc_enc = LocationEncoder(
        spa_enc=spa_enc,
        num_inputs=params['coord_dim'],
        num_classes=params['num_classes'],
        num_filts=params['num_filts'],
        num_users=params.get('num_users', 1)
    )
    wrapper = LocationImageEncoder(
        loc_enc=loc_enc,
        train_loss=params['train_loss'],
        unsuper_loss=params['unsuper_loss'],
        cnn_feat_dim=params.get('cnn_feat_dim', 2048),
        spa_enc_type=params['spa_enc_type']
    ).to(device)
    return wrapper.loc_enc.eval()

def load_location_encoder(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    params = ckpt['params']
    params['coord_dim'] = 2
    spa_enc = build_spa_enc(params, device)
    loc_enc = LocationEncoder(
        spa_enc=spa_enc,
        num_inputs=params['coord_dim'],
        num_classes=params['num_classes'],
        num_filts=params['num_filts'],
        num_users=params.get('num_users', 1)
    )
    wrapper = LocationImageEncoder(
        loc_enc=loc_enc,
        train_loss=params['train_loss'],
        unsuper_loss=params['unsuper_loss'],
        cnn_feat_dim=params.get('cnn_feat_dim', 2048),
        spa_enc_type=params['spa_enc_type']
    ).to(device)
    wrapper.load_state_dict(ckpt['state_dict'])
    return wrapper.loc_enc.eval()

def sample_uniform_spherical(n_points: int) -> np.ndarray:
    u = np.random.rand(n_points)
    v = np.random.rand(n_points)
    lon = 2 * np.pi * u - np.pi         # [−π, π]
    lat = np.arcsin(2 * v - 1)          # [−π/2, π/2]
    return np.degrees(np.stack([lon, lat], axis=1))

def sample_uniform_land(n_points: int, batch_size: int = 10_000) -> np.ndarray:
    shp = shapereader.natural_earth('110m', 'physical', 'land')
    geoms = list(shapereader.Reader(shp).geometries())
    land_poly = unary_union(geoms)
    pts = []
    while len(pts) < n_points:
        lon = np.random.uniform(-180, 180, batch_size)
        u   = np.random.uniform(-1, 1, batch_size)
        lat = np.degrees(np.arcsin(u))
        for φ, θ in zip(lon, lat):
            if land_poly.contains(Point(φ, θ)):
                pts.append((φ, θ))
                if len(pts) >= n_points:
                    break
    return np.array(pts)


def compute_embeddings(loc_enc, coords_deg: np.ndarray, device='cuda') -> np.ndarray:
    # GridCellSpatialRelationEncoder expects degrees, so:
    if isinstance(loc_enc.spa_enc, GridCellSpatialRelationEncoder):
        x = torch.from_numpy(coords_deg.astype(np.float32)).to(device)
    else:
        # other spa_enc (e.g. geo_net) expects radians
        coords_rad = np.radians(coords_deg)
        x = torch.from_numpy(coords_rad.astype(np.float32)).to(device)

    with torch.no_grad():
        emb = loc_enc(x, return_feats=True)
    return emb.cpu().numpy()

def plot_local_mle(coords_deg, emb_cpu, ks, output_prefix="csp", title_prefix=""):
    local_results = {}
    for k in ks:
        mle = MLE(neighborhood_based=True)
        raw, smooth = mle.fit_transform_pw(emb_cpu, n_neighbors=k, smooth=True)
        local_results[k] = (raw, smooth)

    # build symmetrical diverging norm
    all_vals = np.hstack([np.hstack((r,s)) for r,s in local_results.values()])
    vmin, vmax = np.percentile(all_vals, [2, 98])
    vcenter    = np.median(all_vals)
    # norm       = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=False)
    views = [
        (  0,   0, "Equatorial"),
        (  0,  90, "North Pole"),
        (  0, -90, "South Pole"),
        (120,  30, "Rotated 120°E")
    ]

    for k, (raw, smooth) in local_results.items():
        for vals, suffix, suf_text in [
            (raw,        "",         " (raw)"),
            (smooth, "_smooth", " (smoothed)")
        ]:
            fig = plt.figure(figsize=(16,16))
            for i, (lon0, lat0, title) in enumerate(views, start=1):
                ax = fig.add_subplot(
                    2, 2, i,
                    projection=ccrs.Orthographic(
                        central_longitude=lon0,
                        central_latitude=lat0
                    )
                )
                ax.set_global()
                ax.set_facecolor("#f0f0f0")
                circ = plt.Circle(
                    (0.5,0.5), 0.495,
                    transform=ax.transAxes,
                    facecolor="#a6cee3", zorder=0
                )
                ax.add_patch(circ)
                ax.coastlines(resolution="110m", color="black", linewidth=1.0, zorder=2)
                ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=3)
                gl = ax.gridlines(
                    draw_labels=False, color="gray", linestyle="--",
                    linewidth=0.5, alpha=0.7
                )
                gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
                gl.ylocator = mticker.FixedLocator(np.arange(-90, 91,30))

                sc = ax.scatter(
                    coords_deg[:,0], coords_deg[:,1],
                    c=vals, cmap="YlOrRd", norm=norm,
                    s=1, alpha=0.8,
                    transform=ccrs.PlateCarree(), zorder=1
                )
                ax.set_title(title, fontsize=14, pad=8)

            cbar = fig.colorbar(sc, orientation="horizontal", fraction=0.05, pad=0.03)
            cbar.set_label("Intrinsic Dimensionality", fontsize=12)
            fig.suptitle(
                f"{title_prefix}local MLE k={k}{suf_text}",
                fontsize=16, y=0.96
            )

            fname = f"{output_prefix}_local_mle{suffix}_k{k}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

            wandb.log({f"{output_prefix}/local_mle{k}{suffix}": wandb.Image(fname)})


            # ─── Density Plots for Local MLE ─────────────────────────────────────────
            print(f"   Creating density plots...")
            
            # Create density plots for each k value
            fig, axes = plt.subplots(
                len(ks), 2, 
                figsize=(12, 4*len(ks)),
                squeeze=False
            )
            
            for idx, k in enumerate(ks):
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
                f'{title_prefix}{output_prefix} | Local MLE Density Distributions',
                fontsize=16
            )
            plt.tight_layout()
            
            density_fn = f"density_{output_prefix}.png"
            plt.savefig(density_fn, dpi=150, bbox_inches="tight")
            plt.close(fig)
            
            wandb.log({
                f"{output_prefix}/density_plots": wandb.Image(density_fn)
            })
            
            # Create smoother density plots using KDE
            
            fig_kde, axes_kde = plt.subplots(
                1, len(ks), 
                figsize=(6*len(ks), 5),
                squeeze=False
            )
            
            for idx, k in enumerate(ks):
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
                f'{title_prefix}{output_prefix} | Local MLE KDE Distributions',
                fontsize=16
            )
            plt.tight_layout()
            
            kde_fn = f"kde_{output_prefix}.png"
            plt.savefig(kde_fn, dpi=150, bbox_inches="tight")
            plt.close(fig_kde)
            
            wandb.log({
                f"{output_prefix}/kde_plots": wandb.Image(kde_fn)
            })
            
            # ─── Log summary statistics ─────────────────────────────────────────────
            for k in ks:
                raw, smooth = local_results[k]
                wandb.log({
                    f"{output_prefix}/k{k}/raw_mean":   np.mean(raw),
                    f"{output_prefix}/k{k}/raw_std":    np.std(raw),
                    f"{output_prefix}/k{k}/raw_median": np.median(raw),
                    f"{output_prefix}/k{k}/raw_min":    np.min(raw),
                    f"{output_prefix}/k{k}/raw_max":    np.max(raw),
                    f"{output_prefix}/k{k}/smooth_mean":   np.mean(smooth),
                    f"{output_prefix}/k{k}/smooth_std":    np.std(smooth),
                    f"{output_prefix}/k{k}/smooth_median": np.median(smooth),
                    f"{output_prefix}/k{k}/smooth_min":    np.min(smooth),
                    f"{output_prefix}/k{k}/smooth_max":    np.max(smooth),
                })

def load_params(checkpoint_path, device='cuda'):
    ckpt   = torch.load(checkpoint_path, map_location=device)
    params = ckpt['params']
    params['coord_dim'] = 2
    return params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',   required=True)
    parser.add_argument('--n_points',     type=int,       default=200_000)
    parser.add_argument('--ks',           nargs='+', type=int, default=[5,10,20])
    parser.add_argument(
        '--sampling',
        nargs='+',
        choices=['uniform','land','pretrain'],
        default=['uniform'],
        help="One or more sampling schemes; e.g. `--sampling uniform land pretrain`"
    )
    parser.add_argument('--device',       default='cuda')
    parser.add_argument(
        '--random_init',
        action='store_true',
        help='Use a fresh, randomly initialized encoder'
    )
    args = parser.parse_args()

    wandb.init(
        project="EigenSpectrum Analysis",
        name   ="mle_global_local_heatmaps_csp_density"
    )

    params = load_params(args.checkpoint, device=args.device)
    if args.random_init:
        print("→ using RANDOMLY INITIALIZED encoder")
        loc_enc = build_random_location_encoder(params, device=args.device)
    else:
        loc_enc = load_location_encoder(args.checkpoint, device=args.device)

    for scheme in args.sampling:
        # sample
        if   scheme == 'uniform':
            coords_deg = sample_uniform_spherical(args.n_points)
        elif scheme == 'land':
            coords_deg = sample_uniform_land(args.n_points)
        else:  # pretrain
            coords_deg = sample_pretraining_distribution(args.n_points, params)

        # embed
        emb = compute_embeddings(loc_enc, coords_deg, device=args.device)

        # global MLE
        mle = MLE(neighborhood_based=True)
        mle.fit(emb, n_neighbors=max(args.ks), n_jobs=-1)
        wandb.log({f"{scheme}/global_MLE": mle.dimension_})

        # local MLE heatmaps
        plot_local_mle(
            coords_deg, emb, args.ks,
            output_prefix=scheme,
            title_prefix=f"{scheme.upper()} | "
        )

if __name__ == '__main__':
    main()
