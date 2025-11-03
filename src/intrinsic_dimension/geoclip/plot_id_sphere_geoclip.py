#!/usr/bin/env python
"""
mle_global_local_heatmaps_geoclip.py
Compute global & local MLE intrinsic-dimension heatmaps for GeoCLIP
location-encoder variants and plot them on globe views.
"""
import os
import argparse
import numpy as np
import torch
import skdim.id
import matplotlib.pyplot as plt
import wandb
from model.location_encoder import LocationEncoder
from cartopy.io import shapereader
from shapely.geometry import Point
from shapely.ops import unary_union
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from cartopy.feature import NaturalEarthFeature

# ─── Sampling functions ─────────────────────────────────────────────────────────
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

def sample_poisson(N, min_dist=5.0, **kwargs):
    pts = []
    while len(pts) < N:
        cand = sample_sphere(1)[0]
        if all(np.hypot(cand[0]-p[0], cand[1]-p[1])>=min_dist for p in pts):
            pts.append(cand)
    return np.array(pts)

def sample_region(N, region_file=None, **kwargs):
    import geopandas as gpd
    from shapely.geometry import Point
    gdf = gpd.read_file(region_file)
    minx, miny, maxx, maxy = gdf.total_bounds
    pts = []
    while len(pts) < N:
        lon = np.random.uniform(minx, maxx)
        lat = np.random.uniform(miny, maxy)
        if gdf.contains(Point(lon, lat)).any():
            pts.append((lon, lat))
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
    'poisson':   sample_poisson,
    'region':    sample_region,
    'land':      sample_land,
}

def get_sampler(name):
    return SAMPLERS[name]

# ─── Argument parsing ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Global & local MLE ID heatmaps for GeoCLIP location encoders."
    )
    p.add_argument('--sampling', nargs='+', choices=list(SAMPLERS.keys()),
                   default=['land'], help="Sampling methods to use")
    p.add_argument('--n', type=int, default=100000,
                   help="Number of sample points per scheme")
    p.add_argument('--ks', type=int, nargs='+', default=[5,10,20],
                   help="Neighborhood sizes for local MLE")
    p.add_argument('--region-file', type=str, default=None,
                   help="GeoJSON/shapefile for 'region' sampler")
    return p.parse_args()

# ─── Main ────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Set thread limits if needed
    os.environ.update({
        "OMP_NUM_THREADS":"72",
        "MKL_NUM_THREADS":"72", 
        "OPENBLAS_NUM_THREADS":"72"
    })
    
    wandb.init(
        project="EigenSpectrum Analysis",
        name="mle_global_local_heatmaps_geoclip_colormap_smooth"
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define GeoCLIP encoder configurations
    encoders = {
        "hierarchy_3": {
            "sigma_exps": np.linspace(0, 8, 3).tolist(),
            "ckpt": None  # use from_pretrained=True
        },
        "hierarchy_6": {
            "sigma_exps": np.linspace(0, 8, 6).tolist(),
            "ckpt": "checkpoints/hierarchy_6/best_locenc.pt"
        },
        "hierarchy_10": {
            "sigma_exps": np.linspace(0, 8, 10).tolist(),
            "ckpt": "checkpoints/hierarchy_10/best_locenc.pt"
        },
        "extended_maxexp_12": {
            "sigma_exps": [0, 4, 8, 12],
            "ckpt": "checkpoints/extended_maxexp_12/best_locenc.pt"
        },
        "extended_maxexp_16": {
            "sigma_exps": [0, 4, 8, 12, 16],
            "ckpt": "checkpoints/extended_maxexp_16/best_locenc.pt"
        }
    }
    
    for sampling in args.sampling:
        sampler = get_sampler(sampling)
        # Sample points in original lon/lat format
        orig_coords = sampler(args.n, region_file=args.region_file)  # (N,2) lon/lat
        
        for name, cfg in encoders.items():
            print(f"[{sampling}] → {name}")
            
            # Load GeoCLIP location encoder
            sigmas = [2.0**e for e in cfg["sigma_exps"]]
            
            if cfg.get("ckpt") is None:
                loc_enc = LocationEncoder(sigma=sigmas, from_pretrained=True)
            else:
                # Load fine-tuned checkpoint
                loc_enc = LocationEncoder(sigma=sigmas, from_pretrained=False)
                state = torch.load(cfg["ckpt"], map_location=device)
                loc_enc.load_state_dict(state)
            
            loc_enc.to(device).eval()
            
            # Convert to model format: swap to (lat, lon) for GeoCLIP
            model_coords = orig_coords[:, [1, 0]].astype(np.float32)
            coords_t = torch.from_numpy(model_coords).to(device)
            
            # Compute embeddings
            with torch.no_grad():
                emb = loc_enc(coords_t).cpu().numpy()
            
            # -- Global MLE --
            mle_glob = skdim.id.MLE(neighborhood_based=True)
            mle_glob.fit(emb, n_neighbors=max(args.ks), n_jobs=-1)
            wandb.log({f"{name}/{sampling}/global_MLE": mle_glob.dimension_})
            
            # -- Local MLE heatmaps for each k --
            for k in args.ks:
                print(f"   k={k}")
                mle = skdim.id.MLE(neighborhood_based=True)
                
                # compute raw + smoothed pointwise estimates
                dim_pw_raw, dim_pw_smooth = mle.fit_transform_pw(
                    emb,
                    n_neighbors=k,
                    n_jobs=-1,
                    smooth=True
                )
                
                # Compute robust color normalization across both raw and smooth
                all_vals = np.hstack([dim_pw_raw, dim_pw_smooth])
                vmin, vmax = np.percentile(all_vals, [2, 98])
                vcenter = np.median(all_vals)
                norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                
                # two passes: raw and smoothed
                for vals, suffix, title_suffix in [
                    (dim_pw_raw,    "",          " (raw)"),
                    (dim_pw_smooth, "_smooth",   " (smoothed)")
                ]:
                    # define the four viewpoints
                    views = [
                        (  0,   0, "Equatorial"    ),
                        (  0,  90, "North Pole"    ),
                        (  0, -90, "South Pole"    ),
                        (120,  30, "Rotated 120°E" )
                    ]
                    
                    # build a 2×2 grid of globe views
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
                        
                        # ocean disk + land background
                        ax.set_facecolor("#f0f0f0")
                        circ = plt.Circle((0.5,0.5), 0.495,
                                        transform=ax.transAxes,
                                        facecolor="#a6cee3",
                                        zorder=0)
                        ax.add_patch(circ)
                        
                        # coastlines + borders
                        ax.coastlines(resolution="110m", color="black",
                                    linewidth=1.0, zorder=2)
                        borders = NaturalEarthFeature(
                            category='cultural',
                            name='admin_0_boundary_lines_land',
                            scale='110m',
                            facecolor='none',
                            edgecolor='black',
                            linewidth=0.5
                        )
                        ax.add_feature(borders, zorder=3)
                        
                        # dashed graticule
                        gl = ax.gridlines(
                            draw_labels=False,
                            color="gray", linestyle="--",
                            linewidth=0.5, alpha=0.7
                        )
                        gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
                        gl.ylocator = mticker.FixedLocator(np.arange(-90, 91,30))
                        
                        # scatter the ID values - use original lon/lat for plotting
                        sc = ax.scatter(
                            orig_coords[:,0], orig_coords[:,1],
                            c=vals,
                            cmap="YlOrRd",
                            norm=norm,
                            s=1,
                            alpha=0.8,
                            transform=ccrs.PlateCarree(),
                            zorder=1
                        )
                        ax.set_title(title, fontsize=14, pad=8)
                    
                    # shared colorbar below all 4 subplots
                    cbar = fig.colorbar(
                        sc,
                        orientation="horizontal",
                        fraction=0.05,
                        pad=0.03
                    )
                    cbar.set_label("Intrinsic Dimensionality", fontsize=12)
                    
                    # overall title, save, and log
                    fig.suptitle(
                        f"{name} | {sampling} | local MLE k={k}{title_suffix}",
                        fontsize=16, y=0.96
                    )
                    fn = f"geo_multi_view_{name}_{sampling}_k{k}{suffix}.png"
                    plt.savefig(fn, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    
                    wandb.log({
                        f"{name}/{sampling}/multi_view_k{k}{suffix}": wandb.Image(fn)
                    })

if __name__ == "__main__":
    main()