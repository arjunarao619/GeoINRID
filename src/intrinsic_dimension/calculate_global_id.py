#!/usr/bin/env python
"""
Master script for computing global intrinsic dimensionality of location encoders
using multiple estimators from scikit-dimension under different sampling schemes.

Encoders: SatCLIP (L=10, L=40), GeoCLIP, CSP (FMoW, iNat)
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from shapely.geometry import Point
from shapely.ops import unary_union
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm

# Import scikit-dimension estimators
import skdim.id
import matplotlib.path as mpath


# For GeoCLIP
sys.path.append(os.path.join(os.path.dirname(__file__), 'geoclip'))
from model.location_encoder import LocationEncoder as GeoCLIPLocationEncoder

# For CSP
sys.path.append(os.path.join(os.path.dirname(__file__), 'csp/main'))
from models import LocationEncoder as CSPLocationEncoder, LocationImageEncoder
from SpatialRelationEncoder import GridCellSpatialRelationEncoder
from module import MultiLayerFeedForwardNN


import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import LogNorm, Normalize
import matplotlib.patches as patches


CONDITIONAL_NUMBER=10
# ─── Configuration ───────────────────────────────────────────────────────────
CHECKPOINT_PATHS = {
    # GeoCLIP path
    "geoclip": "/projects/arra4944/MMLocEnc/intrinsic_dimension/geoclip/model/weights/location_encoder_weights.pth",
    
    # CSP paths
    "csp_fmow": "/projects/arra4944/MMLocEnc/intrinsic_dimension/csp/weights/model_dir/model_fmow/model_fmow_gridcell_0.0010_32_0.1000000_1_512_gelu_UNSUPER-contsoftmax_0.000050_1.000_1_0.100_TMP1.0000_1.0000_1.0000.pth.tar",
    "csp_inat": "/projects/arra4944/MMLocEnc/intrinsic_dimension/csp/weights/model_dir/model_inat_2018/model_inat_2018_gridcell_0.0010_32_0.1000000_1_512_leakyrelu_UNSUPER-contsoftmax_0.000500_1.000_1_1.000_TMP20.0000_1.0000_1.0000.pth.tar",
}

# Define all global ID estimators to use
# Based on scikit-dimension documentation
ESTIMATORS = {
    'MLE':     lambda: skdim.id.MLE(neighborhood_based=True),
    'TwoNN':   lambda: skdim.id.TwoNN(),
    'MOM':     lambda: skdim.id.MOM(),
    'TLE':     lambda: skdim.id.TLE(),
    'FisherS': lambda: skdim.id.FisherS(conditional_number=CONDITIONAL_NUMBER),
    'CorrInt': lambda: skdim.id.CorrInt(),
    'DANCo':   lambda: skdim.id.DANCo(k=10),
    'ESS':     lambda: skdim.id.ESS(),
}



# ─── Sampling functions ─────────────────────────────────────────────────────────
def sample_land(N, **kwargs):
    """Sample uniformly on sphere but keep only land points."""
    shp = shapereader.natural_earth(resolution='110m', category='physical', name='land')
    reader = shapereader.Reader(shp)
    geoms = list(reader.geometries())
    land_poly = unary_union(geoms)
    
    pts = []
    batch = max(N, 10_000)
    while len(pts) < N:
        lons = np.random.uniform(-180, 180, batch)
        u = np.random.uniform(-1, 1, batch)
        lats = np.degrees(np.arcsin(u))
        for lon, lat in zip(lons, lats):
            if land_poly.contains(Point(lon, lat)):
                pts.append((lon, lat))
                if len(pts) == N:
                    break
    return np.array(pts)

def sample_naive(N, **kwargs):
    """Naive uniform sampling in lat/lon rectangle."""
    lats = np.random.uniform(-90, 90, N)
    lons = np.random.uniform(-180, 180, N)
    return np.stack([lons, lats], axis=1)

def sample_sphere(N, **kwargs):
    """Uniform sampling on sphere surface."""
    u = np.random.uniform(-1, 1, N)
    lats = np.degrees(np.arcsin(u))
    lons = np.random.uniform(-180, 180, N)
    return np.stack([lons, lats], axis=1)

def sample_grid(N, lat_step=1.0, lon_step=1.0, **kwargs):
    """Regular grid sampling."""
    lat_vals = np.arange(-90, 90 + lat_step, lat_step)
    lon_vals = np.arange(-180, 180 + lon_step, lon_step)
    grid = np.stack(np.meshgrid(lon_vals, lat_vals), axis=-1).reshape(-1, 2)
    if len(grid) >= N:
        idx = np.random.choice(len(grid), N, replace=False)
        return grid[idx]
    extra = N - len(grid)
    pad = sample_naive(extra)
    return np.vstack([grid, pad])

def sample_healpix(N, **kwargs):
    """Equal-area tessellation via HEALPix."""
    import healpy as hp
    nside = 1
    while hp.nside2npix(nside) < N:
        nside *= 2
    npix = hp.nside2npix(nside)
    sel = np.random.choice(npix, size=N, replace=False)
    theta, phi = hp.pix2ang(nside, sel)
    lats = 90 - np.degrees(theta)
    lons = np.degrees(phi) - 180
    return np.stack([lons, lats], axis=1)

def sample_fibonacci(N, **kwargs):
    """Fibonacci spiral sampling."""
    i = np.arange(N)
    phi = np.pi * (3. - np.sqrt(5.))
    lats = np.degrees(np.arcsin(2*(i / N) - 1))
    lons = np.degrees((i * phi) % (2*np.pi) - np.pi)
    return np.stack([lons, lats], axis=1)

def sample_sobol(N, **kwargs):
    """Sobol sequence sampling."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=2, scramble=True)
    m = int(np.ceil(np.log2(N)))
    u = sampler.random_base2(m)[:N]
    lons = u[:,0]*360. - 180.
    lats = np.degrees(np.arcsin(2*u[:,1] - 1))
    return np.stack([lons, lats], axis=1)

def sample_latin(N, **kwargs):
    """Latin hypercube sampling."""
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=2)
    u = sampler.random(n=N)
    lons = u[:,0]*360. - 180.
    lats = np.degrees(np.arcsin(2*u[:,1] - 1))
    return np.stack([lons, lats], axis=1)

def sample_stratified(N, bands=10, **kwargs):
    """Stratified sampling by latitude bands."""
    per_band = N // bands
    lats_list, lons_list = [], []
    for i in range(bands):
        lat_min = -90 + i*(180/bands)
        lat_max = lat_min + (180/bands)
        lats_list.append(np.random.uniform(lat_min, lat_max, per_band))
        lons_list.append(np.random.uniform(-180, 180, per_band))
    lats = np.concatenate(lats_list)
    lons = np.concatenate(lons_list)
    if len(lats) < N:
        extra = N - len(lats)
        pad = sample_naive(extra)
        return np.vstack([np.stack([lons, lats], axis=1), pad])
    return np.stack([lons, lats], axis=1)

def sample_poisson(N, min_dist=5.0, **kwargs):
    """Poisson disk sampling."""
    pts = []
    while len(pts) < N:
        cand = sample_sphere(1)[0]
        if all(np.hypot(cand[0]-p[0], cand[1]-p[1]) >= min_dist for p in pts):
            pts.append(cand)
    return np.array(pts)

def sample_region(N, region_file=None, **kwargs):
    """Sample within a specific geographic region."""
    import geopandas as gpd
    gdf = gpd.read_file(region_file)
    minx, miny, maxx, maxy = gdf.total_bounds
    pts = []
    while len(pts) < N:
        lon = np.random.uniform(minx, maxx)
        lat = np.random.uniform(miny, maxy)
        if gdf.contains(Point(lon, lat)).any():
            pts.append([lon, lat])
    return np.array(pts)

def get_sampler(name):
    """Get sampling function by name."""
    return {
        'naive': sample_naive,
        'sphere': sample_sphere,
        'grid': sample_grid,
        'healpix': sample_healpix,
        'fibonacci': sample_fibonacci,
        'sobol': sample_sobol,
        'latin': sample_latin,
        'stratified': sample_stratified,
        'poisson': sample_poisson,
        'region': sample_region,
        'land': sample_land,
    }[name]


# ─── Model loading functions ─────────────────────────────────────────────────
def load_geoclip(device):
    """Load pre-trained GeoCLIP location encoder."""
    print("Loading GeoCLIP...")
    # Use the standard pretrained weights
    loc_enc = GeoCLIPLocationEncoder(
        sigma=[2**0, 2**4, 2**8],  # Standard 3-hierarchy
        from_pretrained=True
    ).to(device).eval()
    return loc_enc

def build_csp_spa_enc(params, device):
    """Build CSP spatial encoder."""
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

def load_csp(checkpoint_path, device):
    """Load CSP location encoder."""
    print(f"Loading CSP from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    params = ckpt['params']
    params['coord_dim'] = 2
    
    spa_enc = build_csp_spa_enc(params, device)
    loc_enc = CSPLocationEncoder(
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


def compute_embeddings(model, coords, model_type, device):
    """Compute embeddings for given coordinates."""
    with torch.no_grad():
        if model_type == "geoclip":
            # GeoCLIP expects (lat, lon) order
            coords_swapped = coords[:, [1, 0]].astype(np.float32)
            coords_t = torch.from_numpy(coords_swapped).to(device)
            emb = model(coords_t)
            
        elif model_type == "csp":
            # CSP expects degrees directly
            coords_t = torch.from_numpy(coords.astype(np.float32)).to(device)
            emb = model(coords_t, return_feats=True)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    return emb.cpu().numpy().astype(np.float64)

import inspect
def _accepts(param, fn) -> bool:
    try:
        return param in inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return False

def compute_all_id_estimates(embeddings, encoder_name, ks):
    """
    Run global ID for all estimators in ESTIMATORS.
    If estimator.fit accepts n_neighbors, run for each k in ks -> keys like 'MLE_k20'.
    Otherwise run once -> key like 'TwoNN'.
    """
    results = {}
    n = embeddings.shape[0]

    for est_name, est_fn in ESTIMATORS.items():
        try:
            est = est_fn()
            fit_fn = est.fit
            per_k = _accepts("n_neighbors", fit_fn)

            if per_k:
                for k in ks:
                    k_eff = min(k, max(n - 1, 1))
                    if k_eff < 2:
                        print(f"    Computing {est_name} (k={k})... -> skipped (n={n} too small)")
                        results[f"{est_name}_k{k}"] = np.nan
                        continue

                    print(f"    Computing {est_name} (k={k_eff})...")
                    est = est_fn()  # fresh instance
                    kwargs = {"n_neighbors": k_eff}

                    # only force comb for MLE; others keep defaults
                    if est_name.upper() == "MLE":
                        kwargs["comb"] = "mle"

                    # optional: parallelize when available
                    if _accepts("n_jobs", est.fit):
                        kwargs["n_jobs"] = -1

                    est.fit(embeddings, **kwargs)
                    id_val = float(getattr(est, "dimension_", np.nan))
                    results[f"{est_name}_k{k}"] = id_val
                    print(f"      → {id_val:.4f}")

            else:
                print(f"    Computing {est_name} (no k)...")
                kwargs = {}
                if _accepts("n_jobs", fit_fn):
                    kwargs["n_jobs"] = -1
                est.fit(embeddings, **kwargs)
                id_val = float(getattr(est, "dimension_", np.nan))
                results[est_name] = id_val
                print(f"      → {id_val:.4f}")

        except Exception as e:
            print(f"      → Failed for {est_name}: {e}")
            if 'per_k' in locals() and per_k:
                for k in ks:
                    results.setdefault(f"{est_name}_k{k}", np.nan)
            else:
                results.setdefault(est_name, np.nan)

    return results


# ─── Visualization functions ─────────────────────────────────────────────────
def plot_sampling_distribution(coords, sampling_name, n_subsample=10000):
    """Create globe views of the sampling distribution."""
    # Subsample for better visualization
    if len(coords) > n_subsample:
        idx = np.random.choice(len(coords), n_subsample, replace=False)
        coords_plot = coords[idx]
    else:
        coords_plot = coords
    
    lons = coords_plot[:, 0]
    lats = coords_plot[:, 1]
    
    views = [
        (0, 0, "Equatorial"),
        (0, 90, "North Pole"),
        (0, -90, "South Pole"),
        (120, 30, "Rotated 120°E"),
    ]
    
    fig = plt.figure(figsize=(12, 12))  # Smaller figure
    plt.subplots_adjust(
        wspace=0.05, hspace=0.15,
        left=0.05, right=0.95,
        top=0.95, bottom=0.05
    )
    
    for i, (lon0, lat0, title) in enumerate(views, start=1):
        ax = fig.add_subplot(
            2, 2, i,
            projection=ccrs.Orthographic(
                central_longitude=lon0,
                central_latitude=lat0
            )
        )
        ax.set_global()
        
        # Lighter background colors
        ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", zorder=1)
        ax.coastlines(resolution='110m', linewidth=0.5, color='gray', zorder=2)
        
        # Plot with larger points and better visibility
        ax.scatter(
            lons, lats,
            transform=ccrs.PlateCarree(),
            s=7,              # Larger points
            c='crimson',      # Darker color for contrast
            alpha=0.6,
            marker='*',
            linewidths=0,
            zorder=3
        )
        ax.set_title(title, pad=5, fontsize=12)
    
    fig.suptitle(f'Sampling Distribution: {sampling_name} (showing {len(coords_plot):,} of {len(coords):,} points)', 
                 fontsize=14, y=0.98)
    
    return fig


def plot_local_id_2d_minimal(coords, local_ids, estimator_name, encoder_name,
                             sampling_name, k, figsize=(10, 5)):
    """
    Create a minimalist 2D world map with PlateCarree projection and transparent background.
    """
    
    # Clean data
    mask = np.isfinite(local_ids)
    coords_clean = coords[mask]
    ids_clean = local_ids[mask]
    
    if len(ids_clean) == 0:
        print(f"Warning: No valid local IDs for {encoder_name}")
        return None
    
    # Color scale - using percentiles for consistency
    vmin, vmax = np.percentile(ids_clean, [5, 95])
    if vmin <= 0:
        vmin = 0.1
    if vmax <= vmin:
        vmax = vmin * 1.1
    
    # Create figure with transparent background
    fig = plt.figure(figsize=figsize, facecolor='none')
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), facecolor='none')
    
    # Set global extent
    ax.set_global()
    
    # Add minimal features - coastlines and country borders only
    ax.coastlines(resolution='110m', color='black', linewidth=0.8, alpha=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray', alpha=0.5)
    
    # Remove outline - try different methods for different cartopy versions
    try:
        ax.outline_patch.set_visible(False)
    except AttributeError:
        # For newer versions of cartopy
        try:
            ax.spines['geo'].set_visible(False)
        except:
            # If that doesn't work either, just continue
            pass
    
    # Determine point size based on data density
    # Aim for even visual coverage
    n_points = len(coords_clean)
    if n_points >= 100000:
        point_size = 2
    elif n_points > 50000:
        point_size = 0.2
    elif n_points > 10000:
        point_size = 0.5
    else:
        point_size = 1.0
    
    # Plot points
    sc = ax.scatter(
        coords_clean[:, 0], coords_clean[:, 1],
        c=ids_clean, cmap='OrRd',
        s=point_size,
        alpha=0.7,
        transform=ccrs.PlateCarree(),
        vmin=vmin, vmax=vmax,
        rasterized=True,  # Better for PDFs
        edgecolors='none'
    )
    
    # Minimal colorbar
    cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', 
                       pad=0.02, shrink=0.6, aspect=40)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('Local Intrinsic Dimension', fontsize=10)
    
    # Format colorbar ticks
    if vmax > 100:
        cbar.formatter = mticker.FormatStrFormatter('%.0f')
    else:
        cbar.formatter = mticker.FormatStrFormatter('%.1f')
    cbar.update_ticks()
    
    # Tight layout to minimize whitespace
    plt.tight_layout(pad=0.2)
    
    # Save with transparent background
    filename = f"local_id_2d_{sampling_name}_{encoder_name}_{estimator_name}_k{k}.png"
    fig.savefig(filename, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    
    return filename


import matplotlib.path as mpath
import matplotlib.ticker as mticker
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_local_id_orthographic(
    coords, local_ids, estimator_name, encoder_name,
    sampling_name, k, smooth=False, shapefile_path=None
):

    # Try to load shapefile if provided
    gdf = None
    if shapefile_path is not None:
        try:
            import geopandas as gpd
            gdf = gpd.read_file(shapefile_path)
            if gdf.crs and gdf.crs.to_string().lower() not in ("epsg:4326", "wgs84", "ogc:crs84"):
                gdf = gdf.to_crs("EPSG:4326")
            print(f"      Loaded shapefile with {len(gdf)} features")
        except Exception as e:
            print(f"      Warning: Could not load shapefile: {e}")

    # extent overlays if no shapefile
    regional_extents = {
        'france':      {'lon_min': -5.0,  'lon_max': 10.0,  'lat_min': 41.0,  'lat_max': 52.0},
        'mountainwest':{'lon_min': -115.0,'lon_max': -102.0,'lat_min': 31.0,  'lat_max': 49.0},
        'denver':      {'lon_min': -106.5,'lon_max': -103.5,'lat_min': 38.5,  'lat_max': 40.5},
    }
    extent_overlay = None
    enc_low = encoder_name.lower()
    if gdf is None:
        if 'france' in enc_low:
            extent_overlay = regional_extents['france']
        elif 'mountain' in enc_low:
            extent_overlay = regional_extents['mountainwest']
        elif 'denver' in enc_low:
            extent_overlay = regional_extents['denver']

    # view set includes USA/France centers
    views = [
        (-90,   0,  "Americas"),
        (-98,  39,  "United States"),
        (2,    46,  "France"),
        (10,   45,  "Europe"),
        (20,    0,  "Africa"),
        (100,  50,  "Asia"),
        (135, -25,  "Oceania"),
        (0,    90,  "North Pole"),
        (0,   -90,  "South Pole"),
    ]

    finite_ids = local_ids[np.isfinite(local_ids)]
    if finite_ids.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = np.percentile(finite_ids, [5, 95])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = float(np.nanmin(finite_ids)), float(np.nanmax(finite_ids))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = 0.0, 1.0

    figures = []

    for lon0, lat0, region_name in views:
        fig = plt.figure(figsize=(6.0, 7.0))
        ax = fig.add_axes([0.16, 0.24, 0.68, 0.68],
                          projection=ccrs.Orthographic(central_longitude=lon0,
                                                       central_latitude=lat0))
        ax.set_global()

        # circular globe boundary
        theta = np.linspace(0, 2*np.pi, 256)
        center, radius = np.array([0.5, 0.5]), 0.5
        circle_verts = np.column_stack([np.cos(theta), np.sin(theta)]) * radius + center
        ax.set_boundary(mpath.Path(circle_verts), transform=ax.transAxes)

        # styling: white ocean
        ax.set_facecolor("#ffffff")
        ax.add_feature(cfeature.LAND,  facecolor="#fafafa", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=0)
        ax.coastlines(resolution="50m", color="black", linewidth=0.6, zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray", zorder=3)

        gl = ax.gridlines(draw_labels=False, color="gray", linestyle=":",
                          linewidth=0.4, alpha=0.5, zorder=1)
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
        gl.ylocator = mticker.FixedLocator(np.arange(-90,  91, 30))

        # scatter FIRST (under overlay)
        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        c=local_ids, cmap="OrRd",
                        vmin=vmin, vmax=vmax,
                        s=1, alpha=1,
                        transform=ccrs.PlateCarree(),
                        zorder=2, rasterized=True, edgecolors="none")

        # overlay training region (shapefile) on TOP
        gdf_to_plot, shp_suffix = None, ""
        if gdf is not None:
            try:
                country_attrs = ['NAME', 'NAME_EN', 'ADMIN', 'SOVEREIGNT', 'NAME_LONG', 'GEOUNIT']
                country_col = next((c for c in country_attrs if c in gdf.columns), None)
                gdf_to_plot = gdf

                if country_col:
                    if region_name == "United States":
                        usa_names = ['United States', 'United States of America', 'USA']
                        gdf_to_plot = gdf[gdf[country_col].isin(usa_names)]
                        if gdf_to_plot.empty:
                            gdf_to_plot = gdf[gdf[country_col].str.contains('United States', case=False, na=False)]
                        shp_suffix = " (USA training region)"
                    elif region_name == "France":
                        france_names = ['France', 'République française']
                        gdf_to_plot = gdf[gdf[country_col].isin(france_names)]
                        if gdf_to_plot.empty:
                            gdf_to_plot = gdf[gdf[country_col].str.contains('France', case=False, na=False)]
                        shp_suffix = " (France training region)"

                if gdf_to_plot is not None and not gdf_to_plot.empty:
                    ax.add_geometries(
                        gdf_to_plot.geometry, crs=ccrs.PlateCarree(),
                        facecolor="none",
                        edgecolor=(0.0, 0.5, 0.0, 0.8),
                        linewidth=1.6,
                        hatch='///', zorder=10
                    )
                    if not shp_suffix:
                        shp_suffix = " (training region)"
            except Exception as e:
                print(f"      Warning: could not draw shapefile: {e}")

        # extent overlay if no shapefile
        if gdf is None and extent_overlay is not None:
            try:
                import matplotlib.patches as mpatches
                lon_min = extent_overlay['lon_min']; lon_max = extent_overlay['lon_max']
                lat_min = extent_overlay['lat_min']; lat_max = extent_overlay['lat_max']
                rect_verts = [[lon_min, lat_min],[lon_max, lat_min],[lon_max, lat_max],[lon_min, lat_max],[lon_min, lat_min]]
                rect_patch = mpatches.Polygon(
                    rect_verts, transform=ccrs.PlateCarree(),
                    facecolor="none",
                    edgecolor=(0.0, 0.5, 0.0, 0.8),
                    linewidth=1.6, hatch='///', zorder=10
                )
                ax.add_patch(rect_patch)
                shp_suffix = " (training extent)"
            except Exception as e:
                print(f"      Warning: could not draw extent overlay: {e}")

        ax.set_title(region_name + shp_suffix, fontsize=16, fontweight="bold", pad=10)

        # horizontal colorbar
        cax = fig.add_axes([0.18, 0.08, 0.64, 0.035])
        cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=11)
        cbar.formatter = (mticker.FormatStrFormatter('%.0f') if vmax > 100
                          else mticker.FormatStrFormatter('%.1f'))
        cbar.update_ticks()
        cbar.set_label("Local Intrinsic Dimension", fontsize=12, labelpad=4)

        smooth_suffix = "_smooth" if smooth else "_raw"
        coverage_suffix = "_wshp" if (gdf is not None or extent_overlay is not None) else ""
        filename = (f"local_id_{sampling_name}_{encoder_name}_{estimator_name}_k{k}_"
                    f"{region_name.lower().replace(' ', '_')}{smooth_suffix}{coverage_suffix}.png")
        fig.savefig(filename, dpi=300, facecolor="white")
        try:
            wandb.log({
                f"{sampling_name}/{encoder_name}/local_{estimator_name}_k{k}/"
                f"{region_name}{smooth_suffix}{coverage_suffix}": wandb.Image(filename)
            })
        except Exception:
            pass
        figures.append((fig, filename))
        plt.close(fig)

    # optional composite (keep if you want a grid panel)
    try:
        create_composite_orthographic(coords, local_ids, estimator_name, encoder_name,
                                      sampling_name, k, [(lon,lat,name) for lon,lat,name in views])
    except Exception:
        pass

    return figures



def compute_local_id_all_views(embeddings, coords, encoder_name, sampling_name, k=10):
    """
    Compute local ID and create all visualizations with separate figures.
    """
    local_results = {}
    
    # Define orthographic views
    ortho_views = [
        (0, 0, "Equatorial"),
        (0, 90, "North Pole"),
        (0, -90, "South Pole"),
        (-90, 30, "Americas"),
        (10, 50, "Europe"),
        (20, 0, "Africa"),
        (100, 30, "Asia"),
        (135, -25, "Oceania"),
        (-60, -20, "South America"),
    ]
    
    # Local estimators
    local_estimators = {
        'MLE': lambda: skdim.id.MLE(neighborhood_based=True),
        # Add other estimators as needed
    }
    
    for estimator_name, estimator_fn in local_estimators.items():
        try:
            print(f"    Computing local {estimator_name} (k={k})...")
            estimator = estimator_fn()
            
            if hasattr(estimator, 'fit_transform_pw'):
                local_ids = estimator.fit_transform_pw(
                    embeddings, 
                    n_neighbors=k, 
                    n_jobs=-1
                )
                
                if isinstance(local_ids, tuple):
                    local_ids = local_ids[0]
                
                local_results[estimator_name] = local_ids
                
                # Create 2D PlateCarree map
                print("      Creating 2D map...")
                map_2d_file = plot_local_id_2d_minimal(
                    coords, local_ids, estimator_name, 
                    encoder_name, sampling_name, k
                )
                
                if map_2d_file:
                    wandb.log({
                        f"{sampling_name}/{encoder_name}/local_{estimator_name}_2d_k{k}": 
                        wandb.Image(map_2d_file)
                    })
                
                # Create individual orthographic views
                print("      Creating orthographic views...")
                for view_params in ortho_views:
                    ortho_file = plot_local_id_orthographic_single(
                        coords, local_ids, estimator_name, 
                        encoder_name, sampling_name, k, view_params
                    )
                    
                    if ortho_file:
                        region_name = view_params[2].lower().replace(' ', '_')
                        wandb.log({
                            f"{sampling_name}/{encoder_name}/local_{estimator_name}_ortho_{region_name}_k{k}": 
                            wandb.Image(ortho_file)
                        })
                
                print(f"      → Mean: {np.nanmean(local_ids):.4f}, "
                     f"Std: {np.nanstd(local_ids):.4f}, "
                     f"Valid points: {np.sum(np.isfinite(local_ids))}")
                     
        except Exception as e:
            print(f"      → Failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return local_results

def compute_local_id_estimates(embeddings, coords, encoder_name, sampling_name, k=10, shapefile_path=None):
    """
    Compute local ID (per-point) and produce orthographic overlays + 2D transparent map,
    matching the SatCLIP script's visuals.
    """
    local_results = {}

    LOCAL_ESTIMATORS = {
        'MLE': lambda: skdim.id.MLE(neighborhood_based=True),
        # add others if they support fit_transform_pw
        # 'MOM': lambda: skdim.id.MOM(),
        # 'TLE': lambda: skdim.id.TLE(),
        # 'FisherS': lambda: skdim.id.FisherS(conditional_number=CONDITIONAL_NUMBER),
    }

    for estimator_name, estimator_fn in LOCAL_ESTIMATORS.items():
        try:
            print(f"    Computing local {estimator_name} (k={k})...")
            est = estimator_fn()

            if hasattr(est, 'fit_transform_pw'):
                out = est.fit_transform_pw(embeddings, n_neighbors=k, n_jobs=-1)
                local_ids = out[0] if isinstance(out, tuple) else out

                local_results[estimator_name] = local_ids

                # Orthographic views with overlay
                _ = plot_local_id_orthographic(
                    coords, local_ids, estimator_name, encoder_name,
                    sampling_name, k, smooth=False, shapefile_path=shapefile_path
                )

                # Minimal 2D world map (transparent background)
                map_2d = plot_local_id_2d_minimal(
                    coords, local_ids, estimator_name,
                    encoder_name, sampling_name, k
                )
                if map_2d:
                    wandb.log({f"{sampling_name}/{encoder_name}/local_{estimator_name}_2d_k{k}": wandb.Image(map_2d)})

                print(f"      → Mean: {np.nanmean(local_ids):.4f}, Std: {np.nanstd(local_ids):.4f}")

        except Exception as e:
            print(f"      → Failed: {e}")
            import traceback; traceback.print_exc()

    return local_results

def create_id_comparison_plot(all_results, sampling_name):
    """Create a comparison plot of ID estimates across encoders and methods."""
    encoders = list(all_results.keys())
    # union of metric names present (e.g., MLE_k5, MOM_k10, TwoNN, ...)
    estimator_labels = sorted({k for enc in encoders for k in all_results[enc].keys()})

    # Build matrix
    id_matrix = np.array([
        [all_results[enc].get(label, np.nan) for label in estimator_labels]
        for enc in encoders
    ])

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(encoders))
    width = 0.8 / max(1, len(estimator_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(estimator_labels)))

    for j, (label, color) in enumerate(zip(estimator_labels, colors)):
        offset = (j - len(estimator_labels)/2 + 0.5) * width
        values = id_matrix[:, j]
        ax.bar(x + offset, values, width, label=label, color=color)

    ax.set_xlabel('Encoder', fontsize=14)
    ax.set_ylabel('Intrinsic Dimension', fontsize=14)
    ax.set_title(f'ID Estimates - {sampling_name} sampling', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(encoders, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig

def create_composite_orthographic(coords, local_ids, estimator_name, encoder_name,
                                 sampling_name, k, views):
    """Create a composite figure with all orthographic views."""
    fig = plt.figure(figsize=(18, 18))
    fig.suptitle(f"{encoder_name} | {estimator_name} | {sampling_name} | k={k}",
                 fontsize=16, fontweight="bold", y=0.98)
    
    # Get consistent color scale
    mask = np.isfinite(local_ids)
    ids_clean = local_ids[mask]
    vmin, vmax = np.percentile(ids_clean, [5, 95])
    if vmin <= 0:
        vmin = 0.1
    if vmax <= vmin:
        vmax = vmin * 1.1
    
    for idx, (lon0, lat0, region_name) in enumerate(views):
        ax = fig.add_subplot(3, 3, idx + 1, 
                            projection=ccrs.Orthographic(central_longitude=lon0,
                                                        central_latitude=lat0))
        ax.set_global()
        
        # Circular boundary
        theta = np.linspace(0, 2*np.pi, 256)
        center, radius = np.array([0.5, 0.5]), 0.5
        verts = np.c_[np.cos(theta), np.sin(theta)] * radius + center
        path = mpath.Path(verts)
        ax.set_boundary(path, transform=ax.transAxes)
        
        # Map features
        ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff", zorder=0)
        ax.coastlines(resolution="110m", color="black", linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray", zorder=2)
        
        # Plot points
        coords_clean = coords[mask]
        sc = ax.scatter(
            coords_clean[:, 0], coords_clean[:, 1],
            c=ids_clean, cmap='OrRd',
            s=0.5, alpha=0.8,
            transform=ccrs.PlateCarree(),
            vmin=vmin, vmax=vmax,
            zorder=3, rasterized=True
        )
        
        ax.set_title(region_name, fontsize=12, pad=5)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Local Intrinsic Dimension", fontsize=12)
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    filename = f"local_id_composite_{sampling_name}_{encoder_name}_{estimator_name}_k{k}.png"
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    wandb.log({
        f"{sampling_name}/{encoder_name}/local_{estimator_name}_composite_k{k}": 
        wandb.Image(filename)
    })

def create_eigenspectrum_plot(eigen_spectra, sampling_name):
    """Create combined eigenspectrum plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(eigen_spectra)))
    
    for (name, spectrum), color in zip(eigen_spectra.items(), colors):
        x = np.arange(1, len(spectrum) + 1)
        ax.plot(x, spectrum, marker='o', markersize=4, 
                label=name, color=color, linewidth=2)
    
    ax.set_xlim(1, 50)
    ax.set_xlabel('Principal Component Index', fontsize=12)
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_title(f'Eigenspectrum - {sampling_name} sampling', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale often better for eigenspectra
    
    plt.tight_layout()
    return fig


# ─── Argument parsing ────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Compute global intrinsic dimensionality of location encoders."
    )
    p.add_argument('--sampling',
                   choices=['naive', 'sphere', 'grid', 'healpix', 'fibonacci',
                            'sobol', 'latin', 'stratified', 'poisson', 'region', 'land'],
                   nargs='+',
                   required=True,
                   help="One or more sampling schemes (space-separated)")
    p.add_argument('--n', type=int, default=100000)
    p.add_argument('--ks', type=int, nargs='+', default=[10, 20, 100],
                help="Neighborhood sizes for global/local ID (space-separated)")
    p.add_argument('--lat-step', type=float, default=1.0)
    p.add_argument('--lon-step', type=float, default=1.0)
    p.add_argument('--bands', type=int, default=10)
    p.add_argument('--min-dist', type=float, default=5.0)
    p.add_argument('--region-file', type=str, default=None)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--shapefile-path', type=str, default=None,
                help="Path to shapefile (.shp) to overlay training region boundaries on plots")
    return p.parse_args()


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    
    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Update CPU worker count! We use scikit-dimension which allows for parallelized compute of ID measures. 
    os.environ.update({
        "OMP_NUM_THREADS": "72",
        "MKL_NUM_THREADS": "72",
        "OPENBLAS_NUM_THREADS": "72"
    })
    
    # Initialize wandb
    sampling_str = "-".join(args.sampling)
    wandb.init(
        project="Ablations-Appendix",
        name=f"global-local-id-{sampling_str}-k{args.ks}_cond{CONDITIONAL_NUMBER}"
    )
    
    # Define encoders to evaluate (no SatCLIP)
    encoders = {
        "GeoCLIP": {
            "type": "geoclip",
        },
        "CSP-FMoW": {
            "type": "csp",
            "path": CHECKPOINT_PATHS["csp_fmow"],
        },
        "CSP-iNat": {
            "type": "csp",
            "path": CHECKPOINT_PATHS["csp_inat"],
        },
    }
    
    # Process each sampling scheme
    for sampling_name in args.sampling:
        print(f"\n{'='*80}")
        print(f"PROCESSING SAMPLING SCHEME: {sampling_name}")
        print('='*80)
        
        # Sample coordinates
        print(f"\nSampling {args.n} points using '{sampling_name}' scheme...")
        sampler = get_sampler(sampling_name)
        coords_np = sampler(
            args.n,
            lat_step=args.lat_step,
            lon_step=args.lon_step,
            bands=args.bands,
            min_dist=args.min_dist,
            region_file=args.region_file,
        )
        
        # Visualize sampling
        fig_sampling = plot_sampling_distribution(coords_np, sampling_name)
        sampling_fn = f"sampling_{sampling_name}_globe_views.png"
        fig_sampling.savefig(sampling_fn, dpi=300, bbox_inches='tight')
        plt.close(fig_sampling)
        wandb.log({f"{sampling_name}_sampling_visualization": wandb.Image(sampling_fn)})
        
        # Store all results for this sampling scheme
        all_results = {}
        eigen_spectra = {}
        
        # Process each encoder
        for name, meta in encoders.items():
            print(f"\n{'-'*60}")
            print(f"Processing {name} ({meta['type']})")
            print('-'*60)
            
            try:
                # Load model
                if meta["type"] == "geoclip":
                    model = load_geoclip(args.device)
                elif meta["type"] == "csp":
                    model = load_csp(meta["path"], args.device)
                else:
                    raise ValueError(f"Unknown model type: {meta['type']}")
                
                # Compute embeddings
                print("  Computing embeddings...")
                embeddings = compute_embeddings(model, coords_np, meta["type"], args.device)
                print(f"  Embedding shape: {embeddings.shape}")
                
                # Compute eigenspectrum for visualization
                # print("  Computing eigenspectrum...")
                # cov = np.cov(embeddings, rowvar=False)
                # eigs = np.linalg.eigvalsh(cov)
                # eigen_spectra[name] = eigs[::-1]  # Largest to smallest
                
                # Compute ID using all estimators
                print("  Computing intrinsic dimensions...")
                id_results = compute_all_id_estimates(embeddings, name, args.ks)
                all_results[name] = id_results
                
                # Log individual results
                for metric_name, id_value in id_results.items():
                    if not np.isnan(id_value):
                        wandb.log({f"{sampling_name}/{name}/{metric_name}_{CONDITIONAL_NUMBER}": id_value})
                
                for k_local in args.ks:
                    print(f"  Computing local intrinsic dimensions (k={k_local})...")
                    _ = compute_local_id_estimates(
                        embeddings, coords_np, name, sampling_name, k=k_local, shapefile_path=args.shapefile_path
                    )
                
                #Clean up memory
                del model
                del embeddings
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  ERROR processing {name}: {str(e)}")
                all_results[name] = {est: np.nan for est in ESTIMATORS.keys()}
        
        # Create comparison visualization for this sampling scheme
        print(f"\nCreating comparison plots for {sampling_name}...")
        fig_comparison = create_id_comparison_plot(all_results, sampling_name)
        comparison_fn = f"id_comparison_{sampling_name}.png"
        fig_comparison.savefig(comparison_fn, dpi=300, bbox_inches='tight')
        plt.close(fig_comparison)
        wandb.log({f"{sampling_name}_id_comparison": wandb.Image(comparison_fn)})
        
        # Create eigenspectrum plot
        fig_eigen = create_eigenspectrum_plot(eigen_spectra, sampling_name)
        eigen_fn = f"eigenspectrum_{sampling_name}.png"
        fig_eigen.savefig(eigen_fn, dpi=300, bbox_inches='tight')
        plt.close(fig_eigen)
        wandb.log({f"{sampling_name}_eigenspectrum": wandb.Image(eigen_fn)})
        
        # Create and log summary table for this sampling scheme
        all_labels = sorted({k for enc in all_results for k in all_results[enc].keys()})

        # Print header
        header = f"{'Encoder':<15}" + "".join(f"{lab:>14}" for lab in all_labels)
        print(header)
        print("-"*80)

        # Print rows
        for encoder, res in all_results.items():
            row = f"{encoder:<15}"
            for lab in all_labels:
                val = res.get(lab, np.nan)
                row += f"{('N/A' if np.isnan(val) else f'{val:.3f}'):>14}"
            print(row)
        
        # Log summary table to wandb
        summary_data = []
        all_labels = sorted({k for enc in all_results for k in all_results[enc].keys()})
        for encoder, res in all_results.items():
            row_data = {"Encoder": encoder, "Sampling": sampling_name}
            for lab in all_labels:
                row_data[lab] = res.get(lab, np.nan)
            summary_data.append(row_data)

        summary_df = pd.DataFrame(summary_data)
        wandb.log({f"{sampling_name}_summary_table": wandb.Table(dataframe=summary_df)})
    
    # Finish wandb run
    wandb.finish()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()