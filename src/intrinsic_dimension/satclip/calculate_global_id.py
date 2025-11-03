#!/usr/bin/env python
"""
SatCLIP-specific script for computing global and local intrinsic dimensionality
using multiple estimators from scikit-dimension under different sampling schemes.

Models: SatCLIP (L=10, L=40)
"""
import pyproj
pyproj.datadir.set_data_dir(
    "/projects/arra4944/arm64/software/miniforge/envs/bg2/share/proj"
)


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
from matplotlib.colors import TwoSlopeNorm, LogNorm
from cartopy.feature import NaturalEarthFeature

# Import scikit-dimension estimators
import skdim.id

# Import SatCLIP utilities
from eval_utils import get_satclip

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm


import inspect
import numpy as np


# ─── Configuration ───────────────────────────────────────────────────────────
CHECKPOINT_PATHS = {
    # SatCLIP paths
    "satclip_l10": "/projects/arra4944/MMLocEnc/pretrain/vanilla_weights/satclip-resnet50-l10.ckpt",
    "satclip_l40": "/projects/arra4944/MMLocEnc/pretrain/vanilla_weights/satclip-resnet50-l40.ckpt",
    "satclip_usa": "/projects/arra4944/MMLocEnc/pretrain/satclip/satclip/satclip_logs/satclip_s2geo_usa/zjldz9aj/checkpoints/last.ckpt",
    "satclip_europe": "/projects/arra4944/MMLocEnc/pretrain/satclip/satclip/satclip_logs/satclip_s2geo_europe/au1nrhyh/checkpoints/last.ckpt",
    "satclip_france": "/projects/arra4944/MMLocEnc/pretrain/satclip/satclip/satclip_logs/satclip_s2geo_france/2ur4w02o/checkpoints/epoch=284-val_loss=5.19_s2geo_france.ckpt",
    "satclip_mountainwest": "/projects/arra4944/MMLocEnc/pretrain/satclip/satclip/satclip_logs/satclip_s2geo_mountainwest/sp0x1a4b/checkpoints/epoch=424-val_loss=4.82_s2geo_mountainwest.ckpt",
    "satclip_africa": "/projects/arra4944/MMLocEnc/pretrain/satclip/satclip/satclip_logs/satclip_s2geo_africa/abquocsf/checkpoints/last.ckpt",
}

# Define all global ID estimators to use
ESTIMATORS = {
    'MLE': lambda: skdim.id.MLE(neighborhood_based=True),
    'TwoNN': lambda: skdim.id.TwoNN(),
    'FisherS': lambda: skdim.id.FisherS(),
    'MOM': lambda: skdim.id.MOM(),
    'TLE': lambda: skdim.id.TLE(),
    'CorrInt': lambda: skdim.id.CorrInt(),
    'DANCo': lambda: skdim.id.DANCo(k=10),
    'ESS': lambda: skdim.id.ESS(),
    'MiND_ML': lambda: skdim.id.MiND_ML(ver='ML'),
    'MiND_KL': lambda: skdim.id.MiND_KL(ver='KL'),
    'MADA': lambda: skdim.id.MADA(),
}

# Estimators that support local ID computation
LOCAL_ESTIMATORS = {
    'MLE': lambda: skdim.id.MLE(neighborhood_based=True),
    'TwoNN': lambda: skdim.id.TwoNN(),
    # 'FisherS': lambda: skdim.id.FisherS(),
    'MOM': lambda: skdim.id.MOM(),
    'TLE': lambda: skdim.id.TLE(),
    'ESS': lambda: skdim.id.ESS(),
}


# ─── Sampling functions ─────────────────────────────────────────────────────────
def sample_land(N, **kwargs):
    """Sample uniformly on sphere but keep only land points."""
    shp = shapereader.natural_earth('110m', 'physical', 'land')
    geoms = list(shapereader.Reader(shp).geometries())
    land_poly = unary_union(geoms)
    pts, batch = [], max(N, 10_000)
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
    return np.vstack([grid, sample_naive(extra)])

def sample_healpix(N, **kwargs):
    """Equal-area tessellation via HEALPix."""
    import healpy as hp
    nside = 1
    while hp.nside2npix(nside) < N:
        nside *= 2
    sel = np.random.choice(hp.nside2npix(nside), N, replace=False)
    theta, phi = hp.pix2ang(nside, sel)
    lats = 90 - np.degrees(theta)
    lons = np.degrees(phi) - 180
    return np.stack([lons, lats], axis=1)

def sample_fibonacci(N, **kwargs):
    """Fibonacci spiral sampling."""
    i = np.arange(N)
    phi = np.pi * (3. - np.sqrt(5.))
    lats = np.degrees(np.arcsin(2*(i/N) - 1))
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
        return np.vstack([np.stack([lons, lats], axis=1), sample_naive(extra)])
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

SAMPLERS = {
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
}

def get_sampler(name):
    """Get sampling function by name."""
    return SAMPLERS[name]



import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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

def plot_local_id_orthographic(
    coords, local_ids, estimator_name, encoder_name,
    sampling_name, k, smooth=False, shapefile_path=None
):
    """
    Camera-ready orthographic globes with USA and France views added.
    Handles both shapefile and lat/lon extent overlays for regional models.
    Ocean is white, training regions shown with green diagonal hatching on top.
    """
    
    # Load shapefile if provided
    gdf = None
    if shapefile_path is not None:
        try:
            import geopandas as gpd
            gdf = gpd.read_file(shapefile_path)
            if gdf.crs and gdf.crs.to_string().lower() not in ("epsg:4326", "wgs84", "ogc:crs84"):
                gdf = gdf.to_crs("EPSG:4326")
            print(f"      Loaded shapefile with {len(gdf)} features")
            
            # Check if this is a countries shapefile by looking for common attribute names
            if gdf is not None and not gdf.empty:
                # Common attribute names for country in Natural Earth data
                country_attrs = ['NAME', 'NAME_EN', 'ADMIN', 'SOVEREIGNT', 'NAME_LONG', 'GEOUNIT']
                country_col = None
                for attr in country_attrs:
                    if attr in gdf.columns:
                        country_col = attr
                        break
                if country_col:
                    print(f"      Detected countries shapefile with column: {country_col}")
                    
        except Exception as e:
            print(f"      Warning: Could not load shapefile: {e}")
            gdf = None
    
    # Define lat/lon extent overlays for models without shapefiles
    regional_extents = {
        'france': {
            'lon_min': -5.0, 'lon_max': 10.0,
            'lat_min': 41.0, 'lat_max': 52.0
        },
        'mountainwest': {
            'lon_min': -115.0, 'lon_max': -102.0, 
            'lat_min': 31.0, 'lat_max': 49.0
        },
        'denver': {
            'lon_min': -106.5, 'lon_max': -103.5,
            'lat_min': 38.5, 'lat_max': 40.5
        }
    }
    
    # Determine if we need extent-based overlay (when no shapefile)
    extent_overlay = None
    if gdf is None:  # No shapefile provided
        if 'france' in encoder_name.lower():
            extent_overlay = regional_extents['france']
        elif 'mountain' in encoder_name.lower():
            extent_overlay = regional_extents['mountainwest']
        elif 'denver' in encoder_name.lower():
            extent_overlay = regional_extents['denver']
    
    # Extended views including USA and France focus
    views = [
        (-90,   0,  "Americas"),
        (-98,  39,  "United States"),  # USA centered view
        (2,    46,  "France"),          # France centered view
        (10,   45,  "Europe"),
        (20,    0,  "Africa"),
        (100,  50,  "Asia"),
        (135, -25,  "Oceania"),
        (0,    90,  "North Pole"),
        (0,   -90,  "South Pole"),
    ]
    
    # Robust color scale
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
        # Square-ish figure
        fig = plt.figure(figsize=(6.0, 7.0))
        ax = fig.add_axes([0.16, 0.24, 0.68, 0.68],
                        projection=ccrs.Orthographic(central_longitude=lon0,
                                                    central_latitude=lat0))
        ax.set_global()
        
        # Force circular globe boundary
        theta = np.linspace(0, 2*np.pi, 256)
        center, radius = np.array([0.5, 0.5]), 0.5
        circle_verts = np.column_stack([np.cos(theta), np.sin(theta)]) * radius + center
        circle_path = mpath.Path(circle_verts)
        ax.set_boundary(circle_path, transform=ax.transAxes)
        
        ax.set_facecolor("#ffffff")
        
        # Base features - WHITE OCEAN
        ax.add_feature(cfeature.LAND, facecolor="#fafafa", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=0)  # White ocean
        ax.coastlines(resolution="50m", color="black", linewidth=0.6, zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray", zorder=3)
        
        # Gridlines
        gl = ax.gridlines(draw_labels=False, color="gray", linestyle=":",
                          linewidth=0.4, alpha=0.5, zorder=1)
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 30))
        
        # Local ID scatter FIRST (lower z-order)
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=local_ids, cmap="OrRd",
            vmin=vmin, vmax=vmax,
            s=1, alpha=1,
            transform=ccrs.PlateCarree(),
            zorder=2, rasterized=True, edgecolors="none"  # Lower zorder
        )
        
        # Overlay training region with green hatching ON TOP
        shp_suffix = ""
        
        # Filter shapefile for specific views
        gdf_to_plot = gdf
        if gdf is not None and not gdf.empty:
            # Check if this is a countries shapefile
            country_attrs = ['NAME', 'NAME_EN', 'ADMIN', 'SOVEREIGNT', 'NAME_LONG', 'GEOUNIT']
            country_col = None
            for attr in country_attrs:
                if attr in gdf.columns:
                    country_col = attr
                    break
            
            if country_col:
                # Filter for specific countries based on view
                if region_name == "United States":
                    # Filter for USA
                    usa_names = ['United States', 'United States of America', 'USA']
                    gdf_to_plot = gdf[gdf[country_col].isin(usa_names)]
                    if gdf_to_plot.empty:
                        # Try partial match
                        gdf_to_plot = gdf[gdf[country_col].str.contains('United States', case=False, na=False)]
                    shp_suffix = " (USA training region)"
                    
                elif region_name == "France":
                    # Filter for France
                    france_names = ['France', 'République française']
                    gdf_to_plot = gdf[gdf[country_col].isin(france_names)]
                    if gdf_to_plot.empty:
                        gdf_to_plot = gdf[gdf[country_col].str.contains('France', case=False, na=False)]
                    shp_suffix = " (France training region)"
        
        # Option 1: Shapefile overlay with green hatching - HIGH Z-ORDER
        if gdf_to_plot is not None and not gdf_to_plot.empty:
            try:
                # Hatched fill with green diagonal lines
                ax.add_geometries(
                    gdf_to_plot.geometry,
                    crs=ccrs.PlateCarree(),
                    facecolor="none",  # No fill color
                    edgecolor=(0.0, 0.5, 0.0, 0.8),  # Green edge
                    linewidth=1.6,
                    hatch='///',  # Diagonal hatching
                    zorder=10  # High zorder to appear on top
                )
                if not shp_suffix:
                    shp_suffix = " (training region)"
            except Exception as e:
                print(f"      Warning: could not draw shapefile: {e}")
        
        # Option 2: Lat/lon extent overlay with green hatching - HIGH Z-ORDER
        elif extent_overlay is not None and gdf is None:
            try:
                import matplotlib.patches as mpatches
                
                # Create rectangle from extent
                lon_min = extent_overlay['lon_min']
                lon_max = extent_overlay['lon_max']
                lat_min = extent_overlay['lat_min']
                lat_max = extent_overlay['lat_max']
                
                # Create vertices for the rectangular region
                rect_verts = [
                    [lon_min, lat_min],
                    [lon_max, lat_min],
                    [lon_max, lat_max],
                    [lon_min, lat_max],
                    [lon_min, lat_min]
                ]
                
                # Add as polygon with green hatching
                rect_patch = mpatches.Polygon(
                    rect_verts,
                    transform=ccrs.PlateCarree(),
                    facecolor="none",  # No fill
                    edgecolor=(0.0, 0.5, 0.0, 0.8),  # Green edge
                    linewidth=1.6,
                    hatch='///',  # Diagonal hatching
                    zorder=10  # High zorder to appear on top
                )
                ax.add_patch(rect_patch)
                shp_suffix = " (training extent)"
                
            except Exception as e:
                print(f"      Warning: could not draw extent overlay: {e}")
        
        # Title
        ax.set_title(region_name + shp_suffix, fontsize=16, fontweight="bold", pad=10)
        
        # Horizontal colorbar
        cax = fig.add_axes([0.18, 0.08, 0.64, 0.035])
        cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=11)
        cbar.formatter = (mticker.FormatStrFormatter('%.0f') if vmax > 100
                          else mticker.FormatStrFormatter('%.1f'))
        cbar.update_ticks()
        cbar.set_label("Local Intrinsic Dimension", fontsize=12, labelpad=4)
        
        # Save
        smooth_suffix = "_smooth" if smooth else "_raw"
        coverage_suffix = "_wshp" if (gdf is not None or extent_overlay is not None) else ""
        filename = (
            f"local_id_{sampling_name}_{encoder_name}_{estimator_name}_k{k}_"
            f"{region_name.lower().replace(' ', '_')}{smooth_suffix}{coverage_suffix}.png"
        )
        fig.savefig(filename, dpi=300, facecolor="white")
        
        try:
            import wandb
            wandb.log({
                f"{sampling_name}/{encoder_name}/local_{estimator_name}_k{k}/"
                f"{region_name}{smooth_suffix}{coverage_suffix}": wandb.Image(filename)
            })
        except:
            pass
        
        figures.append((fig, filename))
        plt.close(fig)
    
    # Composite figure
    try:
        create_composite_figure(
            coords, local_ids, estimator_name, encoder_name,
            sampling_name, k, smooth, vmin, vmax, shapefile_path
        )
    except:
        pass
    
    return figures


def plot_sampling_distribution(coords, sampling_name, n_subsample=10000):
    """Create globe views of the sampling distribution with white ocean."""
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
    
    fig = plt.figure(figsize=(12, 12))
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
        ax.set_aspect('equal')
        ax.set_global()
        
        # White ocean and light land
        ax.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=0)  # White ocean
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

def compute_local_id_estimates(embeddings, coords, encoder_name, sampling_name, k=10, shapefile_path=None):
    """
    Compute local ID and create both orthographic and 2D map visualizations.
    """
    local_results = {}
    
    for estimator_name, estimator_fn in LOCAL_ESTIMATORS.items():
        try:
            print(f"    Computing local {estimator_name} (k={k})...")
            estimator = estimator_fn()
            
            if hasattr(estimator, 'fit_transform_pw'):
                result = estimator.fit_transform_pw(
                    embeddings, 
                    n_neighbors=k, 
                    n_jobs=-1
                )
                
                if isinstance(result, tuple) and len(result) == 2:
                    raw_ids, smooth_ids = result
                    
                    # Create orthographic views for raw
                    figures_raw = plot_local_id_orthographic(
                        coords, raw_ids, estimator_name, 
                        encoder_name, sampling_name, k, smooth=False, 
                        shapefile_path=shapefile_path
                    )
                    
                    # Create 2D map for raw
                    map_2d_raw = plot_local_id_2d_minimal(
                        coords, raw_ids, estimator_name + "_raw", 
                        encoder_name, sampling_name, k
                    )
                    if map_2d_raw:
                        wandb.log({
                            f"{sampling_name}/{encoder_name}/local_{estimator_name}_2d_raw_k{k}": 
                            wandb.Image(map_2d_raw)
                        })
                    
                    # Create orthographic views for smooth
                    figures_smooth = plot_local_id_orthographic(
                        coords, smooth_ids, estimator_name, 
                        encoder_name, sampling_name, k, smooth=True,
                        shapefile_path=shapefile_path
                    )
                    
                    # Create 2D map for smooth
                    map_2d_smooth = plot_local_id_2d_minimal(
                        coords, smooth_ids, estimator_name + "_smooth", 
                        encoder_name, sampling_name, k
                    )
                    if map_2d_smooth:
                        wandb.log({
                            f"{sampling_name}/{encoder_name}/local_{estimator_name}_2d_smooth_k{k}": 
                            wandb.Image(map_2d_smooth)
                        })
                    
                    local_results[estimator_name + "_raw"] = raw_ids
                    local_results[estimator_name + "_smooth"] = smooth_ids
                    
                    print(f"      → Raw: Mean={np.nanmean(raw_ids):.4f}, Std={np.nanstd(raw_ids):.4f}")
                    print(f"      → Smooth: Mean={np.nanmean(smooth_ids):.4f}, Std={np.nanstd(smooth_ids):.4f}")
                    
                else:
                    # Single output
                    local_ids = result if not isinstance(result, tuple) else result[0]
                    
                    # Create orthographic views
                    figures = plot_local_id_orthographic(
                        coords, local_ids, estimator_name, 
                        encoder_name, sampling_name, k, smooth=False,
                        shapefile_path=shapefile_path
                    )
                    
                    # Create 2D map
                    map_2d_file = plot_local_id_2d_minimal(
                        coords, local_ids, estimator_name, 
                        encoder_name, sampling_name, k
                    )
                    if map_2d_file:
                        wandb.log({
                            f"{sampling_name}/{encoder_name}/local_{estimator_name}_2d_k{k}": 
                            wandb.Image(map_2d_file)
                        })
                    
                    local_results[estimator_name] = local_ids
                    
                    print(f"      → Mean: {np.nanmean(local_ids):.4f}, Std: {np.nanstd(local_ids):.4f}")
                
        except Exception as e:
            print(f"      → Failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return local_results



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

                    # only force comb for MLE; others keep their defaults
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




def create_id_comparison_plot(all_results, sampling_name):
    """
    Plot global ID per encoder. Columns are drawn from the union of keys
    in all_results[...] (e.g., MLE_k5, MLE_k10, ...).
    """
    encoders = list(all_results.keys())

    # union of all metric names present
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



# ─── Argument parsing ────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Global & local intrinsic dimension analysis for SatCLIP encoders."
    )
    p.add_argument('--sampling', nargs='+', 
                   choices=list(SAMPLERS.keys()),
                   required=True,
                   help="One or more sampling schemes (space-separated)")
    p.add_argument('--n', type=int, default=100000,
                   help="Number of sample points per scheme")
    p.add_argument('--ks', type=int, nargs='+', default=[20,100],
                   help="Neighborhood sizes for local ID")
    p.add_argument('--lat-step', type=float, default=1.0)
    p.add_argument('--lon-step', type=float, default=1.0)
    p.add_argument('--bands', type=int, default=10)
    p.add_argument('--min-dist', type=float, default=5.0)
    p.add_argument('--region-file', type=str, default=None,
                   help="GeoJSON/shapefile for 'region' sampler")
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--shapefile-path', type=str, default=None,
                help="Path to shapefile (.shp) to overlay training region boundaries on plots")
    return p.parse_args()


# ─── Main ────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    
    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Set thread limits
    os.environ.update({
        "OMP_NUM_THREADS": "72",
        "MKL_NUM_THREADS": "72",
        "OPENBLAS_NUM_THREADS": "72"
    })
    
    # Initialize wandb
    sampling_str = "-".join(args.sampling)
    ks_str = "-".join(map(str, args.ks))
    wandb.init(
        project="Ablations-Appendix",
        name=f"satclip-id-{sampling_str}-ks{ks_str}"
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define encoders
    encoders = {
        "SatCLIP-L10": {
            "path": CHECKPOINT_PATHS["satclip_l10"],
        },
        "SatCLIP-L40": {
            "path": CHECKPOINT_PATHS["satclip_l40"],
        },
        "SatCLIP-USA": {
            "path": CHECKPOINT_PATHS["satclip_usa"]
        },
        "SatCLIP-Europe": {
            "path": CHECKPOINT_PATHS["satclip_europe"]
        },
        "SatCLIP-France": {
            "path": CHECKPOINT_PATHS["satclip_france"]
        },
        "SatCLIP-MountainWest": {
            "path": CHECKPOINT_PATHS["satclip_mountainwest"]
        },
        "SatCLIP-Africa": {
            "path": CHECKPOINT_PATHS["satclip_africa"]
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
        coords_t = torch.from_numpy(coords_np).double().to(device)
        
        # Visualize sampling
        fig_sampling = plot_sampling_distribution(coords_np, sampling_name)
        sampling_fn = f"sampling_{sampling_name}_satclip.png"
        fig_sampling.savefig(sampling_fn, dpi=300, bbox_inches='tight')
        plt.close(fig_sampling)
        wandb.log({f"{sampling_name}_sampling_visualization": wandb.Image(sampling_fn)})
        
        # Store all results for this sampling scheme
        all_results = {}
        
        # Process each encoder
        for name, meta in encoders.items():
            print(f"\n{'-'*60}")
            print(f"Processing {name}")
            print('-'*60)
            
            try:
                # Load model
                print(f"  Loading model from {meta['path']}...")
                model = get_satclip(meta["path"], device, return_all=False)
                model = model.to(device).eval()
                
                # Compute embeddings
                print("  Computing embeddings...")
                with torch.no_grad():
                    embeddings = model(coords_t).cpu().numpy()
                print(f"  Embedding shape: {embeddings.shape}")
                
                # Compute global ID using all estimators
                print("  Computing global intrinsic dimensions...")
                id_results = compute_all_id_estimates(embeddings, name, args.ks)
                all_results[name] = id_results
                
                # Log individual results
                for metric_name, id_value in id_results.items():
                    if not np.isnan(id_value):
                        wandb.log({f"{sampling_name}/{name}/global_{metric_name}": id_value})
                
                # Compute local ID estimates for each k
                for k in args.ks:
                    print(f"\n  Processing k={k}...")
                    local_id_results = compute_local_id_estimates(
                        embeddings, coords_np, name, sampling_name, k=k, 
                        shapefile_path=args.shapefile_path  # Pass the shapefile path
                    )
                
                # Clean up memory
                del model
                del embeddings
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  ERROR processing {name}: {str(e)}")
                all_results[name] = {est: np.nan for est in ESTIMATORS.keys()}
        
        # Create comparison visualization for this sampling scheme
        print(f"\nCreating comparison plots for {sampling_name}...")
        fig_comparison = create_id_comparison_plot(all_results, sampling_name)
        comparison_fn = f"id_comparison_{sampling_name}_satclip.png"
        fig_comparison.savefig(comparison_fn, dpi=300, bbox_inches='tight')
        plt.close(fig_comparison)
        wandb.log({f"{sampling_name}_id_comparison": wandb.Image(comparison_fn)})
        


        # Create and log summary table for this sampling scheme
        print(f"\n{'-'*80}")
        print(f"SUMMARY OF INTRINSIC DIMENSION ESTIMATES - {sampling_name}")
        print(f"Sampling: {sampling_name} | N={args.n}")
        print("-"*80)
        
        # Print header
        all_labels = sorted({k for enc in all_results for k in all_results[enc].keys()})

        # Header
        header = f"{'Encoder':<15}" + "".join(f"{lab:>14}" for lab in all_labels)
        print(header)
        print("-"*80)

        # Rows
        for encoder, res in all_results.items():
            row = f"{encoder:<15}"
            for lab in all_labels:
                val = res.get(lab, np.nan)
                row += f"{('N/A' if np.isnan(val) else f'{val:.3f}'):>14}"
            print(row)
        
        # Log summary table to wandb
        summary_data = []
        for encoder, results in all_results.items():
            row_data = {"Encoder": encoder, "Sampling": sampling_name}
            row_data.update(results)
            summary_data.append(row_data)
        
        # Convert to pandas DataFrame
        summary_df = pd.DataFrame(summary_data)
        wandb.log({f"{sampling_name}_summary_table": wandb.Table(dataframe=summary_df)})
    
    # Finish wandb run
    wandb.finish()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()