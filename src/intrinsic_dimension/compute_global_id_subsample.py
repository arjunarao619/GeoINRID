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
    'MLE': lambda: skdim.id.MLE(),
    'TwoNN': lambda: skdim.id.TwoNN(),
    'MOM': lambda: skdim.id.MOM(),
    'TLE': lambda: skdim.id.TLE(),
    'FisherS': lambda: skdim.id.FisherS(conditional_number=CONDITIONAL_NUMBER),

    'CorrInt': lambda: skdim.id.CorrInt(),
    'DANCo': lambda: skdim.id.DANCo(k=10),  # k parameter required
    'ESS': lambda: skdim.id.ESS(),
    'MiND_ML': lambda: skdim.id.MiND_ML(ver='ML'),
    'MiND_KL': lambda: skdim.id.MiND_KL(ver='KL'),
    'MADA': lambda: skdim.id.MADA(),
}

def compute_id_with_variability(embeddings, coords, encoder_name, n_subsamples=10, 
                                subsample_size=None, subsample_fraction=0.5, seed=42):
    """
    Compute ID estimates with variability via subsampling.
    
    Args:
        embeddings: Full embedding matrix
        coords: Full coordinate matrix
        encoder_name: Name of encoder for logging
        n_subsamples: Number of subsamples to take
        subsample_size: Fixed size for subsamples (if None, uses subsample_fraction)
        subsample_fraction: Fraction of data to use in each subsample
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with mean, std, and all estimates for each estimator
    """
    np.random.seed(seed)
    n_total = len(embeddings)
    
    if subsample_size is None:
        subsample_size = int(n_total * subsample_fraction)
    
    # Store results for each estimator across subsamples
    subsample_results = {est_name: [] for est_name in ESTIMATORS.keys()}
    
    print(f"  Computing ID with variability (n_subsamples={n_subsamples}, size={subsample_size})...")
    
    for i in range(n_subsamples):
        # Random subsample
        idx = np.random.choice(n_total, subsample_size, replace=False)
        emb_sub = embeddings[idx]
        
        # Compute ID for this subsample
        for estimator_name, estimator_fn in ESTIMATORS.items():
            try:
                estimator = estimator_fn()
                if estimator_name == 'MLE':
                    estimator.fit(emb_sub, n_neighbors=20, n_jobs=-1)
                else:
                    estimator.fit(emb_sub)
                id_value = estimator.dimension_
                subsample_results[estimator_name].append(id_value)
            except:
                subsample_results[estimator_name].append(np.nan)
        
        if (i + 1) % 5 == 0:
            print(f"    Processed {i+1}/{n_subsamples} subsamples")
    
    # Compute statistics
    stats_results = {}
    for est_name, values in subsample_results.items():
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            stats_results[est_name] = {
                'mean': np.mean(valid_values),
                'std': np.std(valid_values),
                'median': np.median(valid_values),
                'q25': np.percentile(valid_values, 25),
                'q75': np.percentile(valid_values, 75),
                'min': np.min(valid_values),
                'max': np.max(valid_values),
                'n_valid': len(valid_values),
                'all_values': valid_values
            }
        else:
            stats_results[est_name] = {
                'mean': np.nan, 'std': np.nan, 'median': np.nan,
                'q25': np.nan, 'q75': np.nan, 'min': np.nan, 'max': np.nan,
                'n_valid': 0, 'all_values': []
            }
    
    return stats_results


def plot_id_variability(variability_results, sampling_name):
    """Create boxplot visualization of ID variability across subsamples."""
    
    # Prepare data for plotting
    plot_data = []
    labels = []
    
    for encoder_name, encoder_results in variability_results.items():
        for est_name in ESTIMATORS.keys():
            if est_name in encoder_results:
                values = encoder_results[est_name]['all_values']
                if values:  # Only add if we have valid values
                    plot_data.append(values)
                    labels.append(f"{encoder_name}\n{est_name}")
    
    if not plot_data:
        return None
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Boxplot
    bp = ax1.boxplot(plot_data, labels=labels, patch_artist=True, 
                     showfliers=True, widths=0.6)
    
    # Color by encoder
    encoder_colors = {'GeoCLIP': 'lightblue', 'CSP-FMoW': 'lightgreen', 'CSP-iNat': 'lightcoral'}
    for i, label in enumerate(labels):
        encoder = label.split('\n')[0]
        bp['boxes'][i].set_facecolor(encoder_colors.get(encoder, 'lightgray'))
    
    ax1.set_ylabel('Intrinsic Dimension', fontsize=12)
    ax1.set_title(f'ID Variability - {sampling_name} sampling\n(n_subsamples per estimate)', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Coefficient of variation plot
    cv_data = []
    cv_labels = []
    cv_colors = []
    
    for encoder_name, encoder_results in variability_results.items():
        for est_name in ['FisherS', 'MLE', 'TwoNN']:  # Focus on main estimators
            if est_name in encoder_results:
                mean_val = encoder_results[est_name]['mean']
                std_val = encoder_results[est_name]['std']
                if not np.isnan(mean_val) and mean_val > 0:
                    cv = (std_val / mean_val) * 100  # CV as percentage
                    cv_data.append(cv)
                    cv_labels.append(f"{encoder_name}\n{est_name}")
                    cv_colors.append(encoder_colors.get(encoder_name, 'gray'))
    
    if cv_data:
        bars = ax2.bar(range(len(cv_data)), cv_data, color=cv_colors, alpha=0.7)
        ax2.set_xticks(range(len(cv_data)))
        ax2.set_xticklabels(cv_labels, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Coefficient of Variation (%)', fontsize=12)
        ax2.set_title('Relative Variability (CV = σ/μ × 100)', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, cv_data):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig


# Modified main computation section to include variability
def compute_all_id_estimates_with_variability(embeddings, coords, encoder_name, 
                                              sampling_name, n_subsamples=5):
    """Enhanced version with variability estimation."""
    
    # First compute on full dataset
    print("  Computing ID on full dataset...")
    full_results = compute_all_id_estimates(embeddings, encoder_name)
    
    # Then compute with subsampling for variability
    print("  Computing ID variability via subsampling...")
    var_results = compute_id_with_variability(
        embeddings, coords, encoder_name, 
        n_subsamples=n_subsamples,
        subsample_fraction=0.5
    )
    
    # Combine results
    combined_results = {}
    for est_name in ESTIMATORS.keys():
        combined_results[est_name] = {
            'full_sample': full_results.get(est_name, np.nan),
            'subsample_mean': var_results[est_name]['mean'],
            'subsample_std': var_results[est_name]['std'],
            'subsample_median': var_results[est_name]['median'],
            'iqr': var_results[est_name]['q75'] - var_results[est_name]['q25'],
            'cv': (var_results[est_name]['std'] / var_results[est_name]['mean'] * 100 
                   if var_results[est_name]['mean'] > 0 else np.nan)
        }
    
    return combined_results, var_results

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


def compute_all_id_estimates(embeddings, encoder_name):
    """Compute ID using all available estimators."""
    results = {}
    
    for estimator_name, estimator_fn in ESTIMATORS.items():
        try:
            print(f"    Computing {estimator_name}_{CONDITIONAL_NUMBER}...")
            # if estimator_name != 'FisherS':
            #     print("Not fisherS")
            #     fishers_preproc = skdim.id.FisherS(conditional_number = CONDITIONAL_NUMBER, project_on_sphere=True)
            #     print(fishers_preproc._preprocessing(embeddings, center=True, dimred=True, whiten=True).shape)
            #     embeddings = fishers_preproc._preprocessing(embeddings, center=True, dimred=True, whiten=True)
            estimator = estimator_fn()
            if estimator_name == 'MLE':
                estimator.fit(embeddings, n_neighbors=20, n_jobs=-1)
            estimator.fit(embeddings)
            id_value = estimator.dimension_
            results[estimator_name] = id_value
            print(f"      → {id_value:.4f}")
        except Exception as e:
            print(f"      → Failed: {str(e)}")
            results[estimator_name] = np.nan
    
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


def plot_local_id_orthographic_single(coords, local_ids, estimator_name, encoder_name,
                                     sampling_name, k, view_params):
    """
    Create a single orthographic view.
    """
    lon0, lat0, region_name = view_params
    
    # Clean data
    mask = np.isfinite(local_ids)
    coords_clean = coords[mask]
    ids_clean = local_ids[mask]
    
    if len(ids_clean) == 0:
        return None
    
    # Consistent color scale
    vmin, vmax = np.percentile(ids_clean, [5, 95])
    if vmin <= 0:
        vmin = 0.1
    if vmax <= vmin:
        vmax = vmin * 1.1
    
    # Square figure for orthographic projection
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection=ccrs.Orthographic(central_longitude=lon0,
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
    ax.coastlines(resolution="50m", color="black", linewidth=0.8, zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray", zorder=2)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=False, color="gray", linestyle=":",
                     linewidth=0.4, alpha=0.5, zorder=1)
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 30))
    
    # Determine point size based on visible data density
    n_points = len(coords_clean)
    if n_points > 100000:
        point_size = 2
    elif n_points > 50000:
        point_size = 1.0
    elif n_points > 10000:
        point_size = 2.0
    else:
        point_size = 4.0
    
    # Plot points
    sc = ax.scatter(
        coords_clean[:, 0], coords_clean[:, 1],
        c=ids_clean, cmap='OrRd',
        s=point_size,
        alpha=0.8,
        transform=ccrs.PlateCarree(),
        vmin=vmin, vmax=vmax,
        zorder=3,
        rasterized=True,
        edgecolors='none'
    )
    
    # Title
    ax.set_title(f"{encoder_name} - {region_name}", fontsize=14, fontweight='bold', pad=10)
    
    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.7, aspect=40)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(f'Local ID (k={k})', fontsize=11)
    
    # Format colorbar
    if vmax > 100:
        cbar.formatter = mticker.FormatStrFormatter('%.0f')
    else:
        cbar.formatter = mticker.FormatStrFormatter('%.1f')
    cbar.update_ticks()
    
    plt.tight_layout(pad=1.0)
    
    # Save individual view
    region_clean = region_name.lower().replace(' ', '_').replace('°', '')
    filename = (f"local_id_ortho_{sampling_name}_{encoder_name}_{estimator_name}_"
               f"k{k}_{region_clean}.png")
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return filename


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

def compute_local_id_estimates(embeddings, coords, encoder_name, sampling_name, k=10):
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
    
    # Only use estimators that support local ID computation
    local_estimators = {
        'MLE': lambda: skdim.id.MLE(neighborhood_based=True),
        # You can uncomment these if they support fit_transform_pw:
        # 'TwoNN': lambda: skdim.id.TwoNN(),
        # 'FisherS': lambda: skdim.id.FisherS(conditional_number=CONDITIONAL_NUMBER),
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
                
                # Also create a composite figure with all orthographic views
                create_composite_orthographic(
                    coords, local_ids, estimator_name, encoder_name, 
                    sampling_name, k, ortho_views
                )
                
                print(f"      → Mean: {np.nanmean(local_ids):.4f}, "
                     f"Std: {np.nanstd(local_ids):.4f}, "
                     f"Valid points: {np.sum(np.isfinite(local_ids))}")
                     
        except Exception as e:
            print(f"      → Failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return local_results

def create_id_comparison_plot(all_results, sampling_name):
    """Create a comparison plot of ID estimates across encoders and methods."""
    # Prepare data
    encoders = list(all_results.keys())
    estimators = list(ESTIMATORS.keys())
    
    # Create matrix of ID values
    id_matrix = np.zeros((len(encoders), len(estimators)))
    for i, enc in enumerate(encoders):
        for j, est in enumerate(estimators):
            id_matrix[i, j] = all_results[enc].get(est, np.nan)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot as grouped bar chart
    x = np.arange(len(encoders))
    width = 0.8 / len(estimators)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(estimators)))
    
    for j, (est, color) in enumerate(zip(estimators, colors)):
        offset = (j - len(estimators)/2 + 0.5) * width
        values = id_matrix[:, j]
        ax.bar(x + offset, values, width, label=est, color=color)
    
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
    p.add_argument('--k', type=int, default=20,
                   help="Neighborhood size for local ID computation")
    p.add_argument('--lat-step', type=float, default=1.0)
    p.add_argument('--lon-step', type=float, default=1.0)
    p.add_argument('--bands', type=int, default=10)
    p.add_argument('--min-dist', type=float, default=5.0)
    p.add_argument('--region-file', type=str, default=None)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


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
    wandb.init(
        project="conditionnum_allest_test",
        name=f"global-local-id-{sampling_str}-k{args.k}_cond{CONDITIONAL_NUMBER}_var"
    )
    
    # Define encoders to evaluate
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
        
        # Initialize storage for this sampling scheme
        all_results = {}
        variability_results = {}
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
                
                # Compute ID with variability estimation
                print("  Computing intrinsic dimensions with variability...")
                combined_results, var_results = compute_all_id_estimates_with_variability(
                    embeddings, coords_np, name, sampling_name, n_subsamples=3
                )
                
                # Store results
                all_results[name] = {est: res['full_sample'] for est, res in combined_results.items()}
                variability_results[name] = var_results
                
                # Log enhanced metrics to wandb
                for est_name, res in combined_results.items():
                    if not np.isnan(res['full_sample']):
                        wandb.log({
                            f"{sampling_name}/{name}/{est_name}_full": res['full_sample'],
                            f"{sampling_name}/{name}/{est_name}_mean": res['subsample_mean'],
                            f"{sampling_name}/{name}/{est_name}_std": res['subsample_std'],
                            f"{sampling_name}/{name}/{est_name}_cv": res['cv']
                        })
                
                # Compute local ID estimates (optional - comment out if not needed)
                print(f"  Computing local intrinsic dimensions (k={args.k})...")
                local_id_results = compute_local_id_estimates(
                    embeddings, coords_np, name, sampling_name, k=args.k
                )
                
                # Clean up memory
                del model
                del embeddings
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  ERROR processing {name}: {str(e)}")
                import traceback
                traceback.print_exc()
                all_results[name] = {est: np.nan for est in ESTIMATORS.keys()}
                variability_results[name] = {est: {'all_values': [], 'mean': np.nan, 
                                                   'std': np.nan, 'median': np.nan,
                                                   'q25': np.nan, 'q75': np.nan,
                                                   'min': np.nan, 'max': np.nan,
                                                   'n_valid': 0} 
                                            for est in ESTIMATORS.keys()}
        
        # Create visualizations for this sampling scheme
        print(f"\nCreating comparison plots for {sampling_name}...")
        
        # ID comparison plot
        fig_comparison = create_id_comparison_plot(all_results, sampling_name)
        comparison_fn = f"id_comparison_{sampling_name}.png"
        fig_comparison.savefig(comparison_fn, dpi=300, bbox_inches='tight')
        plt.close(fig_comparison)
        wandb.log({f"{sampling_name}_id_comparison": wandb.Image(comparison_fn)})
        
        # Variability plot
        fig_variability = plot_id_variability(variability_results, sampling_name)
        if fig_variability:
            var_fn = f"id_variability_{sampling_name}.png"
            fig_variability.savefig(var_fn, dpi=300, bbox_inches='tight')
            plt.close(fig_variability)
            wandb.log({f"{sampling_name}_id_variability": wandb.Image(var_fn)})
        
        # Eigenspectrum plot (if computed)
        if eigen_spectra:
            fig_eigen = create_eigenspectrum_plot(eigen_spectra, sampling_name)
            eigen_fn = f"eigenspectrum_{sampling_name}.png"
            fig_eigen.savefig(eigen_fn, dpi=300, bbox_inches='tight')
            plt.close(fig_eigen)
            wandb.log({f"{sampling_name}_eigenspectrum": wandb.Image(eigen_fn)})
        
        # Print and log summary table
        print(f"\n{'-'*80}")
        print(f"SUMMARY OF INTRINSIC DIMENSION ESTIMATES - {sampling_name}")
        print(f"Sampling: {sampling_name} | N={args.n}")
        print("-"*80)
        
        # Print header
        header = f"{'Encoder':<15}"
        for est in ESTIMATORS.keys():
            header += f"{est:>10}"
        print(header)
        print("-"*80)
        
        # Print results with uncertainty
        for encoder, results in all_results.items():
            row = f"{encoder:<15}"
            for est in ESTIMATORS.keys():
                val = results.get(est, np.nan)
                if np.isnan(val):
                    row += f"{'N/A':>10}"
                else:
                    std = variability_results[encoder][est]['std']
                    if not np.isnan(std):
                        row += f"{val:>6.2f}±{std:<3.2f}"
                    else:
                        row += f"{val:>10.3f}"
            print(row)
        
        # Create enhanced summary table for wandb
        summary_data = []
        for encoder in all_results.keys():
            for est in ESTIMATORS.keys():
                row_data = {
                    "Encoder": encoder,
                    "Sampling": sampling_name,
                    "Estimator": est,
                    "ID_Full": all_results[encoder].get(est, np.nan),
                    "ID_Mean": variability_results[encoder][est]['mean'],
                    "ID_Std": variability_results[encoder][est]['std'],
                    "ID_Median": variability_results[encoder][est]['median'],
                    "CV_%": variability_results[encoder][est].get('cv', np.nan) 
                            if variability_results[encoder][est]['mean'] > 0 else np.nan,
                    "N_Valid": variability_results[encoder][est]['n_valid']
                }
                summary_data.append(row_data)
        
        summary_df = pd.DataFrame(summary_data)
        wandb.log({f"{sampling_name}_summary_table_detailed": wandb.Table(dataframe=summary_df)})
    
    # Finish wandb run
    wandb.finish()
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()