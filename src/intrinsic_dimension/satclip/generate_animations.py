#!/usr/bin/env python
"""
Generate rotating globe animation for SatCLIP L40 local intrinsic dimension
"""
import pyproj
pyproj.datadir.set_data_dir(
    "/projects/arra4944/arm64/software/miniforge/envs/bg2/share/proj"
)

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from shapely.geometry import Point
from shapely.ops import unary_union
import matplotlib.path as mpath
import imageio
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import scikit-dimension
import skdim.id

# Import SatCLIP utilities
from eval_utils import get_satclip

# Animation parameters
ANIMATION_PARAMS = {
    'n_frames': 36,      # Number of frames for full rotation
    'fps': 8,            # Frames per second
    'dpi': 80,           # Resolution
    'figsize': (6, 6),   # Figure size
    'n_points': 100000,   # Number of land points to sample
    'k_neighbors': 100,   # k for local MLE
}

# SatCLIP checkpoint path
SATCLIP_PATH = "/projects/arra4944/MMLocEnc/pretrain/vanilla_weights/satclip-resnet50-l40.ckpt"

def sample_land_points(n, max_attempts=None):
    """Sample points only on land masses."""
    print("  Loading land polygons...")
    shp = shapereader.natural_earth(resolution='110m', category='physical', name='land')
    reader = shapereader.Reader(shp)
    land_geoms = list(reader.geometries())
    land_union = unary_union(land_geoms)
    
    print(f"  Sampling {n:,} points on land...")
    points = []
    batch_size = max(n, 10000)
    max_attempts = max_attempts or n * 50
    attempts = 0
    
    with tqdm(total=n, desc="    Sampling land points") as pbar:
        while len(points) < n and attempts < max_attempts:
            # Sample on sphere uniformly
            u = np.random.uniform(-1, 1, batch_size)
            lats = np.degrees(np.arcsin(u))
            lons = np.random.uniform(-180, 180, batch_size)
            
            # Check which points are on land
            for lon, lat in zip(lons, lats):
                if land_union.contains(Point(lon, lat)):
                    points.append([lon, lat])
                    pbar.update(1)
                    if len(points) >= n:
                        break
            attempts += batch_size
    
    if len(points) < n:
        print(f"    Warning: Only found {len(points)} land points")
    
    return np.array(points[:n])

def compute_local_id_mle(embeddings, k=20):
    """Compute local intrinsic dimension using MLE."""
    print(f"  Computing local ID with MLE (k={k})...")
    estimator = skdim.id.MLE(neighborhood_based=True)
    
    # Use parallel computation
    local_ids = estimator.fit_transform_pw(
        embeddings, 
        n_neighbors=k, 
        n_jobs=-1
    )
    
    if isinstance(local_ids, tuple):
        local_ids = local_ids[0]
    
    return local_ids

def create_globe_frame(coords, values, lon_center, lat_center=20, 
                      title="", vmin=None, vmax=None):
    """Create a single globe frame with white ocean."""
    
    fig = plt.figure(figsize=ANIMATION_PARAMS['figsize'], facecolor='none')
    ax = fig.add_subplot(111, projection=ccrs.Orthographic(
        central_longitude=lon_center, 
        central_latitude=lat_center
    ))
    
    # Set circular boundary
    ax.set_global()
    theta = np.linspace(0, 2*np.pi, 256)
    center, radius = [0.5, 0.5], 0.49
    verts = np.c_[np.cos(theta), np.sin(theta)] * radius + center
    path = mpath.Path(verts)
    ax.set_boundary(path, transform=ax.transAxes)
    
    # Add map features - white ocean, light gray land
    ax.add_feature(cfeature.OCEAN, facecolor='white', alpha=1.0)
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', alpha=1.0)
    ax.coastlines(resolution='110m', color='#666666', linewidth=0.5, alpha=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#999999', alpha=0.5)
    
    # Add gridlines
    gl = ax.gridlines(color='gray', linestyle=':', linewidth=0.3, alpha=0.3)
    
    # Plot the data points
    mask = np.isfinite(values)
    sc = ax.scatter(
        coords[mask, 0], coords[mask, 1],
        c=values[mask],
        cmap='OrRd',
        s=2,
        alpha=0.8,
        transform=ccrs.PlateCarree(),
        vmin=vmin, vmax=vmax,
        rasterized=True,
        edgecolors='none'
    )
    
    # Add title
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.7, aspect=30)
    cbar.set_label('Local Intrinsic Dimension', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    return fig

def create_rotating_globe_animation(coords, local_ids, output_path):
    """Create rotating globe animation for SatCLIP."""
    
    print(f"  Creating animation...")
    
    # Get color scale
    mask = np.isfinite(local_ids)
    if mask.sum() == 0:
        print(f"    Warning: No valid local IDs")
        return
    
    vmin, vmax = np.percentile(local_ids[mask], [5, 95])
    
    # Generate frames
    frames = []
    n_frames = ANIMATION_PARAMS['n_frames']
    
    # Create longitude centers for rotation
    lon_centers = np.linspace(0, 360, n_frames, endpoint=False)
    
    # Also vary latitude for more interesting motion (subtle wobble)
    lat_centers = 15 + 10 * np.sin(2 * np.pi * np.arange(n_frames) / n_frames)
    
    for i, (lon, lat) in enumerate(tqdm(zip(lon_centers, lat_centers), 
                                       total=n_frames, 
                                       desc="    Rendering frames")):
        
        fig = create_globe_frame(
            coords, local_ids,
            lon_center=lon,
            lat_center=lat,
            title="SatCLIP L40",
            vmin=vmin, vmax=vmax
        )
        
        # Convert to image
        fig.canvas.draw()
        try:
            # Try new matplotlib API first
            image = np.array(fig.canvas.buffer_rgba())
            image = image[:, :, :3]  # Drop alpha channel
        except AttributeError:
            # Fall back to old API
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        frames.append(image)
        plt.close(fig)
    
    # Save as GIF
    print(f"    Saving animation to {output_path}")
    imageio.mimsave(
        output_path,
        frames,
        fps=ANIMATION_PARAMS['fps'],
        loop=0  # Infinite loop
    )
    
    # Also save as MP4 for better quality
    try:
        mp4_path = output_path.replace('.gif', '.mp4')
        imageio.mimsave(
            mp4_path,
            frames,
            fps=ANIMATION_PARAMS['fps'],
            codec='libx264'
        )
        print(f"    MP4 version saved: {mp4_path}")
    except:
        pass

def main():
    """Main execution function."""
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Sample land coordinates
    print(f"\nSampling {ANIMATION_PARAMS['n_points']:,} land points...")
    coords = sample_land_points(ANIMATION_PARAMS['n_points'])
    coords_t = torch.from_numpy(coords).double().to(device)
    
    # Load SatCLIP model
    print("\nLoading SatCLIP L40...")
    model = get_satclip(SATCLIP_PATH, device, return_all=False)
    model = model.to(device).eval()
    
    # Compute embeddings
    print("Computing embeddings...")
    with torch.no_grad():
        embeddings = model(coords_t).cpu().numpy()
    print(f"Embedding shape: {embeddings.shape}")
    
    # Compute local intrinsic dimension
    local_ids = compute_local_id_mle(
        embeddings, 
        k=ANIMATION_PARAMS['k_neighbors']
    )
    
    # Create animation
    output_path = "globe_animation_satclip_l40.gif"
    create_rotating_globe_animation(coords, local_ids, output_path)
    
    # Print statistics
    mask = np.isfinite(local_ids)
    print(f"\nStatistics:")
    print(f"  Mean ID: {np.mean(local_ids[mask]):.2f}")
    print(f"  Std ID: {np.std(local_ids[mask]):.2f}")
    print(f"  Min ID: {np.min(local_ids[mask]):.2f}")
    print(f"  Max ID: {np.max(local_ids[mask]):.2f}")
    
    print("\n" + "="*60)
    print("SatCLIP animation generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()