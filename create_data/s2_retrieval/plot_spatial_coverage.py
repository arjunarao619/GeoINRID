#!/usr/bin/env python3
"""
Simplified spatial coverage visualization
"""
import pyproj
pyproj.datadir.set_data_dir(
    "/projects/arra4944/arm64/software/miniforge/envs/bg2/share/proj"
)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
import argparse

# Set publication quality
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def plot_coverage(data_dir, region_name, output_path=None, style='scatter'):
    """
    Args:
        data_dir: Directory containing index.csv
        region_name: Name of the region
        output_path: Where to save the figure
        style: 'scatter', 'hexbin', or 'density'
    """
    
    # Load data
    index_path = os.path.join(data_dir, 'index.csv')
    if not os.path.exists(index_path):
        index_path = os.path.join(data_dir, 'index_filtered.csv')
    
    print(f"Loading data from {index_path}")
    df = pd.read_csv(index_path)
    print(f"Loaded {len(df)} points")
    
    # Create GeoDataFrame
    geometry = [Point(xy) for xy in zip(df.lon, df.lat)]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    # Convert to Web Mercator for basemap
    gdf = gdf.to_crs('EPSG:3857')
    
    # Determine figure size based on region
    region_lower = region_name.lower()
    if 'colorado' in region_lower:
        figsize = (10, 8)
        buffer_pct = 0.05  # 5% buffer
    elif 'mountain' in region_lower or 'west' in region_lower:
        figsize = (12, 10)
        buffer_pct = 0.08
    elif 'united' in region_lower or 'states' in region_lower:
        figsize = (14, 8)
        buffer_pct = 0.10
    else:
        figsize = (10, 10)
        buffer_pct = 0.08
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set bounds with buffer
    bounds = gdf.total_bounds
    x_buffer = (bounds[2] - bounds[0]) * buffer_pct
    y_buffer = (bounds[3] - bounds[1]) * buffer_pct
    
    ax.set_xlim(bounds[0] - x_buffer, bounds[2] + x_buffer)
    ax.set_ylim(bounds[1] - y_buffer, bounds[3] + y_buffer)
    
    # Try to add basemap with error handling
    try:
        # Try CartoDB Positron (light, clean basemap)
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.8)
    except:
        try:
            # Fallback to OSM
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.8)
        except:
            print("Warning: Could not load basemap. Continuing without it.")
    
    # Plot data based on style
    if style == 'scatter':
        gdf.plot(
            ax=ax,
            markersize=1,
            color='red',
            alpha=0.6,
            edgecolor='darkred',
            linewidth=0.1
        )
    
    elif style == 'hexbin':
        # Hexagonal binning
        x = gdf.geometry.x
        y = gdf.geometry.y
        
        hexbin = ax.hexbin(
            x, y,
            gridsize=50,
            cmap='YlOrRd',
            alpha=0.7,
            mincnt=1
        )
        
        # Add colorbar
        cbar = plt.colorbar(hexbin, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Sample count', rotation=270, labelpad=15)
    
    elif style == 'density':
        from scipy.stats import gaussian_kde
        
        x = gdf.geometry.x.values
        y = gdf.geometry.y.values
        
        # Calculate density
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)
        
        # Create grid
        xx, yy = np.mgrid[
            ax.get_xlim()[0]:ax.get_xlim()[1]:100j,
            ax.get_ylim()[0]:ax.get_ylim()[1]:100j
        ]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = np.reshape(kde(positions).T, xx.shape)
        
        # Plot contours
        contour = ax.contourf(xx, yy, density, levels=15, cmap='YlOrRd', alpha=0.6)
        plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04, label='Density')
    
    # Add title and annotations
    ax.set_title(f"Sentinel-2 Coverage: {region_name.replace('_', ' ').title()}", 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add sample count
    ax.text(
        0.02, 0.98,
        f"n = {len(df):,} samples",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Clean up axes
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    # Add grid
    ax.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")
    
    plt.show()
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Samples: {len(df):,}")
    print(f"  Lon range: [{df.lon.min():.3f}, {df.lon.max():.3f}]")
    print(f"  Lat range: [{df.lat.min():.3f}, {df.lat.max():.3f}]")
    
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description='Plot spatial coverage of Sentinel-2 dataset')
    parser.add_argument('data_dir', help='Directory containing index.csv')
    parser.add_argument('region', help='Region name')
    parser.add_argument('--output', help='Output filename', default=None)
    parser.add_argument('--style', choices=['scatter', 'hexbin', 'density'], 
                       default='scatter', help='Plot style')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f"{args.region}_coverage.png"
    
    plot_coverage(args.data_dir, args.region, args.output, args.style)


if __name__ == "__main__":
    main()