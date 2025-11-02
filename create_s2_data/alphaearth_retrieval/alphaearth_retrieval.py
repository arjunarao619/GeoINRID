#!/usr/bin/env python3
"""
AlphaEarth Embeddings Extraction Script with Parallel Processing
================================================================
Extracts Google DeepMind's AlphaEarth embeddings for specified lat/lon coordinates.
These are 64-dimensional feature vectors at 10-meter resolution.

Author: Arjun Rao
Date: 2025
"""

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import warnings
warnings.filterwarnings('ignore')

import ee
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

# Try to import cartopy for nice map projections
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    logging.warning("Cartopy not found. Using simple matplotlib for maps.")

import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alphaearth_extraction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AlphaEarthExtractor:
    """Extracts AlphaEarth embeddings for specified coordinates with parallel processing."""
    
    # AlphaEarth band names (64 dimensions)
    BAND_NAMES = [f'A{i:02d}' for i in range(1, 65)]  # A01 through A64
    
    def __init__(self, 
                 index_csv_path: str,
                 output_dir: str = './alphaearth_embeddings',
                 year: int = 2024,
                 buffer_meters: int = 10,
                 n_workers: int = 10,
                 auto_dir: bool = True):
        """
        Initialize the AlphaEarth extractor.
        
        Parameters:
        -----------
        index_csv_path : str
            Path to CSV file with fn, lon, lat columns
        output_dir : str
            Base directory to save extracted embeddings
        year : int
            Year for which to extract embeddings (2017-2024 available)
        buffer_meters : int
            Buffer radius around point in meters (10m = single pixel)
        n_workers : int
            Number of parallel workers for extraction
        auto_dir : bool
            If True, append buffer size to output directory name
        """
        
        self.index_csv_path = index_csv_path
        self.year = year
        self.buffer_meters = buffer_meters
        self.n_workers = n_workers
        self.extraction_lock = threading.Lock()
        
        # Automatically organize directory by buffer size
        if auto_dir:
            self.output_dir = f"{output_dir}_buffer{buffer_meters}m"
        else:
            self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        self.embeddings_dir = os.path.join(self.output_dir, 'embeddings')
        self.metadata_dir = os.path.join(self.output_dir, 'metadata')
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Initialize tracking lists for visualization
        self.successful_points = []
        self.failed_points = []
        
        # Initialize Earth Engine
        self._initialize_ee()
        
        # Load the AlphaEarth collection
        self.collection = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
        
        logger.info(f"Initialized AlphaEarth Extractor")
        logger.info(f"  Input CSV: {index_csv_path}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Year: {year}")
        logger.info(f"  Buffer: {buffer_meters}m")
        logger.info(f"  Workers: {n_workers}")
    
    def _initialize_ee(self):
        """Initialize Google Earth Engine."""
        try:
            # Try with your project ID first
            ee.Initialize(project='sentinel-downloader-468517')
            test = ee.Number(1).getInfo()
            logger.info("Successfully initialized Earth Engine with project")
        except Exception as e:
            logger.warning(f"Project initialization failed: {e}")
            logger.info("Attempting default initialization...")
            try:
                ee.Initialize()
                logger.info("Successfully initialized Earth Engine (default)")
            except:
                logger.info("Attempting authentication...")
                ee.Authenticate()
                ee.Initialize()
                logger.info("Successfully authenticated and initialized Earth Engine")
    
    def verify_coordinates(self, coordinates_df: pd.DataFrame):
        """Verify coordinate columns and show sample."""
        print("\n" + "="*60)
        print("COORDINATE VERIFICATION")
        print("="*60)
        
        print(f"CSV columns: {list(coordinates_df.columns)}")
        
        print("\nFirst 5 coordinates:")
        for idx, row in coordinates_df.head(5).iterrows():
            print(f"  {row['fn']}: lon={row['lon']:.4f}, lat={row['lat']:.4f}")
        
        print("\nCoordinate ranges:")
        print(f"  Longitude: {coordinates_df['lon'].min():.2f} to {coordinates_df['lon'].max():.2f}")
        print(f"  Latitude: {coordinates_df['lat'].min():.2f} to {coordinates_df['lat'].max():.2f}")
        
        if (coordinates_df['lat'].abs() > 90).any():
            print("\nWARNING: Some 'lat' values > 90, coordinates might be swapped!")
        if (coordinates_df['lon'].abs() > 180).any():
            print("\nWARNING: Some 'lon' values > 180, coordinates might be swapped!")
        
        print("="*60)
    
    def load_coordinates(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load coordinates from the index CSV file."""
        df = pd.read_csv(self.index_csv_path)
        
        self.verify_coordinates(df)
        
        if limit and limit > 0:
            df = df.head(limit)
            logger.info(f"Limited to first {limit} coordinates for testing")
        
        logger.info(f"Loaded {len(df)} coordinates from {self.index_csv_path}")
        return df
    
    def get_embedding_image(self, year: int) -> ee.Image:
        """Get the AlphaEarth embedding image for a specific year."""
        start_date = ee.Date.fromYMD(year, 1, 1)
        end_date = start_date.advance(1, 'year')
        
        year_collection = self.collection.filterDate(start_date, end_date)
        embedding_image = year_collection.mosaic()
        
        return embedding_image
    
    def extract_point_embedding(self, 
                               lon: float, 
                               lat: float, 
                               image: ee.Image) -> Optional[Dict]:
        """Extract embedding values for a single point."""
        try:
            point = ee.Geometry.Point([lon, lat])
            
            if self.buffer_meters > 0:
                region = point.buffer(self.buffer_meters)
                values = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region,
                    scale=10,
                    maxPixels=1e9
                )
            else:
                values = image.sample(
                    region=point,
                    scale=10,
                    geometries=False
                ).first()
                
                if values is not None:
                    values = values.toDictionary()
            
            result = values.getInfo()
            
            if result and any(v is not None for v in result.values()):
                embedding_vector = []
                for band in self.BAND_NAMES:
                    val = result.get(band, np.nan)
                    embedding_vector.append(val if val is not None else np.nan)
                
                return {
                    'lon': lon,
                    'lat': lat,
                    'embedding': np.array(embedding_vector, dtype=np.float32),
                    'valid_dims': sum(1 for v in embedding_vector if not np.isnan(v))
                }
            else:
                logger.warning(f"No data available at ({lon:.4f}, {lat:.4f})")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting embedding at ({lon:.4f}, {lat:.4f}): {e}")
            return None
    
    def _extract_point_wrapper(self, row, idx, image):
        """Wrapper function for parallel extraction."""
        embedding_data = self.extract_point_embedding(
            row['lon'], 
            row['lat'], 
            image
        )
        
        if embedding_data:
            embedding_data['fn'] = row.get('fn', f'point_{idx}')
            embedding_data['idx'] = idx
        
        return embedding_data
    
    def extract_batch_embeddings_parallel(self, 
                                         coordinates_df: pd.DataFrame,
                                         n_workers: int = None) -> List[Dict]:
        """
        Parallel extraction of embeddings using ThreadPoolExecutor.
        
        Parameters:
        -----------
        coordinates_df : pd.DataFrame
            DataFrame with lon, lat columns
        n_workers : int
            Number of parallel workers
        """
        if n_workers is None:
            n_workers = self.n_workers
            
        logger.info(f"Loading AlphaEarth embeddings for year {self.year}...")
        embedding_image = self.get_embedding_image(self.year)
        
        results = []
        failed_extractions = []
        
        self.successful_points = []
        self.failed_points = []
        
        start_time = time.time()
        logger.info(f"Starting parallel extraction with {n_workers} workers...")
        
        extract_func = partial(self._extract_point_wrapper, image=embedding_image)
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_row = {}
            for idx, row in coordinates_df.iterrows():
                future = executor.submit(extract_func, row, idx)
                future_to_row[future] = (row, idx)
            
            with tqdm(total=len(coordinates_df), desc="Extracting embeddings") as pbar:
                for future in as_completed(future_to_row):
                    row, idx = future_to_row[future]
                    
                    try:
                        embedding_data = future.result(timeout=30)
                        
                        if embedding_data:
                            with self.extraction_lock:
                                results.append(embedding_data)
                                self.successful_points.append({
                                    'lon': row['lon'],
                                    'lat': row['lat'],
                                    'fn': row.get('fn', f'point_{idx}')
                                })
                        else:
                            with self.extraction_lock:
                                failed_extractions.append({
                                    'fn': row.get('fn', f'point_{idx}'),
                                    'lon': row['lon'],
                                    'lat': row['lat'],
                                    'reason': 'No data available'
                                })
                                self.failed_points.append({
                                    'lon': row['lon'],
                                    'lat': row['lat'],
                                    'fn': row.get('fn', f'point_{idx}')
                                })
                                
                    except Exception as e:
                        logger.error(f"Error processing point {idx}: {e}")
                        with self.extraction_lock:
                            failed_extractions.append({
                                'fn': row.get('fn', f'point_{idx}'),
                                'lon': row['lon'],
                                'lat': row['lat'],
                                'reason': str(e)
                            })
                            self.failed_points.append({
                                'lon': row['lon'],
                                'lat': row['lat'],
                                'fn': row.get('fn', f'point_{idx}')
                            })
                    
                    pbar.update(1)
        
        elapsed = time.time() - start_time
        rate = len(results) / elapsed if elapsed > 0 else 0
        logger.info(f"Completed in {elapsed:.1f} seconds ({rate:.1f} points/second)")
        logger.info(f"Successfully extracted {len(results)} embeddings")
        logger.info(f"Failed to extract {len(failed_extractions)} embeddings")
        
        if failed_extractions:
            self._save_failed_extractions(failed_extractions)
        
        return results
    
    def plot_sampling_map(self, save_path: Optional[str] = None):
        """Create a map showing successful and failed sampling locations."""
        
        # Include buffer info in filename
        buffer_str = f"buffer{self.buffer_meters}m"
        
        if save_path is None:
            save_path = os.path.join(self.plots_dir, 
                                    f'sampling_map_{self.year}_{buffer_str}.png')
        
        total_points = len(self.successful_points) + len(self.failed_points)
        
        # Auto-adjust marker size based on number of points
        if total_points < 100:
            marker_size = 50
            line_width = 1.5
        elif total_points < 1000:
            marker_size = 10
            line_width = 0.5
        elif total_points < 10000:
            marker_size = 3
            line_width = 0.3
        else:
            marker_size = 1
            line_width = 0.1
        
        if HAS_CARTOPY:
            self._plot_with_cartopy(save_path, marker_size, line_width)
        else:
            self._plot_simple_map(save_path, marker_size, line_width)
        
        logger.info(f"Saved sampling map to {save_path}")
    
    def _plot_with_cartopy(self, save_path: str, marker_size: float = 3, line_width: float = 0.3):
        """Create a professional map using cartopy."""
        fig = plt.figure(figsize=(16, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3)
        
        if self.successful_points:
            success_lons = [p['lon'] for p in self.successful_points]
            success_lats = [p['lat'] for p in self.successful_points]
            ax.scatter(success_lons, success_lats, 
                      c='green', s=marker_size, marker='o', 
                      transform=ccrs.PlateCarree(),
                      label=f'Success ({len(self.successful_points)})',
                      alpha=0.6, edgecolors='darkgreen', linewidths=line_width)
        
        if self.failed_points:
            fail_lons = [p['lon'] for p in self.failed_points]
            fail_lats = [p['lat'] for p in self.failed_points]
            ax.scatter(fail_lons, fail_lats, 
                      c='red', s=marker_size*1.5, marker='x', 
                      transform=ccrs.PlateCarree(),
                      label=f'Failed ({len(self.failed_points)})',
                      alpha=0.9, linewidths=line_width*2)
        
        all_lons = []
        all_lats = []
        if self.successful_points:
            all_lons.extend([p['lon'] for p in self.successful_points])
            all_lats.extend([p['lat'] for p in self.successful_points])
        if self.failed_points:
            all_lons.extend([p['lon'] for p in self.failed_points])
            all_lats.extend([p['lat'] for p in self.failed_points])
        
        if all_lons and all_lats:
            lon_buffer = (max(all_lons) - min(all_lons)) * 0.1 or 5
            lat_buffer = (max(all_lats) - min(all_lats)) * 0.1 or 5
            ax.set_extent([min(all_lons) - lon_buffer, max(all_lons) + lon_buffer,
                          min(all_lats) - lat_buffer, max(all_lats) + lat_buffer],
                         crs=ccrs.PlateCarree())
        
        plt.title(f'AlphaEarth Embedding Sampling Locations - Year {self.year}\n'
                 f'Buffer: {self.buffer_meters}m | Resolution: 10m | Workers: {self.n_workers}',
                 fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_simple_map(self, save_path: str, marker_size: float = 3, line_width: float = 0.3):
        """Create a simple map using only matplotlib."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_facecolor('#E6F3FF')
        
        if self.successful_points:
            success_lons = [p['lon'] for p in self.successful_points]
            success_lats = [p['lat'] for p in self.successful_points]
            ax.scatter(success_lons, success_lats, 
                      c='green', s=marker_size, marker='o', 
                      label=f'Success ({len(self.successful_points)})',
                      alpha=0.6, edgecolors='darkgreen', linewidths=line_width)
        
        if self.failed_points:
            fail_lons = [p['lon'] for p in self.failed_points]
            fail_lats = [p['lat'] for p in self.failed_points]
            ax.scatter(fail_lons, fail_lats, 
                      c='red', s=marker_size*1.5, marker='x', 
                      label=f'Failed ({len(self.failed_points)})',
                      alpha=0.9, linewidths=line_width*2)
        
        all_lons = []
        all_lats = []
        if self.successful_points:
            all_lons.extend([p['lon'] for p in self.successful_points])
            all_lats.extend([p['lat'] for p in self.successful_points])
        if self.failed_points:
            all_lons.extend([p['lon'] for p in self.failed_points])
            all_lats.extend([p['lat'] for p in self.failed_points])
        
        if all_lons and all_lats:
            lon_buffer = (max(all_lons) - min(all_lons)) * 0.1 or 10
            lat_buffer = (max(all_lats) - min(all_lats)) * 0.1 or 10
            ax.set_xlim(min(all_lons) - lon_buffer, max(all_lons) + lon_buffer)
            ax.set_ylim(min(all_lats) - lat_buffer, max(all_lats) + lat_buffer)
        
        plt.title(f'AlphaEarth Embedding Sampling Locations - Year {self.year}\n'
                 f'Buffer: {self.buffer_meters}m | Resolution: 10m | Workers: {self.n_workers}',
                 fontsize=14, fontweight='bold')
        
        stats_text = f'Total Sampled: {len(self.successful_points) + len(self.failed_points)}\n'
        if (len(self.successful_points) + len(self.failed_points)) > 0:
            stats_text += f'Success Rate: {len(self.successful_points)/(len(self.successful_points) + len(self.failed_points))*100:.1f}%'
        ax.text(0.02, 0.98, stats_text, 
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_embedding_heatmap(self, embeddings_data: List[Dict], save_path: Optional[str] = None):
        """Create a heatmap visualization of the embeddings."""
        if not embeddings_data:
            logger.warning("No embeddings to plot")
            return
        
        # Include buffer info in filename
        buffer_str = f"buffer{self.buffer_meters}m"
        
        if save_path is None:
            save_path = os.path.join(self.plots_dir, 
                                    f'embedding_heatmap_{self.year}_{buffer_str}.png')
        
        embeddings_array = np.array([d['embedding'] for d in embeddings_data])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        im1 = ax1.imshow(embeddings_array.T, aspect='auto', cmap='viridis')
        ax1.set_xlabel('Sample Index', fontsize=12)
        ax1.set_ylabel('Embedding Dimension', fontsize=12)
        ax1.set_title(f'AlphaEarth Embeddings Heatmap (Buffer: {self.buffer_meters}m)\n'
                     f'({embeddings_array.shape[0]} samples × 64 dimensions)', 
                      fontsize=14)
        plt.colorbar(im1, ax=ax1, label='Embedding Value')
        
        mean_per_dim = np.nanmean(embeddings_array, axis=0)
        std_per_dim = np.nanstd(embeddings_array, axis=0)
        
        x_dims = np.arange(64)
        ax2.errorbar(x_dims, mean_per_dim, yerr=std_per_dim, 
                    fmt='o-', markersize=3, capsize=2, alpha=0.7)
        ax2.set_xlabel('Embedding Dimension', fontsize=12)
        ax2.set_ylabel('Value', fontsize=12)
        ax2.set_title(f'Mean ± Std per Dimension (Buffer: {self.buffer_meters}m)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(np.arange(0, 64, 8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved embedding heatmap to {save_path}")
    
    def save_embeddings(self, embeddings_data: List[Dict]):
        """Save extracted embeddings in multiple formats with buffer info."""
        if not embeddings_data:
            logger.warning("No embeddings to save")
            return
        
        # Include buffer info in all filenames
        buffer_str = f"buffer{self.buffer_meters}m"
        
        # Save numpy array
        embeddings_array = np.array([d['embedding'] for d in embeddings_data])
        np_path = os.path.join(self.embeddings_dir, 
                              f'embeddings_{self.year}_{buffer_str}.npy')
        np.save(np_path, embeddings_array)
        logger.info(f"Saved embeddings array: {embeddings_array.shape} to {np_path}")
        
        # Save pickle with full data
        pickle_path = os.path.join(self.embeddings_dir, 
                                  f'embeddings_full_{self.year}_{buffer_str}.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(embeddings_data, f)
        logger.info(f"Saved full embeddings data to {pickle_path}")
        
        # Save CSV index
        csv_data = []
        for data in embeddings_data:
            csv_data.append({
                'fn': data['fn'],
                'lon': data['lon'],
                'lat': data['lat'],
                'buffer_meters': self.buffer_meters,
                'valid_dims': data['valid_dims'],
                'embedding_mean': np.nanmean(data['embedding']),
                'embedding_std': np.nanstd(data['embedding']),
                'embedding_min': np.nanmin(data['embedding']),
                'embedding_max': np.nanmax(data['embedding'])
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_path = os.path.join(self.output_dir, 
                               f'embeddings_index_{self.year}_{buffer_str}.csv')
        csv_df.to_csv(csv_path, index=False)
        logger.info(f"Saved embeddings index to {csv_path}")
        
        self._generate_statistics(embeddings_data)
        self.plot_sampling_map()
        self.plot_embedding_heatmap(embeddings_data)
    
    def _save_failed_extractions(self, failed: List[Dict]):
        """Save list of failed extractions."""
        buffer_str = f"buffer{self.buffer_meters}m"
        failed_path = os.path.join(self.metadata_dir, 
                                  f'failed_extractions_{buffer_str}.json')
        with open(failed_path, 'w') as f:
            json.dump(failed, f, indent=2)
        logger.info(f"Saved {len(failed)} failed extractions to {failed_path}")
    
    def _generate_statistics(self, embeddings_data: List[Dict]):
        """Generate statistics about the extracted embeddings."""
        embeddings_array = np.array([d['embedding'] for d in embeddings_data])
        
        total_attempted = len(self.successful_points) + len(self.failed_points)
        success_rate = (len(self.successful_points) / total_attempted * 100) if total_attempted > 0 else 0
        
        stats = {
            'extraction_info': {
                'total_points': len(embeddings_data),
                'successful_extractions': len(self.successful_points),
                'failed_extractions': len(self.failed_points),
                'success_rate': success_rate,
                'year': self.year,
                'embedding_dimensions': 64,
                'buffer_meters': self.buffer_meters,
                'n_workers': self.n_workers,
                'extraction_timestamp': datetime.now().isoformat()
            },
            'spatial_coverage': {
                'min_lon': min(d['lon'] for d in embeddings_data) if embeddings_data else None,
                'max_lon': max(d['lon'] for d in embeddings_data) if embeddings_data else None,
                'min_lat': min(d['lat'] for d in embeddings_data) if embeddings_data else None,
                'max_lat': max(d['lat'] for d in embeddings_data) if embeddings_data else None
            },
            'embedding_statistics': {
                'mean_per_dim': np.nanmean(embeddings_array, axis=0).tolist() if embeddings_data else [],
                'std_per_dim': np.nanstd(embeddings_array, axis=0).tolist() if embeddings_data else [],
                'min_per_dim': np.nanmin(embeddings_array, axis=0).tolist() if embeddings_data else [],
                'max_per_dim': np.nanmax(embeddings_array, axis=0).tolist() if embeddings_data else [],
                'global_mean': float(np.nanmean(embeddings_array)) if embeddings_data else None,
                'global_std': float(np.nanstd(embeddings_array)) if embeddings_data else None,
                'valid_pixels': int(np.sum(~np.isnan(embeddings_array))) if embeddings_data else 0
            }
        }
        
        buffer_str = f"buffer{self.buffer_meters}m"
        stats_path = os.path.join(self.metadata_dir, 
                                 f'extraction_statistics_{self.year}_{buffer_str}.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("\n" + "="*60)
        print("ALPHAEARTH EXTRACTION SUMMARY")
        print("="*60)
        print(f"Year: {self.year}")
        print(f"Buffer: {self.buffer_meters}m")
        print(f"Workers: {self.n_workers}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Total points attempted: {total_attempted}")
        print(f"Successful extractions: {len(embeddings_data)} ({success_rate:.1f}%)")
        print(f"Failed extractions: {len(self.failed_points)}")
        if embeddings_data:
            print(f"Embedding dimensions: 64")
            print(f"Embedding shape: {embeddings_array.shape}")
            print(f"Spatial extent: ({stats['spatial_coverage']['min_lon']:.2f}, "
                  f"{stats['spatial_coverage']['min_lat']:.2f}) to "
                  f"({stats['spatial_coverage']['max_lon']:.2f}, "
                  f"{stats['spatial_coverage']['max_lat']:.2f})")
            print(f"Global embedding mean: {stats['embedding_statistics']['global_mean']:.4f}")
            print(f"Global embedding std: {stats['embedding_statistics']['global_std']:.4f}")
        print("="*60)
    
    def run(self, limit: Optional[int] = None):
        """Run the complete extraction pipeline."""
        logger.info("Starting AlphaEarth embedding extraction...")
        logger.info(f"Buffer size: {self.buffer_meters}m")
        
        coordinates_df = self.load_coordinates(limit=limit)
        
        embeddings_data = self.extract_batch_embeddings_parallel(
            coordinates_df,
            n_workers=self.n_workers
        )
        
        if embeddings_data:
            self.save_embeddings(embeddings_data)
            logger.info("Extraction complete!")
            logger.info(f"Results saved to: {self.output_dir}")
        else:
            logger.error("No embeddings were successfully extracted")
            if self.failed_points:
                self.plot_sampling_map()


def main():
    INDEX_CSV_PATH = '/scratch/local/arra4944_images/s2_100k/index.csv'
    BASE_OUTPUT_DIR = '/scratch/local/arra4944_images/alphaearth_embeddings'
    YEAR = 2024
    LIMIT = 20
    WORKERS = 10
    BUFFER = 10
    

    
    parser = argparse.ArgumentParser(description='Extract AlphaEarth embeddings with parallel processing')
    parser.add_argument('--csv', type=str, default=INDEX_CSV_PATH,
                      help='Path to index CSV file')
    parser.add_argument('--output', type=str, default=BASE_OUTPUT_DIR,
                      help='Base output directory (buffer will be appended)')
    parser.add_argument('--year', type=int, default=YEAR,
                      help='Year for embeddings (2017-2024)')
    parser.add_argument('--limit', type=int, default=LIMIT,
                      help='Limit number of points to process (0 for all)')
    parser.add_argument('--buffer', type=int, default=BUFFER,
                      help='Buffer radius in meters')
    parser.add_argument('--workers', type=int, default=WORKERS,
                      help='Number of parallel workers (1-50, default: 10)')
    parser.add_argument('--no-auto-dir', action='store_true',
                      help='Disable automatic directory naming by buffer')
    
    args = parser.parse_args()
    
    # Handle limit=0 as None (process all)
    limit = args.limit if args.limit > 0 else None
    
    extractor = AlphaEarthExtractor(
        index_csv_path=args.csv,
        output_dir=args.output,
        year=args.year,
        buffer_meters=args.buffer,
        n_workers=args.workers,
        auto_dir=not args.no_auto_dir  # Auto-dir by default
    )
    
    try:
        extractor.run(limit=limit)
    except KeyboardInterrupt:
        logger.warning("Extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise


if __name__ == "__main__":
    main()