#!/usr/bin/env python3
"""
Sentinel-2 Imagery Download Script
===================================
Downloads multi-band Sentinel-2 imagery patches from specified regions.
Uses Natural Earth shapefiles for accurate boundaries.

Author: Arjun Rao
Date: 2025
"""
import pyproj
pyproj.datadir.set_data_dir(
    "/projects/arra4944/arm64/software/miniforge/envs/bg2/share/proj"
)

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import ee
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import zipfile
import io
import tempfile

# Import region definitions and sampling logic
from regions import RegionSampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentinel2_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Sentinel2Downloader:
    """Downloads Sentinel-2 imagery patches for any specified region."""
    
    # Sentinel-2 band configuration
    BAND_NAMES = [
        'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 
        'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'
    ]
    
    def __init__(self, 
                 region_name: str,
                 output_dir: str,
                 patch_size: int = 256,
                 target_resolution: int = 10,
                 n_samples: int = None,
                 cloud_threshold: int = None,
                 date_range: tuple = ('2023-01-01', '2024-12-31')):
        """
        Initialize the Sentinel-2 downloader.
        
        Parameters:
        -----------
        region_name : str
            Name of the region (from Natural Earth data)
        output_dir : str
            Directory to save the dataset
        patch_size : int
            Size of patches in pixels
        target_resolution : int
            Target resolution in meters
        n_samples : int
            Number of samples (None = use region default)
        cloud_threshold : int
            Max cloud coverage (None = use region default)
        date_range : tuple
            Date range for imagery
        """
        
        self.region_name = region_name
        self.output_dir = output_dir
        self.patch_size = patch_size
        self.target_resolution = target_resolution
        self.date_range = date_range
        
        # Initialize region sampler
        self.region_sampler = RegionSampler()
        
        # Get region configuration
        region_config = self.region_sampler.get_region_config(region_name)
        self.n_samples = n_samples or region_config['default_samples']
        self.cloud_threshold = cloud_threshold or region_config['default_cloud_threshold']
        
        # Create output directories
        self.images_dir = os.path.join(output_dir, 'images')
        self.metadata_dir = os.path.join(output_dir, 'metadata')
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Initialize Earth Engine
        self._initialize_ee()
        
        # Setup region of interest
        self.roi = self._setup_roi()
        
        # Sample points cache file
        self.sample_points_file = os.path.join(
            self.metadata_dir, 
            f'sample_points_{region_name.lower().replace(" ", "_")}.pkl'
        )
        
        logger.info(f"Initialized Sentinel-2 Downloader")
        logger.info(f"  Region: {region_name}")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"  Samples: {self.n_samples:,}")
        logger.info(f"  Cloud threshold: {self.cloud_threshold}%")
        logger.info(f"  Date range: {date_range[0]} to {date_range[1]}")
    
    def _initialize_ee(self):
        """Initialize Google Earth Engine."""
        try:
            ee.Initialize(project='sentinel-downloader-468517')
            test = ee.Number(1).getInfo()
            logger.info("Successfully initialized Earth Engine")
        except Exception as e:
            logger.warning(f"Default initialization failed: {e}")
            logger.info("Attempting authentication...")
            try:
                ee.Authenticate(auth_mode='notebook')
            except:
                ee.Authenticate()
            
            try:
                ee.Initialize(project='sentinel-downloader-468517')
            except:
                ee.Initialize()
            logger.info("Successfully authenticated and initialized Earth Engine")
    
    def _setup_roi(self) -> ee.Geometry:
        """Setup the region of interest using Natural Earth boundaries."""
        try:
            coords = self.region_sampler.get_region_bounds(self.region_name)
            roi = ee.Geometry.Polygon(coords)
            
            # Calculate area
            area = roi.area()
            area_km2 = area.divide(1e6).getInfo()
            logger.info(f"ROI area: approximately {area_km2:,.0f} km²")
            
            return roi
        except Exception as e:
            logger.error(f"Could not get ROI from Natural Earth data: {e}")
            # Fallback to a global extent
            return ee.Geometry.Polygon([
                [-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]
            ])
    
    def generate_sample_points(self, strategy: str = 'random') -> List[Dict]:
        """Generate or load cached sample points."""
        
        # Check cache
        if os.path.exists(self.sample_points_file):
            logger.info("Loading cached sample points...")
            with open(self.sample_points_file, 'rb') as f:
                points = pickle.load(f)
                logger.info(f"Loaded {len(points)} cached points")
                return points
        
        # Generate new points using Natural Earth boundaries
        points = self.region_sampler.generate_sample_points(
            self.region_name, 
            self.n_samples, 
            strategy
        )
        
        # Cache the points
        with open(self.sample_points_file, 'wb') as f:
            pickle.dump(points, f)
        
        return points
    
    def get_sentinel2_image(self, point: Dict) -> ee.Image:
        """Get a Sentinel-2 image for a specific point."""
        try:
            ee_point = ee.Geometry.Point([point['lon'], point['lat']])
            buffer_meters = (self.patch_size * self.target_resolution) / 2
            region = ee_point.buffer(buffer_meters).bounds()
            
            collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(region) \
                .filterDate(self.date_range[0], self.date_range[1]) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', self.cloud_threshold))
            
            image_count = collection.size().getInfo()
            
            if image_count == 0:
                logger.debug(f"No images for point {point['id']} at ({point['lon']:.4f}, {point['lat']:.4f})")
                return None
            
            # Get least cloudy image
            image = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
            
            # Verify bands
            try:
                band_info = image.bandNames().getInfo()
                available_bands = [b for b in self.BAND_NAMES if b in band_info]
                
                if not available_bands:
                    logger.warning(f"No valid bands for point {point['id']}")
                    return None
                
                return image.select(available_bands).clip(region)
                
            except Exception as e:
                logger.debug(f"Band verification failed: {e}")
                return image.select(self.BAND_NAMES).clip(region)
                
        except Exception as e:
            logger.error(f"Error getting image for point {point['id']}: {e}")
            return None
    
    def download_patch(self, point: Dict, overwrite: bool = False) -> Optional[str]:
        """Download a single patch."""
        
        output_filename = f"patch_{point['id']}.tif"
        output_path = os.path.join(self.images_dir, output_filename)
        
        # Check existing file
        if os.path.exists(output_path) and not overwrite:
            try:
                with rasterio.open(output_path) as src:
                    if src.count >= 11:  # At least 11 bands
                        return output_path
            except:
                pass
        
        try:
            # Get image
            image = self.get_sentinel2_image(point)
            if image is None:
                return None
            
            # Setup download region
            buffer_meters = (self.patch_size * self.target_resolution) / 2
            ee_point = ee.Geometry.Point([point['lon'], point['lat']])
            region = ee_point.buffer(buffer_meters).bounds()
            
            # Download parameters
            download_params = {
                'scale': self.target_resolution,
                'region': region,
                'fileFormat': 'GeoTIFF',
                'formatOptions': {'cloudOptimized': False}
            }
            
            # Get download URL
            url = image.getDownloadURL(download_params)
            
            # Download with retries
            response = None
            for attempt in range(3):
                try:
                    response = requests.get(url, timeout=180)
                    response.raise_for_status()
                    break
                except requests.exceptions.Timeout:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            
            content = response.content
            
            # Handle ZIP files
            if content.startswith(b'PK'):
                return self._process_zip_response(content, output_path, point)
            
            # Handle GeoTIFF
            elif content.startswith(b'II') or content.startswith(b'MM'):
                with open(output_path, 'wb') as f:
                    f.write(content)
                
                with rasterio.open(output_path) as src:
                    if src.count >= 11:
                        self._verify_and_resample(output_path)
                        return output_path
                    else:
                        os.remove(output_path)
                        return None
            
            else:
                logger.error(f"Unknown format for patch {point['id']}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading patch {point['id']}: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return None
    
    def _process_zip_response(self, zip_content: bytes, output_path: str, point: Dict) -> Optional[str]:
        """Process ZIP file containing individual band TIFFs."""
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract ZIP
                with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                    zf.extractall(temp_dir)
                
                # Find band files
                band_files = {}
                for filename in os.listdir(temp_dir):
                    if filename.endswith('.tif'):
                        for band in self.BAND_NAMES:
                            if f'.{band}.' in filename:
                                band_files[band] = os.path.join(temp_dir, filename)
                                break
                
                if not band_files:
                    logger.error(f"No band files in ZIP for patch {point['id']}")
                    return None
                
                # Read and combine bands
                arrays = []
                profile = None
                band_order = []
                
                for band_name in self.BAND_NAMES:
                    if band_name in band_files:
                        with rasterio.open(band_files[band_name]) as src:
                            if profile is None:
                                profile = src.profile.copy()
                            arrays.append(src.read(1))
                            band_order.append(band_name)
                
                if not arrays:
                    return None
                
                # Stack bands
                all_bands = np.stack(arrays, axis=0)
                
                # Update profile
                profile.update({
                    'count': len(arrays),
                    'width': self.patch_size,
                    'height': self.patch_size,
                    'compress': 'deflate'
                })
                
                # Save combined file
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(all_bands)
                    for i, band_name in enumerate(band_order, 1):
                        dst.set_band_description(i, band_name)
                
                self._verify_and_resample(output_path)
                return output_path
                
        except Exception as e:
            logger.error(f"Error processing ZIP for patch {point['id']}: {e}")
            return None
    
    def _verify_and_resample(self, filepath: str):
        """Verify and resample if needed."""
        
        try:
            with rasterio.open(filepath, 'r') as src:
                if src.width != self.patch_size or src.height != self.patch_size:
                    # Resample
                    data = src.read()
                    resampled = np.zeros((src.count, self.patch_size, self.patch_size), dtype=data.dtype)
                    
                    west, south, east, north = src.bounds
                    new_transform = from_bounds(west, south, east, north, 
                                              self.patch_size, self.patch_size)
                    
                    for i in range(src.count):
                        reproject(
                            source=data[i],
                            destination=resampled[i],
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=new_transform,
                            dst_crs=src.crs,
                            resampling=Resampling.bilinear
                        )
                    
                    profile = src.profile.copy()
                    profile.update({
                        'width': self.patch_size,
                        'height': self.patch_size,
                        'transform': new_transform
                    })
                    
                    temp_file = filepath + '.tmp'
                    with rasterio.open(temp_file, 'w', **profile) as dst:
                        dst.write(resampled)
                    
                    os.replace(temp_file, filepath)
                    
        except Exception as e:
            logger.error(f"Error verifying {filepath}: {e}")
    
    def download_dataset(self, n_workers: int = 4, batch_size: int = 100, 
                        sampling_strategy: str = 'random'):
        """Download the complete dataset."""
        
        # Limit workers to avoid connection pool issues
        n_workers = min(n_workers, 5)
        
        logger.info(f"Starting dataset download with {n_workers} workers...")
        
        # Generate sample points
        points = self.generate_sample_points(sampling_strategy)
        
        # Progress tracking
        csv_data = []
        failed_downloads = []
        
        pbar = tqdm(total=len(points), desc="Downloading patches")
        
        # Process in batches
        for batch_start in range(0, len(points), batch_size):
            batch_end = min(batch_start + batch_size, len(points))
            batch_points = points[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}")
            
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                future_to_point = {
                    executor.submit(self.download_patch, point): point 
                    for point in batch_points
                }
                
                for future in as_completed(future_to_point):
                    point = future_to_point[future]
                    
                    try:
                        result = future.result(timeout=300)
                        
                        if result:
                            csv_data.append({
                                'fn': f"patch_{point['id']}.tif",
                                'lon': point['lon'],
                                'lat': point['lat']
                            })
                        else:
                            failed_downloads.append(point)
                            
                    except Exception as e:
                        logger.error(f"Download failed for point {point['id']}: {e}")
                        failed_downloads.append(point)
                    
                    pbar.update(1)
            
            # Save intermediate results
            if csv_data:
                self._save_index_csv(csv_data)
            
            time.sleep(3)  # Rate limiting
        
        pbar.close()
        
        # Handle failures
        if failed_downloads:
            logger.warning(f"{len(failed_downloads)} downloads failed")
            self._save_failed_downloads(failed_downloads)
        
        # Final save
        self._save_index_csv(csv_data)
        self._generate_statistics(csv_data)
        
        logger.info(f"Download complete! {len(csv_data)} successful, {len(failed_downloads)} failed")
    
    def _save_index_csv(self, csv_data: List[Dict]):
        """Save index CSV."""
        csv_path = os.path.join(self.output_dir, 'index.csv')
        df = pd.DataFrame(csv_data)
        df = df[['fn', 'lon', 'lat']]
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved index.csv with {len(df)} entries")
    
    def _save_failed_downloads(self, failed_points: List[Dict]):
        """Save failed downloads list."""
        failed_file = os.path.join(self.metadata_dir, 'failed_downloads.json')
        with open(failed_file, 'w') as f:
            json.dump(failed_points, f, indent=2)
        logger.warning(f"Saved {len(failed_points)} failed downloads to {failed_file}")
    
    def _generate_statistics(self, csv_data: List[Dict]):
        """Generate dataset statistics."""
        
        if not csv_data:
            return
        
        stats = {
            'dataset_info': {
                'total_patches': len(csv_data),
                'region': self.region_name,
                'patch_size': f"{self.patch_size}x{self.patch_size}",
                'resolution_meters': self.target_resolution,
                'n_bands': len(self.BAND_NAMES),
                'band_names': self.BAND_NAMES,
                'date_range': self.date_range,
                'cloud_threshold': self.cloud_threshold
            },
            'spatial_coverage': {
                'min_lon': min(p['lon'] for p in csv_data),
                'max_lon': max(p['lon'] for p in csv_data),
                'min_lat': min(p['lat'] for p in csv_data),
                'max_lat': max(p['lat'] for p in csv_data)
            },
            'download_timestamp': datetime.now().isoformat()
        }
        
        stats_file = os.path.join(self.metadata_dir, 'dataset_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Region: {self.region_name}")
        print(f"Total patches: {stats['dataset_info']['total_patches']:,}")
        print(f"Patch size: {stats['dataset_info']['patch_size']} pixels")
        print(f"Resolution: {stats['dataset_info']['resolution_meters']}m")
        print(f"Spatial extent: ({stats['spatial_coverage']['min_lon']:.2f}, "
              f"{stats['spatial_coverage']['min_lat']:.2f}) to "
              f"({stats['spatial_coverage']['max_lon']:.2f}, "
              f"{stats['spatial_coverage']['max_lat']:.2f})")
        print("="*60)


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description='Download Sentinel-2 imagery dataset')
    parser.add_argument('region', type=str,
                      help='Region to download (use --list to see available regions)', nargs='?')
    parser.add_argument('--list', action='store_true',
                      help='List available regions')
    parser.add_argument('--output', type=str, default=None,
                      help='Output directory (default: auto-generated)')
    parser.add_argument('--samples', type=int, default=None,
                      help='Number of samples (default: region-specific)')
    parser.add_argument('--workers', type=int, default=5,
                      help='Number of parallel workers (max 5 for EE)')
    parser.add_argument('--batch', type=int, default=100,
                      help='Batch size')
    parser.add_argument('--cloud', type=int, default=None,
                      help='Max cloud coverage percentage')
    parser.add_argument('--strategy', type=str, default='random',
                      choices=['random', 'stratified'],
                      help='Sampling strategy')
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                      help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-12-31',
                      help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # List regions if requested
    if args.list:
        regions = RegionSampler.list_available_regions()
        print("\nAvailable regions:")
        print("="*40)
        for category, region_list in regions.items():
            print(f"\n{category.upper()}:")
            for region in sorted(region_list):
                print(f"  - {region}")
        sys.exit(0)
    
    # Normalize region name
    region_normalized = args.region.lower().replace(' ', '_').replace('-', '_')
    
    # Check if region exists
    if region_normalized not in RegionSampler.REGION_CONFIG:
        print(f"Error: '{args.region}' is not a valid region.")
        print("Use --list to see available regions")
        sys.exit(1)
    
    # Setup output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = f'/scratch/local/arra4944_images/s2_{region_normalized}'
    
    # Limit workers
    if args.workers > 5:
        logger.warning("Earth Engine works best with <= 5 workers. Reducing to 5.")
        args.workers = 5
    
    # Create downloader
    downloader = Sentinel2Downloader(
        region_name=region_normalized,
        output_dir=output_dir,
        n_samples=args.samples,
        cloud_threshold=args.cloud,
        date_range=(args.start_date, args.end_date)
    )
    
    # Start download
    try:
        downloader.download_dataset(
            n_workers=args.workers,
            batch_size=args.batch,
            sampling_strategy=args.strategy
        )
        
        logger.info(f"Dataset download completed for {args.region}!")
        
    except KeyboardInterrupt:
        logger.warning("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


if __name__ == "__main__":
    main()