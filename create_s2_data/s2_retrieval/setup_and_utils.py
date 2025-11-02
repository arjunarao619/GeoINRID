#!/usr/bin/env python3
"""
Setup and Utility Scripts for Sentinel-2 Dataset Creation
==========================================================
This file contains setup, validation, and utility functions.
"""

# ============================================================================
# 1. SETUP SCRIPT - setup_environment.py
# ============================================================================

import subprocess
import sys
import os

def setup_environment():
    """Setup the Python environment for Sentinel-2 downloads."""
    
    print("Setting up Sentinel-2 Download Environment...")
    print("=" * 50)
    
    # Required packages
    packages = [
        'earthengine-api',
        'rasterio',
        'geopandas',
        'shapely',
        'requests',
        'tqdm',
        'pandas',
        'numpy',
        'matplotlib',
        'Pillow'
    ]
    
    print("Installing required packages...")
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    
    print("\n✓ All packages installed successfully!")
    
    # Setup Earth Engine authentication
    print("\n" + "=" * 50)
    print("Earth Engine Authentication")
    print("=" * 50)
    print("You need to authenticate with Google Earth Engine.")
    print("Run the following command and follow the instructions:")
    print("\n    earthengine authenticate\n")
    
    return True


# ============================================================================
# 2. DATASET VALIDATOR - validate_dataset.py
# ============================================================================

import rasterio
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import json

class DatasetValidator:
    """Validate the downloaded Sentinel-2 dataset."""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / 'images'
        self.index_file = self.dataset_dir / 'index.csv'
        self.validation_report = {}
        
    def validate_structure(self) -> bool:
        """Validate dataset directory structure."""
        print("Validating dataset structure...")
        
        errors = []
        
        # Check directories
        if not self.dataset_dir.exists():
            errors.append(f"Dataset directory not found: {self.dataset_dir}")
        
        if not self.images_dir.exists():
            errors.append(f"Images directory not found: {self.images_dir}")
            
        if not self.index_file.exists():
            errors.append(f"Index file not found: {self.index_file}")
        
        self.validation_report['structure_errors'] = errors
        
        if errors:
            for error in errors:
                print(f"  ✗ {error}")
            return False
        
        print("  ✓ Directory structure valid")
        return True
    
    def validate_index(self) -> Tuple[bool, pd.DataFrame]:
        """Validate the index.csv file."""
        print("Validating index.csv...")
        
        try:
            df = pd.read_csv(self.index_file)
            
            # Check required columns
            required_columns = ['fn', 'lon', 'lat']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"  ✗ Missing columns: {missing_columns}")
                return False, None
            
            # Check for duplicates
            duplicates = df[df.duplicated(subset=['fn'])]
            if not duplicates.empty:
                print(f"  ✗ Found {len(duplicates)} duplicate filenames")
                self.validation_report['duplicate_files'] = duplicates['fn'].tolist()
            
            # Validate coordinates
            invalid_coords = df[
                (df['lon'] < -180) | (df['lon'] > 180) |
                (df['lat'] < -90) | (df['lat'] > 90)
            ]
            
            if not invalid_coords.empty:
                print(f"  ✗ Found {len(invalid_coords)} invalid coordinates")
                self.validation_report['invalid_coordinates'] = invalid_coords.to_dict('records')
            
            print(f"  ✓ Index contains {len(df)} entries")
            self.validation_report['total_entries'] = len(df)
            
            return True, df
            
        except Exception as e:
            print(f"  ✗ Error reading index.csv: {e}")
            return False, None
    
    def validate_images(self, df: pd.DataFrame, sample_size: int = 100) -> bool:
        """Validate a sample of images."""
        print(f"Validating images (sampling {sample_size})...")
        
        # Sample images to check
        sample_df = df.sample(min(sample_size, len(df)))
        
        invalid_images = []
        missing_images = []
        band_issues = []
        size_issues = []
        
        for _, row in sample_df.iterrows():
            image_path = self.images_dir / row['fn']
            
            if not image_path.exists():
                missing_images.append(row['fn'])
                continue
            
            try:
                with rasterio.open(image_path) as src:
                    # Check bands
                    if src.count != 13:
                        band_issues.append({
                            'file': row['fn'],
                            'bands': src.count,
                            'expected': 13
                        })
                    
                    # Check dimensions
                    if src.width != 256 or src.height != 256:
                        size_issues.append({
                            'file': row['fn'],
                            'size': f"{src.width}x{src.height}",
                            'expected': "256x256"
                        })
                    
                    # Check for valid data
                    data = src.read(1)
                    if np.all(data == 0) or np.all(np.isnan(data)):
                        invalid_images.append(row['fn'])
                        
            except Exception as e:
                invalid_images.append({'file': row['fn'], 'error': str(e)})
        
        # Report results
        self.validation_report['missing_images'] = missing_images
        self.validation_report['invalid_images'] = invalid_images
        self.validation_report['band_issues'] = band_issues
        self.validation_report['size_issues'] = size_issues
        
        if missing_images:
            print(f"  ✗ {len(missing_images)} missing images")
        if invalid_images:
            print(f"  ✗ {len(invalid_images)} invalid images")
        if band_issues:
            print(f"  ✗ {len(band_issues)} images with incorrect bands")
        if size_issues:
            print(f"  ✗ {len(size_issues)} images with incorrect dimensions")
        
        if not (missing_images or invalid_images or band_issues or size_issues):
            print(f"  ✓ All sampled images valid")
            return True
        
        return False
    
    def generate_report(self, output_file: str = 'validation_report.json'):
        """Generate validation report."""
        
        report_path = self.dataset_dir / 'metadata' / output_file
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.validation_report, f, indent=2)
        
        print(f"\nValidation report saved to: {report_path}")
        
        # Print summary
        print("\n" + "=" * 50)
        print("VALIDATION SUMMARY")
        print("=" * 50)
        
        if self.validation_report.get('structure_errors'):
            print(f"Structure Errors: {len(self.validation_report['structure_errors'])}")
        
        print(f"Total Entries: {self.validation_report.get('total_entries', 0)}")
        
        if self.validation_report.get('missing_images'):
            print(f"Missing Images: {len(self.validation_report['missing_images'])}")
            
        if self.validation_report.get('invalid_images'):
            print(f"Invalid Images: {len(self.validation_report['invalid_images'])}")
            
        print("=" * 50)
    
    def run_validation(self):
        """Run complete validation."""
        print("Starting dataset validation...")
        print("=" * 50)
        
        # Validate structure
        if not self.validate_structure():
            self.generate_report()
            return False
        
        # Validate index
        valid_index, df = self.validate_index()
        if not valid_index or df is None:
            self.generate_report()
            return False
        
        # Validate images
        self.validate_images(df)
        
        # Generate report
        self.generate_report()
        
        return True


# ============================================================================
# 3. VISUALIZATION UTILITY - visualize_samples.py
# ============================================================================

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def visualize_samples(dataset_dir: str, n_samples: int = 9):
    """
    Visualize sample images from the dataset.
    
    Creates a grid showing RGB composites and false color composites.
    """
    
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / 'images'
    index_file = dataset_path / 'index.csv'
    
    # Read index
    df = pd.read_csv(index_file)
    sample_df = df.sample(min(n_samples, len(df)))
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        if idx >= n_samples:
            break
            
        image_path = images_dir / row['fn']
        
        if not image_path.exists():
            axes[idx].text(0.5, 0.5, 'Missing', ha='center', va='center')
            axes[idx].set_title(f"Missing: {row['fn']}")
            continue
        
        try:
            with rasterio.open(image_path) as src:
                # Read RGB bands (B4, B3, B2 for Sentinel-2)
                red = src.read(4)    # B4
                green = src.read(3)  # B3  
                blue = src.read(2)   # B2
                
                # Normalize and stack
                rgb = np.dstack([
                    normalize_band(red),
                    normalize_band(green),
                    normalize_band(blue)
                ])
                
                # Display
                axes[idx].imshow(rgb)
                axes[idx].set_title(f"Patch {row.name}\n({row['lon']:.2f}, {row['lat']:.2f})")
                axes[idx].axis('off')
                
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Error: {str(e)[:20]}', 
                          ha='center', va='center')
            axes[idx].set_title(f"Error: {row['fn']}")
    
    plt.suptitle(f"Sample Images from {dataset_dir}", fontsize=16)
    plt.tight_layout()
    
    # Save figure
    output_file = dataset_path / 'sample_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    
    plt.show()

def normalize_band(band: np.ndarray, percentile_min: int = 2, percentile_max: int = 98) -> np.ndarray:
    """Normalize band values for visualization."""
    vmin = np.percentile(band, percentile_min)
    vmax = np.percentile(band, percentile_max)
    normalized = (band - vmin) / (vmax - vmin)
    return np.clip(normalized, 0, 1)


# ============================================================================
# 4. DATASET STATISTICS - compute_statistics.py
# ============================================================================

def compute_band_statistics(dataset_dir: str, sample_size: int = 1000):
    """
    Compute statistics for each band across the dataset.
    
    This is useful for normalization and quality checks.
    """
    
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / 'images'
    index_file = dataset_path / 'index.csv'
    
    # Read index
    df = pd.read_csv(index_file)
    sample_df = df.sample(min(sample_size, len(df)))
    
    # Initialize statistics
    band_names = [f'B{i}' for i in range(1, 13)] + ['B8A']
    stats = {band: {'mean': [], 'std': [], 'min': [], 'max': []} 
             for band in band_names}
    
    print(f"Computing band statistics from {sample_size} samples...")
    
    for _, row in sample_df.iterrows():
        image_path = images_dir / row['fn']
        
        if not image_path.exists():
            continue
            
        try:
            with rasterio.open(image_path) as src:
                for band_idx in range(1, min(src.count + 1, 14)):
                    data = src.read(band_idx)
                    
                    # Remove nodata values
                    valid_data = data[data != src.nodata] if src.nodata else data
                    
                    if len(valid_data) > 0:
                        band_name = band_names[band_idx - 1] if band_idx <= len(band_names) else f'Band_{band_idx}'
                        stats[band_name]['mean'].append(np.mean(valid_data))
                        stats[band_name]['std'].append(np.std(valid_data))
                        stats[band_name]['min'].append(np.min(valid_data))
                        stats[band_name]['max'].append(np.max(valid_data))
                        
        except Exception as e:
            print(f"Error processing {row['fn']}: {e}")
            continue
    
    # Compute overall statistics
    overall_stats = {}
    for band, band_stats in stats.items():
        if band_stats['mean']:
            overall_stats[band] = {
                'mean': np.mean(band_stats['mean']),
                'std': np.mean(band_stats['std']),
                'min': np.min(band_stats['min']),
                'max': np.max(band_stats['max']),
                'samples': len(band_stats['mean'])
            }
    
    # Save statistics
    stats_file = dataset_path / 'metadata' / 'band_statistics.json'
    stats_file.parent.mkdir(exist_ok=True)
    
    with open(stats_file, 'w') as f:
        json.dump(overall_stats, f, indent=2)
    
    print(f"\nBand statistics saved to: {stats_file}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("BAND STATISTICS")
    print("=" * 50)
    
    for band, band_stats in overall_stats.items():
        print(f"\n{band}:")
        print(f"  Mean: {band_stats['mean']:.2f}")
        print(f"  Std:  {band_stats['std']:.2f}")
        print(f"  Min:  {band_stats['min']:.2f}")
        print(f"  Max:  {band_stats['max']:.2f}")
    
    return overall_stats


# ============================================================================
# 5. MAIN UTILITY RUNNER
# ============================================================================

def main():
    """Main utility runner."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Sentinel-2 Dataset Utilities')
    parser.add_argument('action', choices=['setup', 'validate', 'visualize', 'statistics'],
                       help='Action to perform')
    parser.add_argument('--dataset-dir', default='./sentinel2_dataset',
                       help='Dataset directory path')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of samples for validation/statistics')
    
    args = parser.parse_args()
    
    if args.action == 'setup':
        setup_environment()
        
    elif args.action == 'validate':
        validator = DatasetValidator(args.dataset_dir)
        validator.run_validation()
        
    elif args.action == 'visualize':
        visualize_samples(args.dataset_dir, n_samples=9)
        
    elif args.action == 'statistics':
        compute_band_statistics(args.dataset_dir, sample_size=args.samples)


if __name__ == "__main__":
    main()