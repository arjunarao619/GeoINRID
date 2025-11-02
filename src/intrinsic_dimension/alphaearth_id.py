#!/usr/bin/env python3
"""
AlphaEarth Embeddings Intrinsic Dimension Analysis
===================================================
Computes global and local intrinsic dimensions for AlphaEarth embeddings
across different buffer sizes using multiple estimators from scikit-dimension.

Note: AlphaEarth embeddings have dimension 64, but the last dimension is always NaN,
so we use only the first 63 dimensions.
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm
from pathlib import Path
from tqdm import tqdm
import wandb
import warnings
warnings.filterwarnings('ignore')

# Import scikit-dimension estimators
import skdim.id

# Define all global ID estimators
GLOBAL_ESTIMATORS = {
    'MLE': lambda: skdim.id.MLE(neighborhood_based=True),
    'TwoNN': lambda: skdim.id.TwoNN(),
    'FisherS': lambda: skdim.id.FisherS(project_on_sphere=False),  # Already on unit sphere
    'MOM': lambda: skdim.id.MOM(),
    'TLE': lambda: skdim.id.TLE(),
    'CorrInt': lambda: skdim.id.CorrInt(),
    'DANCo': lambda: skdim.id.DANCo(k=10),
    'ESS': lambda: skdim.id.ESS(),
    'MiND_ML': lambda: skdim.id.MiND_ML(ver='ML'),
    'MiND_KL': lambda: skdim.id.MiND_ML(ver='KL'),
    'MADA': lambda: skdim.id.MADA(),
}

# Estimators that support local ID computation
LOCAL_ESTIMATORS = {
    'MLE': lambda: skdim.id.MLE(neighborhood_based=True),
    'TwoNN': lambda: skdim.id.TwoNN(),
    'MOM': lambda: skdim.id.MOM(),
    'TLE': lambda: skdim.id.TLE(),
    'ESS': lambda: skdim.id.ESS(),
}


class AlphaEarthIDAnalyzer:
    """Analyzes intrinsic dimensions of AlphaEarth embeddings."""
    
    def __init__(self, base_dir: str, year: int = 2024):
        """
        Initialize the ID analyzer.
        
        Parameters:
        -----------
        base_dir : str
            Base directory containing AlphaEarth embeddings subdirectories
        year : int
            Year of embeddings to analyze
        """
        self.base_dir = base_dir
        self.year = year
        self.buffer_dirs = self._find_buffer_directories()
        
        if not self.buffer_dirs:
            raise ValueError(f"No buffer directories found in {base_dir}")
        
        print(f"Found {len(self.buffer_dirs)} buffer directories:")
        for dir_info in self.buffer_dirs:
            print(f"  - {dir_info['name']}: buffer={dir_info['buffer_meters']}m")
    
    def _find_buffer_directories(self):
        """Find all directories with different buffer sizes."""
        buffer_dirs = []
        
        # Look for directories matching pattern
        pattern = os.path.join(self.base_dir, "*buffer*m*")
        dirs = glob.glob(pattern)
        
        for dir_path in dirs:
            if os.path.isdir(dir_path):
                dir_name = os.path.basename(dir_path)
                # Extract buffer size from directory name
                try:
                    if 'buffer' in dir_name:
                        buffer_str = dir_name.split('buffer')[1].split('m')[0]
                        buffer_meters = int(buffer_str)
                        buffer_dirs.append({
                            'path': dir_path,
                            'name': dir_name,
                            'buffer_meters': buffer_meters
                        })
                except:
                    continue
        
        # Sort by buffer size
        buffer_dirs.sort(key=lambda x: x['buffer_meters'])
        return buffer_dirs
    
    def load_embeddings(self, buffer_dir: dict):
        """
        Load embeddings and coordinates from a buffer directory.
        
        Parameters:
        -----------
        buffer_dir : dict
            Directory information dictionary
        
        Returns:
        --------
        embeddings : np.ndarray
            Embedding matrix (n_samples, 63) - excluding last dimension
        coordinates : np.ndarray
            Coordinate matrix (n_samples, 2) with [lon, lat]
        """
        dir_path = buffer_dir['path']
        buffer_str = f"buffer{buffer_dir['buffer_meters']}m"
        
        # Try to load embeddings
        embeddings_path = os.path.join(dir_path, 'embeddings', 
                                      f'embeddings_{self.year}_{buffer_str}.npy')
        if not os.path.exists(embeddings_path):
            # Fallback to non-buffer-specific name
            embeddings_path = os.path.join(dir_path, 'embeddings', 
                                          f'embeddings_{self.year}.npy')
        
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings not found in {dir_path}")
        
        embeddings = np.load(embeddings_path)
        print(f"  Original embeddings shape: {embeddings.shape}")
        
        # Use only first 63 dimensions (64th is always NaN)
        embeddings = embeddings[:, :63]
        print(f"  Using first 63 dimensions: {embeddings.shape}")
        
        # Try to load coordinates from CSV
        csv_path = os.path.join(dir_path, f'embeddings_index_{self.year}_{buffer_str}.csv')
        if not os.path.exists(csv_path):
            csv_path = os.path.join(dir_path, f'embeddings_index_{self.year}.csv')
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            coordinates = df[['lon', 'lat']].values
        else:
            # Try pickle file
            pickle_path = os.path.join(dir_path, 'embeddings', 
                                      f'embeddings_full_{self.year}_{buffer_str}.pkl')
            if not os.path.exists(pickle_path):
                pickle_path = os.path.join(dir_path, 'embeddings', 
                                          f'embeddings_full_{self.year}.pkl')
            
            if os.path.exists(pickle_path):
                with open(pickle_path, 'rb') as f:
                    full_data = pickle.load(f)
                coordinates = np.array([[d['lon'], d['lat']] for d in full_data])
            else:
                raise FileNotFoundError(f"Coordinates not found in {dir_path}")
        
        # Check for NaN values per dimension
        nan_per_dim = np.sum(np.isnan(embeddings), axis=0)
        if np.any(nan_per_dim > 0):
            print(f"  NaN values per dimension: {nan_per_dim[nan_per_dim > 0]}")
        
        # Remove samples with any NaN values in the 63 dimensions
        valid_mask = ~np.isnan(embeddings).any(axis=1)
        n_invalid = (~valid_mask).sum()
        
        if n_invalid > 0:
            print(f"  Removing {n_invalid} samples with NaN values ({n_invalid/len(embeddings)*100:.1f}%)")
            embeddings = embeddings[valid_mask]
            coordinates = coordinates[valid_mask]
        
        if len(embeddings) == 0:
            raise ValueError("All samples have NaN values after using first 63 dimensions")
        
        print(f"  Loaded {len(embeddings)} valid embeddings of dimension {embeddings.shape[1]}")
        
        # Verify data
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"  Embedding norms: mean={norms.mean():.4f}, std={norms.std():.4f}, "
              f"min={norms.min():.4f}, max={norms.max():.4f}")
        
        return embeddings, coordinates
    
    def compute_global_id(self, embeddings: np.ndarray, buffer_meters: int, 
                         k_values: list = [5, 10, 20, 50, 100]):
        """
        Compute global intrinsic dimension using multiple estimators.
        
        Parameters:
        -----------
        embeddings : np.ndarray
            Embedding matrix (n_samples, 63)
        buffer_meters : int
            Buffer size in meters
        k_values : list
            List of k values for neighborhood-based methods
        
        Returns:
        --------
        results : dict
            Dictionary of estimator results
        """
        results = {}
        n_samples = embeddings.shape[0]
        
        print(f"    Computing global ID for {n_samples} samples...")
        
        for est_name, est_fn in GLOBAL_ESTIMATORS.items():
            try:
                if est_name in ['MLE', 'MOM', 'TLE']:
                    # These support multiple k values
                    for k in k_values:
                        if k >= n_samples:
                            print(f"    Skipping {est_name} k={k} (k >= n_samples)")
                            continue
                        
                        print(f"    Computing {est_name} (k={k})...")
                        estimator = est_fn()
                        
                        if hasattr(estimator, 'fit'):
                            if 'n_neighbors' in estimator.fit.__code__.co_varnames:
                                estimator.fit(embeddings, n_neighbors=k)
                            else:
                                estimator.fit(embeddings)
                            
                            id_val = float(estimator.dimension_)
                            results[f"{est_name}_k{k}"] = id_val
                            print(f"      → ID = {id_val:.4f}")
                
                elif est_name in ['ESS', 'MADA']:
                    # These need special handling for n_neighbors
                    print(f"    Computing {est_name}...")
                    estimator = est_fn()
                    
                    # ESS and MADA automatically determine n_neighbors internally
                    # We don't need to set it manually
                    estimator.fit(embeddings)
                    id_val = float(estimator.dimension_)
                    results[est_name] = id_val
                    print(f"      → ID = {id_val:.4f}")
                
                else:
                    # Single estimate for other estimators
                    print(f"    Computing {est_name}...")
                    estimator = est_fn()
                    
                    estimator.fit(embeddings)
                    id_val = float(estimator.dimension_)
                    results[est_name] = id_val
                    print(f"      → ID = {id_val:.4f}")
                    
            except Exception as e:
                print(f"      Failed: {str(e)}")
                if est_name in ['MLE', 'MOM', 'TLE'] and 'k' in locals():
                    results[f"{est_name}_k{k}"] = np.nan
                else:
                    results[est_name] = np.nan
        
        return results
    
    def compute_local_id(self, embeddings: np.ndarray, coordinates: np.ndarray,
                        buffer_meters: int, k: int = 20):
        """
        Compute local intrinsic dimension estimates.
        
        Parameters:
        -----------
        embeddings : np.ndarray
            Embedding matrix (n_samples, 63)
        coordinates : np.ndarray
            Coordinate matrix [lon, lat]
        buffer_meters : int
            Buffer size in meters
        k : int
            Number of neighbors for local estimation
        
        Returns:
        --------
        local_results : dict
            Dictionary of local ID arrays for each estimator
        """
        local_results = {}
        n_samples = embeddings.shape[0]
        
        # Ensure k is valid
        k = min(k, n_samples - 1)
        if k < 2:
            print(f"    Skipping local ID (insufficient samples, k={k})")
            return local_results
        
        for est_name, est_fn in LOCAL_ESTIMATORS.items():
            try:
                print(f"    Computing local {est_name} (k={k})...")
                estimator = est_fn()
                
                if est_name == 'ESS':
                    # ESS doesn't support pointwise estimation
                    print(f"      Skipping ESS for local estimation (not supported)")
                    continue
                
                if hasattr(estimator, 'fit_transform_pw'):
                    # Use pointwise estimation
                    result = estimator.fit_transform_pw(
                        embeddings,
                        n_neighbors=k,
                        n_jobs=-1
                    )
                    
                    if isinstance(result, tuple) and len(result) == 2:
                        # Returns (raw, smooth)
                        raw_ids, smooth_ids = result
                        local_results[f"{est_name}_raw"] = raw_ids
                        local_results[f"{est_name}_smooth"] = smooth_ids
                        
                        valid_raw = raw_ids[np.isfinite(raw_ids)]
                        valid_smooth = smooth_ids[np.isfinite(smooth_ids)]
                        
                        if len(valid_raw) > 0:
                            print(f"      Raw: mean={np.mean(valid_raw):.4f}, "
                                 f"std={np.std(valid_raw):.4f}")
                        if len(valid_smooth) > 0:
                            print(f"      Smooth: mean={np.mean(valid_smooth):.4f}, "
                                 f"std={np.std(valid_smooth):.4f}")
                    else:
                        # Single output
                        local_ids = result if not isinstance(result, tuple) else result[0]
                        local_results[est_name] = local_ids
                        
                        valid_ids = local_ids[np.isfinite(local_ids)]
                        if len(valid_ids) > 0:
                            print(f"      Mean={np.mean(valid_ids):.4f}, "
                                 f"std={np.std(valid_ids):.4f}")
                
            except Exception as e:
                print(f"      Failed: {str(e)}")
        
        return local_results
    
    def plot_local_id_map(self, coordinates: np.ndarray, local_ids: np.ndarray,
                          estimator_name: str, buffer_meters: int, k: int,
                          smooth: bool = False):
        """
        Create visualization of local ID estimates on a world map.
        
        Parameters:
        -----------
        coordinates : np.ndarray
            Coordinate array [lon, lat]
        local_ids : np.ndarray
            Local ID values
        estimator_name : str
            Name of the estimator
        buffer_meters : int
            Buffer size in meters
        k : int
            Number of neighbors used
        smooth : bool
            Whether these are smoothed estimates
        """
        # Filter valid values
        mask = np.isfinite(local_ids) & (local_ids > 0)
        coords_clean = coordinates[mask]
        ids_clean = local_ids[mask]
        
        if len(ids_clean) == 0:
            print(f"      No valid local IDs to plot")
            return None
        
        # Create figure with two projections
        fig = plt.figure(figsize=(20, 8))
        
        # Orthographic projection 1 (Americas/Atlantic)
        ax1 = fig.add_subplot(121, projection=ccrs.Orthographic(-30, 30))
        ax1.set_global()
        ax1.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax1.add_feature(cfeature.OCEAN, facecolor='white')
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax1.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax1.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.3)
        
        # Orthographic projection 2 (Asia/Pacific)
        ax2 = fig.add_subplot(122, projection=ccrs.Orthographic(120, 30))
        ax2.set_global()
        ax2.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax2.add_feature(cfeature.OCEAN, facecolor='white')
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax2.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax2.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.3)
        
        # Plot on both projections
        vmin, vmax = np.percentile(ids_clean, [5, 95])
        
        for ax in [ax1, ax2]:
            sc = ax.scatter(
                coords_clean[:, 0], coords_clean[:, 1],
                c=ids_clean,
                cmap='YlOrRd',
                vmin=vmin, vmax=vmax,
                s=2,
                alpha=0.7,
                transform=ccrs.PlateCarree(),
                rasterized=True
            )
        
        # Add colorbar
        cbar = fig.colorbar(sc, ax=[ax1, ax2], orientation='horizontal',
                          pad=0.05, shrink=0.6)
        cbar.set_label(f'Local {estimator_name} (k={k})', fontsize=12)
        
        # Title
        smooth_str = " (smoothed)" if smooth else " (raw)"
        fig.suptitle(f'AlphaEarth Local Intrinsic Dimension - Buffer: {buffer_meters}m{smooth_str}',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        smooth_suffix = "_smooth" if smooth else "_raw"
        filename = f"local_id_alphaearth_buffer{buffer_meters}m_{estimator_name}_k{k}{smooth_suffix}.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return filename
    
    def run_analysis(self, k_values_global: list = [5, 10, 20, 50, 100],
                    k_values_local: list = [10, 20, 50]):
        """
        Run complete intrinsic dimension analysis for all buffer sizes.
        
        Parameters:
        -----------
        k_values_global : list
            k values for global ID estimation
        k_values_local : list
            k values for local ID estimation
        
        Returns:
        --------
        all_results : dict
            Dictionary containing all results
        """
        all_results = {}
        
        for buffer_dir in self.buffer_dirs:
            buffer_meters = buffer_dir['buffer_meters']
            
            print(f"\n{'='*60}")
            print(f"Processing buffer size: {buffer_meters}m")
            print(f"Directory: {buffer_dir['path']}")
            print('='*60)
            
            try:
                # Load embeddings
                embeddings, coordinates = self.load_embeddings(buffer_dir)
                
                if len(embeddings) < 2:
                    print(f"  Skipping: insufficient valid samples ({len(embeddings)})")
                    continue
                
                # Compute global ID
                print("\n  Computing global intrinsic dimensions...")
                global_results = self.compute_global_id(
                    embeddings, buffer_meters, k_values_global
                )
                
                # Store results
                buffer_key = f"buffer_{buffer_meters}m"
                all_results[buffer_key] = {
                    'global': global_results,
                    'local': {},
                    'n_samples': len(embeddings),
                    'embedding_dim': embeddings.shape[1]
                }
                
                # Log to wandb
                for metric_name, id_value in global_results.items():
                    if not np.isnan(id_value):
                        wandb.log({
                            f"alphaearth/buffer_{buffer_meters}m/global_{metric_name}": id_value
                        })
                
                # Compute local ID for each k value
                for k_local in k_values_local:
                    if k_local >= len(embeddings):
                        print(f"  Skipping local ID for k={k_local} (k >= n_samples)")
                        continue
                    
                    print(f"\n  Computing local intrinsic dimensions (k={k_local})...")
                    local_results = self.compute_local_id(
                        embeddings, coordinates, buffer_meters, k=k_local
                    )
                    
                    if local_results:
                        all_results[buffer_key]['local'][f'k{k_local}'] = local_results
                        
                        # Create visualizations
                        for est_name, local_ids in local_results.items():
                            is_smooth = 'smooth' in est_name
                            base_name = est_name.replace('_smooth', '').replace('_raw', '')
                            
                            plot_file = self.plot_local_id_map(
                                coordinates, local_ids, base_name,
                                buffer_meters, k_local, smooth=is_smooth
                            )
                            
                            if plot_file:
                                wandb.log({
                                    f"alphaearth/buffer_{buffer_meters}m/local_{est_name}_k{k_local}": 
                                    wandb.Image(plot_file)
                                })
                
            except Exception as e:
                print(f"  ERROR processing buffer {buffer_meters}m: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        return all_results
    
    def create_comparison_plot(self, all_results: dict):
        """
        Create comparison plot of global ID across buffer sizes.
        
        Parameters:
        -----------
        all_results : dict
            Dictionary of all results
        """
        if not all_results:
            print("No results to plot")
            return None
        
        # Extract data for plotting
        buffer_sizes = []
        estimator_names = set()
        
        for buffer_key in all_results.keys():
            buffer_sizes.append(int(buffer_key.split('_')[1].replace('m', '')))
            estimator_names.update(all_results[buffer_key]['global'].keys())
        
        buffer_sizes.sort()
        estimator_names = sorted(list(estimator_names))
        
        if not estimator_names:
            print("No estimator results to plot")
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        markers = ['o', 's', '^', 'v', 'D', '<', '>', 'p', '*', 'h', '+', 'x']
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(estimator_names), 10)))
        
        for i, est_name in enumerate(estimator_names):
            values = []
            valid_buffers = []
            
            for buffer_size in buffer_sizes:
                buffer_key = f"buffer_{buffer_size}m"
                if buffer_key in all_results:
                    val = all_results[buffer_key]['global'].get(est_name, np.nan)
                    if not np.isnan(val):
                        values.append(val)
                        valid_buffers.append(buffer_size)
            
            if values:
                ax.plot(valid_buffers, values,
                       marker=markers[i % len(markers)],
                       color=colors[i % len(colors)],
                       label=est_name,
                       linewidth=2,
                       markersize=8,
                       alpha=0.7)
        
        ax.set_xlabel('Buffer Size (meters)', fontsize=12)
        ax.set_ylabel('Intrinsic Dimension', fontsize=12)
        ax.set_title('AlphaEarth Global Intrinsic Dimension vs Buffer Size', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
        
        plt.tight_layout()
        
        comparison_file = 'alphaearth_id_comparison.png'
        fig.savefig(comparison_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        wandb.log({"alphaearth/id_comparison": wandb.Image(comparison_file)})
        
        return comparison_file
    
    def save_results(self, all_results: dict):
        """
        Save results to CSV and pickle files.
        
        Parameters:
        -----------
        all_results : dict
            Dictionary of all results
        """
        if not all_results:
            print("No results to save")
            return None
        
        # Create summary DataFrame
        summary_data = []
        
        for buffer_key, results in all_results.items():
            buffer_size = int(buffer_key.split('_')[1].replace('m', ''))
            
            row = {
                'buffer_meters': buffer_size,
                'n_samples': results['n_samples'],
                'embedding_dim': results['embedding_dim']
            }
            
            # Add global ID estimates
            for est_name, id_val in results['global'].items():
                row[f'global_{est_name}'] = id_val
            
            summary_data.append(row)
        
        if not summary_data:
            print("No summary data to save")
            return None
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('buffer_meters')
        
        # Save to CSV
        csv_file = 'alphaearth_id_results.csv'
        df.to_csv(csv_file, index=False)
        wandb.save(csv_file)
        
        # Save full results to pickle
        pickle_file = 'alphaearth_id_results_full.pkl'
        with open(pickle_file, 'wb') as f:
            pickle.dump(all_results, f)
        wandb.save(pickle_file)
        
        # Log summary table to wandb
        wandb.log({"alphaearth/summary_table": wandb.Table(dataframe=df)})
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY OF RESULTS")
        print("="*80)
        print(df.to_string())
        
        return df


def main():
    parser = argparse.ArgumentParser(
        description='Compute intrinsic dimensions for AlphaEarth embeddings'
    )
    parser.add_argument('--base_dir', type=str, 
                       default='.',
                       help='Base directory containing AlphaEarth embeddings')
    parser.add_argument('--year', type=int, default=2024,
                       help='Year of embeddings to analyze')
    parser.add_argument('--k_global', nargs='+', type=int,
                       default=[20, 50, 100],
                       help='k values for global ID estimation')
    parser.add_argument('--k_local', nargs='+', type=int,
                       default=[10, 20, 50],
                       help='k values for local ID estimation')
    parser.add_argument('--wandb_project', type=str,
                       default='AlphaEarth_ID_Analysis',
                       help='WandB project name')
    
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=f'alphaearth_id_analysis_{args.year}',
        config=vars(args)
    )
    
    # Create analyzer
    analyzer = AlphaEarthIDAnalyzer(
        base_dir=args.base_dir,
        year=args.year
    )
    
    # Run analysis
    print("\nStarting intrinsic dimension analysis...")
    all_results = analyzer.run_analysis(
        k_values_global=args.k_global,
        k_values_local=args.k_local
    )
    
    # Create comparison plots
    if all_results:
        print("\nCreating comparison visualizations...")
        analyzer.create_comparison_plot(all_results)
        
        # Save results
        print("\nSaving results...")
        summary_df = analyzer.save_results(all_results)
    else:
        print("\nNo results obtained - analysis failed")
    
    # Finish wandb
    wandb.finish()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()