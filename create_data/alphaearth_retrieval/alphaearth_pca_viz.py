#!/usr/bin/env python3
"""
AlphaEarth Embeddings PCA Visualization Script
==============================================
Loads extracted AlphaEarth embeddings and visualizes them using PCA-RGB mapping on a globe.
The first 3 PCA components are mapped to RGB colors for spatial visualization.

Author: Arjun Rao
Date: 2025
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import cartopy for globe projection
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import ListedColormap
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: Cartopy not found. Will use simple map projection instead of globe.")


class AlphaEarthPCAVisualizer:
    """Visualizes AlphaEarth embeddings using PCA-RGB mapping."""
    
    def __init__(self, embeddings_dir: str, year: int = 2024):
        """
        Initialize the PCA visualizer.
        
        Parameters:
        -----------
        embeddings_dir : str
            Directory containing extracted AlphaEarth embeddings
        year : int
            Year of the embeddings to load
        """
        self.embeddings_dir = embeddings_dir
        self.year = year
        
        # Detect buffer size from directory name
        self.buffer_meters = self._detect_buffer_size()
        
        # File paths
        self.embeddings_path = None
        self.pickle_path = None
        self.csv_path = None
        self._find_embedding_files()
        
        # Data containers
        self.embeddings = None
        self.coordinates = None
        self.pca_embeddings = None
        self.rgb_colors = None
        
        print(f"Initialized AlphaEarth PCA Visualizer")
        print(f"  Directory: {embeddings_dir}")
        print(f"  Year: {year}")
        print(f"  Buffer: {self.buffer_meters}m")
    
    def _detect_buffer_size(self) -> int:
        """Detect buffer size from directory name."""
        dir_name = os.path.basename(self.embeddings_dir)
        if 'buffer' in dir_name:
            try:
                # Extract number between 'buffer' and 'm'
                buffer_str = dir_name.split('buffer')[1].split('m')[0]
                return int(buffer_str)
            except:
                pass
        return 10  # Default
    
    def _find_embedding_files(self):
        """Find the embedding files in the directory."""
        # Look for buffer-specific files first
        buffer_str = f"buffer{self.buffer_meters}m"
        
        # Search in embeddings subdirectory
        embeddings_subdir = os.path.join(self.embeddings_dir, 'embeddings')
        
        # Try to find numpy file
        npy_pattern = f'embeddings_{self.year}_{buffer_str}.npy'
        npy_path = os.path.join(embeddings_subdir, npy_pattern)
        if os.path.exists(npy_path):
            self.embeddings_path = npy_path
        else:
            # Fallback to non-buffer-specific name
            npy_path = os.path.join(embeddings_subdir, f'embeddings_{self.year}.npy')
            if os.path.exists(npy_path):
                self.embeddings_path = npy_path
        
        # Try to find pickle file
        pkl_pattern = f'embeddings_full_{self.year}_{buffer_str}.pkl'
        pkl_path = os.path.join(embeddings_subdir, pkl_pattern)
        if os.path.exists(pkl_path):
            self.pickle_path = pkl_path
        else:
            # Fallback
            pkl_path = os.path.join(embeddings_subdir, f'embeddings_full_{self.year}.pkl')
            if os.path.exists(pkl_path):
                self.pickle_path = pkl_path
        
        # Try to find CSV file
        csv_pattern = f'embeddings_index_{self.year}_{buffer_str}.csv'
        csv_path = os.path.join(self.embeddings_dir, csv_pattern)
        if os.path.exists(csv_path):
            self.csv_path = csv_path
        else:
            # Fallback
            csv_path = os.path.join(self.embeddings_dir, f'embeddings_index_{self.year}.csv')
            if os.path.exists(csv_path):
                self.csv_path = csv_path
        
        print(f"Found files:")
        print(f"  Embeddings: {self.embeddings_path}")
        print(f"  Pickle: {self.pickle_path}")
        print(f"  CSV: {self.csv_path}")
    
    def load_embeddings(self):
        """Load embeddings and coordinates from saved files."""
        # Load embeddings array
        if self.embeddings_path and os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            print(f"Loaded embeddings: {self.embeddings.shape}")
        else:
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_path}")
        
        # Load coordinates from CSV or pickle
        if self.csv_path and os.path.exists(self.csv_path):
            self.coordinates = pd.read_csv(self.csv_path)
            print(f"Loaded coordinates from CSV: {len(self.coordinates)} points")
        elif self.pickle_path and os.path.exists(self.pickle_path):
            with open(self.pickle_path, 'rb') as f:
                full_data = pickle.load(f)
            self.coordinates = pd.DataFrame([{
                'lon': d['lon'],
                'lat': d['lat'],
                'fn': d.get('fn', f"point_{i}")
            } for i, d in enumerate(full_data)])
            print(f"Loaded coordinates from pickle: {len(self.coordinates)} points")
        else:
            raise FileNotFoundError("No coordinate file found (CSV or pickle)")
        
        # Validate data consistency
        if len(self.embeddings) != len(self.coordinates):
            print(f"Warning: Mismatch in data sizes!")
            print(f"  Embeddings: {len(self.embeddings)}")
            print(f"  Coordinates: {len(self.coordinates)}")
            min_size = min(len(self.embeddings), len(self.coordinates))
            self.embeddings = self.embeddings[:min_size]
            self.coordinates = self.coordinates.iloc[:min_size]
    

    def compute_pca(self, n_components: int = 3):
        """
        Compute PCA on embeddings and map to RGB colors.
        
        Parameters:
        -----------
        n_components : int
            Number of PCA components (default 3 for RGB)
        """
        print(f"\nComputing PCA with {n_components} components...")
        
        # Check for NaN patterns
        nan_per_sample = np.isnan(self.embeddings).sum(axis=1)
        nan_per_dim = np.isnan(self.embeddings).sum(axis=0)
        
        print(f"NaN analysis:")
        print(f"  Samples with NaN: {(nan_per_sample > 0).sum()} / {len(self.embeddings)}")
        print(f"  Dimensions with NaN: {(nan_per_dim > 0).sum()} / {self.embeddings.shape[1]}")
        
        # Strategy 1: Remove dimensions that are always NaN
        valid_dims = nan_per_dim < len(self.embeddings) * 0.9  # Keep dims with <90% NaN
        if valid_dims.sum() == 0:
            raise ValueError("All dimensions have too many NaN values")
        
        print(f"  Using {valid_dims.sum()} / {len(valid_dims)} dimensions")
        embeddings_subset = self.embeddings[:, valid_dims]
        
        # Strategy 2: For remaining NaNs, use mean imputation
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        embeddings_imputed = imputer.fit_transform(embeddings_subset)
        
        # Alternative Strategy: Only keep samples with few NaNs
        max_nan_per_sample = 10  # Allow up to 10 NaN dimensions per sample
        valid_samples = nan_per_sample <= max_nan_per_sample
        
        if valid_samples.sum() < 10:
            print(f"Warning: Only {valid_samples.sum()} valid samples. Using imputation instead.")
            valid_embeddings = embeddings_imputed
            valid_samples = np.ones(len(self.embeddings), dtype=bool)
        else:
            valid_embeddings = embeddings_imputed[valid_samples]
            print(f"  Keeping {valid_samples.sum()} samples with <= {max_nan_per_sample} NaN values")
        
        # Apply PCA
        if len(valid_embeddings) == 0:
            raise ValueError("No valid embeddings remaining after filtering")
        
        pca = PCA(n_components=n_components)
        self.pca_embeddings = pca.fit_transform(valid_embeddings)
        
        # Print explained variance
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")
        
        # Scale to 0-1 range for RGB
        scaler = MinMaxScaler()
        pca_scaled = scaler.fit_transform(self.pca_embeddings)
        
        # Convert to RGB (0-255)
        self.rgb_colors = (pca_scaled * 255).astype(np.uint8)
        
        # Update coordinates to match valid embeddings
        self.coordinates = self.coordinates[valid_samples].reset_index(drop=True)
        
        print(f"PCA complete. RGB colors shape: {self.rgb_colors.shape}")
        
        return pca
    
    def plot_globe_view(self, save_path: str = None, figsize: tuple = (20, 10)):
        """
        Create a globe visualization with PCA-RGB colors.
        
        Parameters:
        -----------
        save_path : str
            Path to save the figure
        figsize : tuple
            Figure size
        """
        if not HAS_CARTOPY:
            print("Cartopy not available. Using simple map instead.")
            return self.plot_simple_map(save_path, figsize)
        
        fig = plt.figure(figsize=figsize)
        
        # Create two subplots with different projections
        ax1 = fig.add_subplot(121, projection=ccrs.Orthographic(-30, 30))
        ax2 = fig.add_subplot(122, projection=ccrs.Orthographic(150, 30))
        
        for ax in [ax1, ax2]:
            # Add map features
            ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', alpha=0.3)
            ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff', alpha=0.3)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.3)
            ax.gridlines(alpha=0.2)
            
            # Plot points with PCA colors
            for i, row in self.coordinates.iterrows():
                color = self.rgb_colors[i] / 255.0  # Normalize to 0-1
                ax.scatter(row['lon'], row['lat'],
                          c=[color], s=10,
                          transform=ccrs.PlateCarree(),
                          alpha=0.8, edgecolors='none')
        
        # Add title
        fig.suptitle(f'AlphaEarth Embeddings PCA-RGB Visualization\n'
                    f'Year: {self.year} | Buffer: {self.buffer_meters}m | '
                    f'Points: {len(self.coordinates):,}',
                    fontsize=16, fontweight='bold')
        
        # Add color legend showing PCA components
        self._add_pca_color_legend(fig)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved globe view to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_simple_map(self, save_path: str = None, figsize: tuple = (16, 8)):
        """
        Create a simple 2D map visualization with PCA-RGB colors.
        
        Parameters:
        -----------
        save_path : str
            Path to save the figure
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set up map
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#e6f2ff')
        
        # Plot points with PCA colors
        colors_normalized = self.rgb_colors / 255.0
        scatter = ax.scatter(self.coordinates['lon'], 
                           self.coordinates['lat'],
                           c=colors_normalized,
                           s=20, alpha=0.8, edgecolors='none')
        
        # Add title
        ax.set_title(f'AlphaEarth Embeddings PCA-RGB Visualization\n'
                    f'Year: {self.year} | Buffer: {self.buffer_meters}m | '
                    f'Points: {len(self.coordinates):,}',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved map to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_pca_distribution(self, save_path: str = None):
        """
        Plot the distribution of PCA components.
        
        Parameters:
        -----------
        save_path : str
            Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: 3D scatter of PCA components
        ax = fig.add_subplot(221, projection='3d')
        colors_normalized = self.rgb_colors / 255.0
        ax.scatter(self.pca_embeddings[:, 0],
                  self.pca_embeddings[:, 1],
                  self.pca_embeddings[:, 2],
                  c=colors_normalized, s=1, alpha=0.5)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('PCA Components in 3D Space')
        
        # Plot 2-4: 2D projections
        projections = [(0, 1, 222), (0, 2, 223), (1, 2, 224)]
        labels = ['PC1', 'PC2', 'PC3']
        
        for (i, j, subplot) in projections:
            ax = fig.add_subplot(subplot)
            ax.scatter(self.pca_embeddings[:, i],
                      self.pca_embeddings[:, j],
                      c=colors_normalized, s=1, alpha=0.5)
            ax.set_xlabel(labels[i])
            ax.set_ylabel(labels[j])
            ax.set_title(f'{labels[i]} vs {labels[j]}')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'PCA Component Distribution\n'
                    f'Buffer: {self.buffer_meters}m | Points: {len(self.coordinates):,}',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved PCA distribution to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def _add_pca_color_legend(self, fig):
        """Add a color cube legend showing PCA-RGB mapping."""
        ax = fig.add_axes([0.45, 0.02, 0.1, 0.1])
        ax.set_title('PCA→RGB', fontsize=10)
        
        # Create a small color cube
        n = 20
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        X, Y = np.meshgrid(x, y)
        
        # Create RGB array (using mean of PC3 for visualization)
        RGB = np.zeros((n, n, 3))
        RGB[:, :, 0] = X  # Red = PC1
        RGB[:, :, 1] = Y  # Green = PC2
        RGB[:, :, 2] = 0.5  # Blue = PC3 (fixed at middle)
        
        ax.imshow(RGB, extent=[0, 1, 0, 1])
        ax.set_xlabel('PC1→R', fontsize=8)
        ax.set_ylabel('PC2→G', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def create_interactive_html(self, save_path: str = None):
        """
        Create an interactive HTML visualization using plotly.
        
        Parameters:
        -----------
        save_path : str
            Path to save HTML file
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            print("Plotly not installed. Skipping interactive visualization.")
            print("Install with: pip install plotly")
            return
        
        # Convert RGB to hex colors for plotly
        hex_colors = ['#%02x%02x%02x' % tuple(color) for color in self.rgb_colors]
        
        # Create 3D globe
        fig = go.Figure(data=go.Scattergeo(
            lon=self.coordinates['lon'],
            lat=self.coordinates['lat'],
            mode='markers',
            marker=dict(
                size=4,
                color=hex_colors,
                line=dict(width=0)
            ),
            text=[f"Lon: {row['lon']:.2f}, Lat: {row['lat']:.2f}" 
                  for _, row in self.coordinates.iterrows()],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_geos(
            projection_type="natural earth",
            showland=True,
            landcolor='rgb(243, 243, 243)',
            coastlinecolor='rgb(204, 204, 204)',
            showocean=True,
            oceancolor='rgb(230, 242, 255)',
            showcountries=True,
            countrycolor='rgb(204, 204, 204)',
            showlakes=True,
            lakecolor='rgb(230, 242, 255)'
        )
        
        fig.update_layout(
            title=dict(
                text=f'AlphaEarth PCA-RGB Visualization (Interactive)<br>'
                     f'Year: {self.year} | Buffer: {self.buffer_meters}m | '
                     f'Points: {len(self.coordinates):,}',
                x=0.5
            ),
            geo=dict(
                showframe=False,
                showcoastlines=True,
            ),
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Saved interactive HTML to {save_path}")
        else:
            fig.show()
        
        return fig
    
    def run_visualization(self, output_dir: str = None):
        """
        Run the complete visualization pipeline.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations
        """
        # Load data
        print("\n" + "="*60)
        print("LOADING EMBEDDINGS")
        print("="*60)
        self.load_embeddings()
        
        # Compute PCA
        print("\n" + "="*60)
        print("COMPUTING PCA")
        print("="*60)
        pca = self.compute_pca()
        
        # Create output directory
        if output_dir is None:
            output_dir = os.path.join(self.embeddings_dir, 'pca_visualizations')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # 1. Globe view
        globe_path = os.path.join(output_dir, 
                                 f'pca_globe_{self.year}_buffer{self.buffer_meters}m.png')
        self.plot_globe_view(save_path=globe_path)
        
        # 2. PCA distribution
        dist_path = os.path.join(output_dir, 
                                f'pca_distribution_{self.year}_buffer{self.buffer_meters}m.png')
        self.plot_pca_distribution(save_path=dist_path)
        
        # 3. Interactive HTML
        html_path = os.path.join(output_dir, 
                                f'pca_interactive_{self.year}_buffer{self.buffer_meters}m.html')
        self.create_interactive_html(save_path=html_path)
        
        print("\n" + "="*60)
        print("VISUALIZATION COMPLETE")
        print("="*60)
        print(f"Results saved to: {output_dir}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Visualize AlphaEarth embeddings using PCA-RGB mapping'
    )
    parser.add_argument('embeddings_dir', type=str,
                      help='Directory containing extracted AlphaEarth embeddings')
    parser.add_argument('--year', type=int, default=2024,
                      help='Year of embeddings to visualize (default: 2024)')
    parser.add_argument('--output', type=str, default=None,
                      help='Output directory for visualizations')
    parser.add_argument('--projection', type=str, default='globe',
                      choices=['globe', 'map', 'both'],
                      help='Projection type for visualization')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.embeddings_dir):
        print(f"Error: Directory not found: {args.embeddings_dir}")
        sys.exit(1)
    
    # Create visualizer
    visualizer = AlphaEarthPCAVisualizer(
        embeddings_dir=args.embeddings_dir,
        year=args.year
    )
    
    # Run visualization pipeline
    try:
        visualizer.run_visualization(output_dir=args.output)
    except Exception as e:
        print(f"Error during visualization: {e}")
        raise


if __name__ == "__main__":
    main()