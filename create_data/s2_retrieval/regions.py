#!/usr/bin/env python3
"""
Region definitions and sampling using local Natural Earth shapefiles.
"""

import random
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import os

logger = logging.getLogger(__name__)

class RegionSampler:
    """Handles region definitions and sampling using local Natural Earth shapefiles."""
    
    # Local shapefile paths
    SHAPEFILE_PATHS = {
        'countries': '/projects/arra4944/MMLocEnc/create_data/shapefiles/countries/ne_50m_admin_0_countries.shp',
        'states': '/projects/arra4944/MMLocEnc/create_data/shapefiles/states/ne_50m_admin_1_states_provinces.shp',
        'continents_dir': '/projects/arra4944/MMLocEnc/create_data/shapefiles/continenets'  
    }
    
    # Individual continent shapefiles
    CONTINENT_FILES = {
        'africa': 'Africa.shp',
        'asia': 'Asia.shp',
        'europe': 'Europe.shp',
        'north_america': 'North America.shp',
        'south_america': 'South America.shp',
        'oceania': 'Oceania.shp',
        'antarctica': 'Antarctica.shp'
    }
    
    # Region configurations
    REGION_CONFIG = {
        # Continents (use individual shapefiles)
        'africa': {
            'type': 'continent',
            'shapefile': 'africa',
            'default_cloud_threshold': 30,
            'default_samples': 100000
        },
        'europe': {
            'type': 'continent',
            'shapefile': 'europe',
            'default_cloud_threshold': 25,
            'default_samples': 75000
        },
        'asia': {
            'type': 'continent',
            'shapefile': 'asia',
            'default_cloud_threshold': 35,
            'default_samples': 150000
        },
        'north_america': {
            'type': 'continent',
            'shapefile': 'north_america',
            'default_cloud_threshold': 25,
            'default_samples': 80000
        },
        'south_america': {
            'type': 'continent',
            'shapefile': 'south_america',
            'default_cloud_threshold': 40,
            'default_samples': 80000
        },
        'oceania': {
            'type': 'continent',
            'shapefile': 'oceania',
            'default_cloud_threshold': 30,
            'default_samples': 30000
        },
        
        # Countries (from countries shapefile)
        'united_states': {
            'type': 'country',
            'name_field': 'NAME',
            'name_value': 'United States of America',
            'default_cloud_threshold': 25,
            'default_samples': 50000,
            'exclude_regions': ['Alaska', 'Hawaii']  # Continental US only
        },
        'france': {
            'type': 'country',
            'name_field': 'NAME',
            'name_value': 'France',
            'default_cloud_threshold': 20,
            'default_samples': 10000,
            'metropolitan_only': True
        },
        'brazil': {
            'type': 'country',
            'name_field': 'NAME',
            'name_value': 'Brazil',
            'default_cloud_threshold': 35,
            'default_samples': 40000
        },
        'canada': {
            'type': 'country',
            'name_field': 'NAME',
            'name_value': 'Canada',
            'default_cloud_threshold': 30,
            'default_samples': 50000
        },
        'china': {
            'type': 'country',
            'name_field': 'NAME',
            'name_value': 'China',
            'default_cloud_threshold': 35,
            'default_samples': 60000
        },
        'india': {
            'type': 'country',
            'name_field': 'NAME',
            'name_value': 'India',
            'default_cloud_threshold': 40,
            'default_samples': 40000
        },
        'australia': {
            'type': 'country',
            'name_field': 'NAME',
            'name_value': 'Australia',
            'default_cloud_threshold': 20,
            'default_samples': 30000
        },
        'germany': {
            'type': 'country',
            'name_field': 'NAME',
            'name_value': 'Germany',
            'default_cloud_threshold': 25,
            'default_samples': 8000
        },
        'spain': {
            'type': 'country',
            'name_field': 'NAME',
            'name_value': 'Spain',
            'default_cloud_threshold': 20,
            'default_samples': 8000
        },
        'italy': {
            'type': 'country',
            'name_field': 'NAME',
            'name_value': 'Italy',
            'default_cloud_threshold': 20,
            'default_samples': 7000
        },
        'japan': {
            'type': 'country',
            'name_field': 'NAME',
            'name_value': 'Japan',
            'default_cloud_threshold': 35,
            'default_samples': 8000
        },
        'indonesia': {
            'type': 'country',
            'name_field': 'NAME',
            'name_value': 'Indonesia',
            'default_cloud_threshold': 40,
            'default_samples': 15000
        },
        'mexico': {
            'type': 'country',
            'name_field': 'NAME',
            'name_value': 'Mexico',
            'default_cloud_threshold': 25,
            'default_samples': 15000
        },
        'argentina': {
            'type': 'country',
            'name_field': 'NAME',
            'name_value': 'Argentina',
            'default_cloud_threshold': 25,
            'default_samples': 20000
        },
        'south_africa': {
            'type': 'country',
            'name_field': 'NAME',
            'name_value': 'South Africa',
            'default_cloud_threshold': 20,
            'default_samples': 10000
        },
        'nigeria': {
            'type': 'country',
            'name_field': 'NAME',
            'name_value': 'Nigeria',
            'default_cloud_threshold': 35,
            'default_samples': 8000
        },
        'egypt': {
            'type': 'country',
            'name_field': 'NAME',
            'name_value': 'Egypt',
            'default_cloud_threshold': 15,
            'default_samples': 8000
        },
        'kenya': {
            'type': 'country',
            'name_field': 'NAME',
            'name_value': 'Kenya',
            'default_cloud_threshold': 30,
            'default_samples': 6000
        },
        
        # US States (from states shapefile)
        'colorado': {
            'type': 'state',
            'country': 'United States of America',
            'name_field': 'name',
            'name_value': 'Colorado',
            'default_cloud_threshold': 20,
            'default_samples': 5000
        },
        'california': {
            'type': 'state',
            'country': 'United States of America',
            'name_field': 'name',
            'name_value': 'California',
            'default_cloud_threshold': 20,
            'default_samples': 8000
        },
        'texas': {
            'type': 'state',
            'country': 'United States of America',
            'name_field': 'name',
            'name_value': 'Texas',
            'default_cloud_threshold': 25,
            'default_samples': 10000
        },
        'montana': {
            'type': 'state',
            'country': 'United States of America',
            'name_field': 'name',
            'name_value': 'Montana',
            'default_cloud_threshold': 25,
            'default_samples': 5000
        },
        'wyoming': {
            'type': 'state',
            'country': 'United States of America',
            'name_field': 'name',
            'name_value': 'Wyoming',
            'default_cloud_threshold': 20,
            'default_samples': 4000
        },
        'utah': {
            'type': 'state',
            'country': 'United States of America',
            'name_field': 'name',
            'name_value': 'Utah',
            'default_cloud_threshold': 15,
            'default_samples': 4000
        },
        'arizona': {
            'type': 'state',
            'country': 'United States of America',
            'name_field': 'name',
            'name_value': 'Arizona',
            'default_cloud_threshold': 15,
            'default_samples': 5000
        },
        'new_mexico': {
            'type': 'state',
            'country': 'United States of America',
            'name_field': 'name',
            'name_value': 'New Mexico',
            'default_cloud_threshold': 15,
            'default_samples': 5000
        },
        'nevada': {
            'type': 'state',
            'country': 'United States of America',
            'name_field': 'name',
            'name_value': 'Nevada',
            'default_cloud_threshold': 15,
            'default_samples': 4000
        },
        'oregon': {
            'type': 'state',
            'country': 'United States of America',
            'name_field': 'name',
            'name_value': 'Oregon',
            'default_cloud_threshold': 30,
            'default_samples': 4000
        },
        'washington': {
            'type': 'state',
            'country': 'United States of America',
            'name_field': 'name',
            'name_value': 'Washington',
            'default_cloud_threshold': 35,
            'default_samples': 4000
        },
        'idaho': {
            'type': 'state',
            'country': 'United States of America',
            'name_field': 'name',
            'name_value': 'Idaho',
            'default_cloud_threshold': 25,
            'default_samples': 3500
        },
        'florida': {
            'type': 'state',
            'country': 'United States of America',
            'name_field': 'name',
            'name_value': 'Florida',
            'default_cloud_threshold': 35,
            'default_samples': 6000
        },
        'new_york': {
            'type': 'state',
            'country': 'United States of America',
            'name_field': 'name',
            'name_value': 'New York',
            'default_cloud_threshold': 30,
            'default_samples': 5000
        }
    }
    
    # Sub-region definitions for stratified sampling
    SUB_REGIONS = {
        'africa': [
            {'name': 'North Africa', 'countries': ['Egypt', 'Libya', 'Tunisia', 'Algeria', 'Morocco'], 'weight': 0.15},
            {'name': 'West Africa', 'countries': ['Nigeria', 'Ghana', 'Senegal', 'Mali', 'Burkina Faso'], 'weight': 0.20},
            {'name': 'East Africa', 'countries': ['Ethiopia', 'Kenya', 'Tanzania', 'Uganda', 'Somalia'], 'weight': 0.20},
            {'name': 'Central Africa', 'countries': ['Dem. Rep. Congo', 'Cameroon', 'Chad', 'Central African Rep.'], 'weight': 0.15},
            {'name': 'Southern Africa', 'countries': ['South Africa', 'Zimbabwe', 'Botswana', 'Namibia', 'Zambia'], 'weight': 0.20},
            {'name': 'Madagascar', 'countries': ['Madagascar'], 'weight': 0.10}
        ],
        'united_states': [
            {'name': 'West Coast', 'states': ['California', 'Oregon', 'Washington'], 'weight': 0.15},
            {'name': 'Mountain West', 'states': ['Montana', 'Idaho', 'Wyoming', 'Colorado', 'Utah', 'Nevada'], 'weight': 0.20},
            {'name': 'Southwest', 'states': ['Arizona', 'New Mexico', 'Texas', 'Oklahoma'], 'weight': 0.15},
            {'name': 'Midwest', 'states': ['North Dakota', 'South Dakota', 'Nebraska', 'Kansas', 'Minnesota', 'Iowa', 'Missouri', 'Wisconsin', 'Illinois', 'Michigan', 'Indiana', 'Ohio'], 'weight': 0.20},
            {'name': 'Southeast', 'states': ['Florida', 'Georgia', 'South Carolina', 'North Carolina', 'Virginia', 'Tennessee', 'Kentucky', 'Alabama', 'Mississippi', 'Louisiana', 'Arkansas'], 'weight': 0.20},
            {'name': 'Northeast', 'states': ['Maine', 'New Hampshire', 'Vermont', 'Massachusetts', 'Rhode Island', 'Connecticut', 'New York', 'New Jersey', 'Pennsylvania', 'Delaware', 'Maryland', 'West Virginia'], 'weight': 0.10}
        ],
        'europe': [
            {'name': 'Western Europe', 'countries': ['France', 'Germany', 'Belgium', 'Netherlands', 'Luxembourg', 'Switzerland', 'Austria'], 'weight': 0.25},
            {'name': 'Southern Europe', 'countries': ['Spain', 'Portugal', 'Italy', 'Greece'], 'weight': 0.20},
            {'name': 'Northern Europe', 'countries': ['Norway', 'Sweden', 'Finland', 'Denmark', 'Iceland'], 'weight': 0.15},
            {'name': 'Eastern Europe', 'countries': ['Poland', 'Czech Rep.', 'Slovakia', 'Hungary', 'Romania', 'Bulgaria'], 'weight': 0.20},
            {'name': 'British Isles', 'countries': ['United Kingdom', 'Ireland'], 'weight': 0.10},
            {'name': 'Balkans', 'countries': ['Serbia', 'Croatia', 'Bosnia and Herz.', 'Albania', 'Macedonia', 'Montenegro'], 'weight': 0.10}
        ]
    }
    
    def __init__(self):
        """Initialize the region sampler and load shapefiles."""
        self.shapefiles = {}
        self._load_shapefiles()
    
    def _load_shapefiles(self):
        """Load all required shapefiles from local paths."""
        
        # Load countries shapefile
        if os.path.exists(self.SHAPEFILE_PATHS['countries']):
            try:
                self.shapefiles['countries'] = gpd.read_file(self.SHAPEFILE_PATHS['countries'])
                logger.info(f"Loaded countries shapefile with {len(self.shapefiles['countries'])} features")
            except Exception as e:
                logger.warning(f"Could not load countries shapefile: {e}")
        else:
            logger.warning(f"Countries shapefile not found at {self.SHAPEFILE_PATHS['countries']}")
        
        # Load states shapefile
        if os.path.exists(self.SHAPEFILE_PATHS['states']):
            try:
                self.shapefiles['states'] = gpd.read_file(self.SHAPEFILE_PATHS['states'])
                logger.info(f"Loaded states shapefile with {len(self.shapefiles['states'])} features")
            except Exception as e:
                logger.warning(f"Could not load states shapefile: {e}")
        else:
            logger.warning(f"States shapefile not found at {self.SHAPEFILE_PATHS['states']}")
        
        # Load individual continent shapefiles
        continents_dir = self.SHAPEFILE_PATHS['continents_dir']
        if os.path.exists(continents_dir):
            for continent_key, shapefile_name in self.CONTINENT_FILES.items():
                shapefile_path = os.path.join(continents_dir, shapefile_name)
                if os.path.exists(shapefile_path):
                    try:
                        self.shapefiles[f'continent_{continent_key}'] = gpd.read_file(shapefile_path)
                        logger.info(f"Loaded {continent_key} continent shapefile")
                    except Exception as e:
                        logger.warning(f"Could not load {continent_key} shapefile: {e}")
                else:
                    logger.warning(f"Continent shapefile not found: {shapefile_path}")
        else:
            logger.warning(f"Continents directory not found at {continents_dir}")
    
    def get_region_geometry(self, region_name: str) -> Union[Polygon, MultiPolygon]:
        """Get the geometry for a region from shapefiles."""
        
        if region_name not in self.REGION_CONFIG:
            raise ValueError(f"Unknown region: {region_name}")
        
        config = self.REGION_CONFIG[region_name]
        
        if config['type'] == 'continent':
            # Use individual continent shapefile
            shapefile_key = f"continent_{config['shapefile']}"
            if shapefile_key not in self.shapefiles:
                raise ValueError(f"Continent shapefile not loaded for {region_name}")
            
            gdf = self.shapefiles[shapefile_key]
            # Continent shapefiles typically contain the entire continent as one or few features
            geometry = unary_union(gdf.geometry)
            
        elif config['type'] == 'country':
            if 'countries' not in self.shapefiles:
                raise ValueError("Countries shapefile not loaded")
            
            gdf = self.shapefiles['countries']
            region_gdf = gdf[gdf[config['name_field']] == config['name_value']]
            
            # Handle special cases
            if config.get('metropolitan_only') and region_name == 'france':
                # Filter to metropolitan France (European part)
                if 'SUBREGION' in region_gdf.columns:
                    region_gdf = region_gdf[region_gdf['SUBREGION'] == 'Western Europe']
            
            elif config.get('exclude_regions') and region_name == 'united_states':
                # For US, exclude Alaska and Hawaii
                # This requires checking the subunits or filtering by longitude
                if not region_gdf.empty:
                    bounds = region_gdf.bounds
                    # Continental US is roughly between -125 and -66 longitude
                    continental = region_gdf[(bounds['minx'] > -130) & (bounds['maxx'] < -60)]
                    if not continental.empty:
                        region_gdf = continental
            
            if region_gdf.empty:
                raise ValueError(f"No geometry found for country: {region_name}")
            
            geometry = unary_union(region_gdf.geometry)
        
        elif config['type'] == 'state':
            if 'states' not in self.shapefiles:
                raise ValueError("States/provinces shapefile not loaded")
            
            gdf = self.shapefiles['states']
            # Filter by country first, then state
            country_gdf = gdf[gdf['admin'] == config.get('country', 'United States of America')]
            region_gdf = country_gdf[country_gdf[config['name_field']] == config['name_value']]
            
            if region_gdf.empty:
                raise ValueError(f"No geometry found for state: {region_name}")
            
            geometry = unary_union(region_gdf.geometry)
        
        else:
            raise ValueError(f"Unknown region type: {config['type']}")
        
        return geometry
    
    def get_region_bounds(self, region_name: str) -> List[List[float]]:
        """Get the bounding box for a region as polygon coordinates."""
        geometry = self.get_region_geometry(region_name)
        bounds = geometry.bounds  # (minx, miny, maxx, maxy)
        
        # Return as polygon coordinates
        return [
            [bounds[0], bounds[1]],  # SW
            [bounds[2], bounds[1]],  # SE
            [bounds[2], bounds[3]],  # NE
            [bounds[0], bounds[3]],  # NW
            [bounds[0], bounds[1]]   # Close polygon
        ]
    
    def get_region_config(self, region_name: str) -> Dict:
        """Get the configuration for a region."""
        if region_name not in self.REGION_CONFIG:
            raise ValueError(f"Unknown region: {region_name}")
        
        config = self.REGION_CONFIG[region_name].copy()
        
        # Add bounds from geometry
        try:
            geometry = self.get_region_geometry(region_name)
            bounds = geometry.bounds
            config['bounds'] = [[bounds[0], bounds[1]], [bounds[2], bounds[3]]]
        except Exception as e:
            logger.warning(f"Could not get bounds for {region_name}: {e}")
        
        return config
    
    def generate_sample_points(self, region_name: str, n_samples: int, 
                              strategy: str = 'random') -> List[Dict]:
        """
        Generate sampling points within a region using shapefile boundaries.
        
        Parameters:
        -----------
        region_name : str
            Name of the region
        n_samples : int
            Number of samples to generate
        strategy : str
            Sampling strategy ('random' or 'stratified')
            
        Returns:
        --------
        List[Dict] : List of sample points with coordinates
        """
        
        logger.info(f"Generating {n_samples} sample points for {region_name} using {strategy} strategy")
        
        points = []
        
        if strategy == 'stratified' and region_name in self.SUB_REGIONS:
            # Stratified sampling by sub-regions
            points = self._stratified_sampling(region_name, n_samples)
        else:
            # Simple random sampling within the region
            points = self._random_sampling(region_name, n_samples)
        
        logger.info(f"Generated {len(points)} valid sample points")
        
        return points
    
    def _random_sampling(self, region_name: str, n_samples: int) -> List[Dict]:
        """Generate random points within a region's boundaries."""
        
        try:
            geometry = self.get_region_geometry(region_name)
        except Exception as e:
            logger.error(f"Could not get geometry for {region_name}: {e}")
            return []
        
        bounds = geometry.bounds  # (minx, miny, maxx, maxy)
        points = []
        max_attempts = n_samples * 20  # Increased attempts for difficult geometries
        attempts = 0
        
        while len(points) < n_samples and attempts < max_attempts:
            # Generate random point within bounding box
            lon = random.uniform(bounds[0], bounds[2])
            lat = random.uniform(bounds[1], bounds[3])
            point = Point(lon, lat)
            attempts += 1
            
            # Check if point is within actual geometry
            if geometry.contains(point):
                points.append({
                    'id': len(points),
                    'lon': lon,
                    'lat': lat,
                    'region': region_name
                })
        
        if len(points) < n_samples:
            logger.warning(f"Only generated {len(points)}/{n_samples} points after {attempts} attempts")
        
        return points
    
    def _stratified_sampling(self, region_name: str, n_samples: int) -> List[Dict]:
        """Generate stratified samples across sub-regions."""
        
        if region_name not in self.SUB_REGIONS:
            return self._random_sampling(region_name, n_samples)
        
        sub_regions = self.SUB_REGIONS[region_name]
        points = []
        
        # Normalize weights
        total_weight = sum(sr['weight'] for sr in sub_regions)
        
        for sub_region in sub_regions:
            n_sub_samples = int(n_samples * sub_region['weight'] / total_weight)
            
            logger.info(f"  Generating {n_sub_samples} points for {sub_region['name']}")
            
            # Get geometries for this sub-region
            geometries = []
            
            if 'countries' in sub_region and 'countries' in self.shapefiles:
                # Combine country geometries
                gdf = self.shapefiles['countries']
                for country in sub_region['countries']:
                    country_gdf = gdf[gdf['NAME'] == country]
                    if not country_gdf.empty:
                        geometries.extend(country_gdf.geometry.tolist())
            
            elif 'states' in sub_region and 'states' in self.shapefiles:
                # Combine state geometries
                gdf = self.shapefiles['states']
                for state in sub_region['states']:
                    state_gdf = gdf[gdf['name'] == state]
                    if not state_gdf.empty:
                        geometries.extend(state_gdf.geometry.tolist())
            
            if not geometries:
                logger.warning(f"No geometries found for sub-region {sub_region['name']}")
                continue
            
            sub_geometry = unary_union(geometries)
            
            # Sample within this sub-region
            bounds = sub_geometry.bounds
            sub_points = 0
            max_attempts = n_sub_samples * 20
            attempts = 0
            
            while sub_points < n_sub_samples and attempts < max_attempts:
                lon = random.uniform(bounds[0], bounds[2])
                lat = random.uniform(bounds[1], bounds[3])
                point = Point(lon, lat)
                attempts += 1
                
                if sub_geometry.contains(point):
                    points.append({
                        'id': len(points),
                        'lon': lon,
                        'lat': lat,
                        'region': region_name,
                        'sub_region': sub_region['name']
                    })
                    sub_points += 1
        
        # Log distribution
        if points and 'sub_region' in points[0]:
            distribution = {}
            for p in points:
                sr = p.get('sub_region', 'Unknown')
                distribution[sr] = distribution.get(sr, 0) + 1
            logger.info("Sample distribution:")
            for sr, count in sorted(distribution.items()):
                logger.info(f"  {sr}: {count} samples ({100*count/len(points):.1f}%)")
        
        return points
    
    @classmethod
    def list_available_regions(cls) -> Dict[str, List[str]]:
        """List all available regions organized by type."""
        regions_by_type = {
            'continents': [],
            'countries': [],
            'states': []
        }
        
        for region_name, config in cls.REGION_CONFIG.items():
            if config['type'] == 'continent':
                regions_by_type['continents'].append(region_name)
            elif config['type'] == 'country':
                regions_by_type['countries'].append(region_name)
            elif config['type'] == 'state':
                regions_by_type['states'].append(region_name)
        
        return regions_by_type


# Compatibility layer for existing code
REGIONS = RegionSampler.REGION_CONFIG