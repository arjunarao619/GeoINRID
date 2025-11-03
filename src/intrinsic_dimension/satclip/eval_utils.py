
import sys
sys.path.append('./clip')
sys.path.append('./src')
sys.path.append('./experiments/src')
# sys.path.append('./experiments')
sys.path.append('./experiments/src/csp')

from experiments.src.csp import *
from experiments.src.knn import *
from experiments.src.data_utils import *
from experiments.src.split import *
from experiments.src.pretrained_models import *
import os
from sklearn.ensemble import RandomForestRegressor
from experiments.src.wandb_utils import *

from experiments.src.csp.utils import *
from experiments.src.csp.trainer import *
from experiments.src.csp.models import *
from clip.location_encoder import *

import wandb
from clip.main import *
from clip.load import get_geoclip
from src.sc_main import *
from src.sc_load import get_satclip
from clip.positional_encoding.wrap import Wrap

from sklearn.metrics import (
    accuracy_score, 
    jaccard_score, 
    average_precision_score, 
    roc_auc_score,
    mean_absolute_error,
    r2_score
)

from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import torch.nn as nn
import csv
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, Timer

import optuna
import yaml


import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from skdim.id import MLE, TwoNN, ESS
from scipy.spatial import cKDTree

from skdim.id import (
    MLE, TwoNN, ESS, FisherS, MiND_ML, 
    DANCo, lPCA, CorrInt, MOM, TLE, KNN
)
import time
# from id_error_analysis import analyze_id_error_relationship
import pandas as pd


TUNE_RESULTS_DIR = "results/tune"
TUNE_CONT_RESULTS_DIR = "results/tune_continents"

import logging
logging.getLogger("lightning").setLevel(logging.ERROR)


def deduplicate_embeddings(emb_train, emb_val, emb_test, y_train, y_val, y_test, x_train=None, x_val=None, x_test=None):
    """
    Remove duplicate embeddings within each split and ensure no overlap between splits.
    Returns deduplicated embeddings and corresponding labels.
    """
    def get_unique_indices(embeddings):
        """Get indices of unique embeddings"""
        emb_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        # Round to avoid floating point precision issues
        emb_rounded = np.round(emb_np, decimals=6)
        _, unique_indices = np.unique(emb_rounded, axis=0, return_index=True)
        return sorted(unique_indices)
    
    # Get unique indices for each split
    train_unique_idx = get_unique_indices(emb_train)
    val_unique_idx = get_unique_indices(emb_val)
    test_unique_idx = get_unique_indices(emb_test)
    
    # Filter embeddings and labels
    emb_train_unique = emb_train[train_unique_idx]
    emb_val_unique = emb_val[val_unique_idx]
    emb_test_unique = emb_test[test_unique_idx]
    
    y_train_unique = y_train[train_unique_idx]
    y_val_unique = y_val[val_unique_idx]
    y_test_unique = y_test[test_unique_idx]
    
    # Handle x (images) if provided
    if x_train is not None:
        x_train_unique = x_train[train_unique_idx]
        x_val_unique = x_val[val_unique_idx]
        x_test_unique = x_test[test_unique_idx]
    else:
        x_train_unique, x_val_unique, x_test_unique = None, None, None
    
    # Check for overlaps between splits
    def check_overlap(emb1, emb2, name1, name2):
        emb1_np = emb1.cpu().numpy() if isinstance(emb1, torch.Tensor) else emb1
        emb2_np = emb2.cpu().numpy() if isinstance(emb2, torch.Tensor) else emb2
        
        # Create sets of rounded embeddings for comparison
        emb1_tuples = set([tuple(np.round(e, decimals=6)) for e in emb1_np])
        emb2_tuples = set([tuple(np.round(e, decimals=6)) for e in emb2_np])
        
        overlap = emb1_tuples.intersection(emb2_tuples)
        if overlap:
            print(f"WARNING: Found {len(overlap)} overlapping embeddings between {name1} and {name2}")
            
            # Remove overlapping embeddings from the second set
            overlap_mask = np.array([tuple(np.round(e, decimals=6)) not in overlap for e in emb2_np])
            return overlap_mask
        return np.ones(len(emb2_np), dtype=bool)
    
    # Check and remove overlaps (prioritize train > val > test)
    val_mask = check_overlap(emb_train_unique, emb_val_unique, "train", "val")
    emb_val_unique = emb_val_unique[val_mask]
    y_val_unique = y_val_unique[val_mask]
    if x_val_unique is not None:
        x_val_unique = x_val_unique[val_mask]
    
    test_mask_train = check_overlap(emb_train_unique, emb_test_unique, "train", "test")
    test_mask_val = check_overlap(emb_val_unique, emb_test_unique, "val", "test")
    test_mask = test_mask_train & test_mask_val
    emb_test_unique = emb_test_unique[test_mask]
    y_test_unique = y_test_unique[test_mask]
    if x_test_unique is not None:
        x_test_unique = x_test_unique[test_mask]
    
    # Print statistics
    print(f"Deduplication statistics:")
    print(f"  Train: {len(emb_train)} -> {len(emb_train_unique)} ({len(emb_train) - len(emb_train_unique)} duplicates removed)")
    print(f"  Val: {len(emb_val)} -> {len(emb_val_unique)} ({len(emb_val) - len(emb_val_unique)} duplicates + overlaps removed)")
    print(f"  Test: {len(emb_test)} -> {len(emb_test_unique)} ({len(emb_test) - len(emb_test_unique)} duplicates + overlaps removed)")
    
    return (emb_train_unique, emb_val_unique, emb_test_unique, 
            y_train_unique, y_val_unique, y_test_unique,
            x_train_unique, x_val_unique, x_test_unique)

            
def get_data(hparams): 
    if hparams['dataset']=='air_temp':
        hparams['task'] ='regression'
        df = get_air_temp_data_gdf(hparams['path_to_air_temp'],folder_del='/')
        y = torch.tensor(df['Temperature'].values)
        c = torch.tensor(df[['Lon','Lat']].values)
        train_idx, val_idx, test_idx = random_split(c,train_size=hparams['train_size'], val_size=hparams['val_size'], random_state=hparams['seed'])
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        c_train = c[train_idx]
        c_val = c[val_idx]
        c_test = c[test_idx]
    elif hparams['dataset']=='med_income':
        hparams['task'] = 'regression'
        df = get_election_data_gdf(hparams['path_to_med_income'],folder_del='/')
        y = torch.tensor(df['MedianIncome2016'].values)
        c = torch.tensor(df[['Lon','Lat']].values)
        train_idx, val_idx, test_idx = random_split(c,train_size=hparams['train_size'], val_size=hparams['val_size'], random_state=hparams['seed'])
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        c_train = c[train_idx]
        c_val = c[val_idx]
        c_test = c[test_idx]
    elif hparams['dataset']=='cali_housing':
        hparams['task'] = 'regression'
        df = get_cali_housing_data_gdf()
        c = torch.tensor(df[['Longitude','Latitude']].values)
        y = torch.tensor(df['MedHouseVal'].values)
        train_idx, val_idx, test_idx = random_split(c,train_size=hparams['train_size'], val_size=hparams['val_size'], random_state=hparams['seed'])
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        c_train = c[train_idx]
        c_val = c[val_idx]
        c_test = c[test_idx]
    elif hparams['dataset']=='elevation':
        hparams['task'] = 'regression'
        df = get_elev_data_gdf(hparams['path_to_elevation'])
        df = df.dropna()
        c = torch.tensor(df[['Lon','Lat']].values)
        y = torch.tensor(df['elevation'].values)
        y = y / y.max()
        train_idx, val_idx, test_idx = random_split(c,train_size=hparams['train_size'], val_size=hparams['val_size'], random_state=hparams['seed'])
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        c_train = c[train_idx]
        c_val = c[val_idx]
        c_test = c[test_idx]
    elif hparams['dataset']=='population':
        hparams['task'] = 'regression'
        df = get_pop_data_gdf(hparams['path_to_population'])
        df = df.dropna()
        c = torch.tensor(df[['Lon','Lat']].values)
        y = torch.tensor(df['log_population'].values)
        y = y / y.max()
        train_idx, val_idx, test_idx = random_split(c,train_size=hparams['train_size'], val_size=hparams['val_size'], random_state=hparams['seed'])
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        c_train = c[train_idx]
        c_val = c[val_idx]
        c_test = c[test_idx]
    elif hparams['dataset'] in ['countries','biome','ecoregions']:
        hparams['task'] = 'classification'
        if hparams['dataset']=='countries':
            df = get_countries_data_gdf(hparams['path_to_countries'])
        else:
            df = get_biome_data_gdf(hparams['path_to_biome'])
        c = torch.tensor(df[['lon','lat']].values)
        if hparams['dataset']=='countries':
            y = torch.tensor(df['country'].values)
        elif hparams['dataset']=='biome':
            y = torch.tensor(df['biome_class_index'].values)
        elif hparams['dataset']=='ecoregions':
            y = torch.tensor(df['ecoregion_class_index'].values)
        unique_class_labels, indices = torch.unique(y, return_inverse=True)  
        num_classes = len(unique_class_labels)  
        mapped_class_labels = indices  
        y = F.one_hot(mapped_class_labels, num_classes).unsqueeze(-1)
        train_idx, val_idx, test_idx = random_split(c,train_size=hparams['train_size'], val_size=hparams['val_size'], random_state=hparams['seed'])
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        c_train = c[train_idx]
        c_val = c[val_idx]
        c_test = c[test_idx]
    elif hparams['dataset'] == 'inat':
        hparams['task'] ='imageclassification'
        with open(hparams['path_to_inat'] + '/train2018.json') as f:
            train_dict = json.load(f)
        df_train_annotations = pd.json_normalize(train_dict['annotations'])
        df_train_loc = pd.read_json(hparams['path_to_inat'] + '/inat2018_locations/train2018_locations.json')
        df_train = pd.merge(df_train_annotations, df_train_loc, on='id')
        df_train = df_train[['category_id','lon','lat']].dropna()
        with open(hparams['path_to_inat'] + '/val2018.json') as f:
            val_dict = json.load(f)
        df_val_annotations = pd.json_normalize(val_dict['annotations'])
        df_val_loc = pd.read_json(hparams['path_to_inat'] + '/inat2018_locations/val2018_locations.json')
        df_val = pd.merge(df_val_annotations, df_val_loc, on='id')
        df_val = df_val[['category_id','lon','lat']].dropna()
        c_train = torch.tensor(df_train[['lon','lat']].values)
        c_test = torch.tensor(df_val[['lon','lat']].values)
        y_train = torch.tensor(df_train['category_id'].values)
        y_test = torch.tensor(df_val['category_id'].values)
        x_train = torch.tensor(np.load(hparams['path_to_inat'] + '/features_inception/inat2018_train_net_feats.npy'))
        x_test = torch.tensor(np.load(hparams['path_to_inat'] + '/features_inception/inat2018_val_net_feats.npy'))
        train_idx, val_idx, test_idx = random_split(c_train,train_size=hparams['train_size'], val_size=hparams['val_size'], random_state=hparams['seed'])
        c_val = c_train[val_idx]
        y_val = y_train[val_idx]
        x_val = x_train[val_idx]
        train_idx = np.concatenate((train_idx,test_idx))
        c_train = c_train[train_idx]
        y_train = y_train[train_idx]
        x_train = x_train[train_idx]
        test_idx = torch.arange(c_test.shape[0])
    else:
        raise ValueError('Dataset not found.')
    
    if hparams['dataset'] not in ['inat']:
        x_train, x_val, x_test = None, None, None
    
    # if hparams['dataset'] in ['inat']:
    #     train_idx, val_idx, test_idx = None, None, None

    return y_train, y_val, y_test, c_train, c_val, c_test, x_train, x_val, x_test, train_idx, val_idx, test_idx

def get_data_continents(hparams): 
    if hparams['dataset']=='air_temp':
        hparams['task'] ='regression'
        df = get_air_temp_data_gdf(hparams['path_to_air_temp'],folder_del='/')
        y = torch.tensor(df['Temperature'].values)
        c = torch.tensor(df[['Lon','Lat']].values)
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        continent = world[world['continent']==hparams['continent']]
        train_idx = ~gpd.points_from_xy(c[:,0], c[:,1]).within(continent.unary_union)
        test_idx = gpd.points_from_xy(c[:,0], c[:,1]).within(continent.unary_union)
        train_idx_rel, val_idx_rel = random_split_train_test(c[train_idx],train_size=hparams['train_size'], val_size=hparams['val_size'], random_state=hparams['seed'])
        val_idx = np.flatnonzero(train_idx)[val_idx_rel]  
        train_idx = np.flatnonzero(train_idx)[train_idx_rel]  
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        c_train = c[train_idx]
        c_val = c[val_idx]
        c_test = c[test_idx]
    elif hparams['dataset']=='med_income':
        hparams['task'] = 'regression'
        df = get_election_data_gdf(hparams['path_to_med_income'],folder_del='/')
        y = torch.tensor(df['MedianIncome2016'].values)
        c = torch.tensor(df[['Lon','Lat']].values)
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        continent = world[world['continent']==hparams['continent']]
        train_idx = ~gpd.points_from_xy(c[:,0], c[:,1]).within(continent.unary_union)
        test_idx = gpd.points_from_xy(c[:,0], c[:,1]).within(continent.unary_union)
        train_idx_rel, val_idx_rel = random_split_train_test(c[train_idx],train_size=hparams['train_size'], val_size=hparams['val_size'], random_state=hparams['seed'])
        val_idx = np.flatnonzero(train_idx)[val_idx_rel]  
        train_idx = np.flatnonzero(train_idx)[train_idx_rel]  
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        c_train = c[train_idx]
        c_val = c[val_idx]
        c_test = c[test_idx]
    elif hparams['dataset']=='cali_housing':
        hparams['task'] = 'regression'
        df = get_cali_housing_data_gdf()
        c = torch.tensor(df[['Longitude','Latitude']].values)
        y = torch.tensor(df['MedHouseVal'].values)
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        continent = world[world['continent']==hparams['continent']]
        train_idx = ~gpd.points_from_xy(c[:,0], c[:,1]).within(continent.unary_union)
        test_idx = gpd.points_from_xy(c[:,0], c[:,1]).within(continent.unary_union)
        train_idx_rel, val_idx_rel = random_split_train_test(c[train_idx],train_size=hparams['train_size'], val_size=hparams['val_size'], random_state=hparams['seed'])
        val_idx = np.flatnonzero(train_idx)[val_idx_rel]  
        train_idx = np.flatnonzero(train_idx)[train_idx_rel]  
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        c_train = c[train_idx]
        c_val = c[val_idx]
        c_test = c[test_idx]
    elif hparams['dataset']=='elevation':
        hparams['task'] = 'regression'
        df = get_elev_data_gdf(hparams['path_to_elevation'])
        df = df.dropna()
        c = torch.tensor(df[['Lon','Lat']].values)
        y = torch.tensor(df['elevation'].values)
        y = y / y.max()
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        continent = world[world['continent']==hparams['continent']]
        train_idx = ~gpd.points_from_xy(c[:,0], c[:,1]).within(continent.unary_union)
        test_idx = gpd.points_from_xy(c[:,0], c[:,1]).within(continent.unary_union)
        train_idx_rel, val_idx_rel = random_split_train_test(c[train_idx],train_size=hparams['train_size'], val_size=hparams['val_size'], random_state=hparams['seed'])
        val_idx = np.flatnonzero(train_idx)[val_idx_rel]  
        train_idx = np.flatnonzero(train_idx)[train_idx_rel]  
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        c_train = c[train_idx]
        c_val = c[val_idx]
        c_test = c[test_idx]
    elif hparams['dataset']=='population':
        hparams['task'] = 'regression'
        df = get_pop_data_gdf(hparams['path_to_population'])
        df = df.dropna()
        c = torch.tensor(df[['Lon','Lat']].values)
        y = torch.tensor(df['log_population'].values)
        y = y / y.max()
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        continent = world[world['continent']==hparams['continent']]
        train_idx = ~gpd.points_from_xy(c[:,0], c[:,1]).within(continent.unary_union)
        test_idx = gpd.points_from_xy(c[:,0], c[:,1]).within(continent.unary_union)
        train_idx_rel, val_idx_rel = random_split_train_test(c[train_idx],train_size=hparams['train_size'], val_size=hparams['val_size'], random_state=hparams['seed'])
        val_idx = np.flatnonzero(train_idx)[val_idx_rel]  
        train_idx = np.flatnonzero(train_idx)[train_idx_rel]  
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        c_train = c[train_idx]
        c_val = c[val_idx]
        c_test = c[test_idx]
    elif hparams['dataset']=='countries':
        hparams['task'] = 'classification'
        df = get_countries_data_gdf(hparams['path_to_countries'])
        c = torch.tensor(df[['lon','lat']].values)
        y = torch.tensor(df['country'].values)
        unique_class_labels, indices = torch.unique(y, return_inverse=True)  
        num_classes = len(unique_class_labels)  
        mapped_class_labels = indices  
        y = F.one_hot(mapped_class_labels, num_classes).unsqueeze(-1)
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        continent = world[world['continent']==hparams['continent']]
        train_idx = ~gpd.points_from_xy(c[:,0], c[:,1]).within(continent.unary_union)
        test_idx = gpd.points_from_xy(c[:,0], c[:,1]).within(continent.unary_union)
        train_idx_rel, val_idx_rel = random_split_train_test(c[train_idx],train_size=hparams['train_size'], val_size=hparams['val_size'], random_state=hparams['seed'])
        val_idx = np.flatnonzero(train_idx)[val_idx_rel]  
        train_idx = np.flatnonzero(train_idx)[train_idx_rel] 
        train_idx_rel, test_idx_rel = random_split_train_test(c[test_idx],train_size=0.01, val_size=hparams['val_size'], random_state=hparams['seed'])
        train_add_idx = np.flatnonzero(test_idx)[train_idx_rel]  
        test_idx = np.flatnonzero(test_idx)[test_idx_rel] 
        train_idx = np.concatenate((train_idx,train_add_idx))
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        c_train = c[train_idx]
        c_val = c[val_idx]
        c_test = c[test_idx]
    elif hparams['dataset'] in ['biome','ecoregions']:
        hparams['task'] = 'classification'
        df = get_biome_data_gdf(hparams['path_to_biome'])
        df = df.dropna()
        c = torch.tensor(df[['lon','lat']].values)
        if hparams['dataset']=='biome':
            y = torch.tensor(df['biome_class_index'].values)
            unique_class_labels, indices = torch.unique(y, return_inverse=True)  
            num_classes = len(unique_class_labels)  
            mapped_class_labels = indices  
            y = F.one_hot(mapped_class_labels, num_classes).unsqueeze(-1)
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            continent = world[world['continent']==hparams['continent']]
            train_idx = ~gpd.points_from_xy(c[:,0], c[:,1]).within(continent.unary_union)
            test_idx = gpd.points_from_xy(c[:,0], c[:,1]).within(continent.unary_union)
            train_idx_rel, val_idx_rel = random_split_train_test(c[train_idx],train_size=hparams['train_size'], val_size=hparams['val_size'], random_state=hparams['seed'])
            val_idx = np.flatnonzero(train_idx)[val_idx_rel]  
            train_idx = np.flatnonzero(train_idx)[train_idx_rel]  
            y_train = y[train_idx]
            y_val = y[val_idx]
            y_test = y[test_idx]
            c_train = c[train_idx]
            c_val = c[val_idx]
            c_test = c[test_idx]
        elif hparams['dataset']=='ecoregions':
            y = torch.tensor(df['ecoregion_class_index'].values)
            unique_class_labels, indices = torch.unique(y, return_inverse=True)  
            num_classes = len(unique_class_labels)  
            mapped_class_labels = indices  
            y = F.one_hot(mapped_class_labels, num_classes).unsqueeze(-1)
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            continent = world[world['continent']==hparams['continent']]
            train_idx = ~gpd.points_from_xy(c[:,0], c[:,1]).within(continent.unary_union)
            test_idx = gpd.points_from_xy(c[:,0], c[:,1]).within(continent.unary_union)
            train_idx_rel, val_idx_rel = random_split_train_test(c[train_idx],train_size=hparams['train_size'], val_size=hparams['val_size'], random_state=hparams['seed'])
            val_idx = np.flatnonzero(train_idx)[val_idx_rel]  
            train_idx = np.flatnonzero(train_idx)[train_idx_rel] 
            train_idx_rel, test_idx_rel = random_split_train_test(c[test_idx],train_size=0.01, val_size=hparams['val_size'], random_state=hparams['seed'])
            train_add_idx = np.flatnonzero(test_idx)[train_idx_rel]  
            test_idx = np.flatnonzero(test_idx)[test_idx_rel] 
            train_idx = np.concatenate((train_idx,train_add_idx))
            y_train = y[train_idx]
            y_val = y[val_idx]
            y_test = y[test_idx]
            c_train = c[train_idx]
            c_val = c[val_idx]
            c_test = c[test_idx]
    elif hparams['dataset'] == 'inat':
        hparams['task'] ='imageclassification'
        with open(hparams['path_to_inat'] + '/train2018.json') as f:
            train_dict = json.load(f)
        df_train_annotations = pd.json_normalize(train_dict['annotations'])
        df_train_loc = pd.read_json(hparams['path_to_inat'] + '/inat2018_locations/train2018_locations.json')
        df_train = pd.merge(df_train_annotations, df_train_loc, on='id')
        df_train = df_train[['category_id','lon','lat']].dropna()
        with open(hparams['path_to_inat'] + '/val2018.json') as f:
            val_dict = json.load(f)
        df_val_annotations = pd.json_normalize(val_dict['annotations'])
        df_val_loc = pd.read_json(hparams['path_to_inat'] + '/inat2018_locations/val2018_locations.json')
        df_val = pd.merge(df_val_annotations, df_val_loc, on='id')
        df_val = df_val[['category_id','lon','lat']].dropna()
        c_train = torch.tensor(df_train[['lon','lat']].values)
        c_test = torch.tensor(df_val[['lon','lat']].values)
        y_train = torch.tensor(df_train['category_id'].values)
        y_test = torch.tensor(df_val['category_id'].values)
        x_train = torch.tensor(np.load(hparams['path_to_inat'] + '/features_inception/inat2018_train_net_feats.npy'))
        x_test = torch.tensor(np.load(hparams['path_to_inat'] + '/features_inception/inat2018_val_net_feats.npy'))
        c = torch.cat((c_train,c_test),0)
        y = torch.cat((y_train,y_test),0)
        x = torch.cat((x_train,x_test),0)
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        continent = world[world['continent']==hparams['continent']]
        train_idx = ~gpd.points_from_xy(c[:,0], c[:,1]).within(continent.unary_union)
        test_idx = gpd.points_from_xy(c[:,0], c[:,1]).within(continent.unary_union)
        train_idx_rel, val_idx_rel = random_split_train_test(c[train_idx],train_size=hparams['train_size'], val_size=hparams['val_size'], random_state=hparams['seed'])
        val_idx = np.flatnonzero(train_idx)[val_idx_rel]  
        train_idx = np.flatnonzero(train_idx)[train_idx_rel]  
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        x_train = x[train_idx]
        x_val = x[val_idx]
        x_test = x[test_idx]
        c_train = c[train_idx]
        c_val = c[val_idx]
        c_test = c[test_idx]
    else:
        raise ValueError('Dataset not found.')
    
    if hparams['dataset'] not in ['inat']:
        x_train, x_val, x_test = None, None, None
    
    # if hparams['dataset'] in ['inat']:
    #     train_idx, val_idx, test_idx = None, None, None

    return y_train, y_val, y_test, c_train, c_val, c_test, x_train, x_val, x_test, train_idx, val_idx, test_idx

def get_id_hyperparameters():
    """Get hyperparameters for intrinsic dimension analysis."""
    return {
        # ID analysis settings
        'analyze_intrinsic_dim': False,
        'calculate_local_id': False,
        
        # Methods to use for global ID (start with reliable ones)
        'id_methods': ['MLE'],  # Start with just these two reliable methods
        
        # Methods to use for local ID (must support fit_pw)
        'local_id_methods': ['MLE'],  # Start with just MLE for local
        
        # Sampling and computation settings
        'id_max_samples': 100000,      # Reasonable default
        'id_k_neighbors': 20,        # Number of neighbors for local ID
        'n_jobs': -1,                  # Parallel workers
    }

def get_embeddings(hparams, train_idx, val_idx, test_idx, c_train, c_val, c_test):
    """Streamlined embedding loading with scikit-dimension based ID analysis."""
    le_name = hparams['le_name']
    device = hparams['device']
    
    print(f"\n{'='*60}")
    print(f"Loading embeddings for {le_name} on {hparams['dataset']}")
    print(f"{'='*60}")
    start_time = time.time()
    
    
    # Special cases that don't use models
    if le_name == 'ident':
        emb_train, emb_val, emb_test = c_train.float(), c_val.float(), c_test.float()
    elif le_name == 'wrap':
        pe = Wrap()
        emb_train = pe(c_train.float())
        emb_val = pe(c_val.float())
        emb_test = pe(c_test.float())
    elif le_name in ['gps2vec-tag', 'gps2vec-visual']:
        emb_train, emb_val, emb_test = _load_gps2vec_embeddings(hparams, c_train, c_val, c_test, le_name)
    elif le_name == 'mosaiks':
        emb_train, emb_val, emb_test = _load_mosaiks_embeddings(hparams, c_train, c_val, c_test)
    else:
        # Model-based embeddings
        model = load_model(hparams, le_name, device)
        emb_train, emb_val, emb_test = _generate_model_embeddings(model, c_train, c_val, c_test, le_name, device)
    
    load_time = time.time() - start_time
    print(f"Embeddings loaded in {load_time:.2f} seconds")
    print(f"Embedding shapes - Train: {emb_train.shape}, Val: {emb_val.shape}, Test: {emb_test.shape}")
    
    # Perform intrinsic dimension analysis using scikit-dimension
    id_hparams = get_id_hyperparameters()
    if id_hparams.get('analyze_intrinsic_dim', True):
        id_results, local_ids = analyze_intrinsic_dimensions(emb_train, emb_val, emb_test, c_train, c_val, c_test, le_name, hparams['dataset'], hparams)
        return emb_train, emb_val, emb_test, local_ids
    else:
        return emb_train, emb_val, emb_test, None
    
    return emb_train, emb_val, emb_test


def load_model(hparams, le_name, device):
    """Centralized model loading."""
    # GeoCLIP models
    geoclip_models = {
        'geoclip-resnet50-bs16k-l10': 'path_to_geoclip-resnet50-bs16k-l10',
        'geoclip-resnet50-bs16k-l40': 'path_to_geoclip-resnet50-bs16k-l40',
        'geoclip-resnet50-bs8k-l10': 'path_to_geoclip-resnet50-bs8k-l10',
        'geoclip-resnet50-bs8k-l40': 'path_to_geoclip-resnet50-bs8k-l40',
        'geoclip-resnet18-bs8k-l10': 'path_to_geoclip-resnet18-bs8k-l10',
        'geoclip-resnet18-bs8k-l40': 'path_to_geoclip-resnet18-bs8k-l40',
        'geoclip-vit16-bs8k-l10': 'path_to_geoclip-vit16-bs8k-l10',
        'geoclip-vit16-bs8k-l40': 'path_to_geoclip-vit16-bs8k-l40',
    }
    
    if le_name in geoclip_models:
        return get_geoclip(hparams[geoclip_models[le_name]], device, return_all=False)
    
    # SatCLIP models
    satclip_models = {
        'whzpceym': 'path_to_whzpceym',
        'y0osy2g0': 'path_to_y0osy2g0',
        'mmearth_l10': 'path_to_mmearth_l10',
        'mmearth_l40': 'path_to_mmearth_l40',
        '8341plm1': 'path_to_8341plm1',
        'lxye0ph1': 'path_to_lxye0ph1',
        'dsc-resnet50-bs8k-l45': 'path_to_dsc-resnet50-bs8k-l45',
        'dsc-vit16-bs8k-l45': 'path_to_dsc-vit16-bs8k-l45',
        'dsc-w-vit16-bs8k-l45': 'path_to_dsc-w-vit16-bs8k-l45',
    }
    
    if le_name in satclip_models:
        return get_satclip(hparams[satclip_models[le_name]], device, return_all=False)
    
    # Tom Embedding CLIP models
    if 'tomembeddingclip' in le_name:
        return _load_tom_embedding_model(hparams, le_name, device)
    
    # CSP models
    if le_name == 'csp-inat':
        return get_csp(hparams['path_to_csp_inat'])
    elif le_name == 'csp-fmow':
        return get_csp(hparams['path_to_csp_fmow'])
    
    # GC model
    if le_name == 'gc':
        return get_gc(hparams['path_to_gc'])
    
    raise ValueError(f'Model type {le_name} not found.')


def _load_tom_embedding_model(hparams, le_name, device):
    """Load Tom Embedding CLIP models."""
    tom_models = {
        'tomembeddingclip_L40_2M': 'path_to_tomembeddingclip_L40_2M',
        'tomembeddingclip_L40_200k': 'path_to_tomembeddingclip_L40_200k',
        'tomembeddingclip_L75_200k': 'path_to_tomembeddingclip_L75_200k',
        'tomembeddingclip_L100_200k': 'path_to_tomembeddingclip_L100_200k',
        'tomembeddingclip_L10_200k': 'path_to_tomembeddingclip_L10_200k',
        's1s2_tomembeddingclip_L10_200k': 'path_to_s1s2_tomembeddingclip_L10_200k',
        's1s2_tomembeddingclip_L40_200k': 'path_to_s1s2_tomembeddingclip_L40_200k',
        's1s20p8_tomembeddingclip_L10_200k': 'path_to_s1s20p8_tomembeddingclip_L10_200k',
        's1s20p8mlp_tomembeddingclip_L10_200k': 'path_to_s1s20p8mlp_tomembeddingclip_L10_200k',
    }
    
    if le_name not in tom_models:
        raise ValueError(f'Tom model {le_name} not found.')
    
    checkpoint = torch.load(hparams[tom_models[le_name]])
    ours = checkpoint['hyper_parameters']
    weights = checkpoint['state_dict']
    
    model = EmbeddingProjectionCLIP(
        embed_dim=ours['embed_dim'],
        input_embedding_dim=ours['input_embedding_dim'],
        le_type=ours['le_type'],
        pe_type=ours['pe_type'],
        frequency_num=ours['frequency_num'],
        max_radius=ours['max_radius'],
        min_radius=ours['min_radius'],
        legendre_polys=ours['legendre_polys'],
        harmonics_calculation=ours['harmonics_calculation'],
        num_hidden_layers=ours['num_hidden_layers'],
        capacity=ours['capacity']
    )
    
    return extract_location_encoder(full_model=model, full_state_dict=weights).to(device)


def _generate_model_embeddings(model, c_train, c_val, c_test, le_name, device):
    """Generate embeddings using the model."""
    model.eval()
    
    with torch.no_grad():
        if le_name.startswith('csp-'):
            # CSP expects degrees directly, no conversion
            emb_train = model.loc_enc(c_train.float().to(device), return_feats=True).detach().cpu()
            emb_val = model.loc_enc(c_val.float().to(device), return_feats=True).detach().cpu()
            emb_test = model.loc_enc(c_test.float().to(device), return_feats=True).detach().cpu()
        elif le_name == 'gc':
            # GC model expects flipped coordinates
            emb_train = model(c_train.flip(1).float()).detach().cpu()
            emb_val = model(c_val.flip(1).float()).detach().cpu()
            emb_test = model(c_test.flip(1).float()).detach().cpu()
        else:
            # Standard models
            emb_train = model(c_train.double().to(device)).detach().cpu()
            emb_val = model(c_val.double().to(device)).detach().cpu()
            emb_test = model(c_test.double().to(device)).detach().cpu()
    
    return emb_train, emb_val, emb_test


def _load_gps2vec_embeddings(hparams, c_train, c_val, c_test, le_name):
    """Load GPS2Vec embeddings."""
    suffix = 'tag' if 'tag' in le_name else 'visual'
    dataset = hparams['dataset']
    
    # Handle special dataset cases
    if dataset in ['countries', 'biome', 'ecoregions']:
        dataset = 'countries'
    
    if dataset == 'inat':
        df_train = torch.load(f"{hparams['path_to_gps2vec']}/inat_train_{suffix}.pt")
        df_test = torch.load(f"{hparams['path_to_gps2vec']}/inat_val_{suffix}.pt")
        df_train = pd.DataFrame(df_train).rename(columns={0: 'Lon', 1: 'Lat'})
        df_test = pd.DataFrame(df_test).rename(columns={0: 'Lon', 1: 'Lat'})
        df = pd.concat([df_train, df_test]).drop_duplicates(subset=['Lon', 'Lat'])
    else:
        df = torch.load(f"{hparams['path_to_gps2vec']}/{dataset}_{suffix}.pt")
        df = pd.DataFrame(df).rename(columns={0: 'Lon', 1: 'Lat'}).drop_duplicates(subset=['Lon', 'Lat'])
    
    # Convert coordinates to dataframes
    c_train_df = pd.DataFrame(c_train.numpy()).rename(columns={0: 'Lon', 1: 'Lat'})
    c_val_df = pd.DataFrame(c_val.numpy()).rename(columns={0: 'Lon', 1: 'Lat'})
    c_test_df = pd.DataFrame(c_test.numpy()).rename(columns={0: 'Lon', 1: 'Lat'})
    
    # Merge and extract embeddings
    emb_train = torch.tensor(c_train_df.merge(df, on=['Lon', 'Lat'], how='left').iloc[:, 2:].values.astype('float32'))
    emb_val = torch.tensor(c_val_df.merge(df, on=['Lon', 'Lat'], how='left').iloc[:, 2:].values.astype('float32'))
    emb_test = torch.tensor(c_test_df.merge(df, on=['Lon', 'Lat'], how='left').iloc[:, 2:].values.astype('float32'))
    
    # Handle NaN values
    emb_train[torch.isnan(emb_train)] = 0
    emb_val[torch.isnan(emb_val)] = 0
    emb_test[torch.isnan(emb_test)] = 0
    
    return emb_train, emb_val, emb_test


def _load_mosaiks_embeddings(hparams, c_train, c_val, c_test):
    """Load MOSAIKS embeddings."""
    dataset = hparams['dataset']
    
    # Handle special dataset cases
    if dataset in ['countries', 'biome', 'ecoregions']:
        dataset = 'countries'
    
    if dataset == 'inat':
        df_train = pd.read_csv(f"{hparams['path_to_mosaiks']}/aligned/mosaiks_inat_train.csv")
        df_test = pd.read_csv(f"{hparams['path_to_mosaiks']}/aligned/mosaiks_inat_val.csv")
        df = pd.concat([df_train, df_test]).drop_duplicates(subset=['Lon', 'Lat'])
    else:
        df = pd.read_csv(f"{hparams['path_to_mosaiks']}/aligned/mosaiks_{dataset}.csv")
        df = df.drop_duplicates(subset=['Lon', 'Lat'])
    
    # Convert coordinates to dataframes
    c_train_df = pd.DataFrame(c_train.numpy()).rename(columns={0: 'Lon', 1: 'Lat'})
    c_val_df = pd.DataFrame(c_val.numpy()).rename(columns={0: 'Lon', 1: 'Lat'})
    c_test_df = pd.DataFrame(c_test.numpy()).rename(columns={0: 'Lon', 1: 'Lat'})
    
    # Merge and extract embeddings
    emb_train = torch.tensor(c_train_df.merge(df, on=['Lon', 'Lat'], how='left').iloc[:, 4:].values.astype('float32'))
    emb_val = torch.tensor(c_val_df.merge(df, on=['Lon', 'Lat'], how='left').iloc[:, 4:].values.astype('float32'))
    emb_test = torch.tensor(c_test_df.merge(df, on=['Lon', 'Lat'], how='left').iloc[:, 4:].values.astype('float32'))
    
    # Handle NaN values
    emb_train[torch.isnan(emb_train)] = 0
    emb_val[torch.isnan(emb_val)] = 0
    emb_test[torch.isnan(emb_test)] = 0
    
    return emb_train, emb_val, emb_test


def analyze_intrinsic_dimensions(emb_train, emb_val, emb_test, c_train, c_val, c_test, le_name, dataset, hparams):
    """Analyze intrinsic dimensions using scikit-dimension methods."""
    save_dir = Path('results1/intrinsic_dimensions')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine all embeddings and coordinates
    all_embeddings = torch.cat([emb_train, emb_val, emb_test], dim=0).numpy()
    all_coordinates = torch.cat([c_train, c_val, c_test], dim=0).numpy()
    
    print(f"\n{'='*60}")
    print(f"ID Analysis for {le_name} on {dataset}")
    print(f"Total samples: {len(all_embeddings)}, Embedding dim: {all_embeddings.shape[1]}")
    print(f"{'='*60}")
    
    # Subsample if needed
    max_samples = hparams.get('id_max_samples', None)
    if max_samples and len(all_embeddings) > max_samples:
        print(f"Subsampling to {max_samples} points (from {len(all_embeddings)})")
        indices = np.random.choice(len(all_embeddings), max_samples, replace=False)
        all_embeddings_sample = all_embeddings[indices]
        all_coordinates_sample = all_coordinates[indices]
    else:
        all_embeddings_sample = all_embeddings
        all_coordinates_sample = all_coordinates
        indices = None
    
    # Initialize results dictionary
    results = {
        'le_name': le_name,
        'dataset': dataset,
        'embedding_dim': all_embeddings.shape[1],
        'n_samples': len(all_embeddings),
        'n_samples_used': len(all_embeddings_sample)
    }
    
    # Get ID methods to use
    methods_to_use = hparams.get('id_methods', ['MLE', 'TwoNN', 'ESS'])
    n_neighbors = hparams.get('id_k_neighbors', 20)
    n_jobs = hparams.get('n_jobs', -1)
    

    estimators = {
        'MLE': MLE(neighborhood_based=True),  # Uses default parameters
        # 'TwoNN': TwoNN(discard_fraction=0.1),
        # # 'ESS': ESS(ver='b'),
        # # 'FisherS': FisherS(),
        # # 'MiND_ML': MiND_ML(k=10),
        # 'DANCo': DANCo(k=10, D=10),
        # 'lPCA': lPCA(),
        # # 'CorrInt': CorrInt(k1=10, k2=20),
        # # 'MOM': MOM(),  # Takes no arguments
        # # 'TLE': TLE(),
        # 'KNN': KNN(k=5)
    }
    
    # Calculate global IDs
    print("\nCalculating global intrinsic dimensions:")
    for method_name in methods_to_use:
        if method_name not in estimators:
            print(f"  Warning: Method {method_name} not available")
            continue
        start = time.time()
        estimator = estimators[method_name]
        global_id = estimator.fit_transform(all_embeddings_sample)
        elapsed = time.time() - start
        
        results[f'global_id_{method_name}'] = float(global_id)
        print(f"  {method_name}: {float(global_id):.3f} (took {elapsed:.2f}s)")
            

    # Calculate local IDs if requested
    if hparams.get('calculate_local_id', True):
        print(f"\nCalculating local intrinsic dimensions (k={n_neighbors}):")
        
        local_methods = hparams.get('local_id_methods', ['MLE', 'ESS'])
        
        for method_name in local_methods:
            if method_name not in estimators:
                continue
                
            try:
                print(f"  Computing local IDs with {method_name}...")
                start = time.time()
                estimator = estimators[method_name]

                estimator.fit_transform_pw(
                    all_embeddings_sample,
                    n_neighbors=n_neighbors,
                    n_jobs=n_jobs
                )
                local_ids = estimator.dimension_pw_   
                elapsed = time.time() - start
                
                # If we subsampled, expand back to full size
                if indices is not None:
                    local_ids_full = np.full(len(all_embeddings), np.nan)
                    local_ids_full[indices] = local_ids
                    local_ids = local_ids_full
                
                # Calculate statistics
                valid_mask = ~np.isnan(local_ids)
                if np.any(valid_mask):
                    results[f'local_id_{method_name}_mean'] = float(np.nanmean(local_ids))
                    results[f'local_id_{method_name}_std'] = float(np.nanstd(local_ids))
                    results[f'local_id_{method_name}_min'] = float(np.nanmin(local_ids))
                    results[f'local_id_{method_name}_max'] = float(np.nanmax(local_ids))
                    results[f'local_id_{method_name}_valid'] = int(np.sum(valid_mask))
                    
                    print(f"    {method_name}: mean={np.nanmean(local_ids):.3f}, "
                          f"std={np.nanstd(local_ids):.3f}, "
                          f"range=[{np.nanmin(local_ids):.3f}, {np.nanmax(local_ids):.3f}] "
                          f"(took {elapsed:.2f}s)")
                    
                    # Create visualization for the first local method
                    if method_name == local_methods[0]:
                        print(f"  Creating visualization with {method_name} local IDs...")
                        visualize_local_ids_globe(
                            all_coordinates, local_ids, le_name, dataset, save_dir, method_name
                        )
                else:
                    print(f"    {method_name}: No valid local IDs computed")
                    
            except Exception as e:
                print(f"    {method_name}: Failed - {str(e)}")
    
    # Save results to CSV
    results_file = save_dir / f"{dataset}_id_results.csv"
    df = pd.DataFrame([results])
    
    if results_file.exists():
        df_existing = pd.read_csv(results_file)
        # Remove existing entry for this model
        df_existing = df_existing[df_existing['le_name'] != le_name]
        df = pd.concat([df_existing, df], ignore_index=True)
    
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary of ID estimates:")
    for method in methods_to_use:
        if f'global_id_{method}' in results and results[f'global_id_{method}'] is not None:
            print(f"  {method}: {results[f'global_id_{method}']:.3f}")
    print(f"{'='*60}\n")
    
    return results, local_ids



class EvalLocDataModule(pl.LightningDataModule):
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, num_workers=0, batch_size=1000):
        super().__init__()
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

    def setup(self, stage: str):
        
        self.train_ds = TensorDataset(*[self.x_train, self.y_train])
        self.valid_ds = TensorDataset(*[self.x_val, self.y_val])
        self.evalu_ds = TensorDataset(*[self.x_test, self.y_test])
        self.test_locs = self.x_test.detach()
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.evalu_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def get_test_locs(self):
        return self.test_locs    
    
class EvalImgDataModule(pl.LightningDataModule):
    def __init__(self, x_train, img_train, y_train, x_val, img_val, y_val, x_test, img_test, y_test, num_workers=0, batch_size=1000):
        super().__init__()
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.x_train = x_train
        self.img_train = img_train
        self.y_train = y_train
        self.x_val = x_val
        self.img_val = img_val
        self.y_val = y_val
        self.x_test = x_test
        self.img_test = img_test
        self.y_test = y_test

    def setup(self, stage: str):
        self.train_ds = TensorDataset(*[self.x_train, self.img_train, self.y_train])
        self.valid_ds = TensorDataset(*[self.x_val, self.img_val, self.y_val])
        self.evalu_ds = TensorDataset(*[self.x_test, self.img_test, self.y_test])
        self.test_locs = self.x_test.detach()
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.evalu_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def get_test_locs(self):
        return self.test_locs    

def overwrite_hparams_with_tuned_hparams(hparams, hparams_tuned):
    hparams_tuned = hparams_tuned[hparams['le_name']]
    if "dim_hidden" in hparams.keys() and hparams_tuned["dim_hidden"] is not None:
        hparams["dim_hidden"] = hparams_tuned["dim_hidden"]
    if "num_layers" in hparams.keys() and hparams_tuned["num_layers"] is not None:
        hparams["num_layers"] = hparams_tuned["num_layers"]
    if "lr" in hparams["optimizer"].keys() and hparams_tuned["lr"] is not None:
        hparams["optimizer"]["lr"] = hparams_tuned["lr"]
    if "wd" in hparams["optimizer"].keys() and hparams_tuned["wd"] is not None:
        hparams["optimizer"]["wd"] = hparams_tuned["wd"]
    return hparams

def get_param(hparams, key, default=False):
    """
    Convenience function that indexes the hyperparameter dict but returns a default value if not defined rather than
    an error
    """
    return hparams[key] if key in hparams.keys() else default

def BCE_loss(model, input, label):
    prediction_logits = model.forward(input)
    if prediction_logits.size(1) == 1:
        return nn.functional.binary_cross_entropy_with_logits(prediction_logits, label)
        #return nn.functional.cross_entropy(prediction_logits, label.squeeze())
    else:
        return nn.functional.cross_entropy(prediction_logits, label.squeeze())

def INAT_loss(model, input, img, label):
    prediction_logits = model.forward(input, img)
    # print(label.shape)
    # print(prediction_logits.shape)
    if prediction_logits.size(1) == 1:
        return nn.functional.binary_cross_entropy_with_logits(prediction_logits, label.long())
        #return nn.functional.cross_entropy(prediction_logits, label.squeeze())
    else:
        return nn.functional.cross_entropy(prediction_logits, label.squeeze().long())

def MSE_loss(model, input, label):
    prediction_logits = model.forward(input)
    if prediction_logits.size(1) == 1:
        return nn.functional.mse_loss(prediction_logits, label, reduction='mean')
    else:
        return nn.functional.mse_loss(prediction_logits, label.squeeze(), reduction='mean')

class MLP(nn.Module):
    def __init__(self, input_dim, dim_hidden, num_layers, out_dims):
        super(MLP, self).__init__()

        layers = []
        layers += [nn.Linear(input_dim, dim_hidden, bias=True), nn.ReLU()] # input layer
        layers += [nn.Linear(dim_hidden, dim_hidden, bias=True), nn.ReLU()] * num_layers # hidden layers
        layers += [nn.Linear(dim_hidden, out_dims, bias=True)] # output layer

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)

class DualMLP(nn.Module):
    def __init__(self, input_dim, img_dim, dim_hidden, num_layers, out_dims):
        super(DualMLP, self).__init__()

        layers1 = []
        layers1 += [nn.Linear(input_dim, dim_hidden, bias=True), nn.ReLU()] # input layer
        layers1 += [nn.Linear(dim_hidden, dim_hidden, bias=True), nn.ReLU()] * num_layers # hidden layers

        layers2 = []
        layers2 += [nn.Linear(img_dim, dim_hidden, bias=True), nn.ReLU()] # input layer
        layers2 += [nn.Linear(dim_hidden, dim_hidden, bias=True), nn.ReLU()] * num_layers # hidden layers
        
        
        self.head = nn.Linear(dim_hidden*2, out_dims, bias=True) # output layer

        self.features1 = nn.Sequential(*layers1)
        self.features2 = nn.Sequential(*layers2)

    def forward(self, x1, x2):
        x1 = self.features1(x1)
        x2 = self.features2(x2)
        out = torch.cat((x1, x2), dim=1)
        return self.head(out)

class MLPpretrained(nn.Module):
    def __init__(self, pretrained_model, dim_hidden, num_layers, out_dims):
        super(MLPpretrained, self).__init__()

        layers = []
        layers += [nn.Linear(256, dim_hidden, bias=True), nn.ReLU()] # input layer
        layers += [nn.Linear(dim_hidden, dim_hidden, bias=True), nn.ReLU()] * num_layers # hidden layers
        layers += [nn.Linear(dim_hidden, out_dims, bias=True)] # output layer

        self.features = nn.Sequential(*layers)
        self.pretrained_model = pretrained_model.train()

    def forward(self, x):
        x = self.pretrained_model(x)
        return self.features(x)

class DualMLPpretrained(nn.Module):
    def __init__(self, pretrained_model, img_dim, dim_hidden, num_layers, out_dims):
        super(DualMLPpretrained, self).__init__()

        layers = []
        layers += [nn.Linear(img_dim, dim_hidden, bias=True), nn.ReLU()] # input layer
        layers += [nn.Linear(dim_hidden, dim_hidden, bias=True), nn.ReLU()] * num_layers # hidden layers
        
        
        self.head = nn.Linear(256+img_dim, out_dims, bias=True) # output layer
        self.pretrained_model = pretrained_model.train()

        self.head = nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1 = self.pretrained_model(x1)
        out = torch.cat((x1, x2), dim=1)
        return self.head(out)
    
class SHpretrained(nn.Module):
    def __init__(self, pretrained_model_path, dim_hidden, num_layers, out_dims, hparams=None):
        super(SHpretrained, self).__init__()

        self.pretrained_model = get_geoclip(pretrained_model_path, device=hparams['device'])
        if hparams['le_name']=='siren-sh-l10' or hparams['le_name']=='siren-sh-l40':
            for name, layer in self.pretrained_model.named_children():
                # Check if the layer has a reset_parameters method
                if hasattr(layer, 'reset_parameters'):
                    # Call the method to rseset the parameters
                    layer.reset_parameters()

        self.features = Siren(dim_in = 256, dim_out = out_dims, w0 = 1., use_bias = True, activation = None, dropout = False)
        self.pretrained_model = self.pretrained_model.train()

    def forward(self, x):
        x = self.pretrained_model(x)
        return self.features(x)
    
class DualSHpretrained(nn.Module):
    def __init__(self, pretrained_model_path, img_dim, dim_hidden, num_layers, out_dims, hparams=None):
        super(DualSHpretrained, self).__init__()

        self.pretrained_model = get_geoclip(pretrained_model_path, device=hparams['device'])
        if hparams['le_name']=='siren-sh-l10' or hparams['le_name']=='siren-sh-l40':
            for name, layer in self.pretrained_model.named_children():
                # Check if the layer has a reset_parameters method
                if hasattr(layer, 'reset_parameters'):
                    # Call the method to reset the parameters
                    layer.reset_parameters()

        self.img_head = Siren(dim_in = img_dim, dim_out = 256, w0 = 1., use_bias = True, activation = None, dropout = True)
        self.features = Siren(dim_in = 256*2, dim_out = out_dims, w0 = 1., use_bias = True, activation = None, dropout = False)
        self.pretrained_model = self.pretrained_model.train()

    def forward(self, x1, x2):
        x1 = self.pretrained_model(x1)
        x2 = self.img_head(x2)
        x = torch.cat((x1, x2), dim=1)
        return self.features(x)

class PredictionHead(pl.LightningModule):
    def __init__(self, hparams, pretrained=None):
        super().__init__()

        self.learning_rate = hparams["optimizer"]["lr"]
        self.weight_decay = hparams["optimizer"]["wd"]
        self.task = hparams["task"]

        if self.task=='regression':
            self.loss_fn = MSE_loss
            self.regression = True
        else:
            self.loss_fn = BCE_loss
            self.regression = False

        if pretrained is not None:
            # self.model = MLPpretrained(pretrained_model=pretrained, 
            #                         dim_hidden=hparams["dim_hidden"],
            #                         num_layers=hparams["num_layers"],
            #                         out_dims=hparams["out_dims"]
            #                     ).double()
            self.model = SHpretrained(pretrained_model_path=pretrained, 
                                    dim_hidden=hparams["dim_hidden"],
                                    num_layers=hparams["num_layers"],
                                    out_dims=hparams["out_dims"],
                                    hparams=hparams
                                ).double()
        else:
            self.model = MLP(input_dim=hparams["input_dim"],
                                    dim_hidden=hparams["dim_hidden"],
                                    num_layers=hparams["num_layers"],
                                    out_dims=hparams["out_dims"]
                                ).double()

        # this enables SpatialEncoder.load_from_checkpoint(path)
        self.save_hyperparameters()

    def common_step(self, batch, batch_idx):
        input, label = batch
        return self.loss_fn(self, input.double(), label)

    def forward(self, input):
        return self.model(input.double())

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return {"val_loss":loss}

    def predict_step(self, batch, batch_idx):
        input, label = batch
        prediction_logits = self.forward(input.double())
        return prediction_logits, input, label

    def test_step(self, batch, batch_idx):
        input, label = batch
        prediction_logits = self.forward(input.double())
        
        loss = self.loss_fn(self, input.double(), label)

        # check if binary
        if (prediction_logits.size(1) == 1) and not (self.regression):
            y_pred = (prediction_logits.squeeze() > 0).cpu()
            average =  "binary"
        elif self.regression:
            y_pred = prediction_logits.cpu()
            average= 'None'                                
        else: # take argmax
            y_pred = prediction_logits.argmax(-1).cpu()
            average = "macro"
        #print(average)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        if self.regression:
            MAE = mean_absolute_error(y_true=label.cpu().numpy(), y_pred = y_pred)
            rsq = r2_score(y_true=label.cpu().numpy(), y_pred = y_pred.cpu().numpy())
            self.log("test_MAE", MAE, on_step=False, on_epoch=True)
            self.log("test_r2", rsq, on_step=False, on_epoch=True)
            
            test_results = {"test_loss":loss,
                            "test_MAE":MAE,
                            "test_r2":rsq}
        else:
            # print(label.shape)
            # print(y_pred.shape)
            accuracy = accuracy_score(y_true=label.squeeze().argmax(-1).cpu(), y_pred= y_pred)
            IoU = jaccard_score(y_true=label.squeeze().argmax(-1).cpu(),  y_pred = y_pred, average=average)
            self.log("test_accuracy", accuracy, on_step=False, on_epoch=True)
            self.log("test_IoU", IoU, on_step=False, on_epoch=True)
            
            test_results = {"test_loss":loss,
                          "test_accuracy":accuracy}
        
        if not hasattr(self, 'test_predictions'):
            self.test_predictions = []
            self.test_labels = []
            self.test_inputs = []
    
        self.test_predictions.append(y_pred.cpu())
        self.test_labels.append(label.cpu())
        self.test_inputs.append(input.cpu())

        return test_results
    
    def on_test_epoch_end(self):
        """Concatenate all test predictions at the end of test epoch."""
        if hasattr(self, 'test_predictions'):
            self.all_test_predictions = torch.cat(self.test_predictions)
            self.all_test_labels = torch.cat(self.test_labels)

    def configure_optimizers(self):
        optimizer = optim.Adam([{"params": self.model.parameters(), "weight_decay":0}],
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        return optimizer
    
class PredictionHeadImg(pl.LightningModule):
    def __init__(self, hparams, pretrained=None):
        super().__init__()

        self.learning_rate = hparams["optimizer"]["lr"]
        self.weight_decay = hparams["optimizer"]["wd"]
        self.task = hparams["task"]


        self.loss_fn = INAT_loss
        self.regression = False

        if pretrained is not None:
            # self.model = DualMLPpretrained(pretrained_model=pretrained, 
            #                         img_dim=hparams["img_dim"],
            #                         dim_hidden=hparams["dim_hidden"],
            #                         num_layers=hparams["num_layers"],
            #                         out_dims=hparams["out_dims"]
            #                     ).double()
            self.model = DualSHpretrained(pretrained_model_path=pretrained, 
                        img_dim=hparams["img_dim"],
                        dim_hidden=hparams["dim_hidden"],
                        num_layers=hparams["num_layers"],
                        out_dims=hparams["out_dims"],
                        hparams=hparams
                    ).double()
        else:
            self.model = DualMLP(input_dim=hparams["input_dim"],
                                    img_dim=hparams["img_dim"],
                                    dim_hidden=hparams["dim_hidden"],
                                    num_layers=hparams["num_layers"],
                                    out_dims=hparams["out_dims"]
                                ).double()

        # this enables SpatialEncoder.load_from_checkpoint(path)
        self.save_hyperparameters()

    def common_step(self, batch, batch_idx):
        input, img, label = batch
        return self.loss_fn(self, input.double(), img.double(), label)

    def forward(self, input, img):
        return self.model(input.double(),img.double())

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return {"val_loss":loss}

    def predict_step(self, batch, batch_idx):
        input, img, label = batch
        prediction_logits = self.forward(input.double(), img.double())
        return prediction_logits, input, img, label

    def test_step(self, batch, batch_idx):
        input, img, label = batch
        prediction_logits = self.forward(input.double(), img.double())
        
        loss = self.loss_fn(self, input.double(), img.double(), label)

        # check if binary
        if (prediction_logits.size(1) == 1) and not (self.regression):
            y_pred = (prediction_logits.squeeze() > 0).cpu()
            average = "binary"
        elif self.regression:
            y_pred = prediction_logits.cpu()        
            average = 'none'                        
        else: # take argmax
            y_pred = prediction_logits.argmax(-1).cpu()
            average = "macro"
        # print(average)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        if self.regression:
            MAE = mean_absolute_error(y_true=label.cpu(), y_pred = y_pred)
            rsq = r2_score(y_true=label.cpu().numpy(), y_pred = y_pred.cpu().numpy())
            self.log("test_MAE", MAE, on_step=False, on_epoch=True)
            self.log("test_r2", rsq, on_step=False, on_epoch=True)
            
            test_results = {"test_loss":loss,
                            "test_MAE":MAE,
                            "test_r2":rsq}
        else:
            # print(label)
            # print(y_pred)
            accuracy = accuracy_score(y_true=label.cpu(), y_pred= y_pred)
            IoU = jaccard_score(y_true=label.cpu(),  y_pred = y_pred, average=average)
            self.log("test_accuracy", accuracy, on_step=False, on_epoch=True)
            self.log("test_IoU", IoU, on_step=False, on_epoch=True)
            
            test_results = {"test_loss":loss,
                          "test_accuracy":accuracy}
        
        if not hasattr(self, 'test_predictions'):
            self.test_predictions = []
            self.test_labels = []
            self.test_inputs = []
    
        self.test_predictions.append(y_pred.cpu())
        self.test_labels.append(label.cpu())
        self.test_inputs.append(input.cpu())

        return test_results
    
    def on_test_epoch_end(self):
        """Concatenate all test predictions at the end of test epoch."""
        if hasattr(self, 'test_predictions'):
            self.all_test_predictions = torch.cat(self.test_predictions)
            self.all_test_labels = torch.cat(self.test_labels)

    def configure_optimizers(self):
        optimizer = optim.Adam([{"params": self.model.parameters(), "weight_decay":0}],
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        return optimizer
