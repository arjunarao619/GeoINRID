import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import skdim.id
import wandb
from tqdm import tqdm
from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, './sinr') 
sys.path.insert(0, '.')  # Add this line

import sinr.models as sinr_models
import sinr.utils as sinr_utils

import pyproj
pyproj.datadir.set_data_dir(
    "/projects/arra4944/arm64/software/miniforge/envs/bg2/share/proj"
)
import torchgeo.models as tgm
from torchgeo.models import (
    RCF,
    ScaleMAE, ScaleMAELarge16_Weights,
    croma_base, CROMABase_Weights,
    dofa_base_patch16_224, DOFABase16_Weights,
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    resnet152, ResNet152_Weights,
    swin_v2_t, Swin_V2_T_Weights,
    swin_v2_b, Swin_V2_B_Weights,
    vit_small_patch16_224, ViTSmall16_Weights,
    vit_base_patch16_224, ViTBase16_Weights,
    panopticon_vitb14, Panopticon_Weights
)
from torch.utils.data._utils.collate import default_collate

from utils.s2geo_dataset import S2Geo
from utils.satclip_transforms import get_pretrained_s2_train_transform

import kornia.augmentation as K
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm
import skdim

GLOBAL_ESTIMATORS = {
    'MLE': lambda: skdim.id.MLE(neighborhood_based=True),
    'TwoNN': lambda: skdim.id.TwoNN(),
    'FisherS': lambda: skdim.id.FisherS(project_on_sphere=False),
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


def remove_duplicate_features(features, coords=None, tolerance=1e-10):
    """
    Remove duplicate feature vectors.
    
    Parameters:
    -----------
    features : np.ndarray
        Feature matrix (n_samples, n_features)
    coords : np.ndarray or None
        Coordinate matrix (n_samples, 2) to keep in sync
    tolerance : float
        Tolerance for considering features as duplicates
    
    Returns:
    --------
    features_unique : np.ndarray
        Feature matrix with duplicates removed
    coords_unique : np.ndarray or None
        Coordinate matrix with duplicates removed (if provided)
    n_duplicates : int
        Number of duplicate rows removed
    """
    # Round features to avoid floating point precision issues
    features_rounded = np.round(features / tolerance) * tolerance
    
    # Find unique rows
    _, unique_indices = np.unique(features_rounded, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)  # Maintain original order
    
    n_duplicates = len(features) - len(unique_indices)
    
    features_unique = features[unique_indices]
    coords_unique = coords[unique_indices] if coords is not None else None
    
    return features_unique, coords_unique, n_duplicates

class CoordinateDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.coords = torch.from_numpy(df[['lon', 'lat']].values.astype(np.float32))
    
    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        return {'point': self.coords[idx]}  # Removed 'image': None


class RobustDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper that catches errors during __getitem__ and returns None for failed samples"""
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            return self.dataset[idx]
        except (FileNotFoundError, OSError, IOError) as e:
            return None


CROMA_OPTICAL_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]

MODEL_CONFIG = {
     'rcf':        dict(loader=lambda args: get_rcf_model(13, args.rcf_features), size=None),
     'scalemae':   dict(loader=lambda args: get_scalemae_model(),                  size=224),
     'croma':      dict(loader=lambda args: croma_base(weights=CROMABase_Weights.CROMA_VIT,
                                                      modalities=['optical'], image_size=120), size=120),
     'dofa':       dict(loader=lambda args: dofa_base_patch16_224(weights=DOFABase16_Weights.DOFA_MAE), size=224),
     'resnet18':   dict(loader=lambda args: resnet18(weights=ResNet18_Weights.SENTINEL2_ALL_MOCO),     size=224),
     'resnet18_rgb':dict(loader=lambda args: resnet18(weights=ResNet18_Weights.SENTINEL2_RGB_MOCO),     size=224),
     'resnet50':   dict(loader=lambda args: resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO),     size=224),
     'resnet50_rgb':dict(loader=lambda args: resnet50(weights=ResNet50_Weights.SENTINEL2_RGB_MOCO),     size=224),
    'resnet152':  dict(loader=lambda args: resnet152(weights=ResNet152_Weights.IMAGENET1K_V2),        size=224),
    'swin_t':     dict(loader=lambda args: swin_v2_t(weights=Swin_V2_T_Weights.SENTINEL2_ALL_MOCO), size=224),
    'swin_b':     dict(loader=lambda args: swin_v2_b(weights=Swin_V2_B_Weights.SENTINEL2_ALL_MOCO), size=224),
    'vit_small':  dict(loader=lambda args: vit_small_patch16_224(weights=ViTSmall16_Weights.SENTINEL2_ALL_MOCO), size=224),
    'vit_base':   dict(loader=lambda args: vit_base_patch16_224(weights=ViTBase16_Weights.SENTINEL2_ALL_MOCO), size=224),
    'panopticon': dict(loader=lambda args: panopticon_vitb14(weights=Panopticon_Weights.VIT_BASE14, img_size=224), size=224),
    'sinr': dict(loader=lambda args: load_sinr_model(args.sinr_checkpoint), size=None),
}


def load_sinr_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    params = checkpoint['params']
    model = sinr_models.get_model(params)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model


def _move_croma_buffers_to_device(model, device):
    if hasattr(model, "attn_bias"):
        if torch.is_tensor(model.attn_bias):
            model.attn_bias = model.attn_bias.to(device)
        else:
            model.attn_bias = {k: v.to(device) for k, v in model.attn_bias.items()}


class CromaPrep(nn.Module):
    def __init__(self, out_size=120, use_8_bit=False):
        super().__init__()
        self.out_size = out_size
        self.use_8_bit = use_8_bit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, CROMA_OPTICAL_IDX]
        if x.shape[-2:] != (self.out_size, self.out_size):
            x = F.interpolate(x, size=(self.out_size, self.out_size),
                              mode='bilinear', align_corners=False)
        mean = x.mean(dim=(2,3), keepdim=True)
        std  = x.std(dim=(2,3), keepdim=True)
        mn   = mean - 2 * std
        mx   = mean + 2 * std
        x = (x - mn) / (mx - mn)
        x = torch.clamp(x, 0, 1)
        if self.use_8_bit:
            x = (x * 255).round().to(torch.uint8)
        return x


def get_rcf_model(in_channels, features=512, kernel_size=3, seed=42):
    return RCF(
        in_channels=in_channels,
        features=features,
        kernel_size=kernel_size,
        bias=-1.0,
        seed=seed,
        mode='gaussian'
    )


def get_scalemae_model(weights=None):
    from functools import partial
    if weights is None:
        weights = ScaleMAELarge16_Weights.FMOW_RGB
    model = ScaleMAE(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        img_size=224,
        in_chans=3,
    )
    if weights:
        state_dict = weights.get_state_dict(progress=True)
        model.load_state_dict(state_dict, strict=False)
    return model


def get_croma_model():
    weights = CROMABase_Weights.CROMA_VIT
    model = tgm.croma_base(weights=weights, modalities=['optical'], image_size=120)
    return model, weights


def get_dofa_model():
    weights = DOFABase16_Weights.DOFA_MAE
    model = tgm.dofa_base_patch16_224(weights=weights)
    return model, weights


def get_resnet18_model(use_rgb=False):
    weights = ResNet18_Weights.SENTINEL2_RGB_MOCO if use_rgb else ResNet18_Weights.SENTINEL2_ALL_MOCO
    model = tgm.resnet18(weights=weights)
    return model, weights


def get_resnet50_model(use_rgb=False):
    weights = ResNet50_Weights.SENTINEL2_RGB_MOCO if use_rgb else ResNet50_Weights.SENTINEL2_ALL_MOCO
    model = tgm.resnet50(weights=weights)
    return model, weights


def get_resnet152_model():
    import torchvision.models as tv_models
    model = tv_models.resnet152(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    return model, None


def _pick_weight(enum_cls, preferred=('SENTINEL2_ALL_MOCO',
                                      'SENTINEL2_RGB_MOCO',
                                      'FMOW_RGB_GASSL',
                                      'IMAGENET1K_V1')):
    for name in preferred:
        if hasattr(enum_cls, name):
            return getattr(enum_cls, name)
    return next(iter(enum_cls)) if len(enum_cls) else None


def get_swin_model(variant='b'):
    if variant == 'b':
        enum_cls = Swin_V2_B_Weights
        builder  = swin_v2_b
    else:
        enum_cls = Swin_V2_T_Weights
        builder  = swin_v2_t
    weights = _pick_weight(enum_cls)
    model   = builder(weights=weights) if weights is not None else builder(weights=None)
    return model, weights


def get_vit_small_model():
    weights = ViTSmall16_Weights.SENTINEL2_ALL_MOCO
    model = tgm.vit_small_patch16_224(weights=weights)
    return model, weights


def extract_features_generic(model, dataloader, device, model_name, dataset_name, desc="Extracting"):
    model = model.to(device).eval()
    if model_name == "croma":
        _move_croma_buffers_to_device(model, device)

    transform = get_transform_for_model(model_name, dataset_name)
    if transform is not None:
        transform = transform.to(device)

    s2_wavelengths = torch.tensor(
        [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783,
         0.842, 0.865, 1.610, 2.190, 0.443, 1.375],
        device=device
    )

    feats, coords = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{desc} - {model_name}"):
            if batch is None:
                continue

            imgs = batch['image'].to(device, non_blocking=True)
            if imgs.dim() == 3:
                imgs = imgs.unsqueeze(0)

            if model_name != 'croma':
                imgs = preprocess_for_model(imgs, model_name, dataset_name)

            if transform is not None:
                imgs = transform(imgs)

            if model_name == 'croma':
                out_dict = model(x_optical=imgs)
                if isinstance(out_dict, dict):
                    out = out_dict.get('optical_GAP', None)
                    if out is None:
                        out = out_dict['optical_encodings'].mean(dim=1)
                else:
                    out = out_dict
            elif model_name == 'dofa':
                out = model.forward_features(imgs, wavelengths=s2_wavelengths)
            elif model_name in ['swin', 'vit_small', 'vit_base', 'swin_t', 'swin_b', 'panopticon']:
                tokens = model.forward_features(imgs)
                if hasattr(model, 'num_prefix_tokens') and model.num_prefix_tokens > 0:
                    tokens = tokens[:, model.num_prefix_tokens:, :]
                out = tokens.mean(dim=1)
            elif model_name in ['resnet18', 'resnet18_rgb', 'resnet50', 'resnet50_rgb']:
                model.fc = nn.Identity()
                out = torch.flatten(model(imgs), 1)
            elif model_name == 'resnet152':
                out = model(imgs).squeeze(-1).squeeze(-1)
            else:
                out = model(imgs)

            feats.append(out.cpu().numpy())
            coords.append(batch['point'].cpu().numpy())
    
    feats = np.vstack(feats).astype(np.float32)
    coords = np.vstack(coords)
    bad = ~np.isfinite(feats).all(axis=1)
    if bad.any():
        print(f"[ID] Dropping {bad.sum()} / {feats.shape[0]} samples with NaN/Inf features")
        feats = feats[~bad]
        coords = coords[~bad]
    return feats, coords


def preprocess_for_model(imgs, model_name, dataset_name):
    if model_name in ['resnet152', 'resnet18_rgb', 'resnet50_rgb']:
        imgs = imgs[:, [3, 2, 1], :, :]
    return imgs


def get_transform_for_model(model_name, dataset_name):
    if model_name == 'croma':
        return CromaPrep(out_size=120, use_8_bit=False)

    rgb_models = ['resnet152', 'resnet18_rgb', 'resnet50_rgb', 'scalemae']
    if model_name in rgb_models:
        return K.AugmentationSequential(
            K.Resize((224, 224)),
            K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            data_keys=['image']
        )

    multi_band = ['resnet18', 'resnet50', 'swin', 'vit_small', 'vit_base', 'swin_t',
                  'swin_b', 'panopticon', 'dofa']
    if model_name in multi_band:
        resize = K.CenterCrop((224, 224)) if dataset_name == 's2' else K.Resize((224, 224))
        return K.AugmentationSequential(resize, data_keys=['image'])

    return None


def extract_features_rcf(model, dataloader, device, desc="RCF"):
    model = model.to(device).eval()
    feats, coords = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            if batch is None:
                continue
            imgs = batch['image'].to(device)
            if imgs.dim() == 3:
                imgs = imgs.unsqueeze(0)
            out = model(imgs)
            feats.append(out.cpu().numpy())
            coords.append(batch['point'].numpy())
    return np.vstack(feats), np.vstack(coords)


def extract_features_scalemae(model, dataloader, device, dataset_name, transform=None, desc="ScaleMAE"):
    model = model.to(device).eval()
    if transform is None:
        norm_steps = [
            K.Normalize(mean=torch.tensor(0), std=torch.tensor(255)),
            K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                        std=torch.tensor([0.229, 0.224, 0.225]))
        ]
        if dataset_name == 's2':
            pipeline = [K.CenterCrop((224, 224)), *norm_steps]
        else:
            pipeline = [K.Resize((224, 224)), *norm_steps]
        transform = K.AugmentationSequential(*pipeline, data_keys=['image'])
    
    feats, coords = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            if batch is None:
                continue
            imgs = batch['image'].to(device)
            if imgs.dim() == 3:
                imgs = imgs.unsqueeze(0)
            imgs = imgs[:, [3, 2, 1], :, :]
            imgs = transform(imgs)
            out = model.forward_features(imgs)
            if hasattr(model, 'cls_token'):
                out = out[:, 1:, :].mean(dim=1)
            else:
                out = out.mean(dim=1)
            feats.append(out.cpu().numpy())
            coords.append(batch['point'].numpy())
    return np.vstack(feats), np.vstack(coords)


def calculate_intrinsic_dimension(features, methods, k_values):
    """
    Calculate intrinsic dimension using multiple estimators.
    
    Parameters:
    -----------
    features : np.ndarray
        Feature matrix
    methods : list
        List of method names (e.g., ['MLE', 'TwoNN', 'ESS'])
    k_values : list
        List of k values for neighborhood-based methods
    
    Returns:
    --------
    res : dict
        Dictionary of results
    """
    res = {}
    mask = np.isfinite(features).all(axis=1)
    features = features[mask]
    
    # Remove duplicate features
    features, _, n_duplicates = remove_duplicate_features(features)
    if n_duplicates > 0:
        print(f"  Removed {n_duplicates} duplicate feature vectors")
    
    n_samples = features.shape[0]

    for method in methods:
        if method not in GLOBAL_ESTIMATORS:
            print(f"Warning: Unknown method {method}, skipping")
            continue
        
        try:
            if method in ['MLE', 'MOM', 'TLE']:
                # These support multiple k values
                for k in k_values:
                    if k >= n_samples:
                        print(f"  Skipping {method} k={k} (k >= n_samples)")
                        continue
                    
                    print(f"  Computing {method} (k={k})...")
                    estimator = GLOBAL_ESTIMATORS[method]()
                    
                    if hasattr(estimator, 'fit'):
                        if 'n_neighbors' in estimator.fit.__code__.co_varnames:
                            estimator.fit(features, n_neighbors=k)
                        else:
                            estimator.fit(features)
                        
                        id_val = float(estimator.dimension_)
                        res[f'{method}_k{k}'] = id_val
                        print(f"    → ID = {id_val:.3f}")
            
            elif method in ['ESS', 'MADA']:
                # These automatically determine n_neighbors internally
                print(f"  Computing {method}...")
                estimator = GLOBAL_ESTIMATORS[method]()
                estimator.fit(features)
                id_val = float(estimator.dimension_)
                res[method] = id_val
                print(f"    → ID = {id_val:.3f}")
            
            else:
                # Single estimate for other estimators (TwoNN, FisherS, CorrInt, MiND_ML, MiND_KL, DANCo)
                print(f"  Computing {method}...")
                estimator = GLOBAL_ESTIMATORS[method]()
                estimator.fit(features)
                id_val = float(estimator.dimension_)
                res[method] = id_val
                print(f"    → ID = {id_val:.3f}")
                
        except Exception as e:
            print(f"    Failed: {str(e)}")
            if method in ['MLE', 'MOM', 'TLE']:
                for k in k_values:
                    if k < n_samples:
                        res[f'{method}_k{k}'] = np.nan
            else:
                res[method] = np.nan
    
    return res


def compute_local_id(features, estimator_name='MLE', k=10, n_jobs=-1, coords=None):
    """
    Compute local intrinsic dimension estimates.
    
    Parameters:
    -----------
    features : np.ndarray
        Feature matrix
    estimator_name : str
        Name of estimator to use ('MLE', 'TwoNN', 'MOM', 'TLE', 'ESS')
    k : int
        Number of neighbors for local estimation
    n_jobs : int
        Number of parallel jobs
    coords : np.ndarray or None
        Coordinate matrix to keep in sync with feature filtering
    
    Returns:
    --------
    dims_raw : np.ndarray or None
        Raw local ID estimates
    dims_smooth : np.ndarray or None
        Smoothed local ID estimates (if available)
    coords_filtered : np.ndarray or None
        Filtered coordinates (if coords provided)
    """
    if estimator_name not in LOCAL_ESTIMATORS:
        print(f"Warning: {estimator_name} not supported for local ID")
        return None, None, coords
    
    # Remove duplicate features
    features, coords_filtered, n_duplicates = remove_duplicate_features(features, coords)
    if n_duplicates > 0:
        print(f"  Removed {n_duplicates} duplicate feature vectors for local ID")
    
    n_samples = features.shape[0]
    k = min(k, n_samples - 1)
    
    if k < 2:
        print(f"  Skipping local {estimator_name} (insufficient samples, k={k})")
        return None, None, coords_filtered
    
    try:
        estimator = LOCAL_ESTIMATORS[estimator_name]()
        
        if estimator_name == 'ESS':
            # ESS doesn't support pointwise estimation
            print(f"  Skipping ESS for local estimation (not supported)")
            return None, None, coords_filtered
        
        if hasattr(estimator, 'fit_transform_pw'):
            # Use pointwise estimation
            result = estimator.fit_transform_pw(
                features,
                n_neighbors=k,
                n_jobs=n_jobs
            )
            
            if isinstance(result, tuple) and len(result) == 2:
                # Returns (raw, smooth)
                dims_raw, dims_smooth = result
                return dims_raw.astype(float), dims_smooth.astype(float), coords_filtered
            else:
                # Single output
                dims = result if not isinstance(result, tuple) else result[0]
                return dims.astype(float), None, coords_filtered
        else:
            # Fallback to old method for MLE
            if estimator_name == 'MLE':
                mle = skdim.id.MLE(neighborhood_based=True)
                mle.fit(features, n_neighbors=k, smooth=True, n_jobs=n_jobs)
                dims_raw = mle.dimension_pw_.astype(float)
                dims_smooth = mle.dimension_pw_smooth_.astype(float)
                return dims_raw, dims_smooth, coords_filtered
            
    except Exception as e:
        print(f"  Failed to compute local {estimator_name}: {str(e)}")
        return None, None, coords_filtered
    
    return None, None, coords_filtered


def plot_two_globe_views(coords, values, k, title, output_path=None):
    coords = np.asarray(coords)
    values = np.asarray(values)
    fig = plt.figure(figsize=(16, 8))
    
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.Orthographic(central_longitude=-20, central_latitude=50))
    ax1.set_title(f"{title}\n(NorthPole view)")
    
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.Orthographic(central_longitude=80, central_latitude=0))
    ax2.set_title(f"{title}\n(Asia/Africa view)")
    
    for ax in (ax1, ax2):
        ax.add_feature(cfeature.LAND.with_scale('50m'))
        ax.add_feature(cfeature.OCEAN.with_scale('50m'))
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
        ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5)
        sc = ax.scatter(
            coords[:,0], coords[:,1],
            c=values,
            cmap='YlOrRd',
            norm=LogNorm(),
            transform=ccrs.PlateCarree(),
            s=0.5
        )
    cbar = fig.colorbar(sc, ax=(ax1, ax2), orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.set_label(f'Local MLE (k={k})')
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def collate_ignore_none(batch):
    """Collate function that filters out None samples and handles None values in dicts"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    # Handle case where batch items are dicts with potential None values
    if isinstance(batch[0], dict):
        # Find keys that have at least one non-None value
        valid_keys = set()
        for item in batch:
            for k, v in item.items():
                if v is not None:
                    valid_keys.add(k)
        
        # Filter out None values for each key
        collated = {}
        for key in valid_keys:
            values = [item[key] for item in batch if key in item and item[key] is not None]
            if values:
                collated[key] = default_collate(values)
        return collated
    
    return default_collate(batch)


def prepare_loader(dataset_name, args, model_type=None):
    if dataset_name == 's2':
        if model_type == 'sinr':
            ds = CoordinateDataset(args.s2_index_csv)
            if 0 < args.subset_size < len(ds):
                idx = np.random.choice(len(ds), args.subset_size, replace=False)
                ds = Subset(ds, idx)
            return DataLoader(
                ds,
                batch_size=args.batch_size,
                num_workers=0,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_ignore_none
            )

        if model_type in ['scalemae', 'resnet18_rgb', 'resnet50_rgb']:
            transform = None
        else:
            transform = get_pretrained_s2_train_transform(resize_crop_size=256)

        ds = S2Geo(root=args.s2_data_dir, transform=transform, mode='both')
        ds = RobustDatasetWrapper(ds)

        if 0 < args.subset_size < len(ds):
            idx = np.random.choice(len(ds), args.subset_size, replace=False)
            ds = Subset(ds, idx)

        return DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_ignore_none,
            persistent_workers=True if args.num_workers > 0 else False
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def extract_features_sinr(model, dataloader, device, desc="SINR"):
    model = model.to(device).eval()
    
    class SINRCoordEncoder:
        def encode(self, coords):
            locs = coords.clone()
            locs[:, 0] /= 180.0
            locs[:, 1] /= 90.0
            feats = torch.cat([
                torch.sin(torch.pi * locs),
                torch.cos(torch.pi * locs)
            ], dim=1)
            return feats
    
    encoder = SINRCoordEncoder()
    feats, coords = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            if batch is None:
                continue
            coords_batch = batch['point'].to(device)
            encoded = encoder.encode(coords_batch)
            emb = model(encoded, return_feats=True)
            feats.append(emb.cpu().numpy())
            coords.append(coords_batch.cpu().numpy())
    
    feats = np.vstack(feats).astype(np.float32)
    coords = np.vstack(coords)
    
    bad = ~np.isfinite(feats).all(axis=1)
    if bad.any():
        print(f"[SINR] Dropping {bad.sum()} / {feats.shape[0]} samples with NaN/Inf")
        feats = feats[~bad]
        coords = coords[~bad]
    
    return feats, coords


def get_model_by_name(model_name, args=None):  # Added args parameter
    if model_name == 'rcf':
        return None, None
    elif model_name == 'sinr':  # Added SINR case
        return load_sinr_model(args.sinr_checkpoint), None
    elif model_name == 'scalemae':
        return get_scalemae_model(), None
    elif model_name == 'croma':
        return get_croma_model()
    elif model_name == 'dofa':
        return get_dofa_model()
    elif model_name == 'resnet18':
        return get_resnet18_model(use_rgb=False)
    elif model_name == 'resnet18_rgb':
        return get_resnet18_model(use_rgb=True)
    elif model_name == 'resnet50':
        return get_resnet50_model(use_rgb=False)
    elif model_name == 'resnet50_rgb':
        return get_resnet50_model(use_rgb=True)
    elif model_name == 'resnet152':
        return get_resnet152_model()
    elif model_name in ['swin_b', 'swin']:
        return get_swin_model('b')
    elif model_name == 'swin_t':
        return get_swin_model('t')
    elif model_name == 'vit_small':
        return get_vit_small_model()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--s2_data_dir', type=str, default="/scratch/local/arra4944_images/s2100k")
    p.add_argument('--s2_index_csv', type=str, default="/scratch/local/arra4944_images/s2100k/index.csv")
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=71)
    p.add_argument('--subset_size', type=int, default=-1)
    p.add_argument('--methods', nargs='+', default=['MLE'], 
                choices=['MLE', 'TwoNN', 'FisherS', 'MOM', 'TLE', 'CorrInt', 
                        'DANCo', 'ESS', 'MiND_ML', 'MiND_KL', 'MADA'])
    p.add_argument('--k_values', nargs='+', type=int, default=[5,10,20])
    p.add_argument('--rcf_features', type=int, default=512)
    p.add_argument('--wandb_project', type=str, default="Baseline_ID_Analysis")
    p.add_argument('--models', nargs='+',
                   choices=['rcf', 'scalemae', 'croma', 'dofa',
                            'resnet18', 'resnet18_rgb', 'resnet50', 'resnet50_rgb',
                            'resnet152', 'swin', 'vit_small', 'sinr'],
                   default=['rcf', 'scalemae'])
    p.add_argument('--plot_k', type=int, default=10)
    p.add_argument('--sinr_checkpoint', type=str,
                   default="./sinr/pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt")
    args = p.parse_args()

    wandb.init(project=args.wandb_project, name="torchgeo_baseline_id", config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results = []

    for model_name in args.models:
        print(f"\nProcessing {model_name}...")

        if model_name == 'rcf':
            loader = prepare_loader('s2', args, model_type=model_name)
            model  = get_rcf_model(in_channels=13, features=args.rcf_features)
            feats, coords = extract_features_rcf(model, loader, device)
        elif model_name == 'scalemae':
            loader = prepare_loader('s2', args, model_type=model_name)
            model  = get_scalemae_model()
            feats, coords = extract_features_scalemae(model, loader, device, 's2')
        elif model_name == 'sinr':
            loader = prepare_loader('s2', args, model_type='sinr')
            model, _ = get_model_by_name(model_name, args)
            feats, coords = extract_features_sinr(model, loader, device)
        else:
            loader = prepare_loader('s2', args, model_type=None)
            model, _ = get_model_by_name(model_name, args)
            feats, coords = extract_features_generic(model, loader, device, model_name, 's2')

        assert feats.shape[0] == coords.shape[0], "features/coords length mismatch"

        id_res = calculate_intrinsic_dimension(feats, args.methods, args.k_values)

        for m, v in id_res.items():
            wandb.log({f"{model_name}/{m}": v})
            all_results.append({
                'Model': model_name,
                'Method': m,
                'ID': v,
                'Features': feats.shape[1],
                'Samples': len(loader.dataset)
            })

        print(f"  Model: {model_name}, Feature dim: {feats.shape[1]}, Samples: {len(loader.dataset)}")
        for m, v in id_res.items():
            print(f"    {m}: {v:.3f}")

        for method in args.methods:
            if method not in LOCAL_ESTIMATORS:
                continue
                
            print(f"Computing local {method} for {model_name}...")
            dims_raw, dims_smooth, coords_filtered = compute_local_id(
                feats, estimator_name=method, k=args.plot_k, coords=coords
            )
            
            if dims_raw is None:
                continue

            mask = np.isfinite(dims_raw) & (dims_raw > 0)
            if dims_smooth is not None:
                mask &= np.isfinite(dims_smooth) & (dims_smooth > 0)

            coords_m = coords_filtered[mask]
            raw_m = dims_raw[mask]
            smooth_m = dims_smooth[mask] if dims_smooth is not None else None

            title_stub = f"{model_name} Local {method} (k={args.plot_k})"

            fig_raw = plot_two_globe_views(coords_m, raw_m, k=args.plot_k, title=f"{title_stub} - Raw")
            raw_path = f"local_{method.lower()}_raw_{model_name}_k{args.plot_k}.png"
            fig_raw.savefig(raw_path, dpi=150, bbox_inches='tight')
            wandb.log({f"{model_name}/local_{method.lower()}_raw": wandb.Image(raw_path)})
            plt.close(fig_raw)

            if smooth_m is not None:
                fig_smooth = plot_two_globe_views(coords_m, smooth_m, k=args.plot_k, title=f"{title_stub} - Smooth")
                smooth_path = f"local_{method.lower()}_smooth_{model_name}_k{args.plot_k}.png"
                fig_smooth.savefig(smooth_path, dpi=150, bbox_inches='tight')
                wandb.log({f"{model_name}/local_{method.lower()}_smooth": wandb.Image(smooth_path)})
                plt.close(fig_smooth)

    df = pd.DataFrame(all_results)
    df.to_csv("torchgeo_baseline_id_results.csv", index=False)
    wandb.save("torchgeo_baseline_id_results.csv")
    print("\n=== Final Results ===")
    print(df)
    wandb.log({"results_table": wandb.Table(dataframe=df)})


if __name__ == "__main__":
    main()