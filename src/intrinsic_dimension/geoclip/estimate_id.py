#!/usr/bin/env python3
"""
compare_locenc_id.py

Load the pretrained base and fine-tuned high-resolution LocationEncoders,
sample lat/lon pairs using multiple schemes, compute embeddings, and estimate 
intrinsic dimension via scikit-dimension's FisherS estimator.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from skdim.id import FisherS
from model.location_encoder import LocationEncoder

# For land sampling
import cartopy.io.shapereader as shapereader
from shapely.geometry import Point
from shapely.ops import unary_union

# ─── Configuration ────────────────────────────────────────────────────────────────
N_SAMPLES       = 100_000  # Reduced for land sampling efficiency
BATCH_SIZE      = 4096
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
BASE_SIGMAS     = [2**0, 2**4, 2**8]
HIGHRES_SIGMAS  = [2**0, 2**4, 2**8, 2**12, 2**16]
BASE_CHECKPT    = None
HIGHRES_CHECKPT = "checkpoints/extended_maxexp_16/best_locenc.pt"

# ─── Sampling functions ─────────────────────────────────────────────────────────
def sample_land(N, seed=42):
    """Sample uniformly on sphere but keep only land points."""
    np.random.seed(seed)
    shp = shapereader.natural_earth(resolution='110m', category='physical', name='land')
    reader = shapereader.Reader(shp)
    geoms = list(reader.geometries())
    land_poly = unary_union(geoms)
    
    pts = []
    batch = max(N, 10_000)
    while len(pts) < N:
        lons = np.random.uniform(-180, 180, batch)
        u = np.random.uniform(-1, 1, batch)
        lats = np.degrees(np.arcsin(u))
        for lon, lat in zip(lons, lats):
            if land_poly.contains(Point(lon, lat)):
                pts.append((lat, lon))  # Note: returning (lat, lon) for consistency
                if len(pts) == N:
                    break
    return np.array(pts, dtype=np.float32)

def sample_sphere(N, seed=42):
    """Uniform sampling on sphere surface."""
    np.random.seed(seed)
    u = np.random.uniform(-1, 1, N)
    lats = np.degrees(np.arcsin(u))
    lons = np.random.uniform(-180, 180, N)
    return np.stack([lats, lons], axis=1).astype(np.float32)

def sample_fibonacci(N, seed=42):
    """Fibonacci spiral sampling."""
    np.random.seed(seed)
    i = np.arange(N)
    phi = np.pi * (3. - np.sqrt(5.))
    lats = np.degrees(np.arcsin(2*(i / N) - 1))
    lons = np.degrees((i * phi) % (2*np.pi) - np.pi)
    return np.stack([lats, lons], axis=1).astype(np.float32)

# ─── Embed a batch of coords with a given encoder ─────────────────────────────
def collect_embeddings(loc_enc: torch.nn.Module,
                       coords: np.ndarray,
                       batch_size: int,
                       device: str):
    loc_enc = loc_enc.to(device).eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, coords.shape[0], batch_size):
            batch = torch.from_numpy(coords[start:start+batch_size]).to(device)
            feats = loc_enc(batch)
            embeddings.append(feats.cpu().numpy())
    return np.vstack(embeddings)

def evaluate_sampling_scheme(coords, base_enc, high_enc, scheme_name):
    """Evaluate ID for a specific sampling scheme."""
    print(f"\n{scheme_name} Sampling ({len(coords)} points):")
    print("-" * 50)
    
    # Base encoder embeddings
    print(f"Computing embeddings with base encoder...")
    emb_base = collect_embeddings(base_enc, coords, BATCH_SIZE, DEVICE)
    
    # High-res encoder embeddings  
    print(f"Computing embeddings with high-res encoder...")
    emb_high = collect_embeddings(high_enc, coords, BATCH_SIZE, DEVICE)
    
    # Estimate ID with FisherS
    fishers = FisherS(conditional_number=10)
    
    print(f"Estimating intrinsic dimension...")
    id_base = fishers.fit_transform(emb_base)
    id_high = fishers.fit_transform(emb_high)
    
    print(f"  Base encoder ID:     {id_base:.2f}")
    print(f"  High-res encoder ID: {id_high:.2f}")
    print(f"  ID increase:         {id_high - id_base:.2f}")
    
    return {
        'scheme': scheme_name,
        'id_base': id_base,
        'id_high': id_high,
        'id_delta': id_high - id_base
    }

def main():
    # Load encoders once
    print("Loading encoders...")
    
    # Base (pretrained) LocationEncoder
    base_enc = LocationEncoder(sigma=BASE_SIGMAS, from_pretrained=True)
    if BASE_CHECKPT:
        base_enc.load_state_dict(
            torch.load(BASE_CHECKPT, map_location="cpu")
        )
    
    # High-resolution fine-tuned LocationEncoder
    high_enc = LocationEncoder(sigma=HIGHRES_SIGMAS, from_pretrained=False)
    if os.path.exists(HIGHRES_CHECKPT):
        ckpt = torch.load(HIGHRES_CHECKPT, map_location="cpu")
        high_enc.load_state_dict(ckpt, strict=False)
    else:
        print(f"Warning: Checkpoint {HIGHRES_CHECKPT} not found, using random init")
    
    # Define sampling schemes
    sampling_schemes = [
        ("Sphere", sample_sphere),
        ("Fibonacci", sample_fibonacci),
        ("Land", sample_land),
    ]
    
    results = []
    
    # Evaluate each sampling scheme
    for name, sampler in sampling_schemes:
        try:
            print(f"\nGenerating {name} samples...")
            coords = sampler(N_SAMPLES, seed=42)
            result = evaluate_sampling_scheme(coords, base_enc, high_enc, name)
            results.append(result)
        except Exception as e:
            print(f"Error with {name} sampling: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Intrinsic Dimension by Sampling Scheme")
    print("=" * 60)
    print(f"{'Scheme':<12} {'Base ID':>10} {'High-Res ID':>12} {'Δ ID':>8}")
    print("-" * 42)
    for r in results:
        print(f"{r['scheme']:<12} {r['id_base']:>10.2f} {r['id_high']:>12.2f} {r['id_delta']:>8.2f}")

if __name__ == "__main__":
    main()