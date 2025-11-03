#!/usr/bin/env python3
"""
compare_locenc_id_by_hierarchy.py

For each RFF hierarchy level M ∈ [3,10] and each sampling scheme:
  - constructs LocationEncoder with M log-spaced σ ∈ [2^0, 2^8]
  - loads its fine-tuned checkpoint from checkpoints/hierarchy_{M}/best_locenc.pt
  - samples N_SAMPLES (lat,lon) pairs using different schemes
  - computes embeddings in batches
  - estimates intrinsic dimension via scikit-dimension's FisherS
  - prints the resulting ID for each M and sampling scheme
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

# ─── Configuration ─────────────────────────────────────────────────────────────
N_SAMPLES    = 100_000  
BATCH_SIZE   = 4096
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SIGMA_MIN_EXP = 0      # exponent for σ_min = 2^0
SIGMA_MAX_EXP = 8      # exponent for σ_max = 2^8
M_START      = 3
M_END        = 10
CKPT_ROOT    = "checkpoints"   # expects checkpoints/hierarchy_{M}/best_locenc.pt
SEED         = 42

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
                pts.append((lat, lon))  # (lat, lon) order
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

def evaluate_hierarchy(M, coords, scheme_name):
    """Evaluate ID for a specific M and sampling scheme."""
    # Build σ list
    exps   = np.linspace(SIGMA_MIN_EXP, SIGMA_MAX_EXP, M)
    sigmas = [float(2.0 ** e) for e in exps]
    
    # Instantiate encoder and load checkpoint
    loc_enc = LocationEncoder(sigma=sigmas, from_pretrained=False)
    ckpt_path = os.path.join(CKPT_ROOT, f"hierarchy_{M}", "best_locenc.pt")
    
    if not os.path.isfile(ckpt_path):
        print(f"  [M={M}] checkpoint not found at {ckpt_path}, skipping.")
        return None
    
    state = torch.load(ckpt_path, map_location="cpu")
    loc_enc.load_state_dict(state, strict=False)
    
    # Compute embeddings
    emb = collect_embeddings(loc_enc, coords, BATCH_SIZE, DEVICE)
    
    # Estimate intrinsic dimension via FisherS
    fishers = FisherS(conditional_number=10)
    id_est = fishers.fit_transform(emb)
    
    return id_est

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    sampling_schemes = [
        ("Sphere", sample_sphere),
        ("Fibonacci", sample_fibonacci),
        ("Land", sample_land),
    ]
    
    results = {scheme: {} for scheme, _ in sampling_schemes}
    
    # Generate samples once for each scheme
    print("Generating samples for each scheme...")
    samples = {}
    for name, sampler in sampling_schemes:
        print(f"  Generating {name} samples...")
        samples[name] = sampler(N_SAMPLES, seed=SEED)
    
    # Test each M value with each sampling scheme
    print("\nEvaluating hierarchies...")
    for M in range(M_START, M_END + 1):
        print(f"\n[M={M}] Processing...")
        
        for scheme_name, _ in sampling_schemes:
            print(f"  {scheme_name} sampling...", end=" ")
            id_val = evaluate_hierarchy(M, samples[scheme_name], scheme_name)
            
            if id_val is not None:
                results[scheme_name][M] = id_val
                print(f"ID = {id_val:.2f}")
            else:
                print("skipped")
    
    print("\n" + "="*70)
    print("DETAILED RESULTS - Intrinsic Dimension by M and Sampling Scheme")
    print("="*70)
    
    for scheme_name, _ in sampling_schemes:
        print(f"\n{scheme_name} Sampling:")
        print("-"*30)
        print(f"{'M':>4} {'σ values':>30} {'ID':>8}")
        print("-"*30)
        
        for M in range(M_START, M_END + 1):
            if M in results[scheme_name]:
                exps = np.linspace(SIGMA_MIN_EXP, SIGMA_MAX_EXP, M)
                sigmas = [2.0**e for e in exps]
                sigma_str = f"[{sigmas[0]:.1f}...{sigmas[-1]:.1f}] ({M} values)"
                print(f"{M:>4} {sigma_str:>30} {results[scheme_name][M]:>8.2f}")
    
    print("\n" + "="*70)
    print("SUMMARY - Intrinsic Dimension Comparison")
    print("="*70)
    print(f"{'M':>4} ", end="")
    for scheme_name, _ in sampling_schemes:
        print(f"{scheme_name:>12} ", end="")
    print("\n" + "-"*50)
    
    for M in range(M_START, M_END + 1):
        print(f"{M:>4} ", end="")
        for scheme_name, _ in sampling_schemes:
            if M in results[scheme_name]:
                print(f"{results[scheme_name][M]:>12.2f} ", end="")
            else:
                print(f"{'N/A':>12} ", end="")
        print()
    
    print("\n" + "="*70)
    print("ANALYSIS - ID Change with Hierarchy Depth")
    print("="*70)
    
    for scheme_name, _ in sampling_schemes:
        if len(results[scheme_name]) >= 2:
            m_vals = sorted(results[scheme_name].keys())
            first_id = results[scheme_name][m_vals[0]]
            last_id = results[scheme_name][m_vals[-1]]
            change = last_id - first_id
            print(f"{scheme_name:12}: M={m_vals[0]} ID={first_id:.2f} → M={m_vals[-1]} ID={last_id:.2f} (Δ={change:+.2f})")

if __name__ == "__main__":
    main()