#!/usr/bin/env python3
"""
finetune_geoclip_hierarchy_sweep.py

Sweep GeoCLIP fine-tuning over number of RFF hierarchy levels M ∈ [3,10],
with σ_min=2^0, σ_max=2^8 fixed. For each M:
  - builds LocationEncoder with M log-spaced σ’s in [1,256]
  - fine-tunes only the location encoder (image encoder & logit_scale frozen)
  - early stops on val_loss (patience=3)
  - logs metrics to W&B (one run per M)
  - saves best checkpoints named by M

Usage:
  python finetune_geoclip_hierarchy_sweep.py
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import wandb

from model import GeoCLIP
from model.location_encoder import LocationEncoder

# --- Global settings ---------------------------------------------------------
SIGMA_MIN_EXP = 0      # σ_min = 2^0
SIGMA_MAX_EXP = 8      # σ_max = 2^8
M_START       = 3
M_END         = 10     # inclusive
EPOCHS        = 50
BATCH_SIZE    = 256
LR            = 5e-5
WD            = 1e-6
VAL_FRACTION  = 0.10
PATIENCE      = 20
SEED          = 42

IMG_DIR   = "/scratch/local/arra4944_images/yfcc/images"
META_CSV  = "/scratch/local/arra4944_images/yfcc/yfcc15m.csv"

# --- Transforms ----------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275,  0.40821073),
        std =(0.26862954, 0.26130258, 0.27577711),
    ),
])
val_transform = transforms.Compose([
    transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275,  0.40821073),
        std =(0.26862954,  0.26130258, 0.27577711),
    ),
])

# --- Dataset -------------------------------------------------------------------
class YFCC1kDataset(Dataset):
    def __init__(self, df, transform, sigma_noise_m=5.0):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        # convert meters noise to degrees (~111 km per degree)
        self.noise_deg = sigma_noise_m / 111_000.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.path).convert("RGB")
        img = self.transform(img)
        gps = np.array([row.latitude, row.longitude], dtype=np.float32)
        gps += np.random.normal(scale=self.noise_deg, size=2).astype(np.float32)
        return img, torch.from_numpy(gps)

# --- Load & split metadata ----------------------------------------------------
df = pd.read_csv(META_CSV, low_memory=False)
df["filename"] = df.photoid.astype(str) + "." + df.ext.str.lower().str.strip()
df["path"]     = df.filename.apply(lambda fn: os.path.join(IMG_DIR, fn))
df = df.dropna(subset=["latitude", "longitude", "downloadurl"])
df = df[
    (df.latitude != 0) &
    (df.longitude != 0) &
    (df.downloadurl.str.strip() != "") &
    (df.path.apply(os.path.exists))
].reset_index(drop=True)

# shuffle once
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n_val = int(len(df) * VAL_FRACTION)
train_df, val_df = df.iloc[n_val:], df.iloc[:n_val]

# --- Utility: build dataloaders -----------------------------------------------
def make_loaders():
    train_ds = YFCC1kDataset(train_df, transform=train_transform)
    val_ds   = YFCC1kDataset(val_df,   transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=8, pin_memory=True)
    return train_loader, val_loader

# --- Training function --------------------------------------------------------
def train_for_M(M: int):
    """Fine-tune GeoCLIP with M RFF hierarchies."""
    # 1) compute σ list (log-spaced exponents)
    exps   = np.linspace(SIGMA_MIN_EXP, SIGMA_MAX_EXP, M)
    sigmas = [float(2.0 ** e) for e in exps]

    # 2) init W&B run
    run = wandb.init(
        project="geoclip-hierarchy-sweep",
        reinit=True,
        name=f"hier_{M}",
        config={
            "M": M,
            "sigmas": sigmas,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WD,
        }
    )

    # 3) build model & freeze image encoder
    model = GeoCLIP(from_pretrained=True)
    loc_enc = LocationEncoder(sigma=sigmas, from_pretrained=False)
    model.location_encoder = loc_enc
    for p in model.image_encoder.parameters():
        p.requires_grad = False
    model.logit_scale.requires_grad = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    wandb.watch(model, log="all", log_freq=10)

    # 4) optimizer & loader
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WD
    )
    train_loader, val_loader = make_loaders()

    # 5) training loop with early stopping
    best_val, stale = float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        # — train —
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for imgs, gps in train_loader:
            imgs, gps = imgs.to(device), gps.to(device)
            opt.zero_grad()
            img_feats = F.normalize(model.image_encoder(imgs), dim=1)
            loc_feats = F.normalize(model.location_encoder(gps), dim=1)
            logits    = model.logit_scale.exp() * img_feats @ loc_feats.t()
            labels    = torch.arange(len(imgs), device=device)
            loss      = F.cross_entropy(logits, labels)
            acc       = (logits.argmax(dim=1) == labels).float().mean().item()
            loss.backward()
            opt.step()
            train_loss += loss.item() * imgs.size(0)
            train_acc  += acc * imgs.size(0)
        train_loss /= len(train_loader.dataset)
        train_acc  /= len(train_loader.dataset)

        # — validate —
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for imgs, gps in val_loader:
                imgs, gps = imgs.to(device), gps.to(device)
                img_feats = F.normalize(model.image_encoder(imgs), dim=1)
                loc_feats = F.normalize(model.location_encoder(gps), dim=1)
                logits    = model.logit_scale.exp() * img_feats @ loc_feats.t()
                labels    = torch.arange(len(imgs), device=device)
                loss      = F.cross_entropy(logits, labels)
                acc       = (logits.argmax(dim=1) == labels).float().mean().item()
                val_loss += loss.item() * imgs.size(0)
                val_acc  += acc * imgs.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc  /= len(val_loader.dataset)

        # log to W&B and stdout
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        print(f"[M={M}][{epoch}/{EPOCHS}] "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        # checkpoint on improvement
        if val_loss < best_val:
            best_val = val_loss
            stale    = 0

            ckpt_dir = os.path.join("checkpoints", f"hierarchy_{M}")
            os.makedirs(ckpt_dir, exist_ok=True)

            save_path = os.path.join(ckpt_dir, "best_locenc.pt")
            torch.save(model.location_encoder.state_dict(), save_path)
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"[M={M}] early stopping after {epoch} epochs.")
                break

    run.finish()

# --- Main ----------------------------------------------------------------------
def main():
    for M in range(M_START, M_END + 1):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        train_for_M(M)

if __name__ == "__main__":
    main()
