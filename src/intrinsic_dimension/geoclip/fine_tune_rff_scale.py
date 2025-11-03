#!/usr/bin/env python3
"""
finetune_geoclip_add_branches_capsule.py

Extend a pretrained GeoCLIP LocationEncoderCapsule by appending one (for exp=12)
or two (for exp=16) finer-scale capsules, freezing all original capsules
and training only the new ones.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import wandb

from model import GeoCLIP
from model.location_encoder import LocationEncoder

# ─── Config ────────────────────────────────────────────────────────────────────
SIGMA_EXT_EXPS = [12, 16]
EPOCHS         = 50
BATCH_SIZE     = 256
LR             = 5e-5
WD             = 1e-6
VAL_FRACTION   = 0.10
PATIENCE       = 20
SEED           = 42

IMG_DIR  = "/scratch/local/arra4944_images/yfcc/images"
META_CSV = "/scratch/local/arra4944_images/yfcc/yfcc15m.csv"

# ─── Transforms ────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5,1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466,0.4578275,0.40821073),
        std =(0.26862954,0.26130258,0.27577711),
    ),
])
val_transform = transforms.Compose([
    transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466,0.4578275,0.40821073),
        std =(0.26862954,0.26130258,0.27577711),
    ),
])

# ─── Dataset ───────────────────────────────────────────────────────────────────
class YFCC15MDataset(Dataset):
    def __init__(self, df, transform, sigma_noise_m=5.0):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
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

def make_loaders(df):
    df = df.dropna(subset=["latitude","longitude","downloadurl"])
    df = df[
        (df.latitude != 0) &
        (df.longitude != 0) &
        (df.downloadurl.str.strip() != "") &
        (df.path.apply(os.path.exists))
    ].reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_val = int(len(df) * VAL_FRACTION)
    val_df, train_df = df.iloc[:n_val], df.iloc[n_val:]
    return (
        DataLoader(YFCC15MDataset(train_df, train_transform),
                   batch_size=BATCH_SIZE, shuffle=True,
                   num_workers=71, pin_memory=True),
        DataLoader(YFCC15MDataset(val_df,   val_transform),
                   batch_size=BATCH_SIZE, shuffle=False,
                   num_workers=71, pin_memory=True),
    )

# ─── Training ──────────────────────────────────────────────────────────────────
def train_extended_branch(max_exp):
    model = GeoCLIP(from_pretrained=True).cuda()
    old_loc_enc = model.location_encoder
    # original capsule names, e.g. ['LocEnc1','LocEnc2','LocEnc3']
    orig_caps = [name for name, _ in old_loc_enc.named_children()]

    # build new sigma list and new encoder
    orig_sigmas = old_loc_enc.sigma      # [1.0,16.0,256.0]
    new_sigmas  = orig_sigmas + [2.0**e for e in SIGMA_EXT_EXPS if e <= max_exp]
    new_loc_enc = LocationEncoder(sigma=new_sigmas, from_pretrained=False)
    # copy pretrained weights into first len(orig_caps) capsules
    for name, module in old_loc_enc.named_children():
        if name in orig_caps:
            new_loc_enc._modules[name].load_state_dict(module.state_dict())
    model.location_encoder = new_loc_enc.cuda()

    # freeze image encoder + logit_scale
    for p in model.image_encoder.parameters():
        p.requires_grad = False
    model.logit_scale.requires_grad = False

    # freeze all Capsules, unfreeze only the new ones
    for name, module in model.location_encoder.named_children():
        for p in module.parameters():
            p.requires_grad = False
    # determine new capsule names (the ones beyond orig_caps)
    new_caps = [n for n in model.location_encoder._modules.keys()
                if n not in orig_caps]
    for name in new_caps:
        for p in model.location_encoder._modules[name].parameters():
            p.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WD
    )
    run = wandb.init(
        project="geoclip-extended-freq",
        reinit=True,
        name=f"extended_maxexp_{max_exp}",
        config={"new_caps": new_caps, "epochs":EPOCHS}
    )
    wandb.watch(model, log="all", log_freq=10)

    df = pd.read_csv(META_CSV, low_memory=False)
    df["filename"] = df.photoid.astype(str) + "." + df.ext.str.lower().str.strip()
    df["path"]     = df.filename.apply(lambda fn: os.path.join(IMG_DIR, fn))
    train_loader, val_loader = make_loaders(df)

    best_val, stale = float("inf"), 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        tl, ta = 0.0, 0.0
        for imgs, gps in train_loader:
            imgs, gps = imgs.cuda(), gps.cuda()
            optimizer.zero_grad()
            img_f = F.normalize(model.image_encoder(imgs), dim=1)
            loc_f = F.normalize(model.location_encoder(gps), dim=1)
            logits = model.logit_scale.exp() * img_f @ loc_f.t()
            labels = torch.arange(len(imgs), device='cuda')
            loss   = F.cross_entropy(logits, labels)
            acc    = (logits.argmax(1)==labels).float().mean().item()
            loss.backward()
            optimizer.step()
            tl += loss.item() * imgs.size(0)
            ta += acc * imgs.size(0)
        train_loss, train_acc = tl/len(train_loader.dataset), ta/len(train_loader.dataset)

        model.eval()
        vl, va = 0.0, 0.0
        with torch.no_grad():
            for imgs, gps in val_loader:
                imgs, gps = imgs.cuda(), gps.cuda()
                img_f = F.normalize(model.image_encoder(imgs), dim=1)
                loc_f = F.normalize(model.location_encoder(gps), dim=1)
                logits = model.logit_scale.exp() * img_f @ loc_f.t()
                labels = torch.arange(len(imgs), device='cuda')
                loss   = F.cross_entropy(logits, labels)
                acc    = (logits.argmax(1)==labels).float().mean().item()
                vl += loss.item() * imgs.size(0)
                va += acc * imgs.size(0)
        val_loss, val_acc = vl/len(val_loader.dataset), va/len(val_loader.dataset)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        print(f"[exp={max_exp}][{epoch}/{EPOCHS}] "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        if val_loss < best_val:
            best_val, stale = val_loss, 0
            ckpt_dir = f"checkpoints/extended_maxexp_{max_exp}"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                model.location_encoder.state_dict(),
                os.path.join(ckpt_dir, "best_locenc.pt")
            )
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"[exp={max_exp}] early stopping at epoch {epoch}.")
                break

    run.finish()

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    for exp in SIGMA_EXT_EXPS:
        train_extended_branch(exp)

if __name__ == "__main__":
    main()
