# BrainAI - built by Mohammed Zahid.
"""Train the perception encoders on recorded data, with held-out accuracy.

    python train_sensors.py --modality images --steps 3000
    python train_sensors.py --modality audio  --steps 2000
    python train_sensors.py --modality video  --steps 1500
    python train_sensors.py --modality images --steps 20 --batch 16     # smoke test

Each run trains one encoder (the spiking stack that BrainAI itself uses) plus a
throwaway linear probe on mean-pooled features, and reports top-1 accuracy on a
10% held-out split against chance. Encoder weights are saved to
checkpoints/sensors_<modality>_<scale>.pt for loading into BrainAI.encoder.

Scope: supervised classification only (objects, sounds, actions). Speech->text
(audio_speech.npz) and motor control need different losses and are separate.
"""
import argparse, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from brainai import BrainAI, BrainConfig

SENSORS = Path("sensors")


def load(modality):
    if modality == "images":
        d = np.load(SENSORS / "images.npz"); x, y = d["x"], d["y"]
    elif modality == "audio":
        d = np.load(SENSORS / "audio_sounds.npz"); x, y = d["x"], d["y"]
    else:
        d = np.load(SENSORS / "video.npz"); x, y = d["x"], d["y"]
    return x, y.astype(np.int64)


def to_input(modality, xb, device):
    t = torch.from_numpy(xb).to(device)
    if modality == "audio":
        return (t.float() / 32767.0).unsqueeze(1)             # (B,1,L)
    return t.float() / 255.0                                  # images (B,3,H,W) / video (B,T,3,H,W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["images", "audio", "video"], required=True)
    ap.add_argument("--scale", default="nano")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    x, y = load(a.modality)
    n_cls = int(y.max()) + 1
    perm = rng.permutation(len(x)); cut = int(len(x) * 0.9)
    tr, va = perm[:cut], perm[cut:]
    print(f"{a.modality}: {len(x)} samples, {n_cls} classes, train {len(tr)} / val {len(va)}, device {dev}")

    cfg = BrainConfig.for_scale(a.scale)
    model = BrainAI(cfg).to(dev)
    enc = {"images": model.encoder.image, "audio": model.encoder.audio, "video": model.encoder.video}[a.modality]
    head = nn.Linear(cfg.d_model, n_cls).to(dev)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)

    def logits(xb):
        return head(enc(to_input(a.modality, xb, dev)).mean(dim=1))

    def evaluate():
        enc.eval(); head.eval(); ok = 0
        with torch.no_grad():
            for i in range(0, len(va), a.batch):
                idx = va[i:i + a.batch]
                ok += (logits(x[idx]).argmax(-1).cpu() == torch.from_numpy(y[idx])).sum().item()
        enc.train(); head.train()
        return ok / len(va)

    t0 = time.time()
    for step in range(a.steps):
        lr = a.lr * (min(1, (step + 1) / 50)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups: g["lr"] = lr
        idx = rng.choice(tr, a.batch, replace=False)
        xb = x[idx]
        if a.modality == "images" and rng.random() < 0.5:
            xb = xb[..., ::-1].copy()                          # horizontal flip augmentation
        loss = F.cross_entropy(logits(xb), torch.from_numpy(y[idx]).to(dev))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % max(1, a.steps // 10) == 0 or step == a.steps - 1:
            print(f"step {step:5d}  loss {loss.item():.3f}  val_acc {evaluate():.3f}  ({time.time()-t0:.0f}s)")
    acc = evaluate()
    print(f"FINAL {a.modality}: val top-1 {acc:.3f}   chance {1/n_cls:.3f}")
    out = Path("checkpoints"); out.mkdir(exist_ok=True)
    torch.save({"encoder": enc.state_dict(), "scale": a.scale, "val_acc": acc, "classes": n_cls},
               out / f"sensors_{a.modality}_{a.scale}.pt")


if __name__ == "__main__":
    main()
