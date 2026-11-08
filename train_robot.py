# BrainAI - built by Mohammed Zahid.
"""Behaviour cloning on LeRobot data: past joint states -> next action.

    python train_robot.py --dataset robot_data/aloha_sim_transfer_cube_human --steps 3000
    python train_robot.py --dataset robot_data/unitreeh1_fold_clothes --steps 3000

Reads the parquet files directly (LeRobot v3 layout: data/chunk-*/file-*.parquet).
A window of the last K proprioceptive states goes through BrainAI's TouchEncoder
(the spiking stack the robot uses for body sense) and a linear action head.

Reported on held-out *episodes* (never frames from a training episode), in
normalised units, against two baselines that need no learning:
    mean-action   always output the training-set mean action
    copy-state    output the current state as the action (strong when the
                  action is a joint-position target; only valid if dims match)
A model that does not beat copy-state has learned nothing about the task.

Scope: state-only. Camera frames (observation.images.*) are not used here;
adding the vision encoder needs per-episode video seeking and is a later step.
Training and the real run belong on a GPU (see colab/BrainAI_colab.ipynb).
"""
import argparse, glob, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from brainai import BrainConfig
from brainai.encoders import TouchEncoder


def load_dataset(root):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(Path(root) / "data" / "chunk-*" / "file-*.parquet")))
    if not files:
        raise SystemExit(f"no parquet files under {root}/data")
    cols = ["observation.state", "action", "episode_index", "frame_index"]
    eps = {}
    for f in files:
        t = pq.read_table(f, columns=cols).to_pydict()
        for s, a, e, fi in zip(t["observation.state"], t["action"], t["episode_index"], t["frame_index"]):
            eps.setdefault(int(e), []).append((int(fi), s, a))
    out = {}
    for e, rows in eps.items():
        rows.sort(key=lambda r: r[0])
        out[e] = (np.asarray([r[1] for r in rows], np.float32), np.asarray([r[2] for r in rows], np.float32))
    return out


def windows(eps, ids, k):
    """All (past-k-states, action) pairs; early frames are padded with the first state."""
    X, Y, S = [], [], []
    for e in ids:
        st, ac = eps[e]
        pad = np.concatenate([np.repeat(st[:1], k - 1, 0), st], 0)
        for t in range(len(st)):
            X.append(pad[t:t + k]); Y.append(ac[t]); S.append(st[t])
    return np.stack(X), np.stack(Y), np.stack(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scale", default="nano")
    ap.add_argument("--k", type=int, default=8, help="state history length")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    eps = load_dataset(a.dataset)
    ids = np.array(sorted(eps)); rng.shuffle(ids)
    n_val = max(1, int(len(ids) * 0.1))
    val_ids, tr_ids = ids[:n_val], ids[n_val:]
    Xtr, Ytr, Str = windows(eps, tr_ids, a.k)
    Xva, Yva, Sva = windows(eps, val_ids, a.k)
    ds, da = Xtr.shape[-1], Ytr.shape[-1]
    print(f"{Path(a.dataset).name}: {len(ids)} episodes ({len(tr_ids)} train / {len(val_ids)} val), "
          f"{len(Xtr)} train frames, state {ds}-d, action {da}-d, device {dev}")

    sm, ss = Xtr.reshape(-1, ds).mean(0), Xtr.reshape(-1, ds).std(0) + 1e-6
    am, asd = Ytr.mean(0), Ytr.std(0) + 1e-6
    nx = lambda x: torch.from_numpy((x - sm) / ss).float().to(dev)
    ny = lambda y: torch.from_numpy((y - am) / asd).float().to(dev)

    cfg = BrainConfig.for_scale(a.scale)
    enc = TouchEncoder(cfg, n_sensors=ds).to(dev)
    head = nn.Linear(cfg.d_model, da).to(dev)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    pred = lambda x: head(enc(x)[:, -1])           # last token = "now"

    def val_mse():
        enc.eval(); head.eval(); se = 0.0
        with torch.no_grad():
            for i in range(0, len(Xva), 512):
                se += ((pred(nx(Xva[i:i + 512])) - ny(Yva[i:i + 512])) ** 2).sum().item()
        enc.train(); head.train()
        return se / (len(Xva) * da)

    base_mean = float(((np.zeros_like(Yva) - ((Yva - am) / asd)) ** 2).mean())
    base_copy = float((((Sva - am) / asd - (Yva - am) / asd) ** 2).mean()) if ds == da else None
    t0 = time.time()
    for step in range(a.steps):
        lr = a.lr * min(1, (step + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups: g["lr"] = lr
        i = rng.integers(0, len(Xtr), a.batch)
        loss = F.mse_loss(pred(nx(Xtr[i])), ny(Ytr[i]))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % max(1, a.steps // 10) == 0 or step == a.steps - 1:
            print(f"step {step:5d}  train {loss.item():.4f}  val {val_mse():.4f}  ({time.time()-t0:.0f}s)")
    final = val_mse()
    print(f"FINAL val MSE {final:.4f} | mean-action baseline {base_mean:.4f}"
          + (f" | copy-state baseline {base_copy:.4f}" if base_copy is not None else " | copy-state n/a (dims differ)"))
    out = Path("checkpoints"); out.mkdir(exist_ok=True)
    torch.save({"touch": enc.state_dict(), "head": head.state_dict(), "scale": a.scale, "k": a.k,
                "state_mean": sm, "state_std": ss, "act_mean": am, "act_std": asd,
                "val_mse": final, "baseline_mean": base_mean, "baseline_copy": base_copy},
               out / f"robot_{Path(a.dataset).name}_{a.scale}.pt")


if __name__ == "__main__":
    main()
