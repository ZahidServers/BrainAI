# BrainAI - built by Mohammed Zahid.
"""Fetch image / audio / video datasets for the sensor encoders.

    python tools/fetch_sensors.py --out /content/drive/MyDrive/brainai/sensors --n 20000

Writes uint8 .npy shards the encoders can consume directly:
    images: (N,3,64,64) + labels     audio: (N,16000) 1 s @16 kHz + labels
    video : (N,8,3,64,64) + labels   (clips of 8 frames)
Training happens offline on recorded data; "live" camera/mic is the inference path.
"""
import argparse
from pathlib import Path
import numpy as np


def images(out, n):
    """Tiny-ImageNet, 200 object classes. Not streamed: the stream is sorted by
    label, so the first N rows would cover only a handful of classes."""
    from datasets import load_dataset
    from PIL import Image
    ds = load_dataset("zh-plus/tiny-imagenet", split="train")
    idx = np.random.default_rng(0).permutation(len(ds))[:n]
    X, Y = [], []
    for i in idx:
        r = ds[int(i)]
        im = r["image"].convert("RGB").resize((64, 64), Image.BILINEAR)
        X.append(np.asarray(im, np.uint8).transpose(2, 0, 1)); Y.append(r["label"])
    np.savez_compressed(out / "images.npz", x=np.stack(X), y=np.array(Y))
    print("images", len(X), "samples,", len(set(Y)), "classes")


def _decode(a):
    import io, soundfile as sf
    x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=False)
    if x.ndim > 1: x = x.mean(1)
    if sr != 16000:
        x = np.interp(np.linspace(0, len(x) - 1, int(len(x) * 16000 / sr)), np.arange(len(x)), x)
    return x.astype(np.float32)


def _fix(x, n):
    x = x[:n]
    return np.pad(x, (0, n - len(x)))


def audio(out, n):
    """ESC-50 environmental sounds (50 classes, 4 s) + LibriSpeech speech with transcripts (6 s)."""
    from datasets import load_dataset, Audio
    ds = load_dataset("ashraq/esc50", split="train", streaming=True).cast_column("audio", Audio(decode=False))
    X, Y = [], []
    for r in ds:
        X.append((_fix(_decode(r["audio"]), 64000) * 32767).astype(np.int16)); Y.append(r["target"])
    np.savez_compressed(out / "audio_sounds.npz", x=np.stack(X), y=np.array(Y))
    print("audio_sounds", len(X))
    ds = load_dataset("openslr/librispeech_asr", "clean", split="train.100", streaming=True).cast_column("audio", Audio(decode=False))
    X, T = [], []
    for r in ds:
        X.append((_fix(_decode(r["audio"]), 96000) * 32767).astype(np.int16)); T.append(r["text"].lower())
        if len(X) >= min(n, 5000): break
    np.savez_compressed(out / "audio_speech.npz", x=np.stack(X), text=np.array(T))
    print("audio_speech", len(X))


def video(out, n):
    """UCF101 subset (10 action classes, ~900 clips) -> 8 frames at 64x64.
    Label = class name parsed from the filename. Archive extracted with the
    'data' filter into its own directory."""
    import cv2, tarfile, os
    from huggingface_hub import hf_hub_download
    arc = hf_hub_download("sayakpaul/ucf101-subset", "UCF101_subset.tar.gz", repo_type="dataset")
    dest = out / "_ucf101_raw"; dest.mkdir(exist_ok=True)
    with tarfile.open(arc) as t:
        t.extractall(dest, filter="data")
    X, names = [], []
    for path in sorted(dest.rglob("*.avi")):
        cap = cv2.VideoCapture(str(path)); fr = []
        while True:
            ok, f = cap.read()
            if not ok: break
            fr.append(cv2.resize(f, (64, 64))[:, :, ::-1])
        cap.release()
        if len(fr) >= 8:
            idx = np.linspace(0, len(fr) - 1, 8).astype(int)
            X.append(np.stack([fr[i] for i in idx]).transpose(0, 3, 1, 2)); names.append(path.name.split("_")[1])
        if len(X) >= n: break
    classes = sorted(set(names))
    np.savez_compressed(out / "video.npz", x=np.stack(X), y=np.array([classes.index(c) for c in names]), classes=np.array(classes))
    print("video", len(X), "clips,", len(classes), "classes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sensors"); ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--only", nargs="*", default=["images", "audio", "video"])
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for k in a.only:
        try: globals()[k](out, a.n if k != "video" else min(a.n, 3000))
        except Exception as e: print(f"[fail] {k}: {e}")
