"""
CIFAR-10 loading, stratified train/validation split, normalization and batched
data augmentation (implemented on tensors so no DataLoader worker processes are
needed -- keeps things fast and Windows/Jupyter friendly on CPU).
"""

import io
import os
import urllib.request

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ImageNet statistics: required by the ImageNet-pretrained ResNet-18 and reused
# for the Plain CNN so both models see identically preprocessed inputs.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# Official CIFAR-10 release (Krizhevsky, 2009) as published by its authors' group
# (uoft-cs) on the Hugging Face Hub. Same 50,000 train / 10,000 test images and
# labels as https://www.cs.toronto.edu/~kriz/cifar.html, but served from a CDN
# (the Toronto server was downloading at <1 KB/s during development).
HF_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet"


def _load_split(data_dir, split):
    npz_path = os.path.join(data_dir, f"cifar10_{split}.npz")
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        return d['x'], d['y']

    parquet_path = os.path.join(data_dir, f"cifar10_{split}.parquet")
    if not os.path.exists(parquet_path):
        print(f"[Data] Downloading CIFAR-10 {split} split from {HF_URL.format(split=split)}")
        urllib.request.urlretrieve(HF_URL.format(split=split), parquet_path)

    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path).to_pydict()
    x = np.stack([np.array(Image.open(io.BytesIO(img['bytes'])).convert('RGB')) for img in table['img']])
    y = np.array(table['label'], dtype=np.int64)
    np.savez_compressed(npz_path, x=x, y=y)
    return x, y


def load_cifar10(data_dir, val_size=5000, seed=42):
    """
    Returns uint8 image tensors (N, 3, 32, 32) and int64 labels for
    train / val / test. Train and val come from the official 50,000-image
    training set via a stratified split; test is the official 10,000-image
    test set, never used for training or model selection.
    """
    os.makedirs(data_dir, exist_ok=True)
    x_tr_all, y_tr_all = _load_split(data_dir, 'train')
    x_te, y_te = _load_split(data_dir, 'test')

    x_all = torch.from_numpy(x_tr_all).permute(0, 3, 1, 2).contiguous()
    y_all = torch.from_numpy(y_tr_all)
    idx_train, idx_val = train_test_split(
        np.arange(len(y_all)), test_size=val_size, random_state=seed, stratify=y_tr_all
    )

    return {
        'x_train': x_all[idx_train], 'y_train': y_all[idx_train],
        'x_val': x_all[idx_val], 'y_val': y_all[idx_val],
        'x_test': torch.from_numpy(x_te).permute(0, 3, 1, 2).contiguous(),
        'y_test': torch.from_numpy(y_te),
    }


def normalize(x_uint8):
    return (x_uint8.float() / 255.0 - MEAN) / STD


def denormalize(x):
    return (x * STD + MEAN).clamp(0, 1)


def augment_batch(x_uint8, generator):
    """
    Training-time augmentation on a uint8 batch (B, 3, 32, 32):
      1. Random crop with 4-pixel reflect padding (random translation up to +-4 px)
      2. Random horizontal flip (p = 0.5)
      3. Colour jitter: per-image brightness and contrast factor in [0.8, 1.2]
    All are label-preserving for CIFAR-10 (no vertical flip: upside-down
    vehicles/animals do not occur in the test distribution).
    """
    b = x_uint8.shape[0]
    x = x_uint8.float()

    padded = torch.nn.functional.pad(x, (4, 4, 4, 4), mode='reflect')
    dx = torch.randint(0, 9, (b,), generator=generator)
    dy = torch.randint(0, 9, (b,), generator=generator)
    rows = (dy.view(b, 1) + torch.arange(32).view(1, 32))            # (B, 32)
    cols = (dx.view(b, 1) + torch.arange(32).view(1, 32))            # (B, 32)
    batch_idx = torch.arange(b).view(b, 1, 1)
    x = padded.permute(0, 2, 3, 1)[batch_idx, rows.view(b, 32, 1), cols.view(b, 1, 32)]
    x = x.permute(0, 3, 1, 2)

    flip = torch.rand(b, generator=generator) < 0.5
    x[flip] = x[flip].flip(dims=[3])

    brightness = 0.8 + 0.4 * torch.rand(b, 1, 1, 1, generator=generator)
    contrast = 0.8 + 0.4 * torch.rand(b, 1, 1, 1, generator=generator)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mean) * contrast + mean) * brightness

    return x.clamp(0, 255).to(torch.uint8)
