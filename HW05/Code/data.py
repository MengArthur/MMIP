"""
Datasets (Hugging Face `datasets`, cached in Data/hf_cache):
  - Quiz 1 retrieval : CUB-200-2011 (Wah et al., 2011)  -> bentrevett/caltech-ucsd-birds-200-2011
  - Quiz 2 / Quiz 3  : Flickr8k (Hodosh et al., 2013)    -> jxie/flickr8k (same as Week 4)
"""

import os
import re
from collections import Counter

import numpy as np
import torch

CUB_ID = 'bentrevett/caltech-ucsd-birds-200-2011'
FLICKR_ID = 'jxie/flickr8k'
SEED = 42


def _load(hf_id, data_dir):
    from datasets import load_dataset
    return load_dataset(hf_id, cache_dir=os.path.join(data_dir, 'hf_cache'))


# ----------------------------------------------------------------------------
# Quiz 1: CUB-200-2011 with the standard zero-shot retrieval split
# ----------------------------------------------------------------------------
def load_cub(data_dir, n_train_classes=90, n_val_classes=10):
    """
    Standard metric-learning protocol (Oh Song et al., 2016): classes 1-100 for training,
    classes 101-200 for testing, so every test bird species is unseen during training.
    Classes 91-100 are held out as validation (model selection), also unseen in training.
    """
    from datasets import concatenate_datasets
    ds = _load(CUB_ID, data_dir)
    full = concatenate_datasets([ds['train'], ds['test']])
    names = [re.sub(r'^\d+\.', '', n).replace('_', ' ') for n in ds['train'].features['label'].names]
    labels = np.array(list(full['label']))
    tr_cls, va_cls = n_train_classes, n_train_classes + n_val_classes
    idx = {'train': np.where(labels < tr_cls)[0], 'val': np.where((labels >= tr_cls) & (labels < va_cls))[0],
           'test': np.where(labels >= va_cls)[0]}
    return {'hf': full, 'labels': labels, 'idx': idx, 'class_names': names, 'n_train_classes': tr_cls}


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, indices, labels, transform):
        self.ds, self.idx, self.labels, self.tf = hf_ds, np.asarray(indices), labels, transform

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        return self.tf(self.ds[j]['image'].convert('RGB')), int(self.labels[j])


# ----------------------------------------------------------------------------
# Quiz 2 / 3: Flickr8k
# ----------------------------------------------------------------------------
def load_flickr8k(data_dir):
    ds = _load(FLICKR_ID, data_dir)
    cols = [f'caption_{k}' for k in range(5)]
    captions = {}
    for split in ('train', 'validation', 'test'):
        c = ds[split].select_columns(cols)
        captions[split] = [list(r) for r in zip(*[list(c[col]) for col in cols])]
    return {'hf': ds, 'captions': captions}


_CAPTION_WORD = re.compile(r'[a-z]+')


def tokenize_caption(text):
    return _CAPTION_WORD.findall(text.lower())


class Vocab:
    def __init__(self, token_lists, specials, min_freq=1):
        counter = Counter(t for toks in token_lists for t in toks)
        self.itos = list(specials) + [w for w, c in counter.most_common() if c >= min_freq and w not in specials]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.unk = self.stoi['<unk>']

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, self.unk) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]
