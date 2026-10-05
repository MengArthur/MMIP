"""
Dataset loading for all quizzes (Hugging Face `datasets`, cached under Data/hf_cache)
plus the word-level tokenizer / vocabulary shared by Quiz 2 and Quiz 4.

  - Sentiment Analysis : IMDB Large Movie Review (Maas et al., 2011)   -> stanfordnlp/imdb
  - Image Classification: Oxford-IIIT Pet (Parkhi et al., 2012)        -> timm/oxford-iiit-pet
  - Image Captioning   : Flickr8k (Hodosh et al., 2013)                -> jxie/flickr8k
"""

import os
import re
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import train_test_split

IMDB_ID = 'stanfordnlp/imdb'
PETS_ID = 'timm/oxford-iiit-pet'
FLICKR_ID = 'jxie/flickr8k'
SEED = 42

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load(hf_id, data_dir):
    from datasets import load_dataset
    return load_dataset(hf_id, cache_dir=os.path.join(data_dir, 'hf_cache'))


# ----------------------------------------------------------------------------
# Text
# ----------------------------------------------------------------------------
_WORD = re.compile(r"[a-z0-9']+")
_CAPTION_WORD = re.compile(r"[a-z]+")


def tokenize(text):
    """IMDB reviews: lower-case, drop the <br /> HTML tags, keep words/numbers/apostrophes."""
    return _WORD.findall(text.lower().replace('<br />', ' '))


def tokenize_caption(text):
    return _CAPTION_WORD.findall(text.lower())


class Vocab:
    def __init__(self, token_lists, specials, max_size=None, min_freq=1):
        counter = Counter(t for toks in token_lists for t in toks)
        words = [w for w, c in counter.most_common() if c >= min_freq and w not in specials]
        if max_size:
            words = words[:max_size]
        self.itos = list(specials) + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.unk = self.stoi['<unk>']

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, self.unk) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]


# ----------------------------------------------------------------------------
# Quiz 1 datasets
# ----------------------------------------------------------------------------
def load_imdb(data_dir, val_size=2500):
    """Official 25k train / 25k test; 2,500 reviews of the train split (stratified) become validation."""
    ds = _load(IMDB_ID, data_dir)
    text, y = list(ds['train']['text']), np.array(ds['train']['label'])
    i_tr, i_val = train_test_split(np.arange(len(y)), test_size=val_size, stratify=y, random_state=SEED)
    return {
        'train': ([text[i] for i in i_tr], y[i_tr]),
        'val': ([text[i] for i in i_val], y[i_val]),
        'test': (list(ds['test']['text']), np.array(ds['test']['label'])),
        'class_names': ['negative', 'positive'],
    }


def load_pets(data_dir, val_frac=0.15):
    """Official trainval (3,680) / test (3,669); 15% of trainval (stratified) becomes validation."""
    ds = _load(PETS_ID, data_dir)
    names = [n.replace('_', ' ') for n in ds['train'].features['label'].names]
    y_all = np.array(ds['train']['label'])
    i_tr, i_val = train_test_split(np.arange(len(y_all)), test_size=val_frac, stratify=y_all, random_state=SEED)
    return {'hf_train': ds['train'], 'hf_test': ds['test'], 'idx_train': i_tr, 'idx_val': i_val,
            'y_train': y_all[i_tr], 'y_val': y_all[i_val], 'y_test': np.array(ds['test']['label']),
            'class_names': names}


def load_flickr8k(data_dir):
    """Official Karpathy-style split: 6,000 train / 1,000 val / 1,000 test images, 5 captions each."""
    ds = _load(FLICKR_ID, data_dir)
    cols = [f'caption_{k}' for k in range(5)]
    captions = {}
    for split in ('train', 'validation', 'test'):
        c = ds[split].select_columns(cols)
        columns = [list(c[col]) for col in cols]
        captions[split] = [list(row) for row in zip(*columns)]
    return {'hf': ds, 'captions': captions}


class HFImageDataset(torch.utils.data.Dataset):
    """Wraps a Hugging Face image dataset; returns (transformed image, label)."""

    def __init__(self, hf_ds, indices, transform, label_key='label'):
        self.ds, self.idx, self.tf, self.key = hf_ds, np.asarray(indices), transform, label_key

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        ex = self.ds[int(self.idx[i])]
        label = ex[self.key] if self.key else 0
        return self.tf(ex['image'].convert('RGB')), int(label)
