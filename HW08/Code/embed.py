"""
Retrieval models (both multilingual, strong on Traditional Chinese in the lecture's comparison, slide 10):
  BAAI/bge-m3                 embedding model (dense vector = normalised [CLS] state, 1024-d, up to 8192 tokens)
  BAAI/bge-reranker-v2-m3     cross-encoder re-ranker (slides 23-30): [CLS] query [SEP] chunk [SEP] -> relevance
Implemented with plain transformers (no extra library). Embeddings are cached on disk per text set.
"""

import hashlib
import os

import numpy as np
import torch

from .common import DEVICE, default_dtype

EMB_ID = 'BAAI/bge-m3'
RERANK_ID = 'BAAI/bge-reranker-v2-m3'


class Embedder:
    def __init__(self, model_id=EMB_ID, max_length=512, cache_dir=None):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, dtype=default_dtype()).to(DEVICE).eval()
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.model_id = model_id

    @torch.no_grad()
    def _encode(self, texts, batch_size):
        out = []
        order = np.argsort([-len(t) for t in texts])          # similar lengths per batch
        for s in range(0, len(texts), batch_size):
            idx = order[s:s + batch_size]
            enc = self.tok([texts[i] for i in idx], padding=True, truncation=True, max_length=self.max_length,
                           return_tensors='pt').to(DEVICE)
            h = self.model(**enc).last_hidden_state[:, 0].float()
            out.append((idx, torch.nn.functional.normalize(h, dim=-1).cpu().numpy()))
        E = np.zeros((len(texts), out[0][1].shape[1]), dtype=np.float32) if out else np.zeros((0, 1024), np.float32)
        for idx, e in out:
            E[idx] = e
        return E

    def encode(self, texts, batch_size=32, name=None):
        """L2-normalised embeddings (n, d). With `name`, cached as <cache_dir>/<name>_<hash>.npy."""
        texts = list(texts)
        if name and self.cache_dir:
            h = hashlib.sha1(('\n'.join(texts) + self.model_id).encode()).hexdigest()[:12]
            path = os.path.join(self.cache_dir, f'emb_{name}_{h}.npy')
            if os.path.exists(path):
                return np.load(path)
            E = self._encode(texts, batch_size)
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(path, E)
            return E
        return self._encode(texts, batch_size)


class Reranker:
    def __init__(self, model_id=RERANK_ID, max_length=512):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype=default_dtype()).to(DEVICE).eval()
        self.max_length = max_length

    @torch.no_grad()
    def score(self, query, passages, batch_size=16):
        """Relevance probability (sigmoid of the logit) for each passage; K/B forward passes (slide 25)."""
        scores = []
        for s in range(0, len(passages), batch_size):
            enc = self.tok([query] * len(passages[s:s + batch_size]), passages[s:s + batch_size], padding=True,
                           truncation=True, max_length=self.max_length, return_tensors='pt').to(DEVICE)
            scores.append(torch.sigmoid(self.model(**enc).logits.view(-1).float()).cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)
