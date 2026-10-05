"""
Quiz 2: Sentiment Analysis on IMDB with a vanilla RNN (basic) and an LSTM (advanced).

Both models share exactly the same pipeline and hyper-parameters; the only
difference is the recurrent cell, so the comparison isolates RNN vs LSTM.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve

from .common import DEVICE, plt, set_seed, count_params, fit_classifier, predict_proba, plot_histories, save_fig_path
from .data import Vocab, tokenize

MAX_LEN = 256          # keep the LAST 256 tokens of each review (see README)
VOCAB_SIZE = 20000
EMBED_DIM = 128
HIDDEN = 128
DROPOUT = 0.3
EPOCHS = 10
LR = 1e-3
BATCH = 64
CLIP = 1.0             # gradient clipping (vanilla RNNs are prone to exploding gradients)
LEN_BUCKETS = [(0, 128, '<=128'), (129, 256, '129-256'), (257, 10 ** 9, '>256 (truncated)')]

CUSTOM_REVIEWS = [
    ('This movie was absolutely wonderful, the acting and the story were brilliant.', 1),
    ('A complete waste of time. The plot made no sense and the acting was terrible.', 0),
    ('I expected to hate this film, but by the end I was completely won over.', 1),
    ('The first half is great, but the ending ruins everything and I left the cinema disappointed.', 0),
]


class TextClassifier(nn.Module):
    """Embedding -> single-layer RNN or LSTM -> last hidden state -> Linear(2)."""

    def __init__(self, vocab_size, cell, embed_dim=EMBED_DIM, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.cell = cell
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn = nn.LSTM if cell == 'lstm' else nn.RNN
        self.rnn = rnn(embed_dim, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x, lengths):
        e = self.drop(self.embed(x))
        # Packing makes the recurrence stop at each review's real length, so the
        # final hidden state is never computed over padding tokens.
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        if isinstance(h, tuple):     # LSTM returns (h_n, c_n)
            h = h[0]
        return self.fc(self.drop(h[-1]))


class _SeqDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, labels):
        self.seqs, self.labels = seqs, labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], int(self.labels[i])


def _collate(batch):
    seqs, labels = zip(*batch)
    return (pad_sequence(seqs, batch_first=True, padding_value=0),
            torch.tensor([len(s) for s in seqs]), torch.tensor(labels))


def _forward(model, batch):
    x, lengths, y = batch
    return model(x.to(DEVICE), lengths), y.to(DEVICE)


def encode(texts, vocab, max_len=MAX_LEN):
    """Returns truncated id tensors (last max_len tokens) and the original token counts."""
    seqs, n_tokens = [], []
    for t in texts:
        ids = vocab.encode(tokenize(t))
        n_tokens.append(len(ids))
        seqs.append(torch.tensor(ids[-max_len:] or [vocab.unk]))
    return seqs, np.array(n_tokens)


def _loader(seqs, labels, shuffle):
    return torch.utils.data.DataLoader(_SeqDataset(seqs, labels), batch_size=BATCH, shuffle=shuffle,
                                       collate_fn=_collate)


def binary_metrics(probs, y):
    pred = probs.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average='binary', pos_label=1, zero_division=0)
    return {'accuracy': float((pred == y).mean()), 'precision': float(p), 'recall': float(r), 'f1': float(f1),
            'roc_auc': float(roc_auc_score(y, probs[:, 1]))}


def _evaluation_figure(results, y, n_tokens, path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    for ax, (name, r) in zip(axes[:2], results.items()):
        cm = confusion_matrix(y, r['probs'].argmax(1))
        ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_xticks([0, 1], ['negative', 'positive']); ax.set_yticks([0, 1], ['negative', 'positive'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f"{name}: confusion matrix\nacc {r['metrics']['accuracy']:.4f} | F1 {r['metrics']['f1']:.4f}", fontweight='bold')
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y, r['probs'][:, 1])
        axes[2].plot(fpr, tpr, label=f"{name} (AUC = {r['metrics']['roc_auc']:.4f})")
    axes[2].plot([0, 1], [0, 1], 'k--', lw=0.8)
    axes[2].set_xlabel('False Positive Rate'); axes[2].set_ylabel('True Positive Rate')
    axes[2].set_title('ROC curve (test set, positive = positive review)', fontweight='bold'); axes[2].legend()
    width = 0.38
    for k, (name, r) in enumerate(results.items()):
        accs = [r['bucket_acc'][label] for _, _, label in LEN_BUCKETS]
        bars = axes[3].bar(np.arange(len(LEN_BUCKETS)) + (k - 0.5) * width, accs, width, label=name)
        for b, a in zip(bars, accs):
            axes[3].text(b.get_x() + b.get_width() / 2, a, f'{a:.3f}', ha='center', va='bottom', fontsize=8)
    counts = [int(((n_tokens >= lo) & (n_tokens <= hi)).sum()) for lo, hi, _ in LEN_BUCKETS]
    axes[3].set_xticks(np.arange(len(LEN_BUCKETS)), [f'{l}\n(n = {c:,})' for (_, _, l), c in zip(LEN_BUCKETS, counts)])
    axes[3].set_ylim(0.4, 1.08); axes[3].set_xlabel('review length (tokens)')
    axes[3].set_title('Test accuracy by review length', fontweight='bold')
    axes[3].legend(loc='upper center', ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def run_quiz2(imdb, ckpt_dir, output_dir, force=False):
    print('=' * 70); print('QUIZ 2: SENTIMENT ANALYSIS (RNN vs LSTM)'); print('=' * 70)
    set_seed(42)
    train_texts, y_train = imdb['train']
    vocab = Vocab((tokenize(t) for t in train_texts), specials=('<pad>', '<unk>'), max_size=VOCAB_SIZE, min_freq=2)
    print(f'Vocabulary: {len(vocab):,} tokens (top {VOCAB_SIZE:,} training words with frequency >= 2, plus <pad>/<unk>)')
    enc = {s: encode(imdb[s][0], vocab) for s in ('train', 'val', 'test')}
    loaders = {'train': _loader(enc['train'][0], y_train, True),
               'val': _loader(enc['val'][0], imdb['val'][1], False),
               'test': _loader(enc['test'][0], imdb['test'][1], False)}
    y_test, n_tok_test = imdb['test'][1], enc['test'][1]

    results, histories, models, rows = {}, {}, {}, []
    for name, cell in [('RNN', 'rnn'), ('LSTM', 'lstm')]:
        set_seed(42)
        model = TextClassifier(len(vocab), cell)
        hist = fit_classifier(f'quiz2_{cell}', model, loaders['train'], loaders['val'], _forward, ckpt_dir,
                              epochs=EPOCHS, lr=LR, clip=CLIP, force=force)
        probs, _ = predict_proba(model, loaders['test'], _forward)
        pred = probs.argmax(1)
        bucket_acc = {label: float((pred[(n_tok_test >= lo) & (n_tok_test <= hi)] ==
                                    y_test[(n_tok_test >= lo) & (n_tok_test <= hi)]).mean())
                      for lo, hi, label in LEN_BUCKETS}
        m = binary_metrics(probs, y_test)
        results[name] = {'probs': probs, 'metrics': m, 'bucket_acc': bucket_acc}
        histories[name], models[name] = hist, model
        rows.append({'model': name, 'params': count_params(model), 'best_epoch': hist['best_epoch'],
                     'best_val_acc': hist['best_val_acc'], **{f'test_{k}': round(v, 4) for k, v in m.items()},
                     **{f'acc_len_{k}': round(v, 4) for k, v in bucket_acc.items()},
                     'train_minutes': hist['train_minutes']})

    table = pd.DataFrame(rows)
    table.to_csv(save_fig_path(output_dir, 'quiz2_rnn_vs_lstm.csv'), index=False)
    print(table.to_string(index=False))

    pd.DataFrame({'true_label': y_test, 'n_tokens': n_tok_test,
                  'rnn_prob_positive': results['RNN']['probs'][:, 1].round(4),
                  'lstm_prob_positive': results['LSTM']['probs'][:, 1].round(4)}
                 ).to_csv(save_fig_path(output_dir, 'quiz2_test_predictions.csv'), index_label='test_index')

    # Inference examples: 8 random test reviews + 4 hand-written reviews
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y_test), 8, replace=False)
    texts = [imdb['test'][0][i] for i in idx] + [t for t, _ in CUSTOM_REVIEWS]
    labels = list(y_test[idx]) + [l for _, l in CUSTOM_REVIEWS]
    seqs, _ = encode(texts, vocab)
    ex_loader = _loader(seqs, labels, False)
    names = imdb['class_names']
    ex = pd.DataFrame({'source': [f'test #{i}' for i in idx] + ['custom'] * len(CUSTOM_REVIEWS),
                       'review (first 120 chars)': [t.replace('<br />', ' ')[:120] for t in texts],
                       'true': [names[l] for l in labels]})
    for name, model in models.items():
        p, _ = predict_proba(model, ex_loader, _forward)
        ex[f'{name} prediction'] = [f'{names[k]} ({p[i, k]:.2f})' for i, k in enumerate(p.argmax(1))]
    ex.to_csv(save_fig_path(output_dir, 'quiz2_inference_examples.csv'), index=False)

    figs = {'curves': plot_histories(histories, save_fig_path(output_dir, 'quiz2_training_curves.png'),
                                     'Quiz 2: RNN vs LSTM on IMDB'),
            'evaluation': _evaluation_figure(results, y_test, n_tok_test,
                                             save_fig_path(output_dir, 'quiz2_evaluation.png'))}
    return {'table': table, 'examples': ex, 'figures': figs, 'histories': histories, 'vocab_size': len(vocab)}
