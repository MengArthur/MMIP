"""
Builds every file in Data/ once (run locally; the outputs are committed so all runs use identical data):
  train.json / val.json      720 / 100 template-generated dialogues (dialogues.py), image styles 0-2
  benchmark.json             52 hand-written routing cases (benchmark_data.py), image style 3
  hw3_topic.json, hw3_offtopic.json, hw3_attacks.json   作業三 test sets
  images/                    all images (synthetic drawings + resized COCO val2017 photos)
  coco_sources.json          which COCO photos were used where

COCO photos are chosen from the image list of the previous assignment (HW06/Data/partC_coco_subset.json):
product photos = a laptop / phone / TV is a salient object and no other electronic device is annotated;
other photos = animals or food without electronics.

  python -m Code.prepare_data
"""

import io
import json
import os
import random
import re
import urllib.request
import zlib

from PIL import Image

from . import benchmark_data as B
from . import images as I
from .dialogues import Gen

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'Data')
IMG = os.path.join(DATA, 'images')
HW06_SUBSET = os.path.join(BASE, '..', 'HW06', 'Data', 'partC_coco_subset.json')
ELECTRONICS = {'laptop', 'cell phone', 'tv', 'keyboard', 'mouse', 'remote'}
PRODUCT_NAMES = {'laptop': ['筆電', '筆記型電腦', 'laptop'], 'cell phone': ['手機', 'phone'], 'tv': ['電視', '螢幕', 'tv']}
FOOD = {'cake', 'pizza', 'donut', 'sandwich'}
HELD = {'laptop': ['553788'], 'cell phone': ['506933'], 'dog': ['367195', '251572'], 'cat': ['078426'], 'cake': ['180101'],
        'food': ['533206']}
EXCLUDE = ['494634', '451155', '146489', '263463', '068078', '061108', '419312', '561465', '261116', '095707', '223747', '460841']
ANIMALS = {'cat', 'dog', 'horse', 'giraffe', 'elephant', 'bird', 'sheep', 'cow'}


def _save(img, name):
    os.makedirs(IMG, exist_ok=True)
    img = img.convert('RGB')
    img.thumbnail((512, 512))
    img.save(os.path.join(IMG, name), quality=90)
    return 'images/' + name


def _coco(file_name, tag):
    path = os.path.join(IMG, f'coco_{tag}_{file_name}')
    if not os.path.exists(path):
        try:
            import truststore
            truststore.inject_into_ssl()
        except ImportError:
            pass
        raw = urllib.request.urlopen(f'http://images.cocodataset.org/val2017/{file_name}', timeout=60).read()
        _save(Image.open(io.BytesIO(raw)), f'coco_{tag}_{file_name}')
    return f'images/coco_{tag}_{file_name}'


def _group(it):
    sal, obj = set(it['salient']), set(it['objects'])
    if obj & ELECTRONICS:
        return None
    if 'dog' in sal and 'cat' not in obj:
        return 'dog'
    if 'cat' in sal and 'dog' not in obj:
        return 'cat'
    if 'cake' in sal and not obj & ANIMALS:
        return 'cake'
    if sal & FOOD and not obj & ANIMALS:
        return 'food'
    if sal & ANIMALS:
        return 'animal'
    return None


def coco_pools():
    with open(HW06_SUBSET, encoding='utf-8') as f:
        d = json.load(f)
    items = sorted(d['train'] + d['eval'], key=lambda it: it['file_name'])
    pools = {c: [it['file_name'] for it in items if c in it['salient'] and not (set(it['objects']) & ELECTRONICS - {c})]
             for c in PRODUCT_NAMES}
    for g in ('dog', 'cat', 'cake', 'food', 'animal'):
        pools[g] = [it['file_name'] for it in items if _group(it) == g]
    # evaluation photos were picked by hand after looking at the candidates (the product / animal must be clearly
    # visible); photos where it is not (a reflection, a dark room, a plate of other food) are not used at all
    held = {g: [f'000000{i}.jpg' for i in ids] for g, ids in HELD.items()}
    bad = {f'000000{i}.jpg' for i in EXCLUDE}
    train = {g: [f for f in v if f not in bad and all(f not in h for h in held.values())] for g, v in pools.items()}
    return train, held


class Drawer:
    """make_image callback for dialogues.Gen: draws a training image and returns its relative path."""

    def __init__(self, prefix, coco_train, seed):
        self.prefix, self.coco, self.n = prefix, coco_train, 0
        self.rng = random.Random(seed)

    def __call__(self, kind, **kw):
        self.n += 1
        style, seed = self.rng.randint(0, 2), self.rng.randint(0, 10 ** 6)
        name = f'{self.prefix}_{self.n:04d}_{kind}.jpg'
        if kind == 'receipt':
            return _save(I.receipt(kw['order_id'], kw['items'], kw['date'], style, seed), name)
        if kind == 'error':
            return _save(I.error_screen(kw['code'], kw['message'], kw['app'], style, seed), name)
        if kind == 'damage':
            return _save(I.damage(kw['device'], style, seed), name)
        if kind == 'injection':
            return _save(I.injection(kw['lines'], style, seed), name)
        if kind == 'coco_product':
            cat = self.rng.choice(['laptop', 'laptop', 'cell phone', 'cell phone', 'tv'])
            return _coco(self.rng.choice(self.coco[cat]), 'p'), PRODUCT_NAMES[cat]
        if kind == 'coco_other':   # -> (path, group) so that the user's text matches the photo
            g = self.rng.choice(['dog', 'cat', 'cake', 'food', 'food', 'animal'])
            return _coco(self.rng.choice(self.coco[g]), 'o'), g
        raise ValueError(kind)


class EvalDrawer:
    """Image specs of the hand-written sets -> files in style 3 (held-out COCO photos)."""

    def __init__(self, held):
        self.held, self.used = held, {}

    def __call__(self, spec, name):
        k = spec['kind']
        if k == 'receipt':
            return _save(I.receipt(spec['order_id'], spec['items'], spec['date'], 3, zlib.crc32(name.encode())), name + '.jpg')
        if k == 'error':
            return _save(I.error_screen(spec['code'], spec['message'], spec['app'], 3, 7), name + '.jpg')
        if k == 'damage':
            return _save(I.damage(spec['device'], 3, 11), name + '.jpg')
        if k == 'injection':
            return _save(I.injection(spec['lines'], 3, 5), name + '.jpg')
        if k == 'coco':
            return _coco(self.held[spec['cat']][0], 'eval')
        if k == 'coco_other':      # each held-out photo is used once per group (two dog photos)
            g = spec['cat']
            i = self.used.get(g, 0)
            self.used[g] = i + 1
            return _coco(self.held[g][i % len(self.held[g])], 'eval')
        raise ValueError(k)


def _render_turns(turns, draw, prefix):
    out = []
    for t, turn in enumerate(turns):
        imgs = [draw(spec, f'{prefix}_t{t}_{j}') for j, spec in enumerate(turn['images'])]
        out.append({'role': turn['role'], 'text': turn['text'], 'images': imgs})
    return out


def build():
    os.makedirs(DATA, exist_ok=True)
    coco_train, held = coco_pools()
    eval_oids = set()
    for sets in (B.BENCHMARK, B.TOPIC, B.BENIGN_HARD):
        for case in sets:
            for v in case[-1].values():
                if isinstance(v, str) and v.startswith('SD'):
                    eval_oids.add(v)

    def gen_split(name, n, seed):
        gen = Gen(seed, Drawer(name, coco_train, seed + 1))
        gen.used_oids |= eval_oids
        rows = []
        for i in range(n):
            cat, turns, label = gen.sample()
            rows.append({'id': f'{name}{i:04d}', 'category': cat, 'turns': turns, 'label': label})
        return rows, gen.used_oids

    train, used = gen_split('tr', 720, 1)
    eval_oids |= used
    val, _ = gen_split('va', 100, 2)
    for name, rows in (('train', train), ('val', val)):
        with open(os.path.join(DATA, f'{name}.json'), 'w', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)

    draw = EvalDrawer(held)
    bench = [{'id': i, 'category': c, 'turns': _render_turns(t, draw, f'bench_{i}'), 'label': lab} for i, c, t, lab in B.BENCHMARK]
    topic = [{'id': i, 'category': c, 'turns': _render_turns(t, draw, f'topic_{i}'), 'label': lab} for i, c, t, lab in B.TOPIC]
    off = [{'id': i, 'category': c, 'turns': _render_turns(t, draw, f'off_{i}'), 'pending': B.PENDING.get(i)} for i, c, t in B.OFFTOPIC]
    att = [{'id': i, 'category': c, 'turns': _render_turns(t, draw, f'att_{i}'), 'target': tg} for i, c, t, tg in B.ATTACKS]
    benign = [{'id': i, 'category': 'benign_hard', 'turns': _render_turns(t, draw, f'ben_{i}'), 'label': lab} for i, t, lab in B.BENIGN_HARD]
    for name, rows in (('benchmark', bench), ('hw3_topic', topic), ('hw3_offtopic', off), ('hw3_attacks', {'attacks': att, 'benign_hard': benign})):
        with open(os.path.join(DATA, f'{name}.json'), 'w', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
    with open(os.path.join(DATA, 'coco_sources.json'), 'w', encoding='utf-8') as f:
        json.dump({'source': 'COCO 2017 val (images.cocodataset.org), CC BY 4.0 Flickr photos', 'train': coco_train, 'eval': held}, f, indent=1)

    used = set()
    for name in ('train', 'val', 'benchmark', 'hw3_topic', 'hw3_offtopic', 'hw3_attacks'):
        with open(os.path.join(DATA, f'{name}.json'), encoding='utf-8') as f:
            used |= set(re.findall(r'images/[\w.]+', f.read()))
    for fn in os.listdir(IMG):        # drop images of an earlier build that nothing references any more
        if 'images/' + fn not in used:
            os.remove(os.path.join(IMG, fn))

    from collections import Counter
    for name, rows in (('train', train), ('val', val), ('benchmark', bench)):
        print(name, len(rows), dict(Counter(r['label']['status'] for r in rows)), 'with images:',
              sum(any(t['images'] for t in r['turns']) for r in rows))


if __name__ == '__main__':
    build()
