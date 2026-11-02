"""
One-off data preparation (already run; its outputs are committed in Data/ so every run uses the same
benchmark). Re-run only to rebuild:

  python -m Code.prepare_data --coco_ann <dir with instances_val2017.json, captions_val2017.json>
                              --mmlu <mmlu all/test parquet> --gsm8k <gsm8k main/test parquet>
                              --coco_imgs <dir with the Part B COCO jpgs>

Outputs
  Data/partB_images/*.jpg|png, Data/partB_vlm_benchmark.json   Part B 20-question visual benchmark
  Data/partC_general_benchmark.json                            Part C 100 general questions (60 MMLU + 40 GSM8K)
  Data/partC_coco_subset.json                                  Part C advanced: COCO val2017 train / eval images
                                                               with annotated objects, captions, POPE questions
"""

import argparse
import collections
import json
import os
import random
import shutil

from .common import DATA_DIR, save_json

# ----------------------------------------------------------------------------- Part B
# natural images: COCO val2017 (CC BY 4.0 / Flickr licences); answers checked by looking at each image
PARTB_COCO = [
    ('B01', '000000088250.jpg', '影像分類', 'What kind of animal is shown in this image? Answer with one word.',
     'elephant (two elephants touching trunks)', {'type': 'vqa_keywords', 'groups': [['elephant']]}),
    ('B02', '000000031269.jpg', '物件計數', 'How many zebras are in this image? Answer with a number only.',
     '3 (two in the centre, one partly visible at the left edge)', {'type': 'vqa_number', 'target': 3}),
    ('B03', '000000118209.jpg', '物件計數', 'How many horses are in this image? Answer with a number only.',
     '2', {'type': 'vqa_number', 'target': 2}),
    ('B04', '000000039769.jpg', '小物件計數', 'How many remote controls are in this image? Answer with a number only.',
     '2 (two remotes at the top of the pink couch, next to two sleeping cats)', {'type': 'vqa_number', 'target': 2}),
    ('B05', '000000183648.jpg', '人物計數', 'How many people are standing in front of the elephant? Answer with a number only.',
     '3 (a woman in a cap and two men)', {'type': 'vqa_number', 'target': 3}),
    ('B06', '000000079408.jpg', '場景文字辨識 (OCR)', 'What word is written on the sign?',
     'STOP (red stop sign, black-and-white background)', {'type': 'vqa_keywords', 'groups': [['stop']]}),
    ('B07', '000000427160.jpg', '場景文字辨識 (OCR)', "What number is printed on the back of the jersey of the boy in the front? Answer with a number only.",
     '4 (the "182" in the background is on the fence)', {'type': 'vqa_number', 'target': 4}),
    ('B08', '000000210099.jpg', '屬性辨識（顏色）', 'What color is the cat? Answer in a few words.',
     'orange / ruddy brown (an Abyssinian-type cat on a wicker chair)',
     {'type': 'vqa_keywords', 'groups': [['orange', 'brown', 'ginger', 'tan', 'golden', 'fawn', 'red']]}),
    ('B09', '000000501523.jpg', '空間關係', 'Where is the cat sitting?',
     'inside a (bathroom) sink', {'type': 'vqa_keywords', 'groups': [['sink', 'basin']]}),
    ('B10', '000000022892.jpg', '多物件偵測', 'What two kinds of animals can be seen in this image?',
     'a cat (grey, inside) and a dog (outside the window)', {'type': 'vqa_keywords', 'groups': [['cat', 'kitten'], ['dog', 'puppy']]}),
    ('B11', '000000133418.jpg', '動作 / 運動辨識', 'What sport is the man playing? Answer with one word.',
     'tennis (on a clay court)', {'type': 'vqa_keywords', 'groups': [['tennis']]}),
    ('B12', '000000084170.jpg', '情境推理', 'What is the weather like in this scene? Briefly explain the visual evidence.',
     'rainy: wet road, water splashing, vehicles with headlights on', {'type': 'vqa_keywords', 'groups': [['rain', 'wet', 'drizzl', 'storm']]}),
    ('B13', '000000351331.jpg', '細粒度辨識', 'What type of pizza is this, and what utensil is next to it?',
     'pepperoni pizza; a fork', {'type': 'vqa_keywords', 'groups': [['pepperoni', 'salami'], ['fork']]}),
    ('B14', '000000420472.jpg', '幻覺檢測（不存在物件）', 'Is there a zebra in this image? Answer yes or no.',
     'no (two giraffes eating leaves, no zebra)', {'type': 'vqa_yesno', 'target': 'no'}),
    ('B15', '000000367228.jpg', '形容照片', 'Describe this image in one or two sentences.',
     'a person with raised arms flying a colourful kite on a grassy hill by the sea',
     {'type': 'vqa_keywords', 'groups': [['kite'], ['person', 'man', 'woman', 'boy', 'girl', 'someone', 'people', 'child'],
                                         ['sea', 'ocean', 'water', 'coast', 'beach', 'lake', 'shore']], 'min_hits': 2}),
]
PARTB_SYNTH = [
    ('B16', 'synth_ocr_sign.png', '文件 / 招牌 OCR', 'What is the name of the cafe and what is its phone number?',
     'CAFE LUNA, 03-571-2121', {'type': 'vqa_keywords', 'groups': [['luna'], ['571-2121', '5712121', '571 2121']]}),
    ('B17', 'synth_shapes.png', '物件計數（合成）', 'How many red circles are in the image? Answer with a number only.',
     '7 (plus 4 blue squares as distractors)', {'type': 'vqa_number', 'target': 7}),
    ('B18', 'synth_bar_chart.png', '圖表理解', 'According to the bar chart, which month has the highest sales, and what is the value?',
     'April, 25', {'type': 'vqa_keywords', 'groups': [['apr'], ['25']]}),
    ('B19', 'synth_spatial.png', '空間關係（合成）', 'Is the green triangle to the left or to the right of the yellow star? Answer with one word.',
     'left', {'type': 'vqa_label', 'target': 'left', 'others': ['right']}),
    ('B20', 'synth_math.png', '視覺數學', 'Solve the math problem shown in the image. Answer with a number only.',
     '42 (27 + 15)', {'type': 'vqa_number', 'target': 42}),
]


def _font(size):
    from PIL import ImageFont
    for f in ('arialbd.ttf', 'arial.ttf', 'DejaVuSans-Bold.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_synthetic(out_dir):
    from PIL import Image, ImageDraw
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # sign
    im = Image.new('RGB', (640, 420), (245, 235, 215)); d = ImageDraw.Draw(im)
    d.rectangle([20, 20, 620, 400], outline=(90, 60, 30), width=8)
    d.text((320, 110), 'CAFE LUNA', font=_font(72), fill=(90, 40, 20), anchor='mm')
    d.text((320, 220), 'OPEN 9:00 - 18:00', font=_font(40), fill=(40, 40, 40), anchor='mm')
    d.text((320, 300), 'TEL 03-571-2121', font=_font(40), fill=(40, 40, 40), anchor='mm')
    im.save(os.path.join(out_dir, 'synth_ocr_sign.png'))
    # shapes: 7 red circles, 4 blue squares, no overlap
    rng = random.Random(7); im = Image.new('RGB', (640, 480), 'white'); d = ImageDraw.Draw(im); boxes = []
    for kind, n in (('circle', 7), ('square', 4)):
        k = 0
        while k < n:
            x, y = rng.randint(20, 560), rng.randint(20, 400)
            b = (x, y, x + 60, y + 60)
            if any(not (b[2] + 15 < o[0] or o[2] + 15 < b[0] or b[3] + 15 < o[1] or o[3] + 15 < b[1]) for o in boxes):
                continue
            boxes.append(b); k += 1
            (d.ellipse if kind == 'circle' else d.rectangle)(b, fill=(220, 30, 30) if kind == 'circle' else (30, 70, 220))
    im.save(os.path.join(out_dir, 'synth_shapes.png'))
    # bar chart
    months, vals = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], [12, 18, 9, 25, 15, 20]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=100)
    ax.bar(months, vals, color='#4C72B0'); ax.set_title('Monthly Sales (units)'); ax.set_ylabel('Sales')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.4, str(v), ha='center')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'synth_bar_chart.png')); plt.close(fig)
    # spatial: triangle left, star right
    import math
    im = Image.new('RGB', (640, 400), (235, 240, 245)); d = ImageDraw.Draw(im)
    d.polygon([(130, 270), (230, 270), (180, 150)], fill=(40, 160, 60))
    cx, cy, R, r = 470, 210, 70, 28
    pts = [(cx + (R if i % 2 == 0 else r) * math.sin(math.pi * i / 5), cy - (R if i % 2 == 0 else r) * math.cos(math.pi * i / 5))
           for i in range(10)]
    d.polygon(pts, fill=(240, 200, 20))
    im.save(os.path.join(out_dir, 'synth_spatial.png'))
    # math
    im = Image.new('RGB', (640, 300), 'white'); d = ImageDraw.Draw(im)
    d.text((320, 150), '27 + 15 = ?', font=_font(96), fill='black', anchor='mm')
    im.save(os.path.join(out_dir, 'synth_math.png'))


def build_partB(coco_img_dir):
    img_dir = os.path.join(DATA_DIR, 'partB_images')
    os.makedirs(img_dir, exist_ok=True)
    for _, f, *_ in PARTB_COCO:
        shutil.copy(os.path.join(coco_img_dir, f), os.path.join(img_dir, f))
    make_synthetic(img_dir)
    items = [{'id': i, 'image': f, 'source': 'COCO val2017' if f.startswith('0') else 'synthetic', 'task': task,
              'question': q, 'reference': ref, 'check': chk} for i, f, task, q, ref, chk in PARTB_COCO + PARTB_SYNTH]
    save_json(items, os.path.join(DATA_DIR, 'partB_vlm_benchmark.json'))
    return items


# ----------------------------------------------------------------------------- Part C general benchmark
def build_general(mmlu_parquet, gsm8k_parquet, n_mmlu=60, n_gsm=40, seed=0):
    import pandas as pd
    m = pd.read_parquet(mmlu_parquet)
    m = m[m.question.str.len() + m.choices.map(lambda c: sum(len(x) for x in c)) < 1200]
    m = m.sample(n_mmlu, random_state=seed)
    g = pd.read_parquet(gsm8k_parquet).sample(n_gsm, random_state=seed)
    items = [{'id': f'MMLU{i + 1:02d}', 'source': 'MMLU', 'subject': r.subject, 'question': r.question,
              'choices': list(r.choices), 'answer': 'ABCD'[int(r.answer)]} for i, r in enumerate(m.itertuples())]
    items += [{'id': f'GSM{i + 1:02d}', 'source': 'GSM8K', 'question': r.question,
               'answer': r.answer.split('####')[-1].strip().replace(',', '')} for i, r in enumerate(g.itertuples())]
    save_json(items, os.path.join(DATA_DIR, 'partC_general_benchmark.json'))
    return items


# ----------------------------------------------------------------------------- Part C advanced COCO subset
def build_coco_subset(ann_dir, exclude_files, n_train=150, n_eval=80, seed=0):
    inst = json.load(open(os.path.join(ann_dir, 'instances_val2017.json')))
    caps = json.load(open(os.path.join(ann_dir, 'captions_val2017.json')))
    cats = {c['id']: c['name'] for c in inst['categories']}
    imgs = {i['id']: i for i in inst['images']}
    anns = collections.defaultdict(list)
    for a in inst['annotations']:
        anns[a['image_id']].append(a)
    captions = collections.defaultdict(list)
    for c in caps['annotations']:
        captions[c['image_id']].append(c['caption'].strip())
    # object co-occurrence over all val2017 images (for adversarial POPE negatives)
    present = {i: sorted({cats[a['category_id']] for a in anns[i]}) for i in imgs}
    co = collections.defaultdict(collections.Counter)
    for objs in present.values():
        for o in objs:
            for p in objs:
                if p != o:
                    co[o][p] += 1
    names = sorted(cats.values())

    def salient(i):
        I = imgs[i]; area = I['width'] * I['height']
        big = collections.Counter(cats[a['category_id']] for a in anns[i] if not a['iscrowd'] and a['area'] / area >= 0.02)
        return sorted(big)

    cand = [i for i in imgs if imgs[i]['file_name'] not in exclude_files and len(salient(i)) >= 3
            and not any(a['iscrowd'] for a in anns[i]) and len(present[i]) <= 8]
    rng = random.Random(seed)
    rng.shuffle(cand)
    out = {}
    for split, ids in (('train', cand[:n_train]), ('eval', cand[n_train:n_train + n_eval])):
        rows = []
        for i in ids:
            pres, sal = present[i], salient(i)
            adv = [o for o, _ in co[sal[0]].most_common() if o not in pres]
            for s in sal[1:]:
                adv += [o for o, _ in co[s].most_common(10) if o not in pres and o not in adv]
            rnd = [o for o in names if o not in pres]
            r2 = random.Random(i)
            pos = r2.sample(sal, 3)
            neg_adv = adv[:2]
            neg_rnd = r2.sample([o for o in rnd if o not in neg_adv], 1)
            rows.append({'image_id': i, 'file_name': imgs[i]['file_name'], 'objects': pres, 'salient': sal,
                         'captions': captions[i], 'pope_pos': pos, 'pope_neg_adversarial': neg_adv,
                         'pope_neg_random': neg_rnd, 'adversarial_pool': adv[:8]})
        out[split] = rows
    out['source'] = 'COCO val2017 (instances + captions); images downloaded at run time from images.cocodataset.org'
    save_json(out, os.path.join(DATA_DIR, 'partC_coco_subset.json'))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--coco_ann'); ap.add_argument('--coco_imgs'); ap.add_argument('--mmlu'); ap.add_argument('--gsm8k')
    a = ap.parse_args()
    items = build_partB(a.coco_imgs)
    build_general(a.mmlu, a.gsm8k)
    sub = build_coco_subset(a.coco_ann, {it['image'] for it in items})
    print(len(items), 'Part B items;', len(sub['train']), 'train /', len(sub['eval']), 'eval COCO images')
