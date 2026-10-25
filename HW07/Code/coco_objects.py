"""
CHAIR-style object mention extraction (Rohrbach et al., 2018): map words in a generated description to
COCO categories, so that hallucinated objects (mentioned but not annotated in the image) can be counted.

The synonym lists are a conservative subset of the original CHAIR lists: ambiguous words are left out
(e.g. "glass", "bag", "plant", "screen", "baseball" as a sport) so that a mention is only counted when
it clearly names a COCO object. "orange" is only counted as a fruit ("oranges" / "an orange").
"""

import re

SYNONYMS = {
    'person': ['person', 'people', 'man', 'men', 'woman', 'women', 'boy', 'girl', 'child', 'children', 'kid', 'baby',
               'player', 'guy', 'lady', 'skier', 'surfer', 'skateboarder', 'snowboarder', 'rider', 'pedestrian', 'adult',
               'teenager', 'toddler', 'crowd', 'spectator', 'worker', 'chef', 'cyclist', 'gentleman', 'passenger'],
    'bicycle': ['bicycle', 'bike', 'cycle'], 'car': ['car', 'sedan', 'taxi', 'suv', 'minivan', 'jeep'],
    'motorcycle': ['motorcycle', 'motorbike', 'motor bike', 'scooter', 'moped'],
    'airplane': ['airplane', 'aeroplane', 'plane', 'jet', 'aircraft', 'airliner'], 'bus': ['bus'],
    'train': ['train', 'locomotive', 'tram'], 'truck': ['truck', 'pickup truck', 'lorry'],
    'boat': ['boat', 'ship', 'sailboat', 'kayak', 'canoe', 'yacht', 'ferry'],
    'traffic light': ['traffic light', 'traffic signal', 'stoplight', 'stop light'], 'fire hydrant': ['fire hydrant', 'hydrant'],
    'stop sign': ['stop sign'], 'parking meter': ['parking meter'], 'bench': ['bench'],
    'bird': ['bird', 'pigeon', 'seagull', 'gull', 'duck', 'goose', 'parrot', 'sparrow', 'swan', 'crow', 'eagle', 'owl'],
    'cat': ['cat', 'kitten', 'kitty'], 'dog': ['dog', 'puppy', 'pup'], 'horse': ['horse', 'pony', 'foal'],
    'sheep': ['sheep', 'lamb'], 'cow': ['cow', 'cattle', 'bull', 'calf', 'ox'], 'elephant': ['elephant'],
    'bear': ['bear', 'grizzly'], 'zebra': ['zebra'], 'giraffe': ['giraffe'], 'backpack': ['backpack'],
    'umbrella': ['umbrella', 'parasol'], 'handbag': ['handbag', 'purse'], 'tie': ['necktie', 'tie'],
    'suitcase': ['suitcase', 'luggage'], 'frisbee': ['frisbee'], 'skis': ['ski', 'skis'], 'snowboard': ['snowboard'],
    'sports ball': ['ball', 'soccer ball', 'tennis ball', 'football'], 'kite': ['kite'],
    'baseball bat': ['baseball bat', 'bat'], 'baseball glove': ['baseball glove', 'glove', 'mitt'],
    'skateboard': ['skateboard'], 'surfboard': ['surfboard', 'surf board'], 'tennis racket': ['tennis racket', 'racket', 'racquet'],
    'bottle': ['bottle'], 'wine glass': ['wine glass', 'wineglass'], 'cup': ['cup', 'mug'], 'fork': ['fork'],
    'knife': ['knife'], 'spoon': ['spoon'], 'bowl': ['bowl'], 'banana': ['banana'], 'apple': ['apple'],
    'sandwich': ['sandwich', 'burger', 'hamburger'], 'orange': [], 'broccoli': ['broccoli'], 'carrot': ['carrot'],
    'hot dog': ['hot dog', 'hotdog'], 'pizza': ['pizza'], 'donut': ['donut', 'doughnut'], 'cake': ['cake', 'cupcake'],
    'chair': ['chair', 'stool', 'armchair'], 'couch': ['couch', 'sofa'], 'potted plant': ['potted plant', 'houseplant', 'flowerpot', 'flower pot'],
    'bed': ['bed'], 'dining table': ['dining table', 'table'], 'toilet': ['toilet'], 'tv': ['tv', 'television', 'monitor'],
    'laptop': ['laptop', 'notebook computer'], 'mouse': ['mouse'], 'remote': ['remote', 'remote control', 'controller'],
    'keyboard': ['keyboard'], 'cell phone': ['cell phone', 'cellphone', 'phone', 'smartphone', 'mobile phone'],
    'microwave': ['microwave'], 'oven': ['oven', 'stove'], 'toaster': ['toaster'], 'sink': ['sink'],
    'refrigerator': ['refrigerator', 'fridge'], 'book': ['book'], 'clock': ['clock'], 'vase': ['vase'],
    'scissors': ['scissors'], 'teddy bear': ['teddy bear', 'stuffed bear', 'teddy'], 'hair drier': ['hair drier', 'hair dryer', 'hairdryer', 'blow dryer'],
    'toothbrush': ['toothbrush'],
}
_IRREGULAR = {'people': 'person', 'men': 'man', 'women': 'woman', 'children': 'child', 'mice': 'mouse', 'knives': 'knife',
              'geese': 'goose', 'oxen': 'ox', 'calves': 'calf', 'skis': 'ski', 'scissors': 'scissors', 'sheep': 'sheep'}

_WORD2CAT = {}
for _cat, _words in SYNONYMS.items():
    for _w in _words:
        _WORD2CAT[_w] = _cat
_MULTI = sorted([w for w in _WORD2CAT if ' ' in w], key=len, reverse=True)


def _singular(w):
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    for suf, rep in (('ies', 'y'), ('ves', 'f'), ('ches', 'ch'), ('shes', 'sh'), ('sses', 'ss'), ('xes', 'x'), ('es', 'e'), ('es', ''), ('s', '')):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            cand = w[: -len(suf)] + rep
            if cand in _WORD2CAT:
                return cand
    return w


def mentioned_objects(text):
    """Set of COCO categories mentioned in `text`."""
    t = ' ' + re.sub(r'[^a-z\s]', ' ', text.lower()) + ' '
    t = re.sub(r'\s+', ' ', t)
    found = set()
    if re.search(r'\boranges\b|\ban orange\b(?! (shirt|jacket|wall|color|colour|sky|light|hue|top|vest|dress|and))', t):
        found.add('orange')
    for m in _MULTI:                      # multi-word names first; remove them so "teddy bear" is not also "bear"
        pat = r'\b' + re.escape(m) + r's?\b'
        if re.search(pat, t):
            found.add(_WORD2CAT[m])
            t = re.sub(pat, ' ', t)
    for w in t.split():
        s = _singular(w)
        if s in _WORD2CAT:
            found.add(_WORD2CAT[s])
    return found


def chair(texts, gt_objects, salient_objects=None):
    """CHAIR_s (share of descriptions with >= 1 hallucinated object), CHAIR_i (hallucinated / all mentioned
    objects) and object recall (share of salient annotated objects that are mentioned)."""
    n_h, n_m, n_cap_h, rec = 0, 0, 0, []
    per = []
    for i, (txt, gt) in enumerate(zip(texts, gt_objects)):
        men = mentioned_objects(txt)
        hall = sorted(men - set(gt))
        n_m += len(men); n_h += len(hall); n_cap_h += bool(hall)
        if salient_objects is not None:
            sal = set(salient_objects[i])
            rec.append(len(men & sal) / max(1, len(sal)))
        per.append({'mentioned': sorted(men), 'hallucinated': hall})
    n = max(1, len(texts))
    return {'CHAIR_s': round(n_cap_h / n, 4), 'CHAIR_i': round(n_h / max(1, n_m), 4),
            'recall': round(sum(rec) / max(1, len(rec)), 4) if rec else None,
            'objects_per_caption': round(n_m / n, 2)}, per
