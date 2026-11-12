"""
Synthetic images for the customer-service dialogues (drawn once by prepare_data.py and committed):
  receipt      order summary with the order id (the router must read SDxxxxxxxx from the image)
  error        error dialog of an app / OS with an error code
  damage       phone / laptop / tablet with a cracked screen
  injection    a note whose text gives instructions to the AI (typographic prompt injection)
Styles 0-2 are used for training data, style 3 (different fonts, colours and layouts) only for the
benchmark and the 作業三 test sets, so the evaluation measures generalisation to unseen image layouts.
"""

import math
import os
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont

_FONT_DIRS = ['C:/Windows/Fonts', '/usr/share/fonts/opentype/noto', '/usr/share/fonts/truetype/noto',
              '/usr/share/fonts/noto-cjk', '/System/Library/Fonts']
_FONTS = {'regular': ['msjh.ttc', 'NotoSansCJK-Regular.ttc', 'NotoSansTC-VF.ttf', 'PingFang.ttc'],
          'bold': ['msjhbd.ttc', 'NotoSansCJK-Bold.ttc', 'NotoSansTC-VF.ttf', 'PingFang.ttc'],
          'serif': ['mingliu.ttc', 'NotoSerifCJK-Regular.ttc', 'NotoSerifTC-VF.ttf'],
          'kai': ['kaiu.ttf', 'NotoSerifCJK-Regular.ttc', 'NotoSerifTC-VF.ttf']}


def font(kind, size):
    for name in _FONTS[kind]:
        for d in _FONT_DIRS:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    raise FileNotFoundError(f'no CJK font for {kind}; install fonts-noto-cjk')


STYLES = [
    dict(paper=(255, 255, 255), ink=(30, 30, 30), accent=(220, 70, 40), font='regular', title='bold', bg=(200, 200, 205)),
    dict(paper=(250, 247, 238), ink=(40, 40, 60), accent=(30, 110, 200), font='regular', title='bold', bg=(120, 90, 70)),
    dict(paper=(245, 250, 255), ink=(20, 40, 40), accent=(20, 150, 100), font='regular', title='bold', bg=(60, 70, 80)),
    dict(paper=(255, 252, 230), ink=(60, 30, 20), accent=(150, 40, 150), font='serif', title='kai', bg=(170, 190, 160)),
]


def _finish(img, rng, bg=None):
    """Photograph-like: small rotation on the background colour (if bg is given) and a slight blur."""
    if bg is not None:
        img = img.rotate(rng.uniform(-6, 6), expand=True, fillcolor=bg)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 0.6)))
    return img.convert('RGB')


def receipt(order_id, items, date, style=0, seed=0):
    rng = random.Random(seed)
    s = STYLES[style]
    W, H = 440, 300 + 34 * len(items)
    img = Image.new('RGB', (W, H), s['paper'])
    d = ImageDraw.Draw(img)
    ft, fr, fb = font(s['title'], 26), font(s['font'], 19), font(s['title'], 21)
    if style == 3:     # benchmark layout: e-invoice style, order id at the bottom
        d.rectangle([0, 0, W, 54], fill=s['accent'])
        d.text((20, 12), '電子發票證明聯　交易明細', font=ft, fill=(255, 255, 255))
        y = 72
        d.text((20, y), f'交易時間　{date} 14:{rng.randint(10, 59)}', font=fr, fill=s['ink']); y += 34
        d.text((20, y), '賣方　速達3C股份有限公司', font=fr, fill=s['ink']); y += 40
        for name, price in items:
            d.text((20, y), name, font=fr, fill=s['ink']); d.text((W - 130, y), f'{price:,}', font=fr, fill=s['ink']); y += 34
        d.line([20, y + 4, W - 20, y + 4], fill=s['ink'], width=1); y += 14
        d.text((20, y), f'總計　NT$ {sum(p for _, p in items):,}', font=fb, fill=s['ink']); y += 44
        d.text((20, y), f'訂單號碼：{order_id}', font=fb, fill=s['accent'])
    else:
        d.text((20, 18), '速達 3C 線上商城', font=ft, fill=s['accent'])
        d.text((20, 56), '訂單明細', font=fb, fill=s['ink'])
        y = 96
        d.text((20, y), f'訂單編號：{order_id}', font=fb, fill=s['ink']); y += 36
        d.text((20, y), f'訂購日期：{date}', font=fr, fill=s['ink']); y += 40
        d.line([20, y, W - 20, y], fill=s['ink'], width=2); y += 12
        for name, price in items:
            d.text((20, y), name, font=fr, fill=s['ink']); d.text((W - 140, y), f'NT$ {price:,}', font=fr, fill=s['ink']); y += 34
        d.line([20, y + 2, W - 20, y + 2], fill=s['ink'], width=2); y += 14
        d.text((20, y), f'合計　NT$ {sum(p for _, p in items):,}', font=fb, fill=s['accent'])
    canvas = Image.new('RGB', (W + 60, H + 60), s['bg'])
    canvas.paste(img, (30, 30))
    return _finish(canvas, rng, s['bg'])


def error_screen(code, message, app, style=0, seed=0):
    rng = random.Random(seed)
    s = STYLES[style]
    dark = style in (1, 3)
    W, H = 480, 300
    img = Image.new('RGB', (W, H), (32, 34, 40) if dark else (236, 238, 242))
    d = ImageDraw.Draw(img)
    fg = (235, 235, 235) if dark else (30, 30, 30)
    d.rectangle([0, 0, W, 40], fill=s['accent'])
    d.text((14, 8), app, font=font(s['title'], 20), fill=(255, 255, 255))
    cx, cy = 60, 110
    d.ellipse([cx - 28, cy - 28, cx + 28, cy + 28], fill=(210, 50, 50))
    d.text((cx - 6, cy - 20), '!', font=font('bold', 30), fill=(255, 255, 255))
    d.text((110, 80), '發生錯誤' if style != 3 else '無法完成操作', font=font(s['title'], 24), fill=fg)
    d.text((110, 122), message, font=font(s['font'], 18), fill=fg)
    d.text((110, 160), f'錯誤代碼：{code}', font=font(s['font'], 18), fill=fg)
    d.rounded_rectangle([W - 150, H - 70, W - 30, H - 26], 8, fill=s['accent'])
    d.text((W - 118, H - 62), '確定', font=font(s['title'], 20), fill=(255, 255, 255))
    return _finish(img, rng)


def damage(kind, style=0, seed=0):
    """A device with a cracked screen on a table (drawn, not photographed)."""
    rng = random.Random(seed)
    s = STYLES[style]
    W = H = 448
    table = tuple(max(0, min(255, c + rng.randint(-20, 20))) for c in s['bg'])
    img = Image.new('RGB', (W, H), table)
    d = ImageDraw.Draw(img)
    if kind == 'phone':
        box, scr = [144, 40, 304, 408], [154, 64, 294, 384]
        d.rounded_rectangle(box, 22, fill=(40, 40, 45))
    elif kind == 'tablet':
        box, scr = [64, 84, 384, 364], [82, 100, 366, 348]
        d.rounded_rectangle(box, 18, fill=(50, 50, 55))
    else:   # laptop
        d.polygon([(70, 330), (378, 330), (420, 400), (28, 400)], fill=(170, 170, 175))
        box, scr = [80, 70, 368, 330], [94, 84, 354, 316]
        d.rounded_rectangle(box, 10, fill=(60, 60, 65))
    d.rectangle(scr, fill=(18, 22, 30))
    ox = rng.randint(scr[0] + 30, scr[2] - 30)
    oy = rng.randint(scr[1] + 40, scr[3] - 40)
    for _ in range(rng.randint(9, 14)):   # radial cracks
        ang = rng.uniform(0, 2 * math.pi)
        x, y = ox, oy
        for _ in range(rng.randint(3, 7)):
            ang += rng.uniform(-0.5, 0.5)
            step = rng.uniform(15, 45)
            nx, ny = x + step * math.cos(ang), y + step * math.sin(ang)
            nx, ny = min(max(nx, scr[0]), scr[2]), min(max(ny, scr[1]), scr[3])
            d.line([x, y, nx, ny], fill=(225, 230, 235), width=rng.choice([1, 2]))
            x, y = nx, ny
    d.ellipse([ox - 10, oy - 10, ox + 10, oy + 10], outline=(230, 230, 235), width=2)
    return _finish(img, rng)


def injection(lines, style=0, seed=0):
    """A sticky note / printed page whose text addresses the AI."""
    rng = random.Random(seed)
    s = STYLES[style]
    W = H = 448
    img = Image.new('RGB', (W, H), s['bg'])
    d = ImageDraw.Draw(img)
    note = (255, 240, 140) if style in (0, 3) else s['paper']
    d.rectangle([34, 34, W - 34, H - 34], fill=note)
    f = font(s['title'], 22)
    y = 60
    for ln in lines:
        d.text((56, y), ln, font=f, fill=s['ink']); y += 40
    return _finish(img, rng, s['bg'])
