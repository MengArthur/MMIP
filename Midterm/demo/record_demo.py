"""Record a silent demo video of the web app (backup for the live demo).

Drives the real page with Playwright (system Edge, headless), takes a screenshot at every step, then composes a
1080p H.264 MP4 with a drawn mouse cursor, click ripples and Chinese captions (no audio).

    python app/server.py --port 5001        # server must be running, examples pre-computed (green dots)
    python demo/record_demo.py              # -> demo/RouteVision_demo.mp4  (needs: playwright, imageio-ffmpeg, Pillow)
"""
import argparse
import io
import os

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEW_W, VIEW_H, DPR = 1600, 900, 1.2            # screenshots are 1920 x 1080
OUT_W, OUT_H, FPS = 1920, 1080, 30
FONT = 'C:/Windows/Fonts/msjhbd.ttc'
UPLOAD = os.path.join(ROOT, 'demo', 'upload_examples', 'vortex_D_no_answer.npz')


class Recorder:
    def __init__(self, page, out):
        self.page = page
        self.frame = None                     # current screenshot (PIL, 1920x1080)
        self.cursor = (VIEW_W * 0.55, VIEW_H * 0.55)   # CSS px
        self.caption = ''
        self.font = ImageFont.truetype(FONT, 34)
        self.small = ImageFont.truetype(FONT, 22)
        self.writer = imageio_ffmpeg.write_frames(out, (OUT_W, OUT_H), fps=FPS, codec='libx264', macro_block_size=8,
                                                  pix_fmt_out='yuv420p', output_params=['-crf', '20', '-preset', 'medium'])
        self.writer.send(None)
        self.n = 0

    # ---------------------------------------------------------------- frames
    def shot(self):
        self.frame = Image.open(io.BytesIO(self.page.screenshot(type='png'))).convert('RGB').resize((OUT_W, OUT_H))

    def _compose(self, ripple=None):
        im = self.frame.copy()
        d = ImageDraw.Draw(im, 'RGBA')
        if self.caption:
            w = d.textlength(self.caption, font=self.font)
            x0, y0 = (OUT_W - w) / 2 - 36, OUT_H - 118
            d.rounded_rectangle((x0, y0, x0 + w + 72, y0 + 74), radius=18, fill=(19, 32, 30, 228))
            d.text((x0 + 36, y0 + 15), self.caption, font=self.font, fill=(244, 248, 246))
        cx, cy = self.cursor[0] * DPR, self.cursor[1] * DPR
        if ripple is not None:                                   # click feedback: expanding ring
            r = 14 + 34 * ripple
            d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(15, 118, 110, int(230 * (1 - ripple))), width=5)
        s = 1.25                                                 # arrow cursor, white with dark outline
        pts = [(0, 0), (0, 25), (6, 19), (10.5, 28.5), (14.5, 26.5), (10, 17.5), (18, 17.5)]
        poly = [(cx + x * s, cy + y * s) for x, y in pts]
        d.polygon(poly, fill=(255, 255, 255), outline=(20, 20, 20))
        d.line(poly + [poly[0]], fill=(20, 20, 20), width=2)
        return im

    def hold(self, seconds, ripple=False):
        k = max(1, int(round(seconds * FPS)))
        for i in range(k):
            rp = i / (0.35 * FPS) if ripple and i < 0.35 * FPS else None
            self.writer.send(np.asarray(self._compose(rp)).tobytes())
            self.n += 1

    # ---------------------------------------------------------------- actions
    def say(self, text):
        self.caption = text

    def move(self, x, y, seconds=0.7):
        """Glide the drawn cursor to (x, y) over the current screenshot, then move the real mouse there."""
        x0, y0 = self.cursor
        k = max(1, int(seconds * FPS))
        for i in range(1, k + 1):
            t = i / k
            e = t * t * (3 - 2 * t)
            self.cursor = (x0 + (x - x0) * e, y0 + (y - y0) * e)
            self.writer.send(np.asarray(self._compose()).tobytes())
            self.n += 1
        self.page.mouse.move(x, y)
        self.page.wait_for_timeout(150)
        self.shot()

    def center(self, selector):
        b = self.page.locator(selector).first.bounding_box()
        return b['x'] + b['width'] / 2, b['y'] + b['height'] / 2

    def click(self, selector, wait=600, hold=1.0):
        x, y = self.center(selector)
        self.move(x, y)
        self.page.mouse.click(x, y)
        self.page.wait_for_timeout(wait)
        self.shot()
        self.hold(hold, ripple=True)

    def key(self, key, wait=500):
        self.page.evaluate('document.activeElement && document.activeElement.blur()')
        self.page.keyboard.press(key)
        self.page.wait_for_timeout(wait)
        self.shot()

    def cell(self, canvas, i, j):
        b = self.page.locator(f'#{canvas}').bounding_box()
        return b['x'] + (j + 0.5) * b['width'] / 8, b['y'] + (i + 0.5) * b['height'] / 8

    def hover_cell(self, canvas, i, j, hold=2.2):
        self.move(*self.cell(canvas, i, j))
        self.hold(hold)

    def wait_ready(self):
        self.page.wait_for_function("document.getElementById('busy').classList.contains('hidden') && "
                                    "!document.getElementById('analysis').classList.contains('hidden')", timeout=120000)
        self.page.wait_for_timeout(500)

    def close(self):
        self.writer.close()


def end_card(rec):
    im = Image.new('RGB', (OUT_W, OUT_H), (19, 32, 30))
    d = ImageDraw.Draw(im)
    big, mid = ImageFont.truetype(FONT, 96), ImageFont.truetype(FONT, 40)
    for txt, f, y, col in (('RouteVision', big, 410, (244, 248, 246)),
                           ('placement 一完成，就預測 routing 後哪裡會塞', mid, 550, (215, 228, 224)),
                           ('林孟霆 · 315833033 · MMIP 期中專題', mid, 630, (169, 191, 186))):
        w = d.textlength(txt, font=f)
        d.text(((OUT_W - w) / 2, y), txt, font=f, fill=col)
    rec.frame, rec.caption, rec.cursor = im, '', (-100, -100)
    rec.hold(3)


def run(url, out):
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': VIEW_W, 'height': VIEW_H}, device_scale_factor=DPR)
        page.goto(url)
        page.wait_for_function("typeof S !== 'undefined' && S.info && document.querySelectorAll('.g-item').length > 0", timeout=60000)
        page.wait_for_function("[...document.querySelectorAll('.g-ready')].every(d => !d.classList.contains('pending'))", timeout=600000)
        rec = Recorder(page, out)

        # ---- intro
        rec.shot()
        rec.say('RouteVision：placement 一完成，就預測 routing 後哪裡會塞')
        rec.hold(3.5)
        rec.say('左側「沒看過的晶片」：Vortex-small 完全沒參與訓練')
        rec.move(*rec.center('.g-item[data-k="0"]'))
        rec.hold(1.8)

        # ---- ① hotspots
        rec.say('① 熱點預測：點一張沒看過的晶片')
        rec.click('.g-item[data-k="0"]', wait=300, hold=0.2)
        rec.wait_ready()
        rec.shot()
        rec.hold(2.5)
        rec.say('左：EDA 工具 eGR　中：AI（U-Net 多模態）　右：routing 後的真實熱點')
        rec.move(*rec.center('#verdict'))
        rec.hold(3.5)
        rec.say('紅色＝抓到　斜線＝誤報　藍框＝漏掉：eGR 誤報 8 格，U-Net 只誤報 1 格')
        rec.move(*rec.center('#p0'))
        rec.hold(3.5)
        rec.say('滑鼠移到區塊：三張圖同步標示，顯示真實使用率和各方法的分數')
        rec.hover_cell('cv1', 1, 2)
        rec.hover_cell('cv1', 2, 5)
        rec.hover_cell('cv0', 0, 1)
        rec.say('右上可切換 AI 方法，例如只用影像的 CNN')
        rec.move(*rec.center('#methodSel'))
        page.select_option('#methodSel', 'CNN')
        page.wait_for_timeout(400)
        rec.shot()
        rec.hold(3.0, ripple=True)
        rec.say('切回 U-Net 多模態（三個設計平均最好的方法）')
        page.select_option('#methodSel', 'U-Net (multimodal)')
        page.wait_for_timeout(400)
        rec.shot()
        rec.hold(1.5)

        # ---- ② utilisation maps
        rec.say('② 使用率圖（亮點一）：按 → 換下一步')
        rec.click('.step[data-step="map"]', hold=1.5)
        rec.say('U-Net 直接預測整張使用率圖：相關係數 0.738，eGR 只有 0.604')
        rec.move(*rec.center('#p1'))
        rec.hold(4.0)
        rec.say('虛線框是真實熱點：U-Net 畫出了低使用率方塊和水平繞線條紋')
        rec.hover_cell('cv1', 1, 2, hold=3.0)

        # ---- ③ trust
        rec.say('③ 可信度（亮點二）')
        rec.click('.step[data-step="trust"]', hold=1.5)
        rec.say('紫色框＝RF、Multimodal、ViT 三個模型意見分歧，也就是模型沒把握的區塊')
        rec.move(*rec.center('#p1'))
        rec.hold(3.5)
        rec.say('Ensemble 錯 6 格，其中 4 格落在它標出的 7 個「沒把握」區塊裡')
        rec.hover_cell('cv1', 1, 0, hold=3.5)

        # ---- ④ Grad-CAM
        rec.say('④ 判斷依據：Grad-CAM 顯示模型看哪裡判斷熱點')
        rec.click('.step[data-step="why"]', wait=1500, hold=1.0)
        page.wait_for_function("document.getElementById('pl1').classList.contains('hidden')", timeout=60000)
        rec.shot()
        rec.say('暖色＝最依賴的區域；最熱的地方落在真實熱點（虛線框）上')
        rec.move(*rec.center('#p1'))
        rec.hold(3.5)
        rec.say('也可以切換成 CNN 比較')
        rec.click('#stepCtrl [data-cam="CNN"]', wait=2500, hold=2.5)
        rec.click('#stepCtrl [data-cam="U-Net (multimodal)"]', wait=800, hold=1.0)

        # ---- ⑤ upload
        rec.say('⑤ 上傳新版圖：模擬還沒 routing 的新設計')
        rec.click('#uploadOpen', hold=1.5)
        rec.say('選 demo/upload_examples/vortex_D_no_answer.npz（只有特徵圖和設計參數，沒有答案）')
        page.set_input_files('#fileInput', UPLOAD)
        page.wait_for_timeout(300)
        rec.move(*rec.center('#drop'))
        rec.hold(3.0)
        rec.click('#uploadBtn', wait=300, hold=0.2)
        page.wait_for_function("S.result && S.result.source === 'upload'", timeout=120000)
        rec.wait_ready()
        rec.say('沒有 routing 結果，也能直接預測熱點')
        rec.key('ArrowLeft')
        rec.key('ArrowLeft')
        rec.key('ArrowLeft')
        rec.hold(3.0)
        rec.say('也能預測整張使用率圖')
        rec.key('ArrowRight')
        rec.hold(2.5)
        rec.say('並標出模型沒把握、建議人工確認的區塊')
        rec.key('ArrowRight')
        rec.hold(3.0)

        # ---- overall results
        rec.say('整體成效：三個設計輪流當沒看過的晶片')
        rec.click('.view-tab[data-view="results"]', hold=1.0)
        rec.say('U-Net 多模態 ROC-AUC 0.984 ± 0.013，三個設計都勝過 EDA 工具的粗估')
        rec.move(*rec.center('#lodoChart'))
        rec.hold(4.0)

        end_card(rec)
        n = rec.n
        rec.close()
        browser.close()
    print(f'{n} frames, {n / FPS:.0f} s -> {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:5001/')
    ap.add_argument('--out', default=os.path.join(ROOT, 'demo', 'RouteVision_demo.mp4'))
    a = ap.parse_args()
    run(a.url, a.out)
