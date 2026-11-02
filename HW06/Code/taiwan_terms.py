"""
Part C optimisation aspect: answer Traditional-Chinese questions in *Taiwan* Traditional Chinese.

Qwen models are trained mostly on Simplified-Chinese data; when asked in Traditional Chinese they
often (1) leak Simplified characters and (2) use mainland-China vocabulary written in Traditional
characters (軟件 instead of 軟體, 視頻 instead of 影片, 默認 instead of 預設 ...).

Metric (fully automatic, no LLM judge):
  n_simplified : characters that exist only in Simplified Chinese (OpenCC STCharacters entries that
                 do not map to themselves; e.g. 们 这 软 -- but not 台 / 吃 / 里, which are also Traditional)
  terms        : occurrences of a curated list of mainland terms (after converting Simplified -> Traditional)
  clean        : no Simplified character, no mainland term, and a real Chinese answer (>= 20 CJK characters),
                 so an empty or English answer can never count as an improvement
DPO pairs: chosen = the response with Simplified characters converted and mainland terms replaced;
rejected = the model's own original response (on-policy) when it differs, otherwise the response with
Taiwan terms swapped back to mainland terms, or as a last resort converted to Simplified Chinese.
"""

import os
import re

import opencc

# mainland term (in Traditional characters) -> Taiwan term. Only terms that are clearly non-Taiwanese
# usage are listed; ambiguous ones (支持, 窗口, 菜單, 運行, 項目, 文件, 數據, 優化 ...) are left out.
# Known limitation: matching is by substring without word segmentation, so short entries can fire inside
# other words (的士 in 士林 after 的, 存儲 in 暫存儲器, 打的 in 他打的字). The submitted results were produced
# with this table (2 of 130 chosen answers were altered wrongly, see README); a future run should drop
# such short entries or use a word segmenter.
MAINLAND_TO_TW = {
    '軟件': '軟體', '硬件': '硬體', '網絡': '網路', '互聯網': '網際網路', '移動網絡': '行動網路',
    '視頻通話': '視訊通話', '視頻會議': '視訊會議', '視頻': '影片', '音頻': '音訊', '信息': '資訊', '短信': '簡訊',
    '默認': '預設', '內存': '記憶體', '硬盤': '硬碟', 'U盤': '隨身碟', '優盤': '隨身碟', '鼠標': '滑鼠',
    '打印機': '印表機', '打印': '列印', '筆記本電腦': '筆記型電腦', '屏幕': '螢幕', '博客': '部落格', '博主': '部落客',
    '出租車': '計程車', '的士': '計程車', '打車': '叫車', '公交車': '公車', '公交': '公車', '地鐵': '捷運',
    '摩托車': '機車', '激光': '雷射', '數據庫': '資料庫', '服務器': '伺服器', '芯片': '晶片',
    '智能手機': '智慧型手機', '人工智能': '人工智慧', '智能': '智慧', '攝像頭': '鏡頭', '攝像機': '攝影機',
    '光盤': '光碟', '充電寶': '行動電源', '空調': '冷氣', '便利店': '便利商店', '算法': '演算法', '編程': '程式設計',
    '源代碼': '原始碼', '代碼': '程式碼', '程序員': '程式設計師', '應用程序': '應用程式', '小程序': '小程式',
    '計算機程序': '電腦程式', '變量': '變數', '緩存': '快取', '線程': '執行緒', '調試': '除錯', '字節': '位元組',
    '操作系統': '作業系統', '文件夾': '資料夾', '帖子': '貼文', '刷新': '重新整理', '點贊': '按讚', '網民': '網友',
    '性價比': 'CP值', '酸奶': '優格', '西紅柿': '番茄', '土豆': '馬鈴薯', '菠蘿': '鳳梨', '獼猴桃': '奇異果',
    '方便麵': '泡麵', '三文魚': '鮭魚', '盒飯': '便當', '質量': '品質', '渠道': '管道', '營銷': '行銷',
    '外賣': '外送', '高清': '高畫質', '鏈接': '連結', '二維碼': 'QR碼', '登錄': '登入', '用戶名': '使用者名稱',
    '搜索': '搜尋', '設置': '設定', '激活': '啟用', '卸載': '解除安裝',
    '在線': '線上', '實時': '即時', '信號': '訊號', '移動支付': '行動支付', '移動電話': '行動電話',
    '雲計算': '雲端運算', '數字化': '數位化', '數碼': '數位', '晶體管': '電晶體', '集成電路': '積體電路',
    '分辨率': '解析度', '傳感器': '感測器', '存儲': '儲存', '項目經理': '專案經理', '打的': '搭計程車',
}
# Taiwan term -> mainland term, used only to *create* a rejected answer when the model's own answer
# is already clean (one unambiguous mainland variant per Taiwan term)
TW_TO_MAINLAND = {'軟體': '軟件', '硬體': '硬件', '網路': '網絡', '影片': '視頻', '資訊': '信息', '簡訊': '短信',
                  '預設': '默認', '記憶體': '內存', '硬碟': '硬盤', '滑鼠': '鼠標', '印表機': '打印機', '列印': '打印',
                  '螢幕': '屏幕', '部落格': '博客', '計程車': '出租車', '公車': '公交車', '捷運': '地鐵', '雷射': '激光',
                  '資料庫': '數據庫', '伺服器': '服務器', '晶片': '芯片', '人工智慧': '人工智能', '演算法': '算法',
                  '程式碼': '代碼', '應用程式': '應用程序', '程式設計': '編程', '變數': '變量', '快取': '緩存',
                  '作業系統': '操作系統', '資料夾': '文件夾', '品質': '質量', '行銷': '營銷', '連結': '鏈接',
                  '登入': '登錄', '搜尋': '搜索', '設定': '設置', '線上': '在線', '即時': '實時', '訊號': '信號',
                  '數位': '數碼', '解析度': '分辨率', '儲存': '存儲', '便利商店': '便利店', '冷氣': '空調',
                  '鳳梨': '菠蘿', '馬鈴薯': '土豆', '番茄': '西紅柿', '優格': '酸奶', '便當': '盒飯', '外送': '外賣'}

MIN_CJK = 20      # an answer must contain at least 20 Chinese characters to count as a (clean) Chinese answer
_s2tw = opencc.OpenCC('s2tw')
_t2s = opencc.OpenCC('t2s')


def _load_simplified_only():
    """Simplified characters whose Traditional form differs, excluding characters that are the Taiwan
    standard form of a Traditional variant (e.g. 群: OpenCC Traditional is 羣, but Taiwan writes 群)."""
    def read(name):
        with open(os.path.join(os.path.dirname(opencc.__file__), 'dictionary', name), encoding='utf-8') as f:
            return dict(ln.rstrip('\n').split('\t') for ln in f if '\t' in ln)
    st, tw = read('STCharacters.txt'), read('TWVariants.txt')
    out = set()
    for k, v in st.items():
        cands = v.split(' ')
        if len(k) == 1 and k not in cands and not any(k in tw.get(c, '').split(' ') for c in cands):
            out.add(k)
    return out


SIMPLIFIED_ONLY = _load_simplified_only()


def _alternation(keys):
    return re.compile('|'.join(re.escape(k) for k in sorted(keys, key=len, reverse=True)))


# Taiwan terms are matched too (and kept), so 「算法」 inside 「演算法」 is not counted / replaced.
_FIX_RE = _alternation(set(MAINLAND_TO_TW) | set(MAINLAND_TO_TW.values()))
_INJECT_RE = _alternation(TW_TO_MAINLAND)


def to_traditional(text):
    """Convert only the Simplified-only characters (phrase-aware OpenCC s2tw), leave every other
    character untouched (s2tw alone would also turn 台 into 臺)."""
    if not any(c in SIMPLIFIED_ONLY for c in text):
        return text
    conv = _s2tw.convert(text)
    if len(conv) != len(text):
        return conv
    return ''.join(n if o in SIMPLIFIED_ONLY else o for o, n in zip(text, conv))


def mainland_terms(text):
    return [m.group(0) for m in _FIX_RE.finditer(to_traditional(text)) if m.group(0) in MAINLAND_TO_TW]


# Japanese kana, Hangul and accented Latin letters: should never appear in a Traditional-Chinese answer
# (reported separately; a sign of a degenerated model)
_OTHER_SCRIPT = re.compile('[\u3040-\u30ff\uac00-\ud7a3\u00c0-\u024f]')


def analyse(text):
    n_simp = sum(c in SIMPLIFIED_ONLY for c in text)
    terms = mainland_terms(text)
    n_cjk = sum('一' <= c <= '鿿' for c in text)
    n_other = len(_OTHER_SCRIPT.findall(text))
    return {'n_simplified': n_simp, 'n_terms': len(terms), 'terms': terms, 'n_cjk': n_cjk, 'n_other_script': n_other,
            'chinese': int(n_cjk >= MIN_CJK), 'clean': int(n_simp == 0 and not terms and n_cjk >= MIN_CJK)}


def fix(text):
    t = to_traditional(text)
    return _FIX_RE.sub(lambda m: MAINLAND_TO_TW.get(m.group(0), m.group(0)), t)


def inject(text):
    return _INJECT_RE.sub(lambda m: TW_TO_MAINLAND[m.group(0)], text)


def build_pair(prompt, response):
    chosen = fix(response)
    if chosen != response:
        return {'prompt': prompt, 'chosen': chosen, 'rejected': response, 'kind': 'on_policy'}
    rej = inject(response)
    if rej != response:
        return {'prompt': prompt, 'chosen': response, 'rejected': rej, 'kind': 'injected_terms'}
    rej = _t2s.convert(response)
    if rej != response:
        return {'prompt': prompt, 'chosen': response, 'rejected': rej, 'kind': 'simplified'}
    return None


def summarize(texts):
    rows = [analyse(t) for t in texts]
    n = max(1, len(rows))
    total_cjk = max(1, sum(r['n_cjk'] for r in rows))
    return {'n': len(rows), 'clean_rate': round(sum(r['clean'] for r in rows) / n, 4),
            'chinese_answer_rate': round(sum(r['chinese'] for r in rows) / n, 4),
            'with_simplified_rate': round(sum(r['n_simplified'] > 0 for r in rows) / n, 4),
            'with_mainland_term_rate': round(sum(r['n_terms'] > 0 for r in rows) / n, 4),
            'terms_per_response': round(sum(r['n_terms'] for r in rows) / n, 3),
            'terms_per_1k_chars': round(1000 * sum(r['n_terms'] for r in rows) / total_cjk, 3),
            'simplified_per_1k_chars': round(1000 * sum(r['n_simplified'] for r in rows) / total_cjk, 3),
            'with_other_script_rate': round(sum(r['n_other_script'] > 0 for r in rows) / n, 4),
            'avg_cjk_chars': round(total_cjk / n, 1)}
