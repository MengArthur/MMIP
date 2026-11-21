"""
Builds the corpora in Data/ once (run locally; the outputs are committed so every run uses the same text):
  laws.json      14 Taiwanese labour laws / regulations from the national law database (law.moj.gov.tw),
                 one record per article: law, chapter, article number, text, URL. Laws are not protected by
                 copyright in Taiwan (Copyright Act art. 9).
  selfrag.txt    full text of the Self-RAG paper (Asai et al., arXiv 2310.11511, CC BY 4.0) for 作業四.
  rag_eval.json, graph_benchmark.json   the hand-written evaluation sets (Code/qa_data.py, Code/graph_data.py)

  python -m Code.prepare_data            (fetch + dump)
  python -m Code.prepare_data --dump     (only re-dump the hand-written sets)
"""

import html
import io
import json
import os
import re
import time
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'Data')

LAWS = {   # pcode -> short name used in references
    'N0030001': '勞動基準法', 'N0030002': '勞動基準法施行細則', 'N0030006': '勞工請假規則',
    'N0030014': '性別平等工作法', 'N0030015': '性別平等工作法施行細則', 'N0030020': '勞工退休金條例',
    'N0030028': '最低工資法', 'N0060001': '職業安全衛生法', 'N0090001': '就業服務法', 'N0020007': '勞資爭議處理法',
    'N0020001': '工會法', 'N0050001': '勞工保險條例', 'N0050021': '就業保險法', 'N0050031': '勞工職業災害保險及保護法',
}
PAPER = 'https://arxiv.org/pdf/2310.11511v1'


def _get(url):
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (MMIP coursework)'})
    return urllib.request.urlopen(req, timeout=60).read()


def _text(fragment):
    lines = re.findall(r'<div class="line-\d+[^"]*">(.*?)</div>', fragment, re.S)
    return '\n'.join(html.unescape(re.sub(r'<[^>]+>', '', ln)).strip() for ln in lines)


def parse_law(pcode, name):
    s = _get(f'https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode={pcode}').decode('utf-8')
    date = re.search(r'(修正|公布)日期：</th>\s*<td>([^<]*)<', s)
    body = s[s.index('law-reg-content'):]
    arts, chapter = [], ''
    for m in re.finditer(r'<div class="h3 char-\d">\s*(.*?)</div>|<div class="col-no">.*?name="([^"]+)">(第 [^<]+ 條)</a></div>'
                         r'<div class="col-data"><div class="law-article">(.*?)</div>\s*</div></div>', body, re.S):
        if m.group(1):
            chapter = re.sub(r'\s+', ' ', html.unescape(m.group(1))).strip()
            continue
        no = m.group(3).replace(' ', '')
        text = _text(m.group(4))
        if not text or text.startswith('（刪除）'):
            continue
        arts.append({'id': f'{name}{no}', 'law': name, 'pcode': pcode, 'chapter': chapter, 'article': no, 'text': text,
                     'url': f'https://law.moj.gov.tw/LawClass/LawSingle.aspx?pcode={pcode}&flno={m.group(2)}'})
    return arts, (date.group(2).strip() if date else '')


def build():
    os.makedirs(DATA, exist_ok=True)
    laws, versions = [], {}
    for pcode, name in LAWS.items():
        arts, date = parse_law(pcode, name)
        laws += arts
        versions[name] = {'pcode': pcode, 'articles': len(arts), 'date': date}
        print(f'{name:16s} {len(arts):4d} articles  {date}')
        time.sleep(1)
    with open(os.path.join(DATA, 'laws.json'), 'w', encoding='utf-8') as f:
        json.dump({'source': 'law.moj.gov.tw', 'fetched': time.strftime('%Y-%m-%d'), 'laws': versions, 'articles': laws},
                  f, ensure_ascii=False, indent=1)
    print('total', len(laws), 'articles,', sum(len(a['text']) for a in laws), 'characters')

    from pypdf import PdfReader
    pdf = PdfReader(io.BytesIO(_get(PAPER)))
    text = '\n'.join(p.extract_text() or '' for p in pdf.pages)
    with open(os.path.join(DATA, 'selfrag.txt'), 'w', encoding='utf-8') as f:
        f.write(f'# Asai et al. 2023, Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection\n'
                f'# {PAPER} (CC BY 4.0)\n\n' + text)
    print('paper', len(pdf.pages), 'pages', len(text), 'characters')


def dump_benchmarks():
    from .graph_data import GRAPH_QA
    from .qa_data import DIALOGUES, SINGLE
    with open(os.path.join(DATA, 'rag_eval.json'), 'w', encoding='utf-8') as f:
        json.dump({'single': SINGLE, 'dialogues': DIALOGUES}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(DATA, 'graph_benchmark.json'), 'w', encoding='utf-8') as f:
        json.dump(GRAPH_QA, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    import sys
    if '--dump' not in sys.argv:
        build()
    dump_benchmarks()
