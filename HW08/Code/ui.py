"""
Chat UI (Gradio), two tabs:
  勞動法規 RAG   conversation on the left; on the right the standalone question the system actually searched with,
                the retrieved articles (cited ones marked ★, each with a link to the official law database) and the
                retrieval settings (top-k, prompting technique, query transformation, re-ranker) that can be changed live
  影片 VideoRAG  conversation + the frames that were selected for the answer (with video name and time) + the
                transcript / frame-description units that were used
  demo = build_demo(rag, video_rag); demo.launch(share=True)     (either may be None: one tab only)
Qwen3-4B (8 GB) and Qwen3-VL-2B (4.5 GB) do not fit on a T4 together with the retrieval models, so each tab unloads
the other tab's model before answering (switching tabs costs one model load, about a minute).
"""

import inspect

from .rag import PROMPTS


def _chatbot(gr, **kw):
    if 'type' in inspect.signature(gr.Chatbot.__init__).parameters:
        kw['type'] = 'messages'
    return gr.Chatbot(**kw)


def refs_markdown(rec):
    lines = [f'**檢索用的問題**：{rec["standalone"]}', '']
    for x in rec['references']:
        star = '★ ' if x['cited'] else ''
        arts = '、'.join(f'[{a}]({u})' if u else a for a, u in zip(x['articles'], x['urls'])) or '（條文片段）'
        snippet = x['text'].replace('\n', ' ')[:160]
        lines.append(f'{star}**[{x["n"]}]** {arts}　<sub>相似度 {x["score"]:.3f}</sub><br><sub>{snippet}…</sub>')
    lines.append('\n★ = 回答中有引用')
    return '\n\n'.join(lines)


def build_demo(rag, video_rag=None, title='MMIP HW08：多輪對話 RAG'):
    import gradio as gr
    from .video import fmt

    def law_send(msg, chat, hist, k, prompt, qmode, rerank):
        if not msg:
            return chat, hist, gr.update(), ''
        if video_rag is not None and video_rag.vlm is not None:
            video_rag.vlm.unload()
        rag.top_k, rag.query_mode = int(k), qmode
        keep = rag.reranker
        if not rerank:
            rag.reranker = None
        try:
            rec = rag.answer(msg, hist, prompt=prompt)
        finally:
            rag.reranker = keep
        chat = chat + [{'role': 'user', 'content': msg}, {'role': 'assistant', 'content': rec['answer']}]
        return chat, hist + [(msg, rec['answer'])], refs_markdown(rec), ''

    def video_send(msg, chat, hist):
        if not msg:
            return chat, hist, gr.update(), gr.update(), ''
        if rag is not None:
            rag.llm.unload()
        rec = video_rag.answer(msg, hist)
        chat = chat + [{'role': 'user', 'content': msg}, {'role': 'assistant', 'content': rec['answer']}]
        gallery = [(video_rag.image(f['file']), f'{f["video"]} {fmt(f["t"])}') for f in rec['frames']]
        md = [f'**檢索用的問題**：{rec["standalone"]}', f'**候選影片**：' + '、'.join(f'{v}（{s}）' for v, s in rec['video_scores'].items())]
        for u in rec['text_units']:
            kind = '語音' if u['kind'] == 'speech' else '畫面描述'
            md.append(f'- 【{u["video"]} {fmt(u["start"])}–{fmt(u["end"])}】（{kind}）{u["text"]}')
        return chat, hist + [(msg, rec['answer'])], gallery, '\n\n'.join(md), ''

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f'## {title}')
        if rag is not None:
            with gr.Tab('勞動法規 RAG'):
                hist = gr.State([])
                with gr.Row():
                    with gr.Column(scale=3):
                        chat = _chatbot(gr, height=480, label='對話')
                        box = gr.Textbox(placeholder='例如：我做了兩年被資遣，公司要提早多久通知？', show_label=False)
                        with gr.Row():
                            for ex in ('我在公司做了兩年半，特休有幾天？', '那如果今年沒休完呢？', '老婆要生了，我可以請幾天假？'):
                                gr.Button(ex, size='sm').click(lambda e=ex: e, None, box)
                        clear = gr.Button('清除對話', size='sm')
                    with gr.Column(scale=2):
                        refs = gr.Markdown('檢索到的條文會顯示在這裡。')
                        with gr.Accordion('檢索與生成設定', open=False):
                            k = gr.Slider(1, 10, value=rag.top_k if rag else 5, step=1, label='Top-K')
                            prompt = gr.Dropdown(list(PROMPTS), value=rag.prompt if rag else 'grounded', label='Prompting technique')
                            qmode = gr.Dropdown(['rewrite', 'multi', 'stepback', 'none'], value=rag.query_mode if rag else 'rewrite', label='Query 轉換')
                            has_rr = rag is not None and rag.reranker is not None
                            rerank = gr.Checkbox(value=has_rr, label='使用 Re-ranker', interactive=has_rr)
                box.submit(law_send, [box, chat, hist, k, prompt, qmode, rerank], [chat, hist, refs, box])
                clear.click(lambda: ([], [], '檢索到的條文會顯示在這裡。'), None, [chat, hist, refs])
        if video_rag is not None:
            with gr.Tab('影片 VideoRAG'):
                vhist = gr.State([])
                with gr.Row():
                    with gr.Column(scale=3):
                        vchat = _chatbot(gr, height=420, label='對話')
                        vbox = gr.Textbox(placeholder='問影片裡的內容，例如：第二步要做什麼？', show_label=False)
                        vclear = gr.Button('清除對話', size='sm')
                    with gr.Column(scale=2):
                        gallery = gr.Gallery(label='選出的畫面（Frame Selection）', columns=3, height=260)
                        vrefs = gr.Markdown('使用的逐字稿與畫面描述會顯示在這裡。')
                vbox.submit(video_send, [vbox, vchat, vhist], [vchat, vhist, gallery, vrefs, vbox])
                vclear.click(lambda: ([], [], [], ''), None, [vchat, vhist, gallery, vrefs])
    return demo
