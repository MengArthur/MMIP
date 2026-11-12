"""
作業三 chat UI (Gradio): conversation with photo upload on the left; on the right the ticket that was opened,
the routing trace (path, System One confidences, latency) and the list of tickets. Suggested next topics are
shown as buttons under the conversation (UX: moving on to the next topic).
  demo = build_demo(bot); demo.launch(share=True)      # Colab: public link; lab server: share=False
"""

import inspect
import os

from .chatbot import START_SUGGEST, Session
from .common import DATA_DIR

WELCOME = '您好，我是速達 3C 的客服助理 👋 請告訴我遇到的問題，也可以直接上傳收據、錯誤畫面或產品照片。'
N_SUGGEST = 4


def _chatbot(gr, **kw):
    if 'type' in inspect.signature(gr.Chatbot.__init__).parameters:      # Gradio 4/5 need the messages format flag
        kw['type'] = 'messages'
    return gr.Chatbot(**kw)


def _trace_view(out):
    t = dict(out['trace'])
    d = out['decision'] or {}
    view = {'status': d.get('status'), 'department': d.get('department'), 'urgency': d.get('urgency'),
            'missing_info': d.get('missing_info'), 'path': t.get('path'),
            'latency_s': round(t.get('total_latency', 0), 2)}
    if t.get('system_one'):
        view['system_one_confidence'] = {k: round(v, 3) for k, v in t['system_one']['confidence'].items()}
    if t.get('pattern'):
        view['guard_pattern'] = t['pattern']
    if t.get('output_filter'):
        view['output_filter'] = t['output_filter']
    return view


def build_demo(bot, title='速達 3C 客服（MMIP HW07）', preload=()):
    """preload: optional scripted turns [(text, [image paths relative to Data/])] played when the page opens."""
    import gradio as gr

    def start():
        res = (Session(), [{'role': 'assistant', 'content': WELCOME}], None, None, [], *_buttons(START_SUGGEST))
        for text, images in preload:
            res = send(text, [os.path.join(DATA_DIR, p) for p in images], res[0], res[1], res[4])
        return res

    def _buttons(items):
        items = list(items)[:N_SUGGEST]
        return [gr.update(value=items[i], visible=True) if i < len(items) else gr.update(visible=False) for i in range(N_SUGGEST)]

    def send(text, files, session, chat, tickets):
        if not (text or files):
            return session, chat, gr.update(), gr.update(), tickets, *([gr.update()] * N_SUGGEST)
        files = list(files or [])
        for f in files:
            chat.append({'role': 'user', 'content': {'path': f}})
        if text:
            chat.append({'role': 'user', 'content': text})
        out = bot.respond(session, text or '', files)
        chat.append({'role': 'assistant', 'content': out['reply']})
        if out['ticket']:
            t = out['ticket']
            tickets = tickets + [[t['ticket_id'], t['department_zh'], t['urgency'], t['order_id'] or '', t['product'] or '', t['intent'] or '']]
        return session, chat, out['ticket'], _trace_view(out), tickets, *_buttons(out['suggestions'])

    def submit(msg, session, chat, tickets):
        res = send(msg.get('text', ''), msg.get('files', []), session, chat, tickets)
        return (gr.update(value=None),) + tuple(res)

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f'## {title}\n多輪、多模態客服分流：System One 快速分流 → SFT + 約束解碼建立工單；文字與圖片攻擊會被攔截。')
        session = gr.State()
        tickets = gr.State([])
        with gr.Row():
            with gr.Column(scale=3):
                chat = _chatbot(gr, height=520, label='對話')
                with gr.Row():
                    btns = [gr.Button(visible=False, size='sm') for _ in range(N_SUGGEST)]
                box = gr.MultimodalTextbox(file_types=['image'], file_count='multiple', placeholder='輸入訊息，或上傳圖片…',
                                           show_label=False)
                reset = gr.Button('重新開始對話', size='sm')
            with gr.Column(scale=2):
                ticket = gr.JSON(label='本輪建立的工單')
                trace = gr.JSON(label='分流紀錄（決策、路徑、信心、延遲）')
                table = gr.Dataframe(headers=['工單', '部門', '緊急', '訂單', '產品', '需求'], label='所有工單', interactive=False)
        outs = [session, chat, ticket, trace, tickets] + btns
        box.submit(submit, [box, session, chat, tickets], [box] + outs).then(lambda t: t, tickets, table)
        for b in btns:
            b.click(lambda label, s, c, t: send(label, [], s, c, t), [b, session, chat, tickets], outs).then(lambda t: t, tickets, table)
        reset.click(start, None, outs).then(lambda t: t, tickets, table)
        demo.load(start, None, outs).then(lambda t: t, tickets, table)
    return demo


def scripted_demo(bot, steps):
    """Runs a scripted conversation (list of (text, [image paths])) and returns a readable transcript."""
    s, lines = Session(), []
    for text, images in steps:
        out = bot.respond(s, text, images)
        lines.append({'user': text, 'images': images, 'reply': out['reply'], 'decision': out['decision'],
                      'ticket': out['ticket'], 'path': out['trace'].get('path'),
                      'latency_s': round(out['trace'].get('total_latency', 0), 2), 'suggestions': out['suggestions']})
    return lines
