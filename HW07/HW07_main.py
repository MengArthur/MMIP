"""
MMIP HW07 (lecture 9: structured output of LLMs) -- one-command pipeline.

  python HW07_main.py                      # 作業二 (train + compare A-D) then 作業三 (chatbot experiments)
  python HW07_main.py --parts 2            # only 作業二; --parts 3 needs the 作業二 checkpoints
  python HW07_main.py --no-gemini          # skip the Gemini reference router / judge
  python HW07_main.py --ui                 # afterwards start the chat UI (Gradio)

作業一 (scenario, spec, architecture diagram) is documentation: README.md and docs/.
Data/ is already built; Code/prepare_data.py rebuilds it.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Code.common import gpu_info  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', nargs='+', default=['2', '3'], choices=['2', '3'])
    ap.add_argument('--no-gemini', action='store_true')
    ap.add_argument('--ui', action='store_true')
    a = ap.parse_args()
    print(gpu_info())
    if '2' in a.parts:
        from Code import experiments
        S, C = experiments.run_all(gemini=not a.no_gemini)
        experiments.plots(S, C)
        experiments.training_curves()
        print(S.round(2).to_string())
    bot = None
    if '3' in a.parts:
        from Code import part3
        bot, _, res = part3.run_all(gemini=not a.no_gemini)
        for k in ('topic', 'offtopic', 'attack'):
            print(f'--- {k}'); print(res[k][0].round(3).to_string())
    if a.ui:
        from Code import part3, ui
        bot = bot or part3.build_bots()[0]
        ui.build_demo(bot).launch(share='google.colab' in sys.modules)


if __name__ == '__main__':
    main()
