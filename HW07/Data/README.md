# Data

All benchmark files in this folder were built once by `Code/prepare_data.py` (or written by hand) and are committed, so every run uses exactly the same questions. Models are downloaded from Hugging Face at run time.

| File | Used in | Content | Source |
| :--- | :--- | :--- | :--- |
| `partA_llm_benchmark.json` | 作業一 | 20 self-written instructions, one per ability (math, logic, dates, string handling, coding, knowledge, false-premise, translation, summary, JSON / list / sentence-count formats, sentiment, table reasoning, safety, constrained writing, unit conversion). Each item has a rule-based `check` spec (see `Code/scoring.py`) and a `reference` for the Gemini judge | Written for this assignment |
| `partB_vlm_benchmark.json`, `partB_images/` | 作業 B | 20 visual questions: 15 COCO val2017 photos (answers written after looking at every image) + 5 synthetic images whose answers are exact (sign OCR, shape counting, bar chart, spatial layout, visual math) | Photos: [COCO 2017 val](https://cocodataset.org/) (Flickr images, CC BY licences); synthetic images drawn by `Code/prepare_data.py` |
| `partC_tw_prompts.json` | 作業 C | 170 self-written Traditional-Chinese questions (technology, internet, transport, food, work, study): 130 `train` prompts build the DPO preference pairs, 40 `test` prompts are held out | Written for this assignment |
| `partC_general_benchmark.json` | 作業 C | 100 general-ability questions unrelated to the DPO aspect: 60 MMLU multiple-choice questions (sampled with seed 0 from the test questions of all 57 subjects; 30 subjects appear) + 40 GSM8K math word problems (seed 0) | [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu) test split, [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) main test split |
| `partC_coco_subset.json` | 作業 C 進階 | 150 train + 80 eval COCO val2017 images (disjoint, none from 作業 B) with annotated objects, salient objects (area ≥ 2 %), 5 captions, POPE questions (3 present, 2 adversarial-absent, 1 random-absent object per image) and an adversarial pool of frequently co-occurring absent objects | [COCO 2017](https://cocodataset.org/) `instances_val2017.json`, `captions_val2017.json` (CC BY 4.0) |

`coco_images/` holds the 230 COCO images downloaded at run time from `images.cocodataset.org` (not committed).
