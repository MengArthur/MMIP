"""
作業四 benchmark: 25 questions about the Self-RAG paper (Asai et al. 2023, arXiv 2310.11511v1), written against the
paper text in Data/selfrag.txt. Questions are in Chinese (the paper is English: cross-lingual retrieval), answers
list the key points a correct answer must contain - the judge checks each key point separately.

  local      14  one fact from one place (a table value, a hyper-parameter, a definition)
  multihop    7  needs facts from two or more sections connected (e.g. why the critic exists + how it is trained)
  global      4  whole-paper questions ("what is this paper about", the complete pipeline) - slide 48: the kind of
                 question plain chunk retrieval answers badly
"""


def G(gid, kind, question, answer, key_points):
    return {'id': gid, 'type': kind, 'question': question, 'answer': answer, 'key_points': key_points}


GRAPH_QA = [
    G('G01', 'local', 'Self-RAG 使用哪四種 reflection tokens？', 'Retrieve、ISREL、ISSUP、ISUSE 四種（後三種屬於 Critique tokens）。',
      ['Retrieve', 'ISREL（相關性）', 'ISSUP（是否有根據）', 'ISUSE（有用程度）']),
    G('G02', 'local', 'ISSUP token 有哪些可能的值？', 'fully supported、partially supported、no support。',
      ['fully supported', 'partially supported', 'no support']),
    G('G03', 'local', 'ISUSE token 的輸出是什麼形式？', '1 到 5 的分數，表示回答對輸入有多有用（5 最好）。',
      ['1 到 5 的分數', '評估回答是否有用']),
    G('G04', 'local', 'Retrieve token 有哪些可能的值？', 'yes、no、continue。', ['yes', 'no', 'continue']),
    G('G05', 'local', 'Self-RAG 的 generator 與 critic 分別使用什麼基底模型？', 'Generator 使用 Llama2 7B 與 13B；critic 使用 Llama2 7B。',
      ['generator 為 Llama2 7B 與 13B', 'critic 為 Llama2 7B']),
    G('G06', 'local', 'Self-RAG 預設使用哪個檢索器？每個輸入取幾篇文件？', '預設使用 Contriever-MS MARCO，每個輸入最多檢索 10 篇文件。',
      ['Contriever-MS MARCO', '最多 10 篇']),
    G('G07', 'local', 'Self-RAG 的訓練資料規模與來源為何？', '共 150k 筆 instruction-output pairs，取自 Open-Instruct 處理過的資料與知識密集型資料集。',
      ['150k 筆', 'Open-Instruct', '知識密集型資料集']),
    G('G08', 'local', '推論時 ISREL、ISSUP、ISUSE 的預設權重是多少？', '分別為 1.0、1.0、0.5。', ['ISREL 1.0', 'ISSUP 1.0', 'ISUSE 0.5']),
    G('G09', 'local', '推論時 segment-level beam search 的 beam width 是多少？token 層級怎麼解碼？', 'segment 層級 beam width 為 2，token 層級使用 greedy decoding。',
      ['beam width 2', 'token 層級 greedy decoding']),
    G('G10', 'local', '預設的 retrieval threshold 是多少？', '多數任務為 0.2；ALCE（ASQA）因需要引用設為 0，以鼓勵頻繁檢索。',
      ['多數任務 0.2', 'ALCE 為 0', '因為需要引用']),
    G('G11', 'local', 'Self-RAG 13B 在 PopQA 的準確率是多少？7B 呢？', '13B 為 55.8，7B 為 54.9。', ['13B 55.8', '7B 54.9']),
    G('G12', 'local', 'Self-RAG 13B 在 ASQA 的 citation precision 是多少？與 Ret-ChatGPT 相比如何？',
      'Self-RAG 13B 的 citation precision 為 70.3，高於 Ret-ChatGPT 的 65.1。', ['70.3', 'Ret-ChatGPT 65.1', '超過 ChatGPT']),
    G('G13', 'local', '實驗用的 PopQA 是哪個子集？有多少題？', 'long-tail 子集，1,399 個罕見實體問題，其 Wikipedia 每月瀏覽量低於 100。',
      ['long-tail 子集', '1,399 題', '每月瀏覽量低於 100']),
    G('G14', 'local', '人工評估中，ISREL 與 ISSUP 在 PopQA 上與人工判斷的一致率是多少？', 'ISREL 為 95.0、ISSUP 為 90.0（PopQA）。',
      ['ISREL 95.0', 'ISSUP 90.0']),
    G('G15', 'multihop', '為什麼要訓練一個 critic 模型，而不是直接用 GPT-4 產生 reflection tokens？效果如何？',
      '依賴 GPT-4 這類專有模型會增加 API 成本並降低可重現性；因此用 GPT-4 產生每類 4k–20k 筆監督資料，蒸餾到以 Llama2 7B 初始化的 critic，critic 與 GPT-4 的預測在多數類別有超過 90% 的一致率。',
      ['API 成本', '可重現性', 'GPT-4 產生監督資料（每類 4k–20k）', '蒸餾到 critic', '一致率超過 90%']),
    G('G16', 'multihop', 'Self-RAG 的訓練方式和 RLHF（PPO）相比，成本上有什麼差異？為什麼？',
      'PPO 在訓練時需要獨立的 reward model；Self-RAG 事先用 critic 離線把 critique 插入訓練語料，再用標準的 next-token 目標訓練 generator，因此訓練成本大幅降低。',
      ['PPO 需要 reward model', '離線插入 critique', '標準 LM 目標', '訓練成本較低']),
    G('G17', 'multihop', 'Generator 訓練時如何處理檢索到的段落與 reflection tokens？',
      '檢索段落（以 <p></p> 包住）在計算 loss 時被遮蔽；詞彙表擴充加入 reflection tokens，generator 學習同時預測輸出文字與 reflection tokens。',
      ['檢索段落在 loss 中被遮蔽', '擴充詞彙加入 reflection tokens', '同時預測輸出與 reflection tokens']),
    G('G18', 'multihop', 'Ablation 中「Retrieve top 1」（像傳統 RAG 一樣固定用第一篇）對哪些資料集影響最大？',
      'PopQA 由 45.5 降到 41.8、ASQA 由 32.1 降到 28.6，降幅較大；PubHealth 幾乎不變（73.5→73.1）。',
      ['PopQA 45.5→41.8', 'ASQA 32.1→28.6', 'PubHealth 影響很小']),
    G('G19', 'multihop', '「No Critic」ablation 在 ASQA 的表現如何變化？這代表什麼？',
      'ASQA em 由 32.1 降到 18.1，表示用 critic 產生的 reflection tokens 訓練對效能有很大貢獻。',
      ['32.1 降到 18.1', 'critic 對效能貢獻大']),
    G('G20', 'multihop', '提高 ISSUP 權重會對輸出造成什麼影響？', 'citation precision 提高（更重視是否有證據支持），但 MAUVE（流暢度）下降，因為較長較流暢的生成常含未被引用完全支持的陳述。',
      ['citation precision 提高', 'MAUVE 流暢度下降', '權衡關係']),
    G('G21', 'multihop', '調高 retrieval threshold 會對檢索頻率與準確率造成什麼影響？', 'threshold 越大檢索越少；在 PubHealth 上準確率下降較小，在 PopQA 上下降較大。',
      ['檢索頻率下降', 'PubHealth 下降小', 'PopQA 下降大']),
    G('G22', 'global', '請用 5W1H 整理這篇論文：誰提出、要解決什麼問題、提出什麼方法、怎麼做？',
      'University of Washington、AI2、IBM 的研究者（Asai 等）提出 Self-RAG，解決 LLM 事實錯誤以及傳統 RAG 不分情況固定檢索導致回答不佳的問題；方法是訓練單一 LM 依需要檢索，並用 reflection tokens 評估檢索段落與自身輸出。',
      ['作者／機構（Asai 等；UW、AI2、IBM）', 'LLM 事實錯誤', '傳統 RAG 不分情況固定檢索', '依需要檢索', 'reflection tokens 自我評估']),
    G('G23', 'global', '請完整說明 Self-RAG 從訓練到推論的流程。',
      '先用 GPT-4 產生的資料訓練 critic；critic 與 retriever 離線在訓練語料中插入檢索段落與 reflection tokens；再以 next-token 目標訓練 generator；推論時模型預測 Retrieve，需要時檢索 K 篇、針對每篇平行生成並產生 ISREL／ISSUP／ISUSE，再以加權 critique 分數做 segment-level beam search 選出最佳輸出。',
      ['訓練 critic', '離線插入 reflection tokens', '訓練 generator', '推論時預測 Retrieve', '平行處理多篇段落', '加權 critique 分數的 beam search']),
    G('G24', 'global', 'Self-RAG 在哪些任務上評估？主要結論是什麼？',
      '六個任務：PubHealth、ARC-Challenge（closed-set）、PopQA、TriviaQA（short-form）、傳記生成（FactScore）與 ASQA（long-form 含引用）；Self-RAG 優於參數更多的 LLM 與傳統 RAG，在所有任務都是非專有模型中最好的，並在多項任務超越 ChatGPT。',
      ['六個任務', 'PubHealth／ARC', 'PopQA／TriviaQA', '傳記與 ASQA', '優於傳統 RAG 與更大的 LLM', '部分任務超越 ChatGPT']),
    G('G25', 'global', 'Self-RAG 和傳統 RAG 最主要的差別有哪些？',
      '傳統 RAG 不論需要與否都檢索固定數量的文件並直接生成；Self-RAG 依需要檢索、評估段落相關性與輸出是否有根據並選出最佳片段，且為每個片段提供引用與支持程度的自我評估。',
      ['依需要檢索 vs 固定檢索', '評估段落相關性', '評估輸出是否有根據並選最佳', '提供引用與支持程度']),
]
