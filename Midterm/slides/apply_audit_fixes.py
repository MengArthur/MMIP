p = r'C:\Users\yp455\Downloads\MMIP_project\slides\build_deck.py'
s = open(p, encoding='utf-8').read()
NL = '\\n'
reps = [
    ("'實際資料：一個晶片版圖的繞線壅塞分布（CircuitNet-N14，亮處為壅塞）'",
     "'實際資料：一個晶片版圖的繞線使用率分布（CircuitNet-N14，亮處為繞線擁擠）'"),
    ("[('繞線壅塞', {'bold': True}), ('要等 routing 跑完才看得到；一旦壅塞，就要回頭修改設計再重跑，迭代成本高', {})],",
     "[('真正的壅塞', {'bold': True}), ('要等 routing 跑完才確定；placement 階段只有粗估（RUDY、early global routing），實測和最終結果的相關係數只有約 0.40／0.57', {})],"),
    ("[('和我的研究直接相關：', {'bold': True}), ('HLS 產生的硬體在 FPGA 上也常因繞線壅塞造成 timing 不過', {})],",
     "[('一旦壅塞：', {'bold': True}), ('要回頭修改設計再重跑，迭代成本高；HLS 產生的硬體在 FPGA 上也有同樣問題', {})],"),
    ("('把「routing 後才知道」變成「placement 後就能預測」', {'bold': True, 'color': NAVY})",
     "('在 placement 後就做出比現有粗估更準的壅塞預測', {'bold': True, 'color': NAVY})"),
    ("('效益', '在 routing 之前就提醒設計者修正')", "('效益', '比 RUDY／eGR 粗估更準，提早修正')"),
    ("'已實際下載 Vortex-small 檢查：96 個樣本，每張 459 × 456 格（NumPy 陣列）'",
     "'已實際下載 Vortex-small 檢查：96 個樣本（16 種 floorplan 設定），每張約 400–460 格見方'"),
    ("'只需 routability 特徵（25.6 GB）；先用 3 個小設計，約 3.5 GB，Colab 可處理',",
     "'只需 routability 特徵；先用 3 個小設計（下載約 3.5 GB，解壓後約 10 倍，只解壓需要的特徵並縮成 256×256）',"),
    ("('傳統影像 + 機器學習', 'OpenCV 擷取密度、梯度等特徵" + NL + "＋ Random Forest／SVM'",
     "('傳統方法與 baseline', 'EDA 粗估（RUDY、eGR）；OpenCV" + NL + "特徵 ＋ Random Forest／SVM'"),
    ("('研究延伸', '接入 Agentic HLS" + NL + "先預測再改 pragma')",
     "('未來延伸', '移植到 FPGA（需自建" + NL + "Vivado 資料）接 HLS')"),
    ("('量化比較', '用 ROC-AUC、Macro-F1" + NL + "比較傳統方法與深度學習')",
     "('量化比較', '和 EDA 粗估及 5 種方法" + NL + "比較 ROC-AUC、Macro-F1')"),
    ("('標籤太稀疏', '繞線 overflow 只出現在約 0.05% 的格子', '改用繞線使用率門檻定義熱點')",
     "('標籤太稀疏', '繞線 overflow 只出現在約 1% 的格子', '改用繞線使用率門檻定義熱點')"),
    ("('樣本數有限', 'Vortex-small 只有 96 個樣本', '再加入 nvdla-small、openc910-1')",
     "('尺寸與數量', '每張尺寸不同；Vortex-small 只有 96 筆', '統一縮放並切區塊；加入其他設計')"),
    ("每個樣本是 459 乘 456 格的陣列", "每個樣本是約四百多格見方的陣列，尺寸隨 floorplan 設定不同"),
    ("繞線 overflow 只出現在約萬分之五的格子", "繞線 overflow 只出現在約 1% 的格子"),
    ("繞線壅塞通常要等 routing 跑完才知道，一旦壅塞就得回頭改設計重跑，非常耗時。",
     "真正的壅塞要等 routing 跑完才確定；placement 階段 EDA 工具只有 RUDY 和 early global routing 這類粗估，我用實際資料算過，它們和最終結果的相關係數只有約 0.4 到 0.57。一旦壅塞就得回頭改設計重跑，非常耗時。"),
    ("在 placement 之後就用影像辨識預測哪裡會壅塞", "在 placement 之後用影像辨識做出比這些粗估更準的壅塞預測"),
    ("先用 OpenCV 特徵加上 Random Forest 當傳統 baseline", "先用 EDA 工具內建的粗估、以及 OpenCV 特徵加上 Random Forest 當 baseline"),
    ("之後也希望接回我的研究，讓 agent 寫 HLS 時先預測壅塞再調整 pragma", "未來如果自行產生 FPGA 的資料，也希望把方法接回我的 HLS 研究"),
    ("預期成果有三個：可操作的網頁、五種方法的量化比較", "預期成果有三個：可操作的網頁、和 EDA 粗估及五種方法的量化比較"),
    ("Vortex-small 只有 96 個樣本，會再加入其他設計。", "每張尺寸不同而且 Vortex-small 只有 96 筆，會統一縮放、切成區塊，並加入其他設計。"),
]
missing = [a[:50] for a, _ in reps if s.count(a) != 1]
assert not missing, missing
for a, b in reps:
    s = s.replace(a, b)
open(p, 'w', encoding='utf-8').write(s)
print('applied', len(reps))
