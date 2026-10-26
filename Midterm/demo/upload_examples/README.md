# Demo 上傳用版圖

在網頁左上角按「上傳」，選這裡的檔案。全部來自測試設計 **Vortex-small**（完全沒參與訓練），且不是內建範例。

挑選方式（沒有看模型表現）：排除 7 個內建範例和它們只差 filler 時機的雙胞胎版圖後，依「熱點區塊數」由少到多排序，取第 20%、50%、80% 位置的版圖；D 取第 65% 位置，並拿掉答案。

| 檔案 | 原始版圖 | 熱點區塊 | 內容 | U-Net 多模態 | eGR | Ensemble |
| :--- | :--- | ---: | :--- | :--- | :--- | :--- |
| `vortex_A_hotspots_8.npz` | freq 200, mp 2, fpu 60, p 4, fi ar | 8 | X、params、label、egr | 抓到 7/8、誤報 3（F1 0.778） | 7/8、誤報 9（0.583） | 7/8、誤報 6（0.667） |
| `vortex_B_hotspots_15.npz` | freq 500, mp 2, fpu 60, p 4, fi ap | 15 | X、params、label、egr | 12/15、誤報 0（0.889） | 13/15、誤報 6（0.765） | 14/15、誤報 5（0.824） |
| `vortex_C_hotspots_21.npz` | freq 500, mp 2, fpu 55, p 2, fi ap | 21 | X、params、label、egr | 10/21、誤報 2（0.606） | 9/21、誤報 4（0.529） | 14/21、誤報 1（0.778） |
| `vortex_D_no_answer.npz` | freq 200, mp 2, fpu 65, p 2, fi ar | 17 | **只有** X、params | 只顯示預測（模擬「還沒 routing 的新 placement」） | – | – |

- C 是 U-Net 只抓到一半熱點的例子（Ensemble 在這張比較好），可以用來誠實說明單張版圖會有變動。
- 這些檔案已經在 server 上算過一次，結果存在 `demo/cache/`，demo 時上傳會立即顯示（重新訓練後第一次上傳約 10 秒）。
