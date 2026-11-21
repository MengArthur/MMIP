# 影片資料（作業二）

影片檔（`*.mp4`）較大，不提交到 GitHub；這個資料夾只保留說明。

| 檔名 | 長度 | 聲音 | 來源（自行錄製） |
| :--- | ---: | :--- | :--- |
| `op_patient_inference.mp4` | 19.5 s | 無 | 期末專題 Medical Assistant Model 操作 demo，0–19.5 s |
| `op_review_report.mp4` | 30.1 s | 無 | 同上，19.5–49.5 s |
| `op_database_tables.mp4` | 36.6 s | 無 | 同上，49.5–86.1 s |
| `talk_paper_intro.mp4` | 36.1 s | 有 | 課堂論文報告錄影，8.5–44.5 s |
| `talk_outline.mp4` | 44.5 s | 有 | 同上，44.5–88.85 s |

- 檔名以 `op_` 開頭的是沒有說話的操作影片，以 `talk_` 開頭的是有說話的報告影片。
- 問答與時間標註在 `Data/video_qa.json`：每題記錄影片名稱、答案所在的時間區間（秒）與題型（speech / visual）。
- 重跑時把影片放在這個資料夾（Colab 上為 `MyDrive/MMIP/HW08/Data/videos/`），Notebook 會自動建立索引。
