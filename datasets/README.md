# 資料集清單

本專案用到的每一段影片都列在 [`clips.csv`](clips.csv)。CSV 只放**識別碼與統計量**，不含任何影格、軌跡或 caption。要拿到實際資料，請照下面的步驟從原始來源重新下載。

| split | 筆數 | 來源 | 用途 |
|---|---|---|---|
| `train` | 1950 | [OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M) | 四個模型（main / B0 / B1 / zlat）的訓練資料 |
| `heldout` | 50 | OpenVid-1M | 不參與訓練：validation loss 與 held-out 生成 |
| `bench_pexels` | 68 | Pexels，ZipMo 官方 open-domain benchmark（[Google Drive](https://drive.google.com/drive/folders/1ddt4gPIbfnvnARRjL1YdWYpGS0J3tASN)，見 [CompVis/long-term-motion](https://github.com/CompVis/long-term-motion)） | 所有量化結果（EPE / MSE / VBench） |
| `koala_pilot` | 781 | Koala-36M（隊友整理的 `koala_train_v2`） | 只用在早期的 overfit 測試與資料比較，最終模型沒有使用 |

[`qualitative_images.csv`](qualitative_images.csv) 列出 270 張 Wikimedia Commons 照片，是 qualitative 結果的輸入圖，也就是標註網站用的圖。授權只收 CC0 / PD / CC BY / CC BY-SA；每張的標題、作者、授權和原始網址都在 CSV 裡。這些圖片或用它們生成的影片，出現在任何地方（投影片、報告、網頁）都必須附上 credit。

## 欄位

| 欄位 | 說明 |
|---|---|
| `clip_id` | 片段名稱。OpenVid / Koala 是 `<來源影片>_w<k>`；Pexels 是 benchmark 的檔名 |
| `source_dataset`, `source_video` | 來源資料集與其中的檔名（OpenVid：`OpenVid-1M.csv` 的 `video` 欄） |
| `source_part` | OpenVid 檔案所在的 zip（`OpenVid_partN.zip`）。168 筆來自第一批下載，當時的清單已刪除，所以這欄留空；`--from_csv` 會掃過所有 part，不需要這一欄 |
| `youtube_id` | OpenVid / Koala 片段的 YouTube 來源 |
| `start_frame`, `frame_skip`, `n_frames` | 片段 = 從原影片第 `start_frame` 幀起，每 `frame_skip` 幀取 1 幀，共 `n_frames` 幀（65 幀，約 12–15 fps） |
| `fps`, `height`, `width` | 取樣後的幀率與原始解析度 |
| `motion_score` | OpenVid 自己的 motion score |
| `cam_disp`, `obj_p90`, `face_area` | 我們篩選時算的鏡頭位移、物體運動第 90 百分位、最大正臉面積（見 [docs/data_collection.md](../docs/data_collection.md)） |

## 重建資料

```bash
# OpenVid：只下載 CSV 列出的 2000 支（從 HF 的遠端 zip 用 range read 抽出，不下載整個 zip）
python src/openvid_fetch_train.py --from_csv datasets/clips.csv --out data/openvid_rebuild
python src/openvid_windows.py --root data/openvid_rebuild --shard 0 --n_shards 1   # CoTracker3 + 3 個 ZipMo z
python src/dataset_csv.py windows --root data/openvid_rebuild                        # -> windows/train_windows.csv
python src/prep_koala.py --src data/openvid_rebuild/windows --out data/clips_a14b_openvid
python src/dataset_csv.py split --clips data/clips_a14b_openvid                     # 50 筆 heldout -> *_test/
python src/prep_z.py --clips data/clips_a14b_openvid data/clips_a14b_openvid_test --src data/openvid_rebuild/windows

# Pexels benchmark（需要 pip install gdown；下載後逐檔比對 md5）
python src/download_pexels.py               # -> data/raw/pexels_official/
python src/bench_pexels.py prep              # -> data/bench_pexels/
```

CoTracker3 的 query 點由片段名稱和 seed 決定，所以重跑得到的軌跡與 z 會和我們的幾乎一樣。只有 GPU 浮點運算的差異會造成很小的不同。

在實驗室的 NAS 上可以直接用現成的資料：`/project/YuHsuanLee/poke_to_video/data/`。

## 使用限制

OpenVid-1M 與 Koala-36M 的影片來自 YouTube，**只供課堂專題使用，請勿再分享或公開**。公開的網頁（user study、標註網站）只能用 Pexels 片段和 Wikimedia 照片。

`clips.csv` 由 `python src/dataset_csv.py export` 從本地的 `data/` 產生。
