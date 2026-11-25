# 訓練資料收集方法：OpenVid clean set（2000 筆）

> 2026-11-21。取代 `koala_train_v2`（781 筆）作為 A14B 微調的訓練資料。
> 資料在 `data/openvid_clean_set/`（git 不追蹤）。影片來自 YouTube，**僅供課堂專題使用，請勿再分享或公開**。

## 目標

每筆是一個 65 幀、約 12 fps 的片段，要同時滿足：

1. **鏡頭幾乎不動**：畫面的運動只來自物體本身。
2. **畫面裡真的有東西在動**，最好是手和物體的互動、人或動物在移動，而不是有人對著鏡頭講話。
3. **CoTracker3 的軌跡夠乾淨**，能讓 ZipMo 把軌跡編成 z。

## 結果

| | Koala v2（781） | OpenVid 第一版（2000，只看運動量） | **OpenVid clean（2000）** |
|---|---|---|---|
| 鏡頭位移 `cam_disp`（中位數） | 4.5 | 2.1 | **0.25** |
| `cam_disp` < 1 的比例 | 29% | 42% | **77%** |
| 物體運動第 90 百分位 `obj_p90` | 9.2 | 9.3 | **11.8** |
| 有物體在動的軌跡比例 `obj_frac_moving` | 0.44 | 0.38 | 0.32 |
| 對鏡頭講話的片段（目視抽樣） | — | 約 60% | 約 10% |

- **來源**：1955 個不同的 YouTube 影片。
- **解析度**：1080p 1373 筆、720p 605 筆、其他 22 筆。
- **幀率**：取樣後在 12–15 fps 之間。
- **內容**：以手操作物體為主，例如做菜、手作、玩具、工具、倒液體。
- **caption 關鍵字粗估**：物體互動 30%、走路或運動 6%、動物 6%。

單位說明：`cam_disp`、`obj_p90` 是 ZipMo 的 [0,128] 座標，128 等於畫面寬或高。

## 流程

```
OpenVid-1M metadata（約 100 萬筆）
  │ ① metadata 篩選（不用下載）
  ▼
caption 乾淨的候選池 81,363 筆
  │ ② 隨機抽樣，只從遠端 zip 抽出需要的檔案
  ▼
下載 25,500 支（第一批 5,500 支 + clean 批 20,000 支）
  │ ③ 臉部偵測（在 CoTracker 之前，省 GPU）
  │ ④ CoTracker3 追蹤 + ZipMo 編 z
  │ ⑤ 軌跡篩選：ZipMo 條件、鏡頭、物體運動
  ▼
通過 2,248 支 → 每個來源最多 2 支 → 隨機抽 2,000 筆
```

### ① Metadata 篩選（`src/openvid_fetch_train.py --clean`）

OpenVid-1M 的 `OpenVid-1M.csv` 有 caption、camera motion 標籤、motion score、fps、幀數。條件如下：

| 條件 | 原因 |
|---|---|
| `camera motion == "static"` | OpenVid 自己的鏡頭標籤，先粗篩 |
| 放得下一個 65 幀窗口：`frame ≥ 64·skip + 1`，`skip = round(fps/12)`，取樣後 10–16 fps | 我們要 65 幀、約 12 fps |
| 排除 `celebv*`、`magictime*` | 分別是固定鏡頭的人臉講話、植物縮時攝影 |
| 1.5 ≤ motion score < 25 | ≥ 25 那一層實測反而最靜（見 `docs/openvid_check.md`）；25 以下 motion score 跟實際運動幾乎無關，所以不再細分 |
| caption **不含** `CLEAN_NEG` 的字 | 講話、訪談、看鏡頭、手勢、表情、微笑、vlog、麥克風、動畫、卡通、3D、遊戲畫面、文字、標題等 |

- OpenVid 的 Pexels、Pixabay、Mixkit 素材都被裁成 64 幀、24 fps，不夠 65 幀，所以符合條件的長片段全部來自 YouTube（Panda-70M）。
- 過了前四個條件的有 286,226 筆，其中約 40% 的 caption 在描述有人講話。加上 caption 條件後剩 81,363 筆。

### ② 下載

OpenVid 有 170 個單檔 zip 分卷，每個 20–50 GB。

- 用 `HfFileSystem` 開遠端 zip，讀中央目錄要 20–90 秒，然後用 HTTP range 只抽出被選中的 mp4，不必下載整個分卷。
- 抽樣在全部分卷上均勻隨機進行（分卷是按來源排序的，只抓前幾卷會全是 celebv）。
- 下載 2 萬支大約 2 小時。

### ③ 臉部偵測（`tracking.face_area`，`openvid_windows.py --max_face 0.02`）

- 用 OpenCV Haar 正臉偵測，模型檔在 `third_party/opencv/`，對窗口的第 0、32、64 幀各跑一次，取最大的臉佔畫面的面積比例。
- 超過 2% 就不跑 CoTracker，只記錄原因。
- 靜態鏡頭的 YouTube 片段大多是有人對著鏡頭講話，這一步刷掉的最多（9,297 支）。

### ④ 追蹤與編碼（`src/openvid_windows.py`）

每支影片從第 0 幀起取一個窗口：65 幀、每 `skip` 幀取一幀，寬度超過 1920 時先縮小。

| 步驟 | 細節 |
|---|---|
| CoTracker3 offline | 1,024 個隨機起點、6×6 support grid、前 64 幀；輸出 `tracks_yx`（y, x 正規化到 [-1, 1]）、`visibility`、`confidence` |
| 鏡頭指標 | 跟 Spencer 的 Koala 規則相同：`cam_motion`、`zoom_total`、`cut_ratio` |
| ZipMo z | 從畫面內有效軌跡隨機取 3 組、每組 40 條，連同第一幀編成 `z.npy [3,256,16]`（已 normalize），所用的軌跡索引存在 `enc_idx.npy`。做法跟 Spencer 的 `preprocess_koala.py` 完全一樣，z 解回軌跡的誤差是 1.4–1.8 px（1080p） |

輸出格式跟 `koala_train_v2` 完全相同，`src/prep_koala.py` 可以直接讀。

在 4 張 A6000 上、每張跑 2 個 shard，大約每分鐘 57 筆。

### ⑤ 軌跡篩選（`src/openvid_select_train.py`）

每筆只記第一個沒過的條件，順序如下：

| 條件 | 門檻 | 刷掉 |
|---|---|---|
| caption（只影響第一批沒預篩的 5,500 支） | 不含 `CLEAN_NEG` | 3,397 |
| 臉 | 最大正臉 ≤ 2% | 9,297 |
| ZipMo 靜態鏡頭條件 | `static_frac > 0.5` | 964 |
| 有效軌跡數 | 畫面內有效軌跡 ≥ 80 | 1 |
| Spencer 的鏡頭規則 R1–R3 | 推拉、鏡頭移動太多、場景切換 | 1,065 |
| 物體運動 | `obj_frac_moving ≥ 0.2` 且 `obj_p90 ≥ 4` | 5,774 |
| 鏡頭位移 | `cam_disp < 2.5` | 2,754 |
| **通過** | | **2,248** |

物體運動和鏡頭位移的定義（`src/motion_stats.py`）：

- 在 t = 16、32、48、63，用 RANSAC 擬合第 0 幀 → 第 t 幀的相似變換（平移、縮放、旋轉），代表鏡頭運動。
- `cam_disp`：鏡頭變換讓軌跡移動的距離中位數，四個時間點取最大值。
- 殘差 = 軌跡實際位置減去鏡頭變換預測的位置，代表物體自己的運動。
- `obj_frac_moving`：殘差 > 2 的軌跡比例。
- `obj_p90`：殘差的第 90 百分位。

最後每個 YouTube 來源最多保留 2 支，再用固定 seed 隨機抽 2000 筆。

## 試過但放棄的做法

| 做法 | 結果 |
|---|---|
| Koala v2 | 鏡頭位移大（中位數 4.5），15% 的窗口幾乎沒東西在動 |
| OpenVid，只用運動量篩選（第一版 2000 筆） | 約 60% 是有人對著鏡頭講話 |
| OpenVid，用 caption 正向挑類別（走路、動物、手） | 仍有大量對鏡頭講話的片段，產出率很低 |
| Pexels 原始長度的 stock 素材（Pexels-400k，試驗 258 支） | 內容最符合，但只有 3.5% 鏡頭夠靜（`cam_disp < 2`），走路類是 0；stock 素材幾乎都有運鏡 |

**取捨**：真實影片裡，「鏡頭固定」和「主體大範圍移動（走路、跑步）」很少同時出現，拍走路通常會跟拍。所以這個資料集以手和物體的互動為主，走路和動物大約各佔 6%。

## 已知的限制

- 約 10% 仍是對著鏡頭的人（側臉或小臉偵測不到），另有少數車內鏡頭、卡通、場景切換。
- 每支影片只取開頭一個窗口。
- caption 篩選只看關鍵字，可能誤刪內容其實很好的片段。

## 檔案

```
data/openvid_clean_set/
  windows/<name>_w0/        frames.mp4 (65 幀) tracks_yx.npy visibility.npy confidence.npy z.npy enc_idx.npy meta.json
  windows/train_windows.csv 2000 筆清單：運動統計、face_area、caption
  all_windows.csv           25,500 個候選的所有統計和被刷掉的原因
  candidates/               兩批下載的抽樣清單（可用來重抓原始影片）、第一批的臉部面積、Pexels 試驗統計
```

原始下載影片和沒被選上的窗口都已刪除。訓練只需要 `windows/` 裡的 65 幀 `frames.mp4`。

## 重現

```bash
python src/openvid_fetch_train.py --n 5500 --out data/openvid_train                       # 第一批（無 caption 預篩）
python src/openvid_fetch_train.py --clean --n 20000 --exclude data/openvid_train/selected.csv --out data/openvid_clean
python src/openvid_windows.py --root data/openvid_clean --max_face 0.02 --shard 0 --n_shards 8   # 每個 shard 一個行程
python src/openvid_select_train.py --roots data/openvid_train data/openvid_clean --clean --max_face 0.02 \
    --max_cam 2.5 --talk_frac 0 --n 2000 --out data/openvid_clean_set
python src/prep_koala.py --src data/openvid_clean_set/windows --out data/clips_a14b_openvid
```
