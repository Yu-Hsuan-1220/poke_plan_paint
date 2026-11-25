# Wan2.2-I2V-A14B poke-to-video 微調計畫

> 2026-11-17。依據：隊友的 `專案說明.md`、proposal v2、隊友程式碼 `/project/spencer/ZipMo-Poke2Video/teammate_code/zipmo_poke_code/src/`、ZipMo 原始碼 `/project/YuHsuanLee/long-term-motion/`、本機已下載的 `Wan-AI/Wan2.2-I2V-A14B-Diffusers`。
> 文中「已確認」是讀過程式碼或實際算過；「估計」是還沒在機器上量過的數字，Phase 0 要實測。

## 目錄

0. 結論摘要
1. 隊友文件審查
2. z 與稠密軌跡怎麼注入影片模型（5B 現況 → 14B 設計）
3. 14B 的資源估算
4. 分階段計畫
5. 評估協定的調整
6. 風險與備案
7. 需要和隊友確認的事

---

## 0. 結論摘要

- **隊友的方法和程式整體合理**：ZipMo 的使用方式（z 正規化、encode→decode、poke 格式、座標）和 Pexels 評估協定都對得上官方程式，我逐一核對過。
- **但結果的解讀有幾個漏洞**，最重要的一個：結果表缺少「完全不動」的參考值。我用官方 GT 算出來，**每個點都待在原地**的預測就能拿到 EPE 19.56、MSE 85.5。Baseline 0、Baseline 1（EPE 約 25.7）和 `tracks_zm_d2`（21.45）都**比完全不動還差**；最好的 `tracks_v3`（11.37）才是明顯有效的。
- **14B 不能直接沿用 5B 的注入方式**：A14B 的 VAE（16 ch、空間 8 倍壓縮）、首幀條件的給法（36 ch 輸入 = 雜訊 16 + mask 4 + 條件 16）、兩個 expert 輪流去噪，都和 5B 不同。
- **建議的 14B 注入方式**：沿著軌跡搬移首幀 latent，**寫進 A14B 原本就有的 16 ch 首幀條件通道**（Wan-Move 的做法，Wan-Move 本身就建在 Wan-I2V-14B 上）；位移和佔據遮罩另外經一個 zero-init Conv3d 相加進去。在 320x576 下，A14B 的 latent 格點剛好是 40x72，也就是 5B 實驗裡讓 EPE 降到 11.37 的「2x guide」解析度，等於直接拿到細格點。
- **先做不用訓練的 14B 對照**：本機的 `/project/YuHsuanLee/TTM` 已經能用 A14B 跑 Time-to-Move。把首幀沿 ZipMo 軌跡 warp 成參考影片交給 TTM，一行訓練都不用就有「ZipMo + 14B」的數字。
- **先只訓練 high-noise expert**：A14B 的 transformer 負責 t ≥ 900（已確認：`boundary_ratio = 0.9`），運動的大結構在這段決定。只訓練這個 expert，成本減半；之後再看 low-noise expert 要不要訓練。

---

## 1. 隊友文件審查

### 1.1 正確、已核對的部分

| 項目 | 核對結果 |
|---|---|
| z 的正規化 | `LatentAdapter.normalize`（`motion_cond.py:182`）和 ZipMo `planner.normalize_latents`（`zipmo/planner.py:275`）一致：shift −0.175、乘 √0.5/√11.6 |
| 訓練與推論都走 encode → decode | `prep_data.py:216` 解碼 z_gt，推論時解碼 planner 的 z，同一條路，符合 proposal |
| 首幀的影片 latent 只含第一幀 | 對，Wan 的 VAE 是因果式，第 0 個 latent 幀只看第 0 幀 |
| 「adapter 相加等於擴充輸入 channel」 | 對。patch embedding 和 adapter 都是同 kernel、同 stride 的線性 Conv3d，相加等於把 48 + 51 ch 疊起來做一個大 Conv3d |
| latent 幀對應的軌跡步 | `latent_frame_steps`（`motion_cond.py:53`）：第 f 個 latent 幀取影片第 4f−3..4f 幀的中心，65 幀時影片第 j 幀 = 軌跡第 j 步。正確 |
| Pexels 評估協定 | poke = 前 n 條 GT 軌跡的起終點、seed 43、官方 `compute_metrics_atm`、GT 時間內插到影片長度。我確認過 Pexels 影片是 64 幀、12–15 fps、1280x720，和訓練時 65 幀、約 12 fps 對得上 |
| 座標換算 | `bench_pexels.py` 的 `orig_to_crop` / `crop_to_orig` 處理了 1280x720 → 320x576 的裁切（高度只裁約 9 px） |

### 1.2 需要修正或補充的地方（依重要性排序）

**(a) 缺「完全不動」的參考值，Baseline 的意義要重新解讀**

用官方 `gt_tracks.pt`（68 段）算出，所有點都停在起點時：

| 預測 | EPE（1 poke） | EPE（4 poke） | MSE |
|---|---|---|---|
| 完全不動 | **19.56** | 16.30 | **85.5** |
| Baseline 0 | 25.66 | | 83.2 |
| Baseline 1 | 25.75 | | 94.3 |
| `tracks_zm_d2` | 21.45 | | 98.6 |
| `tracks_c` | 15.98 | | 49.5 |
| `tracks_v3` | 11.37 | | 47.8 |

（完全不動時 K 段都一樣，所以 Min MSE = Mean MSE = 85.5。）

- Baseline 0/1 的 EPE 比「不動」差 6 點，代表影片有在動，但被戳的點往錯的方向走（或 CoTracker 在生成影片上漂移）。
- `tracks_c` 的 EPE 只比「不動」好 18%，`tracks_v3` 好 42%。報告裡「ZipMo 的貢獻」應該拿「不動」當地板來講。
- **做法**：在 `bench_pexels.py eval` 加一行 `static` 條件（GT 起點複製 64 步），每張結果表都放。不需要重新生成影片。

**(b) 2.6 節「Min MSE 47.8 接近 ZipMo-only 44.7，代表影片模型照著 ZipMo 的計畫畫」這個推論不成立**

兩個數字都是和 **GT** 比，不是彼此比。兩個不同的軌跡可以和 GT 一樣遠，卻彼此差很多。而且 oracle（給真實運動）的理想值應該接近 ZipMo VAE 的重建誤差（論文報告重建準確度約 97%），實際卻是 49.5，比 planner 自己的 44.7 還差，這反而說明影片模型這一段損失很大。

- **做法**：加一個「照畫程度」指標：生成影片裡追到的 40 條軌跡，和**同一個 k 的 planner 軌跡**比 MSE / EPE。`bench_pexels.py` 已經把每個 k 的 planner 軌跡存在 `extra.pt` 的 `zipmo_tracks`，生成影片的追蹤結果存在 `tracked.pt`，**現有輸出就能算，不用重跑**。oracle 也一樣：和 VAE 解碼的 GT 軌跡比。

**(c) `tracks_v3` 的 11.37 不能只歸功於「細格點 guide」**

依 6.1 節的表，`tracks_c` 是 550 段、49 幀（`prep_data.py` 的預設 `N_FRAMES = 49`）；`tracks_v3` 是 540 段、**65 幀**、40x72 guide、遮擋 splat。四件事同時變了。49 幀時軌跡要在 64 步之間內插，評分時 GT 也被內插到 49 步；65 幀則是一對一。幀數本身可能就是改善的主要來源。

- **做法**：補一個「65 幀 + 20x36 guide、無遮擋」的版本，才能分開歸因。對 14B 來說這個問題比較小，因為 14B 在 320x576 下本來就是 40x72 格點（見第 2 節）。

**(d) 方案 A 沒有用到 4 個 z_gt 的資料增強**

`prep_data.py:216` 只解碼 `z[:1]`，`dec_tracks` 永遠來自第 0 個 z_gt。4 個 z 只在方案 B（`train.py:105`）被隨機抽。文件 4.2 節和 proposal 都把增強寫成通用設計，實際上方案 A 沒有。14B 的前處理要解碼全部 z。

**(e) 統計顯著性**

每個設定只訓練一次、68 段測試影片，EPE 差 1–2 點可能只是雜訊（例如 `tracks_c` 15.98 對 oracle 16.27）。`bench_pexels.py` 的輸出已經有 `per_clip`，可以做**同一段影片成對的 bootstrap 信賴區間**，報告時只宣稱信賴區間不跨 0 的差異。

**(f) planner 的 K 段和其他條件的 K 段意義不同**

planner 的 4 段影片來自 4 個不同的 z（運動不同）；oracle、line、none 的 4 段是**同一個條件換 4 個 render seed**（`bench_pexels.py:131-141`）。Min MSE 取的是「4 段中最好的」，兩種情況的多樣性來源不同，直接比 Min MSE 對 planner 有利。比 Mean MSE 和 EPE（取平均）比較公平。

**(g) Baseline 1 偏弱**

`line_tracks`（`motion_cond.py:79`）把起點半徑 0.12 內的格點整塊直線平移，其他全部不動；訓練時只給 1 個 poke（`clip_poke(top_frac=0.05)`），評估時有 1 和 4 個 poke。它和 Baseline 0 幾乎一樣，表示模型學會忽略這個條件。報告時要寫明這是「簡單的稀疏 poke 條件」，不是 MOFA-Video 那種稀疏轉稠密的強 baseline。

**(h) 小錯**

3.2 節說「timestep 等條件是加到 token 上的」。Wan 的 timestep 是經 adaLN 調變（`scale_shift_table` 產生的 scale / shift / gate）進入每個 block，不是加到 token 上。residual 相加的說法是對的。

### 1.3 程式本身

沒有看到會讓結果失效的 bug。訓練用固定 prompt embedding（`data/prompt_embeds.pt`），motion CFG 的 unconditional 分支是「同 prompt、不給 guide」，設計一致。

---

## 2. z 與稠密軌跡怎麼注入影片模型

### 2.1 5B 的現況（隊友程式）

**方案 A：稠密軌跡 → guide → 相加到 patch embedding 之後**

```text
z_gt [256,16] ──ZipMo decoder（720 或 2880 個格點中心）──► 稠密軌跡 [N,64,2]
                                                              │ sample_tracks_at：取每個 latent 幀對應的軌跡步
首幀影片 latent [48,20,36] ──────────────────────────────────►│ build_track_guide：雙線性 splat
                                                              ▼
                                guide [51(+1), 17, 20, 36]  ← 48 特徵 + 2 位移(x4) + 1 佔據 (+1 重疊數)
                                                              │ TrackAdapter：Conv3d(k=1x2x2)，zero-init
                                                              ▼
Wan patch_embedding(x_t) ──(+)── 經 forward hook 相加（motion_cond.py:218）──► 30 層 transformer
```

- 程式：`build_track_guide`（`motion_cond.py:101`）、`TrackAdapter`（`:161`）、`MotionConditioner._patch_hook`（`:218`）。
- 第 0 個 latent 幀的 guide 直接設為首幀 latent（`:153`）。
- 首幀本身的條件是 TI2V 的作法：把 x_t 的第 0 個 latent 幀換成乾淨的首幀 latent，並讓那些 token 的 timestep = 0（`train.py:211-214`）。

**方案 B：z → MLP → 接在 text token 後面**

```text
z [256,16] ──normalize──► Linear(16→1024) + 可學位置 ──► MLP ──► [256, 4096]
                                                                   │ 經 forward pre-hook 接到 encoder_hidden_states 後面
text token [512, 4096] ─────────────────────────────────────────── concat ──► 每層 cross-attention 讀
```

- 程式：`LatentAdapter`（`motion_cond.py:172`）、`_text_hook`（`:223`）。結果是幾乎沒學會。

### 2.2 A14B 和 5B 的架構差異（已確認：讀過本機 config 與 diffusers 0.37 的 pipeline）

| 項目 | Wan2.2-TI2V-5B（隊友） | Wan2.2-I2V-A14B（我們） |
|---|---|---|
| transformer | 1 個，30 層，hidden 3072 | **2 個 expert**，各 40 層，hidden 5120（40 heads × 128），FFN 13824 |
| expert 切換 | 無 | `transformer` 負責 t ≥ 900（high noise），`transformer_2` 負責 t < 900（`boundary_ratio = 0.9`） |
| VAE | Wan2.2 VAE：48 ch，空間 /16 | **Wan2.1 VAE：16 ch，空間 /8**，時間一樣 /4 |
| 320x576、65 幀的 latent | [48, 17, 20, 36] | **[16, 17, 40, 72]** |
| patch | 1x2x2 → 10x18 token/幀 | 1x2x2 → **20x36 token/幀**，17 幀共 12,240 token |
| 首幀怎麼給 | 換掉 x_t 的第 0 幀，那些 token 的 timestep 設 0 | **通道拼接**：輸入 36 ch = x_t 16 + mask 4 + 條件 latent 16 |
| 條件 latent 的內容 | | VAE 編碼 `[首幀, 0, 0, …, 0]` 這段影片；mask 只在第 0 幀為 1 |
| text | UMT5，4096 維 | 相同 |
| 磁碟大小 | | 每個 expert 54 GB（fp32 存），bf16 載入約 28 GB |

條件 latent 的組成見 diffusers `pipeline_wan_i2v.py` 的 `prepare_latents`（`video_condition = cat([image, zeros])` → VAE encode → 正規化，再和 4 ch mask 拼接）。

### 2.3 A14B 的注入方式選項

**I1：Wan-Move 式，寫進條件 latent（不加任何新參數）**

```text
條件 latent 原本：[首幀 latent, E0, E0, …, E0]      E0 = 「全黑幀」編碼出的 latent
改成：            [首幀 latent, G1, G2, …, G16]
                   Gf = 首幀 latent [16,40,72] 沿軌跡 splat 到第 f 個 latent 幀；沒有東西搬過來的格子填 E0
mask 4 ch：維持不變（只有第 0 幀為 1）
```

- 好處：A14B 預訓練時已經學過「條件通道裡的 latent 特徵代表那個位置該長什麼樣」，這條路直接重用，資料少時最容易學。不加新參數，兩個 expert 都能直接吃同一份條件。
- 前例：Wan-Move（NeurIPS 2025）就是在 Wan-I2V-14B 上這樣做。
- 細節：沒被覆蓋的格子要填 **E0**（全黑幀的 latent），不能填 0，否則和預訓練的分佈不一樣。E0 對每段影片都一樣，算一次就好。
- mask 不改成 1：mask = 1 對模型的意思是「這格的內容是確定的」，但搬過來的特徵只是指引，不是硬性約束。

**I2：隊友的 adapter 相加（5B 方法直接移植）**

- `TrackAdapter(in_ch=16+3, hidden=5120, patch=(1,2,2))`，在 40x72 格點上直接做，輸出 20x36 token，zero-init，相加到 `patch_embedding` 輸出。
- 兩個 expert 的 token 空間不同，**每個 expert 各要一個 adapter**。
- 好處：和 5B 實驗設計一致，可以直接比較。

**I3：混合（建議的主方法）**

- 搬過來的特徵寫進條件 latent（I1）；
- 位移 2 ch + 佔據 1 ch（+ 遮擋時的重疊數 1 ch）經一個 zero-init Conv3d 相加（I2 的通道部分）。
- 理由：I1 只能放 16 ch 特徵，沒有地方放「往哪裡移、這格有沒有東西」；這些資訊交給 adapter。

**方案 B 在 14B 上**：`LatentAdapter` 的 text_dim 是 4096，14B 一樣，可以直接用。但 5B 已經證明它學不起來，14B 的訓練又貴；**建議只在時間有剩時跑一次**，結果照實報告即可。

### 2.4 兩個 expert 怎麼處理

- 訓練時兩個 expert 的 timestep 範圍不重疊，可以**分開訓練**，每次只載入一個，記憶體只要一份 14B。
- **E-high**：只訓練 `transformer`，sigma 只抽在 [0.9, 1]。low-noise expert 不訓練、吃原本的條件（只有首幀）。推論時運動的大結構在 high-noise 段決定，low-noise 只補細節。
- **E-both**：兩個都訓練（各自的 LoRA + adapter）。
- **E-high + I1 給 low-noise**：low-noise expert 不訓練，但條件 latent 用 I1 的版本。這是分佈外的輸入，可能造成畫面瑕疵，當作一個小實驗。
- 推論時的 timestep 分配：shift 5、30 步時，約 36% 的步數 t ≥ 900（sigma = 5u/(1+4u) ≥ 0.9 ⇔ u ≥ 0.643），大約 11 步走 high-noise expert。

### 2.5 格點和時間對齊

- 320x576 下 A14B 的 latent 是 40x72，所以 ZipMo 在 **2880 個格點中心**解碼（就是 5B 的 `dec_tracks_hr`）。Mixkit 的 `--hires` 前處理已經有這份軌跡，但只來自 `z[0]`。
- 65 幀 = 17 個 latent 幀，影片第 j 幀 = 軌跡第 j 步，沿用 `latent_frame_steps`。
- A14B 原生是 16 fps、81 幀；我們用約 12 fps、65 幀。微調會適應這個差異，但 Phase 0 要先看零樣本在 65 幀、320x576 下畫質是否可接受。

---

## 3. 14B 的資源估算

> 以下都是**估計**（A6000 bf16 實際約 70 TFLOPS 估），Phase 0 第一件事就是實測。

| 項目 | 估計 |
|---|---|
| 訓練記憶體（1 個 expert bf16 + LoRA r32 + gradient checkpointing，12,240 token，latent 與 prompt 事先算好） | 約 38–44 GB，A6000 48 GB 可以放下但很緊 |
| 訓練速度 | 每步約 15–25 秒（batch 1） |
| 2,000 步 | 約 8–14 小時／expert |
| 推論（30 步，只有 high-noise 段做 motion CFG，約 41 次 forward） | 約 4–5 分鐘／段影片（含 expert 在 CPU/GPU 間切換） |
| Pexels 全量（68 段 × K=4 × 1 poke = 272 段） | 約 18–23 GPU 小時；3 張卡平行約 6–8 小時 |
| 開發用子集（20 段 × K=4 × 1 poke = 80 段） | 約 6 GPU 小時；3 張卡約 2 小時 |

- 目前機器上 10 張 A6000，約 3 張閒置。
- 兩個 expert bf16 加起來 56 GB > 48 GB，推論時要把不用的 expert 放 CPU（機器有 1 TB RAM）。比較省的做法：一批影片先全部跑完 high-noise 段，存下中間 latent，再換 low-noise expert 跑完。
- 記憶體不夠時的備案，依序：(1) 解析度維持 320x576（別升到 480x832，token 會變 2.2 倍）；(2) block swap（把部分 block 放 CPU）；(3) base 權重以 fp8 儲存、bf16 計算（A6000 沒有 fp8 tensor core，只省記憶體不加速）。

---

## 4. 分階段計畫

時程對齊 proposal：12/6 是方式 A 的檢查點，12/14–12/20 寫報告。今天 11/17。

### Phase 0：環境與實測（11/18，半天到一天）

- 環境：`ttm` env 已有 diffusers 0.37、torch 2.9、peft、accelerate，能載入 A14B。需要另外裝 CoTracker3（評估用）、確認 ZipMo 權重路徑（隊友 env `zipmo-p2v` 有）。
- 實測：載入 high-noise expert（bf16），12,240 token 下跑一步 forward + backward，記錄峰值記憶體和秒數。若超過 46 GB，照第 3 節的備案。
- 零樣本畫質：原版 A14B 在 320x576、65 幀、固定 prompt 下，對 5 張 Pexels 首幀生成，肉眼看畫質。若太差，改用 384x672（latent 48x84）。
- 評估程式：在 `bench_pexels.py eval` 加 `static` 參考值和「照畫程度」指標（1.2 節 a、b），先用隊友現有 5B 的輸出跑一遍，更新 5B 的結果表。

**完成條件**：有記憶體與速度的實測數字；決定解析度；評估程式能出 static 和照畫程度。

### Phase 1：不用訓練的 14B 對照（11/18–11/20）

| 條件 | 做法 | 回答 |
|---|---|---|
| A14B 原版 | 首幀 + 固定 prompt，不給運動 | 14B 的 Baseline 0（不微調） |
| TTM + 直線 poke | 首幀依 `line_tracks` warp 成參考影片 → TTM dual-clock | 不用 ZipMo、不用訓練 |
| **TTM + ZipMo** | 首幀依 planner 的稠密軌跡 forward-warp（逐像素，40x72 軌跡上採樣）→ TTM | 不用訓練就把 ZipMo 接到 14B |

- 用 `/project/YuHsuanLee/TTM/run_wan_batch.py`（已支援 A14B、單卡 CPU offload）。要寫的只有「軌跡 → warp 影片 + mask」，mask 取「有在動的像素」。
- 先在開發子集（20 段）上跑。

**完成條件**：三列數字。如果 TTM + ZipMo 已經明顯好過「不動」，這本身就是報告的一個結果，也是訓練版要超越的對象。

### Phase 2：14B 資料前處理（11/19–11/22，可和 Phase 1 平行）

每段影片要新算的東西（ZipMo 的軌跡、z 可以沿用）：

| 欄位 | 形狀 | 來源 |
|---|---|---|
| `latents_w21` | [16, 17, 40, 72] | Wan2.1 VAE 編碼 65 幀（320x576） |
| `cond_first` | [16, 40, 72] | 就是 `latents_w21[:, 0]` |
| `dec_tracks_40x72` | [N_z, 2880, 64, 2] | **每個** z_gt 都在 40x72 格點解碼（修正 1.2 d） |
| `E0` | [16, 40, 72] | 全黑幀的 latent，全資料集共用一份 |

- **資料來源**：
  - Koala：`/project/spencer/ZipMo-Poke2Video/data/processed/koala/<id>/` 已有 `frames.mp4`、`tracks_yx.npy`、`z.npy`。要確認 `frames.mp4` 的幀數和解析度，裁成 320x576。
  - Mixkit：隊友的 `.pt` 只存了 5B 的 latent 和首幀，**沒有原始畫面**，要向隊友拿原始影片（`data/raw/mixkit`）才能重新用 Wan2.1 VAE 編碼。
- 主訓練集：先用 `tracks_zm_c` 那 453 段（和 5B 能直接比），Koala 的 `koala_train_v1` 第二輪加入。
- 存檔估計：每段約 latent 2.5 MB + 軌跡（4 × 2880 × 64 × 2 × fp16）約 3 MB。

**完成條件**：視覺檢查 10 段：把 `dec_tracks_40x72` 畫回影片，線跟著物體走；把 splat 後的條件 latent 用 VAE 解碼，看起來像「首幀被拖著走」。

### Phase 3：訓練 high-noise expert（11/22–11/28）

1. **過擬合測試**（半天）：8 段影片、I3、500 步。看 loss 下降，且用訓練影片的 oracle 條件生成時，物體照軌跡動。這一步失敗就先除錯，不要上全量。
2. **主實驗 `w14_I3_high`**：453 段、I3、只訓練 `transformer`、sigma ∈ [0.9, 1]、LoRA r32（to_q/k/v/out，self 和 cross attention 都有，和 5B 一致）、adapter 和 LoRA 的 lr 都是 1e-4、條件 dropout 10%、2,000 步，每 500 步存檔。
3. 每 500 步在開發子集上評估（planner 條件、1 poke、K=4），畫 EPE 對步數的曲線，決定要不要加步數。

**完成條件（12/6 檢查點的判準）**：開發子集上 planner 條件的 EPE 明顯低於 static（19.56）和 5B 的 `tracks_c`（15.98）。

### Phase 4：low-noise expert 與注入方式的消融（11/29–12/5）

用 3 張卡平行：

| run | 改什麼 | 回答 |
|---|---|---|
| `w14_I3_both` | 加訓練 `transformer_2` | low-noise expert 要不要訓練 |
| `w14_I1_high` | 只寫條件 latent，無 adapter | 位移 / 佔據通道的貢獻 |
| `w14_I2_high` | 只有 adapter，條件 latent 維持原版 | 「重用預訓練條件通道」有沒有比較好 |
| `w14_poke_high` | Baseline 1：`line_tracks` 經 I3 | ZipMo 在 14B 上的貢獻 |

推論端的小消融（不用重訓練）：motion CFG 只在 high-noise 段 vs 全程；CFG 2 / 3 / 4。

### Phase 5：資料（12/1–12/8，視時間）

- `w14_koala_high`：Koala 資料、同配方，和 Mixkit 版比。若兩組數量差很多，補一個隨機抽成同數量的版本（隊友文件 6.3 的待決定事項）。
- 鏡頭規則要統一才能公平比較（同樣是 6.3 的待決定事項）。

### Phase 6：最終評估與報告（12/7–12/20）

- 最終入選的 3–4 個設定跑 Pexels 全量（68 段、K=4、1 和 4 poke）。
- 每張表都放：static、ZipMo-only、Baseline 0/1、TTM + ZipMo、14B 主方法、5B 最佳版本。
- 指標：EPE、Min/Mean MSE、照畫程度、成對 bootstrap 信賴區間。
- 質性圖：同一個 poke，5B vs 14B vs TTM，各放首幀 + 軌跡 + 中間幀 + 末幀。

---

## 5. 評估協定的調整

- **開發子集**：固定 20 段 Pexels（用 seed 抽，名單存檔），只跑 1 poke、K=4。最終設定才跑全量。
- **新增兩個參考與指標**：static（地板）、照畫程度（生成軌跡 vs 條件軌跡）。
- **K 段的公平性**：planner 用 4 個 z；oracle / line / none 的 4 段只差 render seed，報告時說明，主比較用 EPE 和 Mean MSE。
- **畫質**：Pexels 評估量不到畫質。5B 的 v3 在舊測試集上出現格紋、FVD 變差。14B 至少要做肉眼檢查，時間允許就在 Pexels 上算首幀 LPIPS 和 FVD（68 段太少，FVD 只能當參考）。

---

## 6. 風險與備案

| 風險 | 影響 | 應對 |
|---|---|---|
| 1 個 expert 訓練放不進 48 GB | 不能訓練 | Phase 0 就量；block swap；fp8 儲存；縮短到 49 幀（13 個 latent 幀） |
| 每步太慢，2,000 步跑不完 | 實驗數量變少 | 只做 E-high；消融改成 1,000 步；多卡 DDP（每張卡一份 expert，batch 3） |
| 320x576 對 A14B 太低，畫質差 | 結果難看 | Phase 0 先看；改 384x672 |
| 14B 推論太慢，評估跑不完 | 對照不齊 | 開發子集；CFG 只在 high-noise 段；批次切換 expert |
| low-noise expert 把運動「修」掉 | 只訓練 high 不夠 | Phase 4 的 `w14_I3_both` |
| Mixkit 原始影片拿不到 | 只能用 Koala | 主訓練集改成 Koala，5B 對照改用隊友的 Koala 版 |
| GPU 被別人佔用 | 時程延後 | 依隊友的規則只用閒置 ≥ 1 分鐘的卡；排程用腳本 |

---

## 7. 需要和隊友確認的事

1. Mixkit 原始影片在哪裡？（`data/raw/mixkit`，不在共享路徑上）
2. 隊友 5B 的輸出（`outputs/bench/*/tracked.pt`、`extra.pt`）在哪裡？要拿來算 static 和照畫程度，更新 5B 的表。
3. 14B 要不要用和 5B 一模一樣的訓練集（453 段），還是直接用 Koala？建議先用 453 段，比較才乾淨。
4. 鏡頭規則要統一成哪一套？
5. GPU 分配：14B 訓練每個 run 要一整張 A6000 跑 8–14 小時，要和隊友協調。
