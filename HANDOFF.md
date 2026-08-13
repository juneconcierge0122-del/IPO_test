# FinText TSFM × IPO / 市場預測 — 交接文件

最後更新 2026-08-13。這份文件記錄 checkpoint 的性質、環境的坑、已完成的實驗與結論。
新 session 從這裡讀起。

---

## 0. 環境（一定要先讀）

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fintext
cd ~/IPO_test
```

conda env `fintext`：python 3.11 / torch 2.7.1+cu126 / transformers 4.51.3 /
chronos-forecasting 1.5.2 / timesfm 1.2.7（`--no-deps`，只為了拿它的 torch decoder）。

兩個**必須知道**的環境陷阱：

1. **`~/.pip/pip.conf` 設了 `install.user = true`**。在 conda env 裡 `pip install` 會裝到
   `~/.local/lib/pythonX.Y/site-packages`，而且那個 user-site 還會**滲透進 conda env**。
   裝套件一律用：
   ```bash
   PIP_USER=0 PYTHONNOUSERSITE=1 ~/miniconda3/envs/fintext/bin/pip install ...
   ```
   env 已經設好 `PYTHONNOUSERSITE=1`（`conda env config vars`）。
   `~/.local` 的 python3.12 那棵樹已經壞掉了（缺 joblib、torchao .so symbol error），別用。

2. **CUDA driver 版本錯配**。kernel module 是 535.161.08，但
   `/etc/ld.so.conf.d/00-cuda-compat.conf` 讓 `libcuda.so.1` 優先解析到
   `/usr/local/cuda/compat/lib` 裡的 **595.58.03**，導致 context 建不起來
   （`cuDevicePrimaryCtxRetain()=999`）。因為它在 ldconfig cache 裡，**清空
   `LD_LIBRARY_PATH` 沒有用**，必須把真 driver 目錄放到最前面。
   已寫進 `envs/fintext/etc/conda/activate.d/00-cuda-driver-fix.sh`：
   ```bash
   export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
   ```
   `conda activate fintext` 後 V100 32GB 正常。**其他 env 和系統 python 沒修，會踩到同樣的雷。**

GPU：Tesla V100-SXM2-32GB（sm_70，只用 float32）。

---

## 0.5 論文協定（先讀，我的實驗有偏離）

論文：Rahimikia, Ni & Wang, *Re(Visiting) Time Series Foundation Models in Finance*,
arXiv:**2511.18578**（2025-11）。它定義了這些 checkpoint 的正確用法：

- **excess return = 含息日總報酬 − 當地日無風險利率**，`r_ex = (S_t + D_t − S_{t−1})/S_{t−1} − r_f`。
  **不是市場調整**。（本專案 `--returns exrf` 就是這個。）
- **模型是 one-step-ahead**：學的是 `P(r_ex_{t+1} | r_ex_{t−C+1..t})`，點預測取
  條件**期望值（樣本平均）**，不是中位數。
- 論文研究的 window 是 5 / 21 / 252 / 512 天，**window 越長越準**；釋出的 ckpt context_len = 512。
- `Augmented` = **JKP-augmented** = 全球超額報酬 + Jensen-Kelly-Pedersen (2023) 的 153 個因子。
  **不是合成資料**。（論文提到兩種 augmented，但 HF 只發佈一種，無法確定是哪個。）
- 論文自己的成績：Chronos-small 從頭訓練在 window 512 是 **R²_oos = −0.59%**、方向正確率
  ~51.7%；經濟結果（年化 36.84%、Sharpe 5.42）來自**每日再平衡的橫斷面 long-short、
  用次日預測、且未計交易成本**。

> **⚠️ 因此：R²_oos 為負是預期中的，不是失敗。** 本文件 §3 裡「沒有模型的 R²_oos > 0」
> 這件事本身不構成負面結論 —— 要看的是它跟 −0.59% 這個基準比起來如何。

> **⚠️ 我的實驗偏離協定兩處**：(1) 用 30 步 / 20 步 horizon，而非 1 步；
> (2) Chronos 取樣本**中位數**而非平均。所以 §3.1 / §3.2 的結果**不是對模型能力的裁決**，
> 只是對「在這個非標準設定下能不能用」的回答。要下結論請照協定重跑。

## 1. FinText checkpoint 的真相

HuggingFace `FinText/*` 共 **360 個 repo**：

| 家族 | 尺寸 | 年份 | 變體 |
|---|---|---|---|
| TimesFM | 8M / 20M | 2000–2023 | Global / US / Augmented |
| Chronos | Tiny / Mini / Small(46M) | 2000–2023 | Global / US / Augmented |

- **年份 = 訓練截止點**（資料 1990→該年）。測 Y 年就用 `min(2023, Y-1)` 的 ckpt。
- 訓練資料是 **excess return**，餵報酬序列，不要餵價格。
- `Global` = 全球、`US` = 美股、`Augmented` = 資料增強。

已下載到 HF cache（`~/.cache/huggingface/hub`）：TimesFM_20M 與 Chronos_Small 的
2007–2023 × 3 變體共 102 個，加上零星 2023/2019 的其他尺寸。

### 1.1 TimesFM 載入方式（重要）

`config.json` 寫 `architectures: ["TimesFMForHF"]`，但 **transformers 沒有這個類別**，
repo 裡也沒附 modeling 程式碼。實際權重就是 google-research `timesfm` 1.x 的
`PatchedTimeSeriesDecoder`，只多一層 `model.` prefix。loader 在
`scripts/fintext_tsfm.py`，`strict=True` 完全對上。

參數：`patch_len=32`、`context_len=512`、`horizon_len=128`、
`use_positional_embedding=True`（實測比 False 誤差低）。

### 1.2 已知的 checkpoint 缺陷

| 問題 | 證據 | 影響 |
|---|---|---|
| **quantile head 壞掉** | 9 個分位數單調比例 **0%**，值到 ±3.0（日報酬 300%） | 只能用 channel 0（point head）；要分布請用 Chronos |
| **`self_attn.scaling` 是未初始化記憶體** | 掃過的 35 個 TimesFM ckpt 有 **29 個**最大值到 1e27~1e38，對照 `qkv_proj.weight` std≈0.03 正常。只有 `TimesFM_20M_2019_Global`、`TimesFM_8M_2023_Global` 看起來真的訓練過 | `load_timesfm(fix_scaling=True)`（預設）把它歸零，`softplus(0)=ln2=1/1.442695`，還原成標準 `1/sqrt(head_dim)`。**context < 64 天時無影響**（見 1.3） |
| **`TimesFM_20M_*_Augmented` 數值發散** | 30 天 context 下 blowup（\|預測日報酬\|>50%）達 **15~96%**，R²_oos 到 −1.9 萬 % | 排除，或至少 clip |

### 1.3 TimesFM 的 patch 退化（實驗設計的關鍵限制）

`patch_len=32`，所以 context 少於 32 天時只有 **1 個有效 patch**，attention 的 softmax 對
單一位置做 = 恆等於 1，**9 層 transformer 退化成一個 MLP**。實測：

| context | 有效 patch | 擾動 scaling 參數對輸出的影響 |
|---|---|---|
| 30 天 | **1/16** | **0.0000** |
| 64 天 | 2/16 | 0.4380 |
| 128 天 | 4/16 | 0.4892 |
| 256 天 | 8/16 | 0.4239 |

**任何 context < 64 天的 TimesFM 結果都不能解讀成「模型能力不足」** —— 它根本沒在運作。
Chronos 沒這個問題（逐時點 token，30 天 = 30 個 token）。

### 1.4 兩個模型的原始輸出

**TimesFM**：`forward()` → `(B, 16 patches, 128 steps, 10 channels)`。
channel 0 = point，1~9 = q10~q90。單位與輸入相同（前處理用 context 的 mean/std 標準化，
輸出反轉回來）。`decode()` 取最後一個 patch 的前 H 步；H≤128 時**不需自迴歸**。

**Chronos**：T5 語言模型，把報酬量化成 token（4093 個 bin 均勻切在 mean-scaled 的
[−15,+15]；實測 1 bin ≈ 0.018% 日報酬、可表達 ±36.5%，解析度不是瓶頸）。
`predict()` → `(B, num_samples, H)`，是**同一條序列的 N 條可能路徑**（不是 N 檔股票），
點預測是你自己取的統計量。因此它的分位數必然單調 —— 這是它勝過 TimesFM 的結構性原因。

---

## 2. 資料

| 用途 | 來源 | 檔案 |
|---|---|---|
| 台股還原股價 | finlab | `~/finlab_db/etl#adj_close.feather`（4719 天 × 2759 檔，2007-04-23→2026-07-03） |
| 上市/上櫃日期 | TWSE + TPEx openapi | `build_tw_ipo_panel.py` 內建抓取 |
| 無風險利率 | **央行金融業隔夜拆款利率** | `data/overnight.csv`（Big5 需 iconv）+ `data/gap_*.html` 補到今天 |

**無風險利率別用重貼現率**：重貼現率是政策利率（1.125~2.0%），實際隔夜拆款只有
0.08~0.83%，差一個量級（台灣銀行間結構性資金寬鬆）。取得方式見 `scripts/tw_riskfree.py`
docstring。CSV 只更新到 2026-03-18，之後用 CBC 的分頁 HTML 補。

### 2.1 IPO panel

`scripts/build_tw_ipo_panel.py` → `data/tw_ipo_panel.npz`（908 檔，2008–2026）。
IPO 日 = 面板中第一個非 NaN 日（與 TWSE/TPEx 上市日中位數差 0 天、93% 在 7 天內）。
篩選：4 碼純數字（排除 ETF `00xx`、TDR `91xx`、權證）、2008-07 之後上市（避開左設限）、
首日價 ≥ 1 元、前 61 個交易日跨度 ≤ 150 天。

存了四種報酬定義（`ctx_*` / `hor_*`，各 30 天）：

| key | 定義 | 與 `ret` 相關 | 平均絕對差 |
|---|---|---|---|
| `ret` | `(P_t − P_{t−1} + 股利) / P_{t−1}`（adj_close 已含股利） | — | — |
| **`exrf`** | `ret − 日無風險利率` ← **FinText 的定義** | **1.00000** | 0.000066 |
| `exmkt` | `ret − 橫斷面中位數報酬` | 0.95507 | 0.005350 |
| `retpt` | 分母改用 `P_t` | 0.99644 | 0.001073 |

**`exrf` 實質上等於原始總報酬**（日無風險 0.007%，日報酬 std 3.3%）。真正有差別的是 `exmkt`。

---

## 3. 實驗與結論

### 3.1 IPO setup A：前 30 天 → 後 30 天（`run_experiment.py`）

881 檔、17 個 point-in-time cohort（2009–2025）。指標見腳本 docstring。

**核心結果 — 報酬定義決定結論**：

| 模型 | `exmkt` IC (t) | **`exrf` IC (t)** | 正 IC 年份 (exrf) |
|---|---|---|---|
| Chronos_Small_Augmented | 0.139 (3.39) | **0.061 (1.50)** | 53% |
| Chronos_Small_Global | 0.087 (1.81) | **0.074 (2.05)** | 76% |
| Chronos_Small_US | 0.069 (1.86) | **0.079 (2.71)** | 71% |
| TimesFM_20M_Global | −0.019 (−0.45) | −0.071 (−1.55) | 35% |
| TimesFM_20M_US | −0.032 (−0.89) | −0.068 (−1.41) | 29% |
| baseline:momentum | 0.012 (0.26) | −0.037 (−0.68) | 35% |

> **最初報告的「Chronos IC 0.142、t=3.24」大半是市場調整造成的假象。**
> 改用 FinText 自己的定義（`exrf`）後只剩 `Chronos_Small_US` t=2.71 撐住，
> 而這是 6 個變體中挑出來的 → Bonferroni 後 p≈0.09，**不顯著**。
> 目前的立場應該是：**IPO 上沒有證實的訊號**。

其他：
- 沒有任何模型的 R²_oos > 0（Chronos_Small_US 最接近，−0.6%）。**注意這其實正好落在論文
  自報的 −0.59% 上**（見 §0.5），所以不是失敗訊號。
- TimesFM 全滅，但受限於 1.3 的 patch 退化，**這不是公平比較**。

### 3.2 大盤 walk-forward（`market_backtest.py`）

0050，context 512 天、horizon 20 天、每 5 天重測；14 個 ckpt 年 × 各自後續所有年份。
TimesFM 在這裡有 16 個 patch 全滿，是公平比較。

**(a) 擇時完全打不贏買進持有** —— 配對檢定（每個 ckpt 年一個觀測）：

| 模型 | 擇時 − B&H | 勝場 | p |
|---|---|---|---|
| Chronos_Small_Global | −2.65pp | 2/14 | 0.000 |
| Chronos_Small_US | −4.43pp | 1/13 | 0.001 |
| TimesFM_20M_Global | −4.83pp | 2/14 | 0.003 |
| TimesFM_20M_US | −3.56pp | 2/14 | 0.022 |

模型 **64~100% 的時間都在看多**，二值化的「持有/空手」規則把連續訊號的資訊丟光了。
（曾經看到 2021 ckpt 去均值規則 Sharpe 2.01，攤開 14 個 ckpt 年後是 **0~1 勝、落後
10~14pp**，確認是事後挑規則的假象。）

**(b) 但方向性有微弱真訊號** —— 預測 vs 實際 20 日累積報酬的 Spearman：

| 模型 | 平均 corr | t | 正值年份 |
|---|---|---|---|
| Chronos_Small_Global | +0.114 | 3.85 | 13/14 |
| Chronos_Small_US | +0.093 | 3.92 | 11/13 |
| TimesFM_20M_Global | +0.053 | 3.84 | 11/14 |
| TimesFM_20M_US | +0.052 | 2.92 | 12/14 |

四個都顯著為正，**Chronos 約為 TimesFM 的兩倍**，且這是在 TimesFM 的主場測的。

**(c) ckpt 新舊沒有差別**：corr vs 模型年齡 rho = +0.02~+0.13，全部不顯著（p 0.13~0.81）。
2010 年的 ckpt 測 2026 跟 2023 年的 ckpt 測 2026 一樣好。
同一測試年、跨 ckpt 年的 corr 標準差（0.09~0.29）**大於模型平均 corr（0.11）**
→ 「哪一年測試」的影響遠大於「用哪個 ckpt」。point-in-time 選 ckpt 是防 look-ahead 的
必要衛生，但別期待新 ckpt 比較準。

### 3.3 美股：照論文協定複現（`build_us_panel.py` / `replicate_paper.py`）

這是**唯一照論文協定跑的實驗**（1 步 horizon、條件期望、十分位每日再平衡多空）。
資料：yfinance 3,886 檔現存美股共普通股、2014-06→2023-12，無風險利率用 Ken French
日頻 RF。測試年 2017–2023，逐年 point-in-time。

**(a) TimesFM_20M_US，全市場（剔除成交金額後 5%），window 掃描：**

| window | R²_OOS | acc | 年化 LS | Sharpe | 論文 |
|---|---|---|---|---|---|
| 5 | −28.2% | 49.31% | −42.4% | −1.62 | −18.22% |
| 21 | −14.1% | 49.86% | +13.6% | 0.69 | — |
| 252 | −21.0% | 49.95% | +50.1% | 2.31 | — |
| **512** | **−3.9%** | **50.58%** | **+64.0%** | **3.06** | **+30.36% / Sharpe 3.66** |

**論文的核心型態重現成功**：window 越長越好的單調關係、以及「5 天為負、512 天為正」
的兩個端點都對上。絕對值偏高，與存活者偏誤方向一致（yfinance 沒有已下市股票，
而下市的多半最差 → 系統性美化空頭腳）。

> ⚠️ 但依 §1.3，window 5 / 21 的 TimesFM 處於 patch 退化區間。論文自己的 −18.22%
> 也在同一區間，所以我重現到的是**同一個假象**，不是模型在短視窗下的真實能力。

**(b) 效益全部來自小型／低流動性股票** —— 同樣 window 512、同樣協定，只把宇宙
換成流動性前 500 檔：

| 模型 | 宇宙 | R²_OOS | 年化 LS | Sharpe |
|---|---|---|---|---|
| TimesFM_20M_US | 全市場 ~3.9k | −3.9% | +64.0% | 3.06 |
| TimesFM_20M_US | 前 500 檔 | −0.9~−6.7% | −28~+31% | −0.72~2.32 |
| Chronos_Small_US | 前 500 檔 | **−0.54~−0.76%** | **−2.6~−17.8%** | −0.27~−1.37 |

**Chronos 的 R² 幾乎完美命中論文的 −0.59%，但多空報酬是負的。** 換句話說，
統計面複現得很好，經濟面在大型股上完全不成立。這跟論文自述「小型股可預測性較高」
一致，但反過來說：**論文那個 Sharpe 5.42 依賴的正是流動性最差、交易成本最高的那一段**。
Chronos 全市場尚未跑（3.9k 檔 × 7 年 ≈ 17 GPU 小時），是最該補的一塊。

**(c) `Augmented` 是校準壞掉、不是檔案壞掉**：預測日報酬 std 0.010~0.043，
而實際 std 約 0.03~0.04；US/Global 只有 0.004~0.013。也就是它「很有自信地猜錯」，
R²_OOS 被打到 −223%。正弦波健檢下 17 個年份只有 2023 真的發散，其餘數值正常。

---

## 4. 檔案地圖

```
scripts/
  fintext_tsfm.py         loader：load_timesfm() / load_chronos()（含 fix_scaling）
  tw_riskfree.py          央行隔夜拆款利率 → 日無風險利率
  build_tw_ipo_panel.py   台股 IPO panel（四種報酬定義）
  run_experiment.py       IPO setup A 逐年評估（--returns exrf|exmkt|ret|retpt）
  market_backtest.py      大盤 walk-forward（ckpt 年 × 測試年）
  inspect_raw_output.py   兩個家族的原始輸出解剖
  smoke_test.py           ckpt 載入 + 推論健檢
  check_signal_source.py  IC 是否只是動能（寫好但未跑）
  build_us_panel.py       美股 panel（非本線產出，未驗證）
  replicate_paper.py      美股橫斷面實驗（非本線產出，未驗證）
out/
  results_exrf.csv        ★ IPO 主結果（FinText 定義）
  results_exmkt.csv       IPO 對照（減市場）
  results_by_year.csv     最初那版（減市場，已被 exrf 推翻）
  market_backtest.csv     ★ 大盤全表
```

---

## 5. 下一步的候選

依重要性排序：

1. **先照協定重跑**（§0.5）：1 步 horizon、Chronos 取樣本**平均**而非中位數。現有 §3 全部
   結果都是非標準設定，不能拿來評價模型。這是所有後續工作的前提。
2. **IPO setup 60→30**：讓 TimesFM 至少有 2 個 patch，才是對它公平的比較。目前 IPO 那組
   TimesFM 全滅的結論**不可引用**。
3. **`check_signal_source.py`** 已寫好未跑：IC 是否只是動能的偽裝（rank 空間殘差化）。
   在 `exrf` 定義下跑，才知道剩下的那點訊號是什麼。
4. **大盤改用連續部位**：既然方向有訊號但二值化後消失，試 `position ∝ 預測值`
   或分位數分層，看訊號能不能轉成收益。論文的經濟結果也是橫斷面 long-short，不是擇時。
5. **交易成本**：目前全部結果都是**稅前、零成本**（論文也是）。台股 IPO 流動性差又有
   漲跌停，T3−T1 那幾個百分點很可能吃不到。
6. 若要下結論寫報告，記得 §3.1 的多重檢定問題（6 個變體 × 4 種報酬定義）。
