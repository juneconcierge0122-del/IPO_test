# FinText TSFM 評估報告

最後更新 2026-08-14。單一事實來源：結論以本文件為準；作廢結果在 `out/archive/`。
論文：Rahimikia, Ni & Wang, *Re(Visiting) Time Series Foundation Models in Finance*,
arXiv:2511.18578。checkpoint：huggingface.co/FinText（360 個）+ fintext.ai（Synthetic 版）。

---

## 0. Metric 定義

單位一律是**超額報酬** = 含息日總報酬 − 當日無風險利率（論文定義，非市場調整）。

- **R²_OOS** `= 1 − Σ(實際−預測)²/Σ(實際)²`：比「一律猜 0」好多少，**越高越好**；
  日頻個股上負值是常態（論文最佳也只有 −0.59%）
- **多空（LS）**：每日按預測值排序，買最高 1/10（TW50 為 1/5、IPO 為 1/3）、放空最低
  同比例，等權、每日再平衡（IPO 為單筆 30 日持有）、不計成本。多頭腳/空頭腳 = 兩側分拆
- **年化 / Sharpe**：`mean(LS_d)×252`；`mean/std×√252`。IPO 版以 30 日為一期（×252/30）
- **IC**：預測值 vs 實際報酬的橫斷面 Spearman；大盤 = 逐日算再平均，IPO = 逐 cohort
- **t 值** = 平均值 ÷ 標準誤（誤差條）：|t|>2 約當 5% 顯著；多模型挑最好要做多重檢定校正

---

## 1. Checkpoint 本身的驗證

### 1.1 輸入/輸出
| | TimesFM (20M) | Chronos (Small, 46M) |
|---|---|---|
| 架構 | patched decoder 9 層 | T5，報酬量化成 4093 個 token |
| 輸入 | 過去 C 日超額報酬，**32 天一個 patch** | 同左，**每天一個 token** |
| 輸出 | `(B,16,128,10)`：ch0 點預測、ch1–9 分位數 | `(B,樣本數,H)` 抽樣路徑 |
| 點預測 | ch0（僅此可用） | 樣本**平均**（=條件期望，論文取法） |
| 分位數 | **全壞**（單調率 0%） | 樣本分位數，必然單調 |

### 1.2 缺陷清單（全部由主動健檢發現，無一報錯）
1. **`self_attn.scaling` 為未初始化記憶體**（35 個 ckpt 中 29 個，1e−32~1e38）。
   softplus(0)=ln2 恰好還原標準 scaling，多數垃圾值被洗成中性 → 實測影響小
   （loader 預設 `fix_scaling=True` 歸零）。意義：訓練/匯出 pipeline 有系統性疏漏。
2. **quantile head（ch1–9）全壞**：10 通道共用骨幹、僅輸出頭分岔，ch0 正常而
   ch1–9 全滅 → 損壞精確落在 quantile 輸出頭。只用 ch0。
3. **`TimesFM_20M_2023_Augmented` 整顆發散**（預測 |0.85|/日）；其餘 Augmented
   為校準失敗（預測 std ≈ 實際 std，自信地猜錯，R² 至 −223%）。
4. **patch 退化**：context < 32 天 = 1 個 patch，attention softmax ≡ 1，
   **9 層 transformer 退化成 MLP**（實測擾動 scaling 影響 = 0.0000）。
   → 所有 30 天 context 的 TimesFM 數字不可解讀為模型能力。Chronos 不受影響。
5. **HF `Augmented` ≠ fintext.ai `Synthetic`**：逐張權重比對（Chronos_Tiny_2015）
   89/89 張量全不同 → 兩者是不同模型。論文有 4 個資料 regime
   （US/Global/JKP/Synthetic），HF 發佈前三個（`Augmented`=JKP，排除法），
   Synthetic 只在 fintext.ai（.rar，2000–2022）。官網說明文字與此矛盾，判斷為筆誤。

---

## 2. 複現論文（美股，協定：1 步 horizon、十分位、每日再平衡）

### 2.1 宇宙梯度（window 512，2017–2023，@=Sharpe）
| 宇宙 | TimesFM | Chronos | 論文（CRSP 全市場） |
|---|---|---|---|
| 973 檔混合（含中小型股）* | 年化 +64.0% @**3.06** | +28.3% @**1.60** | TimesFM +30.36% @3.66 |
| 其中前 500 流動股 | @0.26 | @−0.76 | Chronos +36.84% @5.42 |
| **S&P 500（point-in-time 成分）** | +5.6% @**0.27** | +3.9% @**0.25** | |

\* 973 檔為第一次 yfinance 下載被限流的字母截斷樣本（A–M），留作「含中小型股」對照。

### 2.2 t+2 微結構檢驗（973 宇宙、TimesFM @512）
把交易延後一天（預測不變）：+64.0% → **+20.7%**（Sharpe 3.06 → 0.91）。
**68% 的報酬是微結構收割**（買賣價差跳動＋單日反轉），非預測能力。

### 2.3 統計複現 vs 經濟結論（核心發現）
- 統計面**精確複現**：Chronos @512 R²_OOS 論文 −0.59% vs 我們 SP500 −1.0%
  （2023 單年 **−0.594%**）、前 500 檔 −0.54~−0.76%
- 經濟面**不成立**：同一模型同一協定在 S&P 500 上 Sharpe 0.25 vs 論文 5.42

分解：論文 Sharpe 5.4 = 大型股上的模型能力（**≈0.25**，實測）＋小型股集中
（宇宙梯度）＋微結構收割（t+2：68%）＋零成本假設（論文自己的敏感度：40bps 時轉負）。

---

## 3. 四宇宙成果矩陣（主要交付物）

全部：測試 2017–2023、逐年 point-in-time ckpt（測 Y 年用 Y−1 版）與
point-in-time 成分（測 Y 年用 Y−1 年底成員）。論文參照欄：美國值來自論文正文/
使用者提供的完整表；台灣僅 Sharpe 且論文未測 US 變體 × 台灣（引用時標注）。

### 3.1 TimesFM_20M_US on SPY500
| context | R²_OOS | 方向 | 年化 | Sharpe | 論文年化 | 論文 Sharpe |
|---|---|---|---|---|---|---|
| 5 | −36.0% | 49.6% | −13.1% | −0.29 | −18.22% | −1.84 |
| 21 | −17.1% | 49.6% | −10.0% | −0.43 | 10.50% | 1.24 |
| 252 | −22.0% | 51.9% | +10.8% | 0.55 | 30.50% | 3.51 |
| 512 | −4.4% | 50.8% | +5.6% | **0.27** | 30.36% | 3.66 |

### 3.2 Chronos_Small_US on SPY500
| context | R²_OOS | 方向 | 年化 | Sharpe | 論文年化 | 論文 Sharpe |
|---|---|---|---|---|---|---|
| 5 | −6.2% | 49.3% | −1.4% | −0.21 | −0.32% | −0.07 |
| 21 | −1.4% | 50.2% | −2.4% | −0.17 | 13.99% | 2.56 |
| 252 | −1.0% | 51.1% | +4.1% | 0.37 | 33.61% | 4.89 |
| 512 | −1.0% | 51.0% | +3.9% | **0.25** | 36.84% | 5.42 |

### 3.3 TimesFM_20M_US on TW50
| context | R²_OOS | 方向 | 年化 | Sharpe | IC (avg 單年 t) |
|---|---|---|---|---|---|
| 5 | −34.9% | 50.0% | −4.2% | −0.33 | −0.023 (−1.7) |
| 21 | −17.1% | 50.8% | +8.2% | 0.68 | +0.022 (1.7) |
| 252 | −18.1% | 47.8% | +13.0% | 1.17 | +0.023 (1.9) |
| 512 | −4.5% | 51.2% | **+26.0%** | **2.01** | **+0.059 (4.6)** |

### 3.4 Chronos_Small_US on TW50
| context | R²_OOS | 方向 | 年化 | Sharpe | IC (avg 單年 t) |
|---|---|---|---|---|---|
| 5 | −3.6% | 51.0% | +0.2% | −0.01 | +0.004 (0.4) |
| 21 | −1.6% | 50.1% | +11.4% | 1.03 | +0.019 (1.9) |
| 252 | −1.8% | 50.0% | +12.6% | 1.14 | +0.024 (2.3) |
| 512 | −1.8% | 50.3% | +12.6% | 1.18 | +0.021 (1.9) |

（SPY500 的 IC：TimesFM @512 = +0.015 (1.3)、Chronos @512 = +0.009 (1.1) — 全不顯著。）

### 3.5 IPO（單一設定：上市第 1–30 天 context → 預測第 31–60 天；三分位）
| | TimesFM_TW | Chronos_TW | TimesFM_US | Chronos_US |
|---|---|---|---|---|
| IPO 數 | 286 | 286 | 1,017 | 1,017 |
| R²_OOS | −13.6% | −1.7% | −21.1% | −8.1% |
| 方向 | 53.0% | 52.2% | 47.2% | 48.1% |
| IC | −0.037 | +0.097 | +0.052 | +0.059 |
| 年化* | +30.5% | +35.9% | +20.8% | +8.6% |
| Sharpe* | 0.95 | 2.70 | 0.88 | 0.58 |

\* 每 cohort 一筆 30 日交易 ×(252/30) 年化，僅 7 個觀測。TimesFM_TW 出現
IC 為負、Sharpe 為正的指標打架 → 報酬非來自模型（靠 2020/2022 兩年運氣）。
Chronos_TW 內部一致（IC 正的年份賺、負的年份虧）但為 6 變體之最佳，
全樣本（2009–2025、17 cohort）+Bonferroni 後 p≈0.09，**不顯著**。
**結論：兩市場、所有變體，IPO 上均無可用訊號。**

### 3.6 Synthetic-Augmented（fintext.ai 版）on 台股
tw50：
| context | TimesFM_Syn Sharpe | 論文（台灣、同變體）| Chronos_Syn Sharpe (IC, t) |
|---|---|---|---|
| 252 | **0.83** | **0.85** | 0.82 (+0.022, 2.1) |
| 512 | **0.98** | **0.96** | **1.69** (+0.030, 2.8) |

台股 IPO：TimesFM_Syn IC +0.016 / Chronos_Syn IC −0.059 — 仍無訊號。

美股（補齊四宇宙）：
| @512 | R²_OOS | 年化 | Sharpe | IC |
|---|---|---|---|---|
| SPY500 × TimesFM_Syn | −35.9% | −0.6% | **−0.04** | 0.006 |
| SPY500 × Chronos_Syn | −1.6% | +3.0% | 0.18 | 0.009 |
| US IPO × TimesFM_Syn | −25.7% | +20.7% | 0.98 | +0.032 |
| US IPO × Chronos_Syn | −2.0% | +17.8% | 0.84 | +0.086 |

四個要點：(a) **同市場同變體的論文對照（tw50），差距 <0.02** — 第二次量級精確複現；
(b) Synthetic 沒有比 US 變體強（tw50 @512：TimesFM 0.98 vs US 版 2.01）；
(c) Synthetic 校準正常（Chronos_Syn R² −1.6%），反襯 HF Augmented（JKP）的匯出異常；
(d) **論文自稱美國最強的 TimesFM-Synthetic 在 S&P 500 上 Sharpe −0.04** —
    四個變體在大型股上無一存活，「效益只在中小型股」對所有變體閉環。

---

## 4. 總結論

1. **checkpoint 是真的、統計行為完全如論文所述**（R² 兩度對到小數點）
2. **經濟效益在可交易宇宙上不存在**：大型股（美台）Sharpe 0.25–1.18；
   論文級數字只出現在含中小型股＋零成本＋每日換手的設定，且 2/3 是微結構
3. **IPO 完全沒有訊號**（兩市場 × 所有變體 × 校正後）— 本專案原始問題的最終答案
4. **唯一過多重檢定的正結果**：TimesFM_20M_US × TW50 × 512 的 IC 0.059
   （單年 t≈4.6、62% 天數為正、與 Sharpe 2.01 相互印證）。為事後發現，
   需 2024–2026 樣本外驗證才可當真
5. 視窗單調性（越長越好）在三個宇宙重現，是這批 ckpt 最穩健的性質；
   其中 5/21 天列對 TimesFM 是退化區間（論文的對應列亦然）

## 5. 限制（適用於本文件所有數字）
- 全部**未扣交易成本**、假設可無限量每日放空（與論文同）
- 美股 panel 來自 yfinance（snapshot 2026-08-13/14），**無已下市股票** →
  存活者偏誤，IPO 實驗最嚴重；S&P 500 成分中 93 檔已下市者缺價格
- 成分為「年度 point-in-time」（年中的季度調整不跟）
- IPO 僅 7 個 cohort 觀測；台股 IPO 的最佳變體不過校正
- TW50 五分位（50 檔）、IPO 三分位 — 與論文的十分位不同

## 6. 資料與環境
| 項目 | 來源 / snapshot |
|---|---|
| 美股價格 | yfinance，3,515 檔 A–Z（`data/us_panel.npz`，2026-08-14 重建；973 檔舊版存 `us_panel_973broken.npz`） |
| 台股價格 | finlab `~/finlab_db/etl#adj_close.feather`（snapshot 2026-07-04） |
| 美股 rf | Ken French 日頻 RF |
| 台股 rf | 央行隔夜拆款利率（`scripts/tw_riskfree.py`；勿用重貼現率，差一個量級） |
| S&P500 成分 | Wikipedia 現行名單+歷史異動表回滾（2016–2022 年底，`data/sp500_membership.json`） |
| TW50 成分 | zh.wikipedia 同法（每年恰 50 檔，`data/tw50_membership.json`） |
| Synthetic ckpt | fintext.ai .rar（2016–2022，`data/synthetic/`；TimesFM 的 rar 內多一層目錄） |

環境：conda `fintext`。兩個機器陷阱（pip user-site 滲透、CUDA compat 595/535 錯配）
見 activate.d 與 `python-env-pip-user-quirk` memory；GPU V100 32GB，
TimesFM ~10k 筆/s、Chronos ~110 筆/s（20 樣本）為一切排程的瓶頸。

## 7. 腳本地圖
```
fintext_tsfm.py     loader（fix_scaling、短 context、本地路徑）
replicate_paper.py  美股協定複現（--lag t+2 檢驗、--data 指定 panel、metrics 含 IC）
run_sp500.py        SPY500 point-in-time 成分
run_tw50.py         TW50 point-in-time 成分（五分位、--ckpt-dir 支援 Synthetic）
run_ipo.py          IPO 美台共用（30→31-60、三分位、--ckpt-dir）
build_us_panel.py   美股 panel（加固：小 chunk+退避+字母 gate+原子寫入）
build_sp500_panel.py / build_tw_ipo_panel.py / tw_riskfree.py
market_backtest.py  台股 0050 擇時（⚠️ 輸出含 test_year="ALL" 混合列，聚合時先濾掉）
```

## 8. 下一步候選
1. TW50 TimesFM@512 訊號的 **2024–2026 樣本外驗證**（finlab 資料已涵蓋）
2. 交易成本敏感度（論文用 10/20/40bps 三檔）
3. Chronos SPY500 的 5/21/252 IC 補齊（各 ~80 分鐘 GPU）
4. 寫信向作者確認 HF `Augmented` 的身分與 quantile head 問題
