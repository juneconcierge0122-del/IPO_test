# FinText TSFM 評估報告

最後更新 2026-08-13。單一事實來源：實驗結論以本文件為準，`out/` 下只保留有效結果
（作廢的在 `out/archive/`，附說明）。

論文：Rahimikia, Ni & Wang, *Re(Visiting) Time Series Foundation Models in Finance*,
arXiv:2511.18578。checkpoint：huggingface.co/FinText（360 個 = {TimesFM 8M/20M,
Chronos Tiny/Mini/Small} × 訓練截止年 2000–2023 × {Global, US, Augmented}）。

---

## 0. Metric 定義

單位一律是**超額報酬** = 含息日總報酬 − 當日無風險利率
（`r_ex = (S_t + D_t − S_{t−1})/S_{t−1} − r_f`，論文的定義；**不是**減市場報酬）。

**R²_OOS**（Gu et al. 2020）：

```
R²_OOS = 1 − Σ(實際 − 預測)² / Σ(實際)²
```

分子 = 預測誤差平方和；分母 = 「一律預測 0」的誤差平方和。衡量「比猜 0 好多少」，
>0 才算有預測力。基準用 0 而非歷史平均，因為個股歷史平均雜訊太大，用它會讓爛模型顯得好。
**在日頻個股報酬上，R²_OOS 為負是常態**：論文自己最好的成績也只有 −0.59%。

**多頭腳／空頭腳**：每個交易日把當天所有股票依**預測值**排序，取 k = n/10：

```
多頭腳_d = (預測最高 k 檔的實際超額報酬總和) / k        ← 買進
空頭腳_d = −(預測最低 k 檔的實際超額報酬總和) / k       ← 放空（負號=股票跌則賺）
多空_d   = 多頭腳_d + 空頭腳_d
```

排序用預測值、計酬用實際值；等權重、每日再平衡、不計交易成本。

**年化與 Sharpe**：

```
年化 = mean(多空_d) × 252
Sharpe = mean(多空_d) / std(多空_d) × √252
```

多空是零成本自融資部位，本身已是超額報酬，不再減 r_f。

**IC**（IPO 實驗用）：預測累積報酬 vs 實際累積報酬的橫斷面 Spearman 相關。

---

## 1. 先驗證模型本身（美股複現）

### 1.1 兩個家族的輸入／輸出

| | TimesFM (20M) | Chronos (Small, 46M) |
|---|---|---|
| 架構 | patched decoder，9 層 | T5 seq2seq，報酬量化成 token |
| 輸入 | 過去 C 日超額報酬，**32 天切一個 patch** | 過去 C 日超額報酬，**每天一個 token** |
| 輸出 | `(B, 16 patch, 128 步, 10 通道)`；ch0=點預測、ch1–9=分位數 | `(B, num_samples, H)`：同一序列的 N 條抽樣路徑 |
| 點預測 | ch0 | 抽樣路徑的**平均**（論文取條件期望） |
| 分位數 | **壞掉**（單調比例 0%） | 由抽樣得到，必然單調 |

模型是 **one-step-ahead**：學的是 `P(r_{t+1} | r_{t−C+1..t})`。載入方式見
`scripts/fintext_tsfm.py`（TimesFM 的 `TimesFMForHF` 不是 transformers 類別，
實際權重是 google timesfm 1.x 的 `PatchedTimeSeriesDecoder` 加 `model.` prefix）。

### 1.2 `self_attn.scaling` 未初始化記憶體 — 是什麼問題

掃過 35 個 TimesFM ckpt，**29 個**的 `self_attn.scaling` 最大值落在 1e27–1e38，
帶重複位元組樣式與其他張量的殘留值（layernorm 的 ≈1.0、linear 的 ≈0.01）
→ 這是 `torch.empty()` 配置後**從未被訓練寫入**的記憶體，不是權重。

為什麼理論上嚴重：forward 計算 `q × 1.442695/√d × softplus(scaling)`，
softplus(1e38)≈1e38 會讓 attention logits 爆掉。
為什麼實際影響小：`softplus(0) = ln2 = 1/1.442695`，歸零剛好還原標準 `1/√d` scaling，
而多數 entry 的有效值本來就近 0；實測歸零前後正弦波 MAE 只差 0.6314→0.6343。
loader 預設 `fix_scaling=True` 歸零處理。

**結論：釋出的權重不乾淨（訓練 pipeline 沒存這個參數），但不是結果失效的原因。**
另外 context < 32 天時只有 1 個 patch、attention softmax 恆等於 1，這個參數完全無作用
（實測擾動影響 = 0.0000）—— 也因此 **context < 32 天的 TimesFM 是一個退化成 MLP 的模型**，
任何短 context 結果都不能解讀為模型能力。

### 1.3 有問題的 checkpoint

| checkpoint | 症狀 | 處置 |
|---|---|---|
| `TimesFM_20M_2023_Augmented` | 預測日報酬平均 \|0.85\|（實際~0.013）、逐層 hidden 200–640 vs 同族 ~4 | **排除** |
| 全部 `TimesFM_*_Augmented` | 預測 std 0.010–0.043 vs 實際 0.03–0.04：自信地猜錯，R² 到 −223% | 標註校準失敗 |
| 全部 TimesFM | quantile head 單調比例 0% | 只用 ch0 |
| 全部 TimesFM | §1.2 的 scaling 問題 | `fix_scaling=True` |

（論文提到 JKP-augmented 與 synthetic-augmented 兩種增強，HF 只發佈一種 `Augmented`，
無法確定對應哪個 —— 論文說增強對 TimesFM 幫助最大，與我們觀察到的相反。）

### 1.4 複現設計

- **固定**：horizon = 1 天（照協定）。逐年 point-in-time：測 Y 年用 min(2023, Y−1) 的 ckpt
- **變動**：context C ∈ {5, 21, 252, 512}
- 每個 (C, 年份, 模型) 算 R²_OOS、方向準確率、十分位多空

### 1.5 測試資料（如實標註）

- 來源：yfinance（snapshot **2026-08-13**），`nasdaqtraded.txt` 過濾 ETF/測試股/權證後的共普通股
- 無風險利率：Ken French 日頻 RF（與論文 CRSP 系列同源）
- 測試期間：2017-01-01 ~ 2023-12-31
- **⚠️ 資料缺陷（重要）**：下載被 Yahoo 限流，實得 **973 檔**且依字母截斷
  （A 331 / B 187 / C 268 佔 81%，**N–Z 完全沒有**）。先前文件寫的「3,886 檔」是錯的。
  首字母與報酬應無系統關聯，可視為 ~25% 的準隨機子樣本，但每個十分位只剩 ~80 檔
  → 十分位報酬雜訊放大。**不能宣稱代表美股市場。**
- **存活者偏誤**：yfinance 無下市股票，會系統性美化空頭腳（論文用 CRSP 含下市報酬）

### 1.6 複現結果

TimesFM_20M_US（973 檔子集、剔除成交金額後 5%、2017–2023 七年合計）：

| context | R²_OOS | 方向準確率 | 多空年化 | Sharpe | 多頭腳 | 空頭腳 | 論文對照 |
|---|---|---|---|---|---|---|---|
| 5 | −28.2% | 49.3% | −42.4% | −1.62 | −0.2% | −42.2% | 年化 −18.22% |
| 21 | −14.1% | 49.9% | +13.6% | 0.68 | +24.0% | −10.4% | — |
| 252 | −21.0% | 49.9% | +50.1% | 2.31 | +46.2% | +3.9% | — |
| **512** | **−3.9%** | **50.6%** | **+64.0%** | **3.06** | **+58.3%** | **+5.7%** | **+30.36% / Sharpe 3.66** |

**✅ 論文核心型態重現成功**：
1. context 越長越好的單調關係（Sharpe −1.62 → 0.68 → 2.31 → 3.06）
2. 兩個端點正負號與論文一致（5 天賠錢、512 天賺錢）
3. 多頭腳主導（58.3 vs 5.7），與論文 "long leg consistently outperforms short leg" 一致

（注意 C=5, 21 的 TimesFM 處於 §1.2 的 patch 退化區間，論文的 −18.22% 也在同一區間，
所以連「短 context 很爛」這件事本身都復現了 —— 但那是退化行為，不是模型能力。）

流動性前 500 檔、2017–2023 七年合計（完整）：

| window | 模型 | R²_OOS | 多空年化 | Sharpe |
|---|---|---|---|---|
| 512 | Chronos_Small_US | **−1.40%** | **−14.7%** | −0.76 |
| 512 | TimesFM_20M_US | −4.79% | −1.5% | 0.26 |
| 5 | Chronos_Small_US | −3.53% | −8.4% | −0.51 |
| 5 | TimesFM_20M_US | −30.9% | −10.5% | −0.39 |

（Chronos 逐年 R²_OOS 落在 −0.54% ~ −4.50%，2017–2019/2022 四年都在 −0.76% 以內，
幾乎命中論文的 −0.59%。）

**在流動性前 500 檔上，兩個模型的多空報酬全負或近零。**
同一協定下 TimesFM 從 973 檔全樣本（Sharpe 3.06）換到前 500 檔掉到 0.26。
→ 多空報酬**全部來自排名 500–973 的較不流動股票**。
（先前描述為「3,886 vs 500」是錯的；實際對比只有 973 vs 其中前 500。）

### 1.7 兩個未結案的疑點

1. **多頭腳 +58%/年 可能是微結構假象**：等權重＋每日再平衡＋收盤價會機械性吃到
   買賣價差跳動與短期反轉，低價小型股尤甚。檢驗方式：預測目標從 t+1 改 t+2（跳一天），
   大幅衰減即證實。**未跑。**
2. **逐年 vs 合併推論**：把多個測試年混在一起算相關會吸收「年度之間」的成分，
   系統性高估（台股上已證實會翻轉結論，見 §3）。美股目前是逐年算的，安全；
   但 ckpt 2020 逐年測 2021/2022/2023 vs 合併 2021–23 的正式對照**未跑**。

---

## 2. 美股 IPO（前 30 天 → 後 30 天）— 未執行

現有 973 檔 panel 中各年新上市（以 panel 首個有效報酬日認定）：
2015: 27、2016: 17、2017: 31、2018: 37、2019: 28、2020: 47、2021: 103、2022: 28。
單年最多 103 檔 → 十分位不可行，只能做 IC 或三分位。
30 天 context 對 TimesFM 是退化區間（§1.2），需同時做 60→30 版本才公平。

---

## 3. 台股大盤（0050 擇時）

資料：`~/finlab_db/etl#adj_close.feather`（snapshot **2026-07-04**，4719 天 × 2759 檔，
2007-04-23 → 2026-07-03，本地完整）。無風險利率：央行**隔夜拆款利率**
（`scripts/tw_riskfree.py`；別用重貼現率，差一個量級）。

設計：context 512 → horizon 20，每 5 天重測；14 個 ckpt 年 × 各自之後所有測試年。
這裡 TimesFM 有 16 個 patch 全滿，是對它公平的比較。

**(a) 擇時打不贏買進持有**：四個模型都輸 2.7–4.8pp（p 0.000–0.022，14 年只贏 1–2 年）。
模型 59–64% 時間看多，二值化持有/空手把訊號丟光。

**(b) 方向性訊號（已修正）** — 預測 vs 實際 20 日累積報酬的 Spearman，
**逐年計算、按 ckpt 年聚合**：

| 模型 | corr | t | 正值 ckpt 年 |
|---|---|---|---|
| Chronos_Small_Global | **+0.079** | +2.57 | 10/14 |
| Chronos_Small_US | **+0.053** | +2.32 | 9/13 |
| TimesFM_20M_Global | **−0.074** | −5.27 | **0/14** |
| TimesFM_20M_US | −0.077 | −3.90 | 1/14 |

> ⚠️ 本文件前一版寫「四個模型都顯著為正」，**是錯的** —— 那是用
> `market_backtest.csv` 裡 `test_year="ALL"` 的混合列算的（把該 ckpt 所有年份的預測
> 混在一起算相關，吸收了年度間成分：TimesFM 混合 +0.053 → 逐年 **−0.074**）。
> 交易在特定時點發生，吃不到跨年度水準差異，逐年才有意義。
> **修正後結論：台股大盤上 Chronos 有微弱正向方向訊號，TimesFM 是顯著反向。**

**(c) ckpt 新舊無差別**：corr vs 模型年齡全部不顯著（p 0.13–0.81）；
同一測試年跨 ckpt 的 corr 標準差（0.09–0.29）大於平均 corr（0.11）。
point-in-time 是防 look-ahead 的衛生要求，不是準確度來源。

**尚未做**：台股權值股**橫斷面十分位多空**（對得上論文協定的那種）。0050 擇時
不是論文的用法，論文的經濟效益全部來自橫斷面。

---

## 4. 台股 IPO（前 30 天 → 後 30 天）— 已完成

881 檔（2009–2025）、17 個 point-in-time cohort，`out/results_exrf.csv`。

| 模型 | IC | t | 正 IC 年份 |
|---|---|---|---|
| Chronos_Small_US | 0.079 | 2.71 | 71% |
| Chronos_Small_Global | 0.074 | 2.05 | 76% |
| Chronos_Small_Augmented | 0.061 | 1.50 | 53% |
| TimesFM_20M_Global | −0.071 | −1.55 | 35% |
| TimesFM_20M_US | −0.068 | −1.41 | 29% |
| baseline:momentum | −0.037 | −0.68 | 35% |

**結論：沒有證實的訊號。**
- 唯一撐住的 Chronos_Small_US 是 6 個變體挑出來的 → Bonferroni 後 p≈0.09
- 最初報告的「IC 0.14、t=3.2」來自誤用市場調整報酬（`out/archive/`），已作廢
- TimesFM 全負但處於 patch 退化區間（§1.2），**不是公平比較**；需 60→30 版本
- 台股 IPO 另有漲跌停、承銷制度等因素，模型未見過

---

## 5. 環境（操作前必讀）

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fintext && cd ~/IPO_test
```

1. **pip 陷阱**：`~/.pip/pip.conf` 設了 `install.user=true`，conda env 裡裝套件一律
   `PIP_USER=0 PYTHONNOUSERSITE=1 ...`。env 已設 `PYTHONNOUSERSITE=1`。
2. **CUDA 陷阱**：kernel driver 535 vs compat lib 595 錯配，
   `activate.d/00-cuda-driver-fix.sh` 已把 `/usr/lib/x86_64-linux-gnu` 前置。
   其他 env / 系統 python 沒修。
3. GPU：V100-SXM2-32GB（sm_70，float32）。TimesFM 推論 ~10k 筆/秒；
   Chronos ~110 筆/秒（20 樣本）是瓶頸。

## 6. 檔案地圖

```
scripts/
  fintext_tsfm.py         loader（fix_scaling、短 context 處理）
  tw_riskfree.py          央行隔夜拆款 → 日無風險利率
  build_tw_ipo_panel.py   台股 IPO panel（四種報酬定義）
  build_us_panel.py       美股 panel（⚠️ 有限流問題，見 §1.5）
  run_experiment.py       台股 IPO 逐年評估（--returns exrf）
  replicate_paper.py      美股論文協定複現
  market_backtest.py      台股 0050 walk-forward（⚠️ 輸出含 test_year="ALL" 混合列，勿直接用）
  其他                    健檢與解剖工具
out/
  results_exrf.csv        ★ 台股 IPO 主結果
  results_exmkt.csv       台股 IPO 對照（減市場，用於展示定義敏感性）
  market_backtest.csv     ★ 台股 0050 全表（聚合時先濾掉 test_year=="ALL"）
  us_timesfm.csv          ★ 美股 TimesFM 複現（973 檔子集）
  us_chronos.csv          美股 Chronos 前 500 檔（進行中）
  archive/                作廢結果 + 說明
```

## 7. 下一步

1. t+2 檢驗（§1.7-1，成本極低）— 判斷 Sharpe 3.06 的多頭腳是否微結構假象
2. 逐年 vs 合併的美股正式對照（§1.7-2）
3. 台股權值股橫斷面十分位多空（§3，對齊論文協定）
4. 美股 IPO（§2）與台股 IPO 60→30 補測（§4）
5. 若要更強的美股宇宙：重寫 `build_us_panel.py` 加退避重試 + 覆蓋率驗證 + 原子寫入
