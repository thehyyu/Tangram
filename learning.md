# Tangram — Learning Notes

> 做中學的概念紀錄。每個概念只記「是什麼」和「為什麼重要」，不複製文件。

---

## 環境

### uv
Python 套件管理工具，用 `uv run script.py` 執行程式，不需要手動 activate `.venv`。比 pip + virtualenv 快，也不會污染全域環境。

---

## b0：推論基礎

### transformers
Hugging Face 核心函式庫，把模型從網路下載下來，讓你用幾行程式碼跑推論。

### pipeline
`transformers` 的高階 API，把「tokenize → 推論 → decode」三步包成一個函式。快速驗證用，不適合微調。

### MPS（Metal Performance Shaders）
Apple Silicon 的 GPU 運算介面，PyTorch 透過 `device="mps"` 呼叫，讓 Mac 的 GPU 參與運算，加速訓練與推論。

**為什麼 bitsandbytes 在 MPS 上跑不了？**

先理解兩件事：

1. **GPU 不是通用的**：NVIDIA GPU 用 CUDA 這套程式語言寫底層運算；Apple Silicon 用 Metal。兩套系統語言不同、架構不同，程式碼不能互通。

2. **bitsandbytes 是 CUDA 專用的**：它的核心是用 CUDA C++ 寫的低階 GPU 程式碼，只能跑在 NVIDIA GPU 上。就像 macOS 的 .app 沒辦法直接在 Windows 上執行一樣。

**對這個專案的影響：**

```
b0-b3（setup / 爬蟲 / SFT / LoRA）
  → 不需要 bitsandbytes → Mac mini MPS 可以跑 ✅

b4（QLoRA）
  → 需要 bitsandbytes 做 4-bit 量化 → Mac mini 跑不了 → 必須切換到 Google Colab T4（NVIDIA GPU）❌

b6（Ollama 本地推論）
  → 用 llama.cpp 的 GGUF 格式，跟 bitsandbytes 無關 → Mac mini 可以跑 ✅
```

**一句話總結**：bitsandbytes 是 NVIDIA 專屬工具，Apple 的 GPU 看不懂它，所以 4-bit 訓練只能外包給 Colab。推論（b6）不需要它，所以 Mac 跑得動。

---

## b0b：曳光彈

### Tracer Bullet（曳光彈）
出自《The Pragmatic Programmer》。指在正式開發前，用最少的材料打通系統的整條主幹路徑，確認架構可行。

**和 prototype 的差別**：prototype 是探索完就丟的；曳光彈是真實程式碼，後續的 branch 直接在它的基礎上擴展，不重寫。

**Tangram 的曳光彈做什麼**：跳過爬蟲，手寫 10 筆假 Q&A，跑完 format → 1-epoch 微調 → 推論，確認 fine-tune 之後模型的回答和 base model 有可見差異。這一刀切過整個系統最核心的假設，之後才值得花時間去爬 3000 筆真實資料。

### b0b 實作概覽

#### 關鍵套件
- **`transformers` + `trl`**：載入模型並使用 `SFTTrainer` 簡化微調流程。
- **`datasets`**：管理訓練資料的格式轉換與 Tokenization。
- **`torch` (MPS)**：呼叫 Mac 的 GPU 加速，使用 `bfloat16` 確保數值穩定。

#### 執行流程
1. **載入與測試**：以 `bfloat16` 載入 Llama 3.2-3B 並記錄 Base Model 的原始回答。
2. **格式化資料**：手寫 10 筆唐鳳風格 Q&A，套用 Llama 3 Chat Template。
3. **微調訓練**：執行 1 epoch 的監督式微調（SFT）。
4. **驗證對比**：再次詢問同一個問題，將結果存入 `outputs/b0b_comparison.txt` 以供比對。

#### 微調效果：Base vs Fine-tuned (b0b)
**測試問題**：請解釋什麼是開放資料，讓沒有技術背景的人也能理解。

| 版本 | 模型回答（節錄） | 風格差異觀察 |
| :--- | :--- | :--- |
| **Base Model** | 「開放資料是一種讓所有人都可以自由使用、分享和改變的資訊。想像你在圖書館裡面... 任何人都可以借來看...」 | **百科全書感**。使用較長的圖書館比喻，語氣親切但較為瑣碎。 |
| **Fine-tuned** | 「開放資料是指讓任何人都可以自由使用、分享、改變、發布的資訊... **原則是讓資訊不被某個單位或個人的控制，而是讓它成為公共資源**...」 | **唐鳳味顯現**。語氣變精煉，強調「去中心化控制」與「公共資源」等核心價值。 |

#### 訓練機制細節
- **梯度下降 (Gradient Descent)**：計算模型回答與範例的差距（Loss），微調 30 億個參數中的關鍵權重。
- **學習率預熱 (Warmup)**：前 10% 步驟緩慢提升學習率，防止初期梯度過大導致模型崩潰。
- **梯度檢查點 (Gradient Checkpointing)**：不保留所有中間運算結果，用運算時間換取記憶體空間，是 Mac 16GB RAM 跑微調的必要技術。
- **梯度裁剪 (Max Grad Norm)**：限制單次更新的最大幅度，防止數值暴衝。

### 看懂訓練成績單：鸚鵡特訓比喻

訓練時看到的大括號數據（Logs），可以想成是「教鸚鵡說話」的紀錄：

| 參數 | 白話解釋 | 趨勢 |
| :--- | :--- | :--- |
| **`loss`** | **錯誤分數**。鸚鵡模仿得「像不像」。 | 越低越好 📉 |
| **`grad_norm`** | **修正力道**。教練糾正鸚鵡的力氣。太高代表鸚鵡學到崩潰。 | 穩定較好 ⚖️ |
| **`learning_rate`** | **學習節奏**。暖身開始、衝刺、最後收尾的步伐大小。 | 動態變化 🏃 |
| **`entropy`** | **猶豫程度**。鸚鵡對下一個字有多困惑（亂猜程度）。 | 越低越好 📉 |
| **`mean_token_accuracy`** | **猜題正確率**。鸚鵡猜中唐鳳下一個字會說什麼的機率。 | 越高越好 📈 |
| **`epoch`** | **課程進度**。所有資料看過一遍就是 1 epoch。 | 穩定增加 🆙 |

**常見警告：`pin_memory`**
在 Mac 上看到這個警告代表「快速搬運工」休假。因為 Mac 是統一記憶體（大家住同一個客廳），不需要額外的搬運工。**直接忽略即可**。

### epoch（訓練輪次）
**1 epoch = 讓模型把所有訓練資料「看一遍」**。

用鋼琴練習比喻：桌上有 10 首曲子，1 epoch 就是把 10 首各彈一次；3 epochs 就是各彈三次。每「看一筆」資料，模型就根據錯誤調整一次參數（梯度下降）。

**b0b 為什麼只跑 1 epoch？**
目的只是驗證「這條路能不能走通」，不是真正學好風格。10 筆資料跑太多 epoch 反而會 overfitting（模型背答案而不是學風格）。真正的訓練在 b2-sft 用幾百筆真實演講資料跑，那時才需要觀察跑幾個 epoch 最合適。

---

## b1：資料準備

### 為什麼 Q&A pair 適合作為訓練資料？
SFT 的學習形式是「輸入 → 輸出」的對應。Llama-3.2-3B-Instruct 已經是對話模型，chat template 的結構正是 `user / assistant`。唐鳳演講逐字稿天然就是「記者問、她答」的 Q&A 格式，直接對上：

```
user:      記者的提問
assistant: 唐鳳的回答（模型要學的目標風格）
```

模型已經會對話了，Q&A pair 只是在告訴它「遇到問題，要用這種方式回答」。它學的是風格與論述結構，不是從零學對話能力。若改用沒有切分的完整演講稿，模型不知道哪裡是輸入、哪裡是輸出，學不到東西。

### 訓練集 / 驗證集 / 測試集
三份資料集各司其職，必須在 b1b 就切好，**不能等到 b5 再切**：

| 資料集 | 比例 | 用途 | 使用時機 |
|--------|------|------|---------|
| **訓練集** | 85% | 模型學習 | b2–b4 |
| **驗證集** | 10% | 訓練中監控 overfitting | b2–b4 訓練過程中 |
| **測試集** | 5% | 最終評估 | 只有 b5，訓練全程不碰 |

測試集的鐵律：**看過就污染了**。只要訓練時用到任何測試集的資料，b5 的數字就不可信。

### Data Leakage（資料洩漏）
指測試集的資訊在訓練中洩漏進去，導致評估結果虛高。

**Tangram 的具體風險**：同一場活動的 Q&A，前幾題可能在訓練集、後幾題在測試集。隨機切分無法避開這個問題——模型看過同場活動的其他問答，等於偷看過她的思路。

**解法：時間切分**，用年份區隔：
```
訓練集：2020–2024 年演講   ← 模型學習
驗證集：2025 年演講         ← 監控 overfitting
測試集：2026 年演講         ← 最終評估（從未見過）
```
不同年份的演講在內容和議題上獨立，不會互相洩漏。

### 爬蟲（requests + BeautifulSoup）
用程式自動瀏覽網頁、抽取內容。`requests` 抓 HTML，`BeautifulSoup` 解析標籤。

### 爬蟲設計原則（b1a 實戰整理）

**先看結構再動手**：爬蟲寫之前要先 fetch 幾頁看 HTML 結構，確認標籤、class、speaker href 的格式。archive.tw 的每個發言是 `<li>` 包 `<a href="/speaker/...">` + 發言文字。

**多層過濾取代單一嚴格條件**：
```
第一層：URL 含中文字元      → 過濾純英文演講
第二層：'唐鳳' in speaker  → 過濾她不在場的頁面
第三層：qa_pairs 非空      → 過濾無 Q&A 結構的頁面
```
三層各司其職，任何一層漏掉的，後面會補上。

**資料集性質決定過濾嚴謹度**：archive.tw 是有策展的封閉資料集，不是對全網爬蟲。這種情況不需要 langdetect 或 HTML lang 屬性——多層業務邏輯過濾已經夠用，加語言偵測套件是過度設計。

**重試機制值得加，但要簡單**：用 `requests.Session` + `urllib3.Retry` 五行搞定，遇到 502/503/504 自動退指數等待重試。長時間爬蟲一定要有，否則一個短暫的伺服器錯誤就會中斷整個任務。

**不需要增量爬取的判斷標準**：資料集有上限（幾百篇）且重跑代價低 → 直接覆寫 JSON 即可。如果資料集是動態成長的（新聞、社群媒體），才需要增量爬取。

**爬蟲和格式化永遠分開**：b1a 只存 raw JSON，b1b 才做解析和格式化。分開的好處：爬錯了只需重跑爬蟲，格式改了只需重跑格式化，不用兩個都重來。

### 工具建議的取捨框架（b1b 實戰整理）

外部建議不是都要加，判斷標準：**「這個改動解決的問題，在這個資料集上真的存在嗎？」**

| 建議 | 決定 | 理由 |
|------|------|------|
| tqdm 進度條 | **加** | 55k 筆 tokenization 跑 1–2 分鐘無輸出，使用者無法判斷是否當機，UX 問題真實存在 |
| 測試集資料量警告 | **加** | 2026 年只有 4 個月資料，量可能不足，b5 評估結果會失真，值得明確提示 |
| 內容去重 | **不加** | URL 去重在 b1a 已做；archive.tw 是策展資料集，同內容不會出現兩次；加了只增加複雜度 |
| 語言偵測套件 | **不加** | 三層業務邏輯過濾（URL 含中文 → 唐鳳出現 → Q&A 非空）已足夠；資料集是封閉來源，不是對全網爬蟲 |

**核心原則**：建議越「通用」，就越需要對照「這個專案的具體情境」再判斷。通用工具解決通用問題，但這個資料集有它的特殊性（有策展、有限規模、單一來源），通用做法不一定是最好的做法。

### Quality over Quantity（品質優於數量）
對 LLM 微調來說，1,000 條高品質對話的效果通常遠好於 10,000 條雜訊過多的資料。Tangram 的具體做法：過濾掉回答少於 50 個 token 的 Q&A pair——太短代表沒有完整論述，模型學不到她的說話方式。

### raw JSON 策略
原始資料先存成 JSON，讓後續格式化步驟可以重跑，不需要再爬一次。爬蟲跟格式化永遠分開做。

### chat_template
每個模型都有自己的對話格式。Llama 3 用 `<|begin_of_text|>...<|eot_id|>` 結構。`tokenizer.apply_chat_template()` 自動插入正確的特殊符號。格式錯了模型就學不到東西。

### tokenize
把文字切成模型看得懂的數字序列（token ID）。模型實際吃的是數字不是文字。

### special tokens
對話中有特殊意義的符號（對話開始、結束、角色切換）。模型靠這些分辨「現在是 user 說話還是 assistant 說話」。

### datasets（Hugging Face）
把格式化好的資料轉成可直接送進 Trainer 的格式。

---

## b2：監督式微調（SFT）

### SFT（Supervised Fine-Tuning）
用「問題 + 正確答案」配對資料，讓模型學會「遇到這類問題要這樣回答」。最基礎的微調方式。

### trl（SFTTrainer）
Hugging Face 的訓練函式庫，把 PyTorch 原本要自己寫的訓練迴圈、loss 計算全部包好。

### loss curve（損失曲線）
訓練過程中每個 step 的錯誤率變化圖。loss 持續下降 = 模型在學習。

### learning rate（學習率）與 warmup
LLM 微調的學習率要設得極小（`2e-4` 或 `5e-5`），比一般深度學習小很多。原因是模型已經有大量知識，步伐太大會把原本學好的東西蓋掉。

**Warmup steps**：訓練開始的前 10% steps 先用很小的學習率「預熱」，再逐漸升到目標值。防止訓練初期 loss 暴衝崩潰。

### max_seq_length（最大序列長度）
輸入給模型的最大 token 數。設太長：記憶體爆；設太短：長回答被截斷，模型學不到完整論述。

Tangram 從 1024 開始測試——唐鳳的回答有時很長，如果發現截斷嚴重再調高。長度加倍，記憶體占用約呈平方增長，要謹慎。

### WandB（Weights & Biases）
訓練監控工具。把 `report_to="wandb"` 加進 `TrainingArguments`，就能在瀏覽器上即時看到 train loss 和 validation loss 的曲線圖。

**為什麼不能只看最後數字**：如果 val loss 在第 3 個 epoch 就開始回升，但你跑了 10 個 epoch，最後得到的是一個過擬合的模型。WandB 讓你看到「哪一刻」該停。

### Catastrophic Forgetting 的解法
概念在 b5 學習筆記裡有說。解法是在訓練資料中混入 **5% 的通用對話資料**（如 Alpaca dataset），讓模型在學唐鳳風格的同時，不忘記原本的語言能力。

LoRA 天生比 Full SFT 更抗 Catastrophic Forgetting（因為沒有動到原始權重），但混入通用資料仍是好習慣。

### 全參數微調的記憶體需求（為什麼需要 24–32GB？）

微調一個 3B 模型時，記憶體不只是「存模型本身」而已，實際分三塊：

| 佔用來源 | 大小估算 | 說明 |
|---------|---------|------|
| 模型權重（BF16） | ~6GB | 30億 參數 × 2 bytes |
| 梯度 | ~6GB | 每個參數都要算梯度，dtype 同模型 |
| AdamW 優化器狀態 | ~12GB | 每個參數存一階矩 + 二階矩，各 4 bytes |
| 合計 | **~24GB** | 還不含激活值 |

**為什麼優化器狀態這麼大？**

AdamW 記錄的不只是「現在要往哪走」（梯度），還記錄「過去平均走了多快」（一階矩）和「過去波動有多大」（二階矩），這兩份統計讓它走得更穩、更快收斂。代價就是每個參數額外存兩個 float32 數字。

**gradient_checkpointing 解決的是哪塊？**

上表沒算進激活值——前向傳播時每一層的中間結果，這是記憶體第四大來源。`gradient_checkpointing=True` 會丟掉大部分激活值，需要時重算，讓訓練記憶體降低 60–70%，但多花約 30% 計算時間。**這是 b2 能在 Mac 上跑的關鍵之一。**

**OOM 時怎麼辦？**

直接跳 b3-lora——LoRA 只訓練約 1% 的參數，優化器狀態縮小 99%，記憶體需求從 24GB 降到 ~7GB。這也是 LoRA 在消費級硬體上普及的最主要原因。

### 訓練時間估算

b2 的訓練量：11,681 筆（Tangram） + ~584 筆（Alpaca） ≈ 12,265 筆，跑 3 epochs。

影響時間的關鍵參數：
- `per_device_train_batch_size=1`：每次只看 1 筆（記憶體限制）
- `gradient_accumulation_steps=4`：累積 4 筆才更新一次，等效 batch size = 4
- 每 epoch 約 12,265 個 forward + backward pass

**Mac mini Apple Silicon 上的粗估**：每個 step（前向 + 反向 + 梯度累積）約 1–3 秒，3 epochs 約 12,000–36,000 步，合計 **3–10 小時不等**。序列長度越長、模型越大就越慢。

**實際建議**：

```bash
# 跑之前先估一下速度：看前 50 steps 的 it/s
# WandB dashboard 的 "samples/sec" 就是這個數字
```

`it/s < 0.5` 代表非常慢，考慮：
1. 確認 `gradient_checkpointing=True` 有開
2. 縮短 `max_seq_length`（從 1024 降到 512）
3. 或接受現實讓它跑過夜

### EarlyStopping（早停機制）

**問題**：訓練太多 epoch，模型就只會「背」訓練資料，對沒見過的問題反而表現更差——這叫 overfitting。

**訊號**：train loss 繼續下降，但 **val loss 開始回升**，兩線分叉的那一刻就是過擬合開始的地方。

```
loss
  │  train ──────────────────
2 │                          ─────────
  │  val ────────────────
1 │                      ──
  │                          ↑ val 開始回升：該停了
  └──────────────────────────────── epoch
                           1    2    3
```

**b2 的設定**：

```python
EarlyStoppingCallback(early_stopping_patience=2)
# → val loss 連續 2 個 epoch 沒改善，自動停止訓練
```

搭配 `load_best_model_at_end=True`，訓練結束後自動載回「val loss 最低那個 checkpoint」，不是最後一個 epoch 的。最佳 checkpoint 存在 `checkpoints/b2/best/`。

**patience=2 的意思**：給模型兩次機會。有時 val loss 小幅回升只是暫時波動，patience=1 太敏感會過早停止；patience=2 確認連續兩次都沒改善才停。

### 梯度下降（Gradient Descent）
模型訓練的核心機制。每次看完一筆資料，計算答案有多錯（loss），然後往「讓 loss 變小的方向」調整參數。

比喻：你站在山上，目標是走到山谷（loss 最低點）。每一步都朝腳下最陡的下坡方向走一小步（learning rate），再重新判斷方向，再走一步。

**learning rate 的影響：**
- 太大 → 步伐越過山谷，loss 爆炸崩潰
- 太小 → 走太慢，或卡在半山腰的小凹陷出不來
- LLM 微調用 `2e-4`，極小——因為模型已經在山谷附近，走太大步會破壞原有知識

### batch size 與有效 batch size

`per_device_train_batch_size × gradient_accumulation_steps = 有效 batch size`

b2 的設定：`1 × 4 = 4`。

| 小 batch（1–4）| 大 batch（16–32）|
|--------------|----------------|
| 梯度雜訊多、更新頻繁 | 梯度方向穩、更新次數少 |
| 容易探索細微模式 | 收斂穩但可能過於保守 |
| 記憶體少 | 記憶體多 |

**風格微調選小 batch 的理由**：風格是「細節」不是「規律」，帶雜訊的梯度更新幫助模型探索細微語言模式，而不是只記住最顯眼的詞彙。11,681 筆資料 ÷ 有效 batch 4 ≈ 每 epoch 3,000 次更新，已足夠。

**調整時機**：看完 smoke test 的 WandB，如果 `grad_norm` 持續大幅波動，才考慮把 `gradient_accumulation_steps` 從 4 調高到 8，不需要提前改。

### gradient accumulation（梯度累積）
記憶體不夠時，把多個小批次的梯度加總才更新一次參數，效果等同更大的 batch size。

### gradient checkpointing（梯度檢查點）
解決「訓練時記憶體不夠」的問題。

正常訓練時，每一層的計算結果（激活值）都要全部保留在記憶體，等反向傳播時用來算梯度。一個 3B 模型有幾十層，全部留住記憶體吃很重。

checkpointing 的做法：只保留部分層的結果，其他的丟掉，需要時重新算。代價是多花約 30% 計算時間，但記憶體降低 60–70%。

**Mac 特別需要開**：Apple Silicon 的 RAM 和 GPU 記憶體共用同一池，開 checkpointing 是跑大模型訓練的必要手段。

---

## b3：LoRA

### PEFT（Parameter-Efficient Fine-Tuning）
不更新模型全部參數，只訓練少量附加結構，大幅降低記憶體與運算需求。

### LoRA（Low-Rank Adaptation）
在模型某些層旁邊插入兩個很小的矩陣（A、B），只訓練這兩個矩陣，原始模型權重完全不動。訓練完後 A×B 的結果可以合回原始權重。

### adapter（適配器）
LoRA 訓練出來的那組小矩陣，可以單獨儲存（只有幾十 MB），不同任務可以有不同 adapter，套在同一個 base model 上。

### LoraConfig 關鍵參數

| 參數 | 意義 | 常用值 |
|------|------|--------|
| `r`（rank） | 兩個小矩陣的維度，越大表達能力越強但記憶體用越多 | 8 或 16 |
| `lora_alpha` | 縮放係數，控制 adapter 對原始模型的影響幅度 | 通常設 r 的兩倍 |
| `target_modules` | 要在哪些層插入 adapter | 通常選注意力機制的 q、v 矩陣 |

---

## b4：QLoRA

### 量化（Quantization）
把模型權重從 32-bit 或 16-bit 浮點數壓縮成更少位元，大幅降低記憶體用量。

### QLoRA
Quantization + LoRA 的組合。先把 base model 量化成 4-bit 減少記憶體，再用 LoRA 只訓練 adapter，讓消費級 GPU 也能微調大模型。

### bitsandbytes
實現 4-bit 量化的函式庫。**目前不支援 Mac MPS，只能在 CUDA GPU 上跑** → 需切換到 Google Colab T4。

### NF4（NormalFloat 4-bit）
QLoRA 論文提出的 4-bit 量化格式，比一般 INT4 在訓練時數值更穩定。

### prepare_model_for_kbit_training()
量化後模型有些層需要特別處理才能穩定訓練，這個函式自動做完所有前置設定。

---

## b5：評估

### 初始基準（Initial Baseline）的重要性
在開始微調之前，要先記錄 base model 在測試集上的表現，這份數字就是「改進前的參考點」。沒有它，之後的數字沒有意義——不知道是進步了還是退步了。

### cross-entropy（交叉熵）
訓練時的 loss 本質上就是交叉熵：衡量模型預測的分佈與真實答案之間的差距。loss 持續下降 = 交叉熵縮小 = 模型越來越接近訓練資料的分佈。和 perplexity 的關係：`perplexity = e^(cross-entropy)`，兩者測的是同一件事，只是尺度不同。

### ROUGE
衡量生成文字與參考答案的 n-gram 重疊程度（0–1 分），值越高代表回答越接近標準答案。

**Tangram 的限制**：ROUGE 抓得到詞彙重疊，抓不到「唐鳳味」。她換個說法表達同樣意思，ROUGE 就給低分。所以 ROUGE 是輔助指標，不能當唯一標準。

### perplexity（困惑度）
模型對某段文字「有多困惑」的度量，越低代表模型越能預測這段文字的走向。

**Tangram 的用法**：在 held-out 唐鳳演講集上測 perplexity。fine-tuned 的 perplexity 低於 base model，代表模型學會了她的說話方式。

### LLM-as-a-Judge
用更強的語言模型（Claude、GPT-4）來評分，而不是只靠 n-gram 指標。

**Tangram 的做法**：對同一個問題，拿 base model 和 fine-tuned 的回答各一份，送給 judge 這樣的 prompt：
```
請評估以下回答有多像唐鳳的說話風格，1 分（完全不像）到 10 分（非常像）。
只給分數和一句理由。

回答 A：{base_model_response}
回答 B：{finetuned_response}
```
這是目前對「風格」最直接、最實用的量測方式。

### Overfitting（過擬合）與 Catastrophic Forgetting（災難性遺忘）
兩個不同但相關的風險：

| 問題 | 症狀 | 怎麼檢查 |
|------|------|---------|
| **Overfitting** | 在訓練集表現好，但測試集（沒見過的演講）表現差 | 比較 train loss 與 validation loss 的差距 |
| **Catastrophic Forgetting** | 學了唐鳳風格，但忘了原本的語言能力 | 拿少量 MMLU 題目比較 base vs fine-tuned 的正確率 |

LoRA 天生比 Full SFT 更抗 catastrophic forgetting，因為它沒有動到原始權重。

### 標準化 Benchmark 的取捨
GLUE、MMLU、GSM8K 這類 benchmark 測的是通用語言理解和數學推理，**不適合直接用來評估 Tangram**（風格微調 ≠ 知識能力）。但 MMLU 可以拿來做 Catastrophic Forgetting 的快速檢查：fine-tuned 之後分數不應該明顯下降。

---

## b6：部署

### adapter merge（適配器合併）
把訓練好的 LoRA adapter 數學上合回 base model，產生一個「已微調好的完整模型」，方便後續轉檔。

**精度陷阱**：`merge_and_unload()` 之前，base model 必須以 **FP16 或 BF16** 載入（`torch_dtype=torch.float16`）。若以 FP32 載入再合併，精度不一致會造成性能下降，且 GGUF 檔案體積也會不必要地變大。

### llama.cpp
C++ 實作的推論引擎，支援將 HuggingFace 格式模型轉換成 GGUF，並在 CPU / Apple Silicon 上高效推論。

### GGUF
llama.cpp 使用的模型檔案格式，把模型壓縮成單一檔案，Ollama 能直接讀取並在本機跑。

---

## b7：RAG

### RAG（Retrieval-Augmented Generation）
回答前先從知識庫撈出相關段落，把段落塞進 prompt，讓模型基於真實資料回答，避免捏造。

### 向量資料庫
把文字轉成數字向量儲存，查詢時找出語意最相近的段落（不是關鍵字比對，而是意思相近就能找到）。

### 本專案 RAG 工具
- `chromadb`：本地向量資料庫
- `sentence-transformers`：文字轉向量

---

## b8：發布

### HuggingFace Hub
類似 GitHub 但專門放 AI 模型的平台。上傳後有公開連結可以分享。

### model card
模型的 README，說明模型用途、訓練資料、評估結果與使用方式。

---

## 思考紀錄

> 做完一個 branch 後，把「踩到什麼坑」和「為什麼這樣解」記在這裡。

### b2（2026-04-29）

**OOM：全參數微調在 64GB M4 Pro 上仍然不夠**

實測數字：
```
MPS（訓練）：     ~27 GB（模型 BF16 + 梯度 + AdamW 優化器狀態）
其他（macOS + Python CPU 端 + apps）：~60 GB
合計：            ~87 GB → 超過 MPS 上限 88.13 GB → OOM
```

**為什麼預估錯了**：只算了 MPS 端的訓練記憶體（~24GB），忘了 macOS 本身吃 10–15GB、Python 載入模型和資料到 CPU 端又吃 15–20GB。64GB 統一記憶體是 CPU 和 GPU 共用，不是 GPU 獨享。

**解法：修改規格仍保留練習效果**

| 修改項目 | 原始規格 | 修改後 | 記憶體節省 |
|---------|---------|--------|-----------|
| 優化器 | AdamW | Adafactor | ~23GB（最大贏） |
| max_seq_length | 1024 | 512 | 激活值減半 |

Adafactor 不儲存完整的一階矩和二階矩，而是用因式分解的方式壓縮，optimizer states 從 ~25GB 降到 ~2GB。代價是收斂行為略不同於 AdamW，但「全參數更新 vs LoRA 1% 參數」這個核心對比依然成立。

**學到的原則**：估算 MPS 訓練記憶體時，要把 `系統常駐 + Python CPU 端` 一起算進去，不能只看模型本身。

**最終結論：b2 無法在此機器上執行**

三種修改都試過，仍然 OOM：

| 嘗試 | MPS 降低 | 結果 |
|------|---------|------|
| Adafactor 取代 AdamW | 27GB → 16.7GB | 仍 OOM |
| max_seq_length 512 | 小幅降低激活值 | 仍 OOM |
| low_cpu_mem_usage=True | 16.7GB → 15.8GB | 仍 OOM |

根本問題：`other allocations: 71.07 GB` 三次都完全相同，是 Python runtime + 記憶體映射虛擬記憶體的固定開銷，程式碼層面無法控制。剩餘空間只有 1.26GB，但 lm_head 的 `grad²` 每次需要 1.47GB，差 0.21GB，永遠卡在第 27 步。

**決定**：跳過 b2 實際執行，直接進 b3-lora。b2 的學習目標「全參數更新的概念與限制」已透過這次完整的排查過程體驗到，記錄在此作為 b5 比較時的背景說明。

---

### b3（2026-04-30）

**訓練完成：3 epochs，9201 steps，17.6 小時**

```
train_runtime: 6.328e4 秒 ≈ 17.6 小時
train_steps_per_second: 0.145 it/s（4.30 s/it）
```

**最終數字**

| 指標 | 訓練集 | 驗證集 |
|------|--------|--------|
| `loss` | 2.119 | 2.391 |
| `mean_token_accuracy` | 62.61% | 56.18% |
| `entropy` | 2.191 | 2.478 |

**DoD 達成**：可訓練參數 **0.071%**，遠低於 1% 門檻。

**如何讀懂最後幾行 log**

| 參數 | epoch 2.984 | epoch 3（最終）| 趨勢說明 |
|------|-------------|----------------|---------|
| `loss` | 2.207 | 2.119 | 訓練集持續下降，正常 |
| `grad_norm` | 1.215 | 1.376 | 1–2 屬健康範圍，沒有爆炸 |
| `learning_rate` | 1.946e-08 | 2.879e-11 | cosine decay 末段，幾乎歸零——學習接近停止，符合預期 |
| `entropy` | 2.303 | 2.191 | 模型輸出越來越「有把握」，不確定性降低 |
| `mean_token_accuracy` | 60.75% | 62.61% | 訓練末段每個 token 預測正確率持續提升 |

**overfitting 分析**

train_loss (2.12) < eval_loss (2.39)，差距約 0.27，token accuracy 差距約 6%——有輕微 overfitting 但在可接受範圍內。差距沒有持續擴大，說明模型在學習唐鳳風格，不是在背答案。

**為什麼 eval_loss 比 train_loss 高是正常的？**

驗證集是 2025 年演講（模型從未見過），訓練集是 2020–2024 年（反覆看過多次）。模型對見過的資料自然更熟悉，這個差距是時間切分策略的正常結果，不是警示訊號。若 eval_loss 開始上升而 train_loss 繼續下降，才是真正的 overfitting。

**訓練時長說明**

9201 steps × 4.30 s/it ≈ 39,564 秒 ≈ 11 小時（pure compute）；加上 eval（566 秒）和各種系統開銷，total runtime 來到 6.328e4 秒 ≈ 17.6 小時。`eval_samples_per_second: 1.337` 說明驗證集推論速度正常。

---

### b1a（2026-04-29）

**資料規模**：1213 篇中文演講，55834 組 Q&A，存於 `data/raw_speeches.json`

**網站結構確認**：
- 無分頁，所有演講在同一頁面一次載入
- 每個發言：`<li>` 包 `<a href="/speaker/...">名字</a>` + 純文字
- 唐鳳的 speaker href 為 `/speaker/%E5%94%90%E9%B3%B3-3`

**決策紀錄：不加語言偵測套件**
資料來源是有策展的封閉資料集，三層業務邏輯過濾（URL 中文 → 唐鳳出現 → Q&A 非空）已足夠。加 langdetect 是過度設計。

---

### b0b（2026-04-29）

**坑 1：`apply_chat_template` 回傳 `BatchEncoding`，不是純 tensor**
- 問題：`model.generate(input_ids, ...)` 報 `AttributeError`，找不到 `.shape`
- 原因：新版 transformers 的 `apply_chat_template(return_tensors="pt")` 回傳的是 `BatchEncoding` 物件（像 dict），不是純 tensor；`model.generate` 期待純 tensor
- 解法：兩步走——先 `tokenize=False` 取格式字串，再用 `tokenizer(text, return_tensors="pt")["input_ids"]` 轉 tensor
- 學到：API 行為會隨版本變動，遇到 `AttributeError` 先確認型別

**坑 2：`max_seq_length` 在 TRL 1.3.0 已從 `SFTConfig` 移除**
- 問題：`TypeError: SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'`
- 解法：改成在 tokenizer 上設定 `tokenizer.model_max_length = 512`
- 學到：套件升版時參數名稱會搬家，遇到 `unexpected keyword argument` 查 changelog

**坑 3：float16 在 MPS 上訓練數值不穩定**
- 問題：即使加了 `max_grad_norm=1.0`，loss 仍從正常值暴衝到 500+，最後 `grad_norm: nan`、`loss: 0`，模型完全崩潰
- 原因：float16 的動態範圍太窄（最大約 65504），梯度稍大就溢位成 inf/NaN；CUDA 有硬體層的 loss scaling 保護，MPS 沒有
- 解法：改用 `dtype=torch.bfloat16`——bfloat16 和 float32 有一樣的動態範圍（不溢位），只是尾數精度低一點，MPS 完全支援
- 學到：MPS 上訓練一律用 bfloat16，不用 float16；同時 `torch_dtype` 已棄用，改用 `dtype`

**坑 4：梯度爆炸（gradient accumulation 壓縮步驟數）**
- 問題：loss 在第 3 步從 3.67 暴衝到 9597，`grad_norm: inf`，fine-tuned 輸出亂碼
- 原因：10 筆資料 ÷ `gradient_accumulation_steps=4` = 只有 3 個更新步驟，`warmup_ratio=0.1` 對應到 0.3 步，等於沒有 warmup；學習率從一開始就全速，模型無法適應
- 解法：`gradient_accumulation_steps=1`（讓 10 筆資料變成 10 步），`learning_rate` 從 `2e-4` 降到 `2e-5`，加上 `max_grad_norm=1.0` 裁剪梯度
- 學到：`gradient_accumulation_steps` 會壓縮步驟數，資料量少時要特別注意；warmup 需要足夠的步驟才能發揮作用；小資料集用保守的 learning rate

---

### b4（2026-04-30）

**坑 1：`max_seq_length` 在 Colab 版 TRL 已完全移除，兩處都不接受**
- 問題一：`TypeError: SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'`
- 問題二：改傳給 `SFTTrainer` 後仍報 `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'max_seq_length'`
- 原因：這個版本的 TRL，`max_seq_length` 在 `SFTConfig` 和 `SFTTrainer` 都不接受了
- 解法：直接把 `max_seq_length` 整行刪掉。tokenizer 在載入時已設定 `model_max_length = 1024`，tokenization 時會自動截斷，不需要另外傳
- 學到：遇到 `unexpected keyword argument`，不要只是搬到另一個地方，要先確認這個版本根本還支不支援這個參數；`tokenizer.model_max_length` 是兜底機制，一定要設

**坑 2：LoRA cell 必須在 smoke test 之前執行**
- 問題：notebook 的 cell 順序跑掉，smoke test 擺在 LoRA 設定之前
- 原因：`prepare_model_for_kbit_training()` 和 `get_peft_model()` 沒跑的話，模型沒有可訓練參數，SFTTrainer 會試圖更新被凍結的 4-bit 量化權重
- 正確順序：載入模型（BitsAndBytesConfig） → 套 LoRA adapter → smoke test → 正式訓練
- 學到：在 Colab notebook 裡，cell 的執行順序就是程式的執行順序，寫 notebook 時要確認每個 cell 用到的變數在前面已經定義過

**坑 3：WandB API key 需 40 字元以上**
- 問題：`AuthenticationError: API key must have 40+ characters, has 36`
- 原因：Colab Secrets 填入的是舊 key 或截斷的字串，WandB 拒絕認證
- 解法：去 wandb.ai → User Settings → API keys 重新複製完整 key，或點 New key 生成新的
- 學到：貼 token/key 時要確認字數，Colab Secrets 輸入框不會顯示長度提示

**坑 5：Colab 免費配額不夠跑全量訓練（實際比預估更緊）**
- 問題：smoke test 實測 19 秒/step，推算全量 1 epoch（~3000 steps）需要約 16 小時；第一次縮到 5000 筆估算 2–2.5 小時，但實際帳號配額只有 2.5 小時（非預估的 3.5 小時），正式訓練速度又降至 50 秒/step，估算直接跳到 8.5 小時
- 解法：再砍至 1000 筆（約 125 steps），實測約 1.5–1.7 小時，在 2.5 小時配額內完成
- b5 影響：b4 的 ROUGE/Perplexity 數字會比 b3 低（訓練資料少），但「量化損失幅度」的對比仍然有效，因為兩組都在同樣的評估集上測
- 學到：Colab 的實際可用 GPU 時數因帳號使用狀況而異，比官方說明更少；跑大訓練前先用 smoke test 估算 step 時間，並保留至少 30% 緩衝；資料量要分兩段估：「煙霧測試速度」≠「正式訓練速度」（正式訓練有 eval、checkpoint 儲存等額外開銷）
- **決策原則（成本驅動）**：在免費資源有限的情況下，優先確保「流程跑通」而非「資料量充足」。b4 的核心價值是驗證 QLoRA 工程流程，不是產出最好的模型。犧牲資料量換取在配額內完成，是合理的取捨。若未來要比較完整的量化損失，可在 Colab Pro 或有更多配額時重跑全量版本。

**坑 6：cell 排隊執行兩次導致 `train_result` 遺失**
- 問題：訓練過程中不小心再按了一次 cell 10 的執行鍵 → Colab 排隊機制讓第一次跑完後立刻重跑 → 第二次被中斷，`train_result` 未定義 → cell 11 的 `train_result.metrics.get(...)` 拋出 `NameError`
- 解法：adapter 已存（save_pretrained 在 NameError 之前執行），手動補存 log，將 `train_runtime_sec` 和 `train_loss` 設為 `None`，`history` 用 `trainer.state.log_history` 補入
- 學到：Colab 的「排隊」不是即時執行而是等前一個 cell 完成才觸發，看起來像「自動重跑」；避免方法是訓練跑完前不要碰執行鍵，或在 cell 11 加 `try/except` 保護 `train_result` 的存取

**坑 4（原坑 4）：Colab 工作階段過多**
- 問題：`執行中的工作階段過多，請終止一個現有的工作階段以繼續`
- 原因：之前開過的 Colab session 沒有手動關閉，免費版有同時 session 數量上限
- 解法：Runtime → Manage sessions → 終止舊 session
- 學到：Colab 的 session 不會自動關閉，每次用完要養成手動 Terminate 的習慣，否則 GPU 資源會被舊 session 佔用
