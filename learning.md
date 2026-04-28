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

## b1：資料準備

### 爬蟲（requests + BeautifulSoup）
用程式自動瀏覽網頁、抽取內容。`requests` 抓 HTML，`BeautifulSoup` 解析標籤。

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

### gradient accumulation（梯度累積）
記憶體不夠時，把多個小批次的梯度加總才更新一次參數，效果等同更大的 batch size。

### gradient checkpointing（梯度檢查點）
訓練時不把所有中間計算結果留在記憶體，用時間換空間。記憶體吃緊時開。

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

### ROUGE
衡量生成文字與參考答案重疊程度的指標（0-1 分），值越高代表回答越接近標準答案。

### perplexity（困惑度）
模型對某段文字「有多困惑」的度量，越低代表模型越確信自己的輸出，通常用來衡量語言流暢度。

---

## b6：部署

### adapter merge（適配器合併）
把訓練好的 LoRA adapter 數學上合回 base model，產生一個「已微調好的完整模型」，方便後續轉檔。

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
類似 GitHub 但專門放 AI 模型的平台。上傳後有公開連結，面試時可以直接秀。

### model card
模型的 README，說明模型用途、訓練資料、評估結果與使用方式。AI 作品集的標準格式。

---

## 思考紀錄

> 做完一個 branch 後，把「踩到什麼坑」和「為什麼這樣解」記在這裡。

<!-- 格式參考：
### b0（2026-04-28）
- 問題：`device="mps"` 時 pipeline 報錯
- 原因：model.half() 在 MPS 上有問題，改用 float32
- 學到：MPS 不是 CUDA，某些操作要特別注意 dtype
-->
