# Tangram — AGENTS.md

> Fine-tuned on Audrey Tang's speeches — because AI policy should sound like a conversation, not a whitepaper.

## 願景

以「做中學」方式走完 LLM 微調的完整工程流程，親手建出一個**唐鳳風格 AI 政策溝通助理**。

目標是把一個技術概念丟進去，模型能以唐鳳的方式輸出一段讓非技術背景的人也能讀懂的說明。

---

## 最終架構

整個系統分兩條線：**離線訓練** 和 **線上推論**。

### 離線訓練流程

```
archive.tw/speeches（唐鳳演講逐字稿，CC0）
     │
     ▼ b1a-scrape
  raw JSON（標題 / 日期 / 內文）
     │
     ▼ b1b-format
  Q&A 對 → apply_chat_template → tokenized dataset
     │
     ├─ b2-sft ──→ SFTTrainer 全參數微調（驗證流程可跑）
     │
     └─ b3-lora ─→ LoRA adapter（r=8, target: q_proj / v_proj）
                        │
                        ▼ b4-qlora（Colab T4）
                   4-bit QLoRA adapter
                        │
                        ▼ b6-deploy
                   adapter merge → llama.cpp → tangram.gguf
                        │
                        ▼
                   ollama create tangram（本地可對話）
```

### 線上推論流程（b7 加入 RAG 後）

```
使用者輸入（一段技術概念）
     │
     ▼ sentence-transformers（embedding）
  查詢向量
     │
     ▼ chromadb 向量搜尋
  相關演講段落 top-k chunks
     │
     ▼ 組裝 prompt
  system: 你是唐鳳，以下是她說過的話：{chunks}
  user:   {使用者輸入}
     │
     ▼ Tangram（HF Spaces，CPU 推論）
  輸出：以唐鳳風格解釋的說明
     │
     ▼ Gradio UI（b9）
  公開 demo 頁面（每 session 限制 10 次）
```

### 為什麼要兩個機制並用？

| 機制 | 負責的事 | 沒有它會怎樣 |
|------|---------|------------|
| Fine-tune | 學唐鳳「怎麼說」（風格、句式、論述結構） | 輸出像一般 AI，沒有她的味道 |
| RAG | 引用她真實說過的話（事實錨定） | 模型會「幻覺」，捏造她沒說過的內容 |

---

## Tech Stack

| 層級 | 工具 | 說明 |
|------|------|------|
| 環境 | `uv` + `.venv` | `uv run` 唯一入口，不污染全域 |
| 推論 | `transformers` | 模型載入與 pipeline |
| 爬蟲 | `requests` + `BeautifulSoup` | 爬取演講逐字稿 |
| 資料 | `datasets` | 格式化與 tokenize |
| 訓練 | `trl` SFTTrainer | 包好訓練迴圈與 loss 計算 |
| 微調 | `peft` LoRA | 參數高效微調，adapter < 1% 參數 |
| 量化 | `bitsandbytes` | 4-bit QLoRA，僅 Colab T4（MPS 不支援） |
| 評估 | `evaluate` + `rouge_score` | ROUGE / perplexity |
| 部署 | `llama.cpp` → `ollama` | GGUF 轉換 + 本地推論 |
| RAG | `chromadb` + `sentence-transformers` | 本地向量資料庫 |
| 發布 | `huggingface_hub` | 上傳 adapter + model card |
| Demo | `gradio` | HF Spaces 公開 demo，含 rate limiting |

**起始模型**：`meta-llama/Llama-3.2-3B-Instruct`（Meta 出品，3B 參數，MPS 相容）  
**硬體**：Mac mini Apple Silicon（MPS），b4 需 Google Colab T4

---

## 前置準備

開始任何 branch 前需確認：

**HuggingFace Token**
Llama 3 是 gated model，需先到 HF 申請存取權限，再設定環境變數：
```bash
# .env（不 commit）
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

**`.gitignore`**
以下絕對不進 git（單個檔案動輒數 GB）：
```
.env
.venv/
__pycache__/
*.gguf
*.safetensors
checkpoints/
~/.cache/huggingface/   # HF 模型快取在本機，不在 repo 裡
runs/                   # tensorboard logs
```

---

## Branch DAG

```
b0-setup
  └── b0b-tracer         手寫假資料 → format → 1-epoch 微調 → base vs fine-tuned 對比
        └── b1a-scrape    爬取唐鳳演講頁面 → raw JSON
              └── b1b-format    解析 Q&A → chat template 格式
                    └── b2-sft        SFTTrainer 跑通微調
                          └── b3-lora       換 LoRA，降低可訓練參數
                                └── b4-qlora*     4-bit QLoRA（Colab T4）
                                      └── b5-eval       before/after 效果對比
                                            └── b6-deploy     GGUF → Ollama 本地跑
                                                  └── b7-rag        加上 RAG 事實錨定
                                                        └── b8-hub        HF Hub + model card
                                                              └── b9-spaces  Gradio demo + blog
```

`*b4-qlora 需 Google Colab T4，Mac MPS 不支援 bitsandbytes 4-bit 量化`

---

## Branch 規格

### b0-setup｜載入模型 + 推論驗證
- **任務**：用 `transformers` 載入 Llama-3.2-3B-Instruct，問它一個問題，看到回答
- **關鍵 API**：`pipeline("text-generation", model=..., device="mps")`
- **前置**：需設定 `HF_TOKEN` 環境變數（Llama 3 為 gated model，需先在 HF 申請存取）
- **DoD**：終端機印出模型回答，無報錯

### b0b-tracer｜曳光彈（端對端最薄切片）
- **任務**：跳過爬蟲，手寫 10 筆假 Q&A，跑完 format → 1-epoch 微調 → 推論，印出 base model 與 fine-tuned 的同題回答
- **目的**：在投入真實資料前，驗證「fine-tune → 推論」這條主幹路徑可以跑通，以及模型能否學到風格差異
- **DoD**：base model 與 fine-tuned 的回答出現可見差異，訓練全程無報錯

### b1a-scrape｜爬取演講原文
- **任務**：爬取 `archive.tw/speeches` 所有中文演講頁面，儲存成 raw JSON
- **限制**：只取繁體中文版；「商周專欄」等非 Q&A 格式需跳過
- **DoD**：本地有一份 JSON 檔，包含所有中文演講的標題、日期、內文

### b1b-format｜資料格式化
- **任務**：把逐字稿解析成 Q&A 對，套用 chat_template，tokenize，切分資料集，印出一筆樣本
- **關鍵**：`tokenizer.apply_chat_template()` 插入 Llama 3 的 `<|begin_of_text|>` / `<|eot_id|>` 特殊符號
- **資料切分**（時間切分，避免 data leakage）：
  - 訓練集 85%：2020–2024 年演講
  - 驗證集 10%：2025 年演講
  - 測試集 5%：2026 年演講（b5 唯一用到的時機，訓練全程不碰）
- **品質過濾**：過濾掉回答少於 50 個 token 的 Q&A pair（太短代表沒有完整論述，學不到風格）
- **DoD**：印出一筆完整的 tokenized 樣本，可看到 input_ids 數列；三份資料集筆數確認無誤

### b2-sft｜SFTTrainer 跑通微調
- **任務**：用 `trl` 的 SFTTrainer 做第一次微調，觀察 loss 下降
- **關鍵超參數**：
  - `learning_rate=2e-4`（LLM 微調學習率要極小，太大訓練直接崩）
  - `warmup_ratio=0.1`（前 10% steps 做預熱，防止訓練初期 loss 爆炸）
  - `max_seq_length=1024`（唐鳳的回答有時很長，先從 1024 開始，不夠再調）
- **記憶體策略**：`gradient_accumulation_steps=4` + `gradient_checkpointing=True`
- **可重現性**：固定 `seed=42`，確保 b2 vs b3 結果可比較
- **防止 Catastrophic Forgetting**：訓練資料中混入 5% Alpaca 通用對話資料，維持模型原有語言能力
- **驗證集**：`SFTTrainer(eval_dataset=val_dataset)`，訓練中同步監控 validation loss
- **監控**：用 WandB（`report_to="wandb"`）觀察 train loss 與 val loss 曲線；val loss 開始回升即停止訓練
- **DoD**：train loss 下降；validation loss 同步下降且未出現上揚

### b3-lora｜LoRA 參數高效微調
- **任務**：用 `peft` 加上 LoRA adapter，比較可訓練參數量
- **關鍵參數**：`r=8`、`lora_alpha=16`、`target_modules=["q_proj","v_proj"]`
- **DoD**：可訓練參數量 < 模型總參數量的 1%

### b4-qlora｜4-bit 量化微調（Colab）
- **任務**：在 Google Colab T4 GPU 上跑 QLoRA，體驗完整量化微調流程
- **關鍵**：`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` + `prepare_model_for_kbit_training()`
- **DoD**：Colab notebook 跑通，有 loss 數值輸出

### b5-eval｜效果評估

**比較矩陣**：這個專案產出 3 個訓練變體，評估時要回答三個問題：

| 比較組 | 問的問題 | 主要指標 |
|--------|---------|---------|
| base → b2-sft | 全參數微調有沒有學到唐鳳風格？ | Perplexity、ROUGE、LLM-as-a-Judge |
| b2-sft → b3-lora | LoRA 用 1% 參數能達到接近效果嗎？ | 同上 |
| b3-lora → b4-qlora | 4-bit 量化損失多少品質？ | 同上 |
| b6（無 RAG）→ b7（有 RAG） | RAG 有沒有減少幻覺？ | 引用準確度、人工核對 |

**指標說明**：

- **Perplexity**：在 held-out 唐鳳演講集上測，數值越低代表模型越「預測得到」她的說話方式，即風格契合度越高
- **ROUGE**：微調後回答與她真實回答的 n-gram 重疊率，有效但抓不到風格的深層特徵
- **LLM-as-a-Judge**：請 Claude 或 GPT-4 對「這段回答有多像唐鳳」打 1–10 分，是最直接量測風格的方式；對同一問題取 base model 與 fine-tuned 的回答各一份送給 judge 評分
- **Overfitting 檢查**：在少量 MMLU 題目上比較 base 與 fine-tuned 分數，確認一般語言能力沒有明顯退化（catastrophic forgetting）

**任務**：對 b1b 切出的**測試集**（2026 年演講，訓練全程未見過）跑以上四組比較，整理成表格

**視覺化產出（Blog 用）**：以下 5 張圖為 b5 必交付物，存入 `blog/assets/`

**圖 1 — Loss Curve（訓練健康度）**
- 從 b2-sft / b3-lora 訓練 log 取 `trainer.state.log_history`，畫 train loss 與 val loss 雙折線
- X 軸：steps；Y 軸：loss；標出 early stopping 點（val loss 開始回升處）
- 工具：`matplotlib`；WandB 截圖可作為補充，但需額外輸出靜態 PNG 存檔

**圖 2 — 三欄 Side-by-Side 對比（最直觀的說服工具）**
- 選 5–8 個代表性問題，涵蓋政策類、科技倫理類、開放政府類
- 每題對比三欄：Base Model / 微調後（b3-lora）/ 微調 + RAG（b7）
- 格式：Markdown 表格（blog 直接嵌入）或 HTML 三欄 div
- 問題需包含至少一題讓 Base Model 明顯失敗的 edge case

**圖 3 — 風格詞頻對比圖（Tangram 獨有亮點）**
- 定義唐鳳「簽名詞」清單（至少含：協作、透明、信任、公民、參與、開放、審議、數位）
- 三組文本各自統計詞頻：原始演講語料 / Base Model 輸出 / 微調後輸出
- 圖形：Grouped Bar Chart；工具：`jieba` 分詞 + `matplotlib`
- 可額外附 Word Cloud 作為視覺點綴（非必要）

**圖 4 — 雷達圖（多維風格能力對比）**
- 5 個評分維度，需與唐鳳溝通特性直接對應：
  1. 政策術語準確度（專業詞彙比例）
  2. 包容性語言（「我們」/ 第一人稱複數使用頻率）
  3. 類比解釋能力（遇抽象概念是否轉化為具體例子，LLM-as-Judge 評分）
  4. 問題針對性（回答是否切題，LLM-as-Judge 評分）
  5. 語氣一致性（台語借詞、特色句型符合度，LLM-as-Judge 評分）
- 對比兩條線：Base Model vs b3-lora（可選加入 b3-lora+RAG 第三條線）
- 評分方式：請 Claude 或 GPT-4o 對每題每維度打 1–5 分，取 5 題平均
- 工具：`matplotlib` polar chart

**圖 5 — ROUGE / Perplexity 跨版本對比圖**
- X 軸：4 個模型版本（Base / b2-sft / b3-lora / b4-qlora）
- Y 軸左：ROUGE-L score；Y 軸右：Perplexity（雙軸折線或分組條形圖）
- 工具：`evaluate` 套件 + `rouge_score`；Perplexity 用 held-out 唐鳳演講集計算
- 目的：讓讀者看出 LoRA 以 1% 參數達到接近 SFT 的效果，QLoRA 的量化損失幅度

**DoD**：
- 有 base vs b3-lora 的 perplexity、ROUGE 量化數字
- 有 LLM-as-a-Judge 的風格評分（至少 5 題）
- Overfitting 檢查：fine-tuned 的 MMLU 分數未大幅低於 base model
- `blog/assets/` 資料夾內有圖 1–5 的 PNG 檔，命名規則：`fig1_loss_curve.png`、`fig2_sidebyside.md`、`fig3_word_freq.png`、`fig4_radar.png`、`fig5_metrics.png`

### b6-deploy｜本地部署
- **任務**：把微調模型轉成 GGUF 格式，用 Ollama 在 Mac mini 本地跑
- **流程**：adapter merge → `llama.cpp` 轉 GGUF → `ollama create`
- **合併注意**：`merge_and_unload()` 前須確認 base model 以 FP16 或 BF16 載入，否則合併後精度損失會導致性能下降
- **DoD**：`ollama run tangram` 可以對話，現場 demo 能跑

### b7-rag｜加上 RAG 事實錨定
- **任務**：把唐鳳演講原文建成向量資料庫，讓模型回答時引用真實說過的話
- **工具**：`chromadb`（本地向量庫）+ `sentence-transformers`（文字轉向量）
- **DoD**：問一個問題，回答中能引用她演講的原文段落

### b8-hub｜發布與記錄
- **任務**：push adapter 到 HuggingFace Hub，寫 model card
- **DoD**：HF Hub 上有公開連結，model card 記錄訓練資料、評估結果與使用方式

### b9-spaces｜公開 Demo + Blog
- **任務**：在 HF Spaces 建 Gradio demo，實作 session-based rate limiting，寫 blog 文章
- **Rate limiting 策略**：
  ```python
  # 同時請求上限
  demo.queue(max_size=5, default_concurrency_limit=1)

  # 每 session 限制 10 次
  def predict(prompt, count: int, request: gr.Request):
      if count >= 10:
          return "今日已達使用上限，明天再來。", count
      ...
      return response, count + 1
  ```
- **Blog 敘事架構**（按此順序撰寫，技術敘事而非技術報告）：
  1. **為什麼是唐鳳？** — 動機 + 挑戰說明（她的溝通風格為何值得學習？）
  2. **資料工程** — 從演講到訓練格式（b1a/b1b 流程截圖 + 資料量統計）
  3. **三階段訓練** — SFT → LoRA → QLoRA（圖 1 Loss Curve + 可訓練參數量對比）
  4. **效果如何？** — 圖 2 Side-by-Side + 圖 3 詞頻圖 + 圖 4 雷達圖 + 圖 5 ROUGE/Perplexity
  5. **RAG 加持前後差異** — 引用準確度對比（圖 2 第三欄 vs 第二欄）
  6. **反思：哪裡還不夠好** — 錯誤案例分析，說明下一步優化方向（展示 critical thinking）
- **圖表來源**：所有圖直接引用 `blog/assets/` 內的 b5-eval 產出，不重新生成
- **DoD**：HF Spaces 公開 URL 可測試、rate limiting 有效、blog 文章草稿完成，6 個章節皆有對應內容

---

## AI 協作守則

1. **每個 branch 獨立問**：不要一次問跨 branch 的問題，每個 branch 有自己的 DoD，做完才進下一個。
2. **程式碼優先，解釋其次**：先給可以跑的 code snippet，術語解釋放在 comment 或事後問。
3. **MPS 限制要說清楚**：遇到 `bitsandbytes` / CUDA-only 的東西，直接告知需要切換到 Colab，不要試著在 Mac 上繞行。
4. **debug 時給完整 traceback**：不要只貼最後一行錯誤，完整 traceback 才能定位問題。
5. **資料格式驗證**：b1b 做完後，幫驗證至少一筆 tokenized 樣本的格式是否符合 Llama 3 chat template 規格。
6. **不要跳步驟**：每個 branch 有 DoD，DoD 沒達到不要往下一個 branch 推。
7. **選擇性測試**：只對純函數寫 assert（資料格式化、tokenization 檢查、RAG 查詢回傳筆數）。訓練迴圈和模型輸出風格靠 DoD 人工驗收，不強行套 TDD。
8. **關鍵輸出存檔**：每個 branch 完成後，把關鍵輸出存成檔案（不只印在終端機），方便跨 branch 比較與 blog 取材。命名規則：`outputs/b0b_comparison.txt`、`outputs/b5_eval.json`，依此類推。

## 工作流規約：/ship (發布任務)

當使用者下達 `/ship` 或要求「整理並提交變更」時，必須嚴格執行以下流程：

1. **分組盤點**：使用 `git status` 與 `git diff` 分析目前所有未提交的變更。
2. **邏輯拆分**：禁止 `git add .`。必須按功能邏輯（如：infra, feat, docs, fix）分組，確保每個 commit 僅包含相關檔案。
3. **格式規範**：使用 **繁體中文** 撰寫 commit message，並遵循 **Conventional Commits** 格式：
   - `feat:` 新功能
   - `fix:` 修補 bug
   - `docs:` 文件變更
   - `build:` 構建系統、依賴項變更
   - `refactor:` 重構
4. **逐批提交**：分次執行 `git add` + `git commit`。
5. **最終發布**：所有分組提交完成後，執行 `git push`。

---

## 參考資料對應

| Branch | 主要參考 |
|--------|---------|
| b0–b2 | Hands-On Guide（有完整 code snippet） |
| b3 | Ultimate Guide §PEFT 章節 |
| b5 | Enhancing §Evaluation 章節 |
