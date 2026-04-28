# Tangram — AGENTS.md

> Fine-tuned on Audrey Tang's speeches — because AI policy should sound like a conversation, not a whitepaper.

## 願景

以「做中學」方式走完 LLM 微調的完整工程流程，最終產出**唐鳳風格 AI 政策溝通助理**作為 AI 導入職缺的面試作品集。

**核心痛點**：企業導入 AI 時，技術人員說不清楚、決策者聽不懂，溝通斷層普遍。  
**解法**：把技術概念丟進去，模型以唐鳳的溝通風格輸出「給非技術主管的一頁式 AI 說明」。

**最終架構**

```
技術概念輸入
     ↓
Fine-tuned 唐鳳模型（負責風格與論述結構）
     ↓
RAG 過她的演講原文（負責事實錨定，避免捏造）
     ↓
輸出：給主管的一頁式 AI 說明
```

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

**起始模型**：`meta-llama/Llama-3.2-3B-Instruct`（Meta 出品，3B 參數，MPS 相容）  
**硬體**：Mac mini Apple Silicon（MPS），b4 需 Google Colab T4

---

## Branch DAG

```
b0-setup
  └── b1a-scrape       爬取唐鳳演講頁面 → raw JSON
        └── b1b-format  解析 Q&A → chat template 格式
              └── b2-sft   SFTTrainer 跑通微調
                    └── b3-lora    換 LoRA，降低可訓練參數
                          └── b4-qlora*  4-bit QLoRA（Colab T4）
                                └── b5-eval    before/after 效果對比
                                      └── b6-deploy  GGUF → Ollama 本地跑
                                            └── b7-rag  加上 RAG 事實錨定
                                                  └── b8-hub  HF Hub + blog
```

`*b4-qlora 需 Google Colab T4，Mac MPS 不支援 bitsandbytes 4-bit 量化`

---

## Branch 規格

### b0-setup｜載入模型 + 推論驗證
- **任務**：用 `transformers` 載入 Qwen2.5-0.5B，問它一個問題，看到回答
- **關鍵 API**：`pipeline("text-generation", model=..., device="mps")`
- **DoD**：終端機印出模型回答，無報錯

### b1a-scrape｜爬取演講原文
- **任務**：爬取 `archive.tw/speeches` 所有中文演講頁面，儲存成 raw JSON
- **限制**：只取繁體中文版；「商周專欄」等非 Q&A 格式需跳過
- **DoD**：本地有一份 JSON 檔，包含所有中文演講的標題、日期、內文

### b1b-format｜資料格式化
- **任務**：把逐字稿解析成 Q&A 對，套用 chat_template，tokenize，印出一筆樣本
- **關鍵**：`tokenizer.apply_chat_template()` 插入 Llama 3 的 `<|begin_of_text|>` / `<|eot_id|>` 特殊符號
- **DoD**：印出一筆完整的 tokenized 樣本，可看到 input_ids 數列

### b2-sft｜SFTTrainer 跑通微調
- **任務**：用 `trl` 的 SFTTrainer 做第一次微調，觀察 loss 下降
- **記憶體策略**：`gradient_accumulation_steps` + `gradient_checkpointing`
- **DoD**：loss 數值下降，訓練完成不報錯

### b3-lora｜LoRA 參數高效微調
- **任務**：用 `peft` 加上 LoRA adapter，比較可訓練參數量
- **關鍵參數**：`r=8`、`lora_alpha=16`、`target_modules=["q_proj","v_proj"]`
- **DoD**：可訓練參數量 < 模型總參數量的 1%

### b4-qlora｜4-bit 量化微調（Colab）
- **任務**：在 Google Colab T4 GPU 上跑 QLoRA，體驗完整量化微調流程
- **關鍵**：`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` + `prepare_model_for_kbit_training()`
- **DoD**：Colab notebook 跑通，有 loss 數值輸出

### b5-eval｜效果評估
- **任務**：量化微調前後的模型差異
- **指標**：ROUGE（0-1，越高越接近標準答案）、perplexity（越低越流暢）
- **DoD**：有 before/after 的量化數字可放進 blog

### b6-deploy｜本地部署
- **任務**：把微調模型轉成 GGUF 格式，用 Ollama 在 Mac mini 本地跑
- **流程**：adapter merge → `llama.cpp` 轉 GGUF → `ollama create`
- **DoD**：`ollama run tangram` 可以對話，現場 demo 能跑

### b7-rag｜加上 RAG 事實錨定
- **任務**：把唐鳳演講原文建成向量資料庫，讓模型回答時引用真實說過的話
- **工具**：`chromadb`（本地向量庫）+ `sentence-transformers`（文字轉向量）
- **DoD**：問一個問題，回答中能引用她演講的原文段落

### b8-hub｜發布與記錄
- **任務**：push adapter 到 HuggingFace Hub，寫 model card，整理成 blog 文章
- **DoD**：公開連結可分享，blog 文章草稿完成

---

## AI 協作守則

1. **每個 branch 獨立問**：不要一次問跨 branch 的問題，每個 branch 有自己的 DoD，做完才進下一個。
2. **程式碼優先，解釋其次**：先給可以跑的 code snippet，術語解釋放在 comment 或事後問。
3. **MPS 限制要說清楚**：遇到 `bitsandbytes` / CUDA-only 的東西，直接告知需要切換到 Colab，不要試著在 Mac 上繞行。
4. **debug 時給完整 traceback**：不要只貼最後一行錯誤，完整 traceback 才能定位問題。
5. **資料格式驗證**：b1b 做完後，幫驗證至少一筆 tokenized 樣本的格式是否符合 Qwen chat template 規格。
6. **不要跳步驟**：每個 branch 有 DoD，DoD 沒達到不要往下一個 branch 推。

---

## 參考資料對應

| Branch | 主要參考 |
|--------|---------|
| b0–b2 | Hands-On Guide（有完整 code snippet） |
| b3 | Ultimate Guide §PEFT 章節 |
| b5 | Enhancing §Evaluation 章節 |
