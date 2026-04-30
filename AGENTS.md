# Tangram — AGENTS.md

> Fine-tuned on Audrey Tang's speeches — because AI policy should sound like a conversation, not a whitepaper.

## 願景

以「做中學」方式走完 LLM 微調的完整工程流程，親手建出一個**唐鳳風格 AI 政策溝通助理**。

目標是把一個技術概念丟進去，模型能以唐鳳的方式輸出一段讓非技術背景的人也能讀懂的說明。

**專案轉向聲明 (2026-04-30)**：
本專案已完成從爬蟲、微調到部署的完整工程鏈路。然而，由於訓練資料中包含 HTML 殘留標記（如 `Link in context`）且資料量經量化壓縮後不足，導致模型輸出品質未達預期（出現重複迴圈與雜訊）。
我們決定將專案目標從「產出完美助理」轉向**「初次微調模型的失敗經驗與工程反思」**。這是一個極具價值的學習型專案，記錄了所有遇到的技術坑洞與決策權衡。

---

## 最終架構（已完成部分）

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
| 量化 | `bitsandbytes` | 4-bit QLoRA，僅 Colab T4 |
| 評估 | `rouge_score` + `google-generativeai` | ROUGE-L (CJK) / LLM-as-a-Judge |
| 部署 | `llama.cpp` → `ollama` | GGUF 轉換 + 本地推論 |

**起始模型**：`meta-llama/Llama-3.2-3B-Instruct`  
**硬體**：Mac mini Apple Silicon (M4 Pro)，Colab T4 (b4 專用)

---

## Branch 規格與實作紀錄

### b0–b4｜從環境到 QLoRA 訓練
*(詳見 git log 與各腳本註釋，已完成全線跑通)*

### b5-eval｜效果評估
- **任務**：對測試集跑指標對比，並產出視覺化資產
- **完成狀態**：✅ 主要 DoD 達成（2026-04-30）

| 指標 | base | b3-lora | b4-qlora |
|------|------|---------|----------|
| Perplexity ↓ | 27.44 | **17.15** | 19.52 |
| ROUGE-L ↑ | 0.105 | **0.139** | 0.099 |

- **關鍵發現**：數值指標（PPL/ROUGE）雖然提升，但實際對話出現重複迴圈與爬蟲雜訊，顯示指標與真實體感存在鴻溝。

### b6-deploy｜本地部署
- **任務**：模型轉換為 GGUF 並在 Ollama 運行
- **完成狀態**：✅ DoD 達成（2026-04-30）
- **產出**：`tangram.gguf` (Q4_K_M, 2.0GB)，可在本地實現秒開對話。

---

## 🚫 未竟之路：實驗設計保留區

*以下規格為原計畫執行但因專案方向轉向而叫停的內容，保留作為下次專案的藍圖參考。*

### b7-rag｜檢索增強生成
- **原始任務**：把演講原文建成向量庫，解決幻覺與重複迴圈問題
- **預計工具**：`chromadb` + `sentence-transformers`
- **設計策略**：
  - Embedding：`paraphrase-multilingual-MiniLM-L12-v2`
  - Chunking：按清洗後的 Q&A 對儲存
  - 推論流程：使用者輸入 → 向量搜尋 → 組裝 Prompt (含 Audrey's context) → 生成

### b8-hub｜發布與 Model Card
- **原始任務**：將 Adapter 推送到 HuggingFace Hub 並撰寫完整 Model Card
- **預計步驟**：`huggingface-cli login` → `model.push_to_hub()`

---

## 📝 b9-blog｜失敗經驗與工程鏈路全紀錄 (Final Mission)

- **任務**：將本次專案整理成誠實的技術 Blog，定調為「學習紀錄」與「技術回顧」。
- **DoD**：Blog 文章草稿完成，包含 8 個深度區塊，並在首次出現時解釋微調術語。

### Blog 敘事架構

1.  **開場 Hook**：直接放 `ollama` 的鬼打牆輸出，建立「發身了什麼事」的懸念。
2.  **術語科普**：解釋 **Fine-tuning**、**SFT**、**LoRA**、**QLoRA** 的本質與差異。
3.  **工程鏈路閉環**：
    *   **資料工程**：從 1213 篇逐字稿到 **Chat Template** 的轉換。
    *   **架構決策**：M4 Pro 記憶體限制如何逼出 **LoRA** 與 **Gradient Checkpointing**。
    *   **指標解讀**：解釋 **Perplexity (PPL)** 與 **ROUGE-L** 的意義。
4.  **深度反思：數據是靈魂**：
    *   **失敗分析**：爬蟲殘留標記如何污染模型靈魂。
    *   **重複陷阱**：資料多樣性不足導致的生成崩潰。
5.  **下次我會怎麼做**：
    *   建立 **資料清洗 Checklist**。
    *   引入 **Synthetic Data (合成數據)** 增強質量。
    *   將 **b7 (RAG)** 納入初期設計而非補救措施。

---

## AI 協作守則 (修訂版)

1. **實戰優先**：優先嘗試 MPS 本機執行，若遇 bitsandbytes 等限制，評估後切換 Colab，不要無限繞行。
2. **誠實紀錄**：每個失敗的 Traceback 都是 Blog 的素材，需完整保留與分析。
3. **資料為王**：在未來的專案中，EDA (探索性資料分析) 與清洗佔比需提升至 50% 以上。
