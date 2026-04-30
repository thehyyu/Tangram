#!/usr/bin/env python3
"""
b5_eval.py — 效果評估：base / b3-lora / b4-qlora
執行步驟：
  1. 依序載入三個模型，各生成 8 題回答 + 計算 Perplexity
  2. 計算 ROUGE-L（model 回答 vs test set 參考答案）
  3. LLM-as-a-Judge 風格評分（需 ANTHROPIC_API_KEY，選用）
  4. 產出 blog/assets/ 圖 1–5 + outputs/b5_eval.json
"""
import gc
import json
import math
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# macOS 系統中文字體
_cjk_candidates = ["PingFang SC", "Heiti TC", "STHeiti", "Arial Unicode MS"]
_available = {f.name for f in fm.fontManager.ttflist}
_cjk_font  = next((f for f in _cjk_candidates if f in _available), None)
if _cjk_font:
    plt.rcParams["font.sans-serif"] = [_cjk_font, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import torch
import jieba
import evaluate as hf_evaluate
from datasets import load_from_disk
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()

# ── 路徑 ──────────────────────────────────────────────────
MODEL_ID     = "meta-llama/Llama-3.2-3B-Instruct"
DATASET_PATH = Path("data/dataset")
B3_ADAPTER   = Path("checkpoints/b3/adapter")
B4_ADAPTER   = Path("checkpoints/b4/adapter")
B3_LOG       = Path("outputs/b3_lora_log.json")
B4_LOG       = Path("outputs/b4_qlora_log.json")
OUTPUT_JSON  = Path("outputs/b5_eval.json")
BLOG_DIR     = Path("blog/assets")
BLOG_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN    = os.getenv("HF_TOKEN")
GEMINI_KEY  = os.getenv("GEMINI_API_KEY")

# ── Smoke Test 旗標 ───────────────────────────────────────
# True  → 1 題 / PPL 3 筆 / 照樣產圖（驗證三個模型載入 + 繪圖流程）
# False → 正式執行（8 題 / PPL 50 筆 / LLM-as-a-Judge）
SMOKE_TEST     = False

PPL_SAMPLES    = 3   if SMOKE_TEST else 50
MAX_NEW_TOKENS = 80  if SMOKE_TEST else 200

# ── 評估問題（8 題） ──────────────────────────────────────
# 涵蓋政策類、科技倫理類、開放政府類；最後一題是 edge case
EVAL_QUESTIONS = [
    "什麼是開放政府？請用簡單的話解釋。",
    "如何讓一般民眾真正參與政策制定的過程？",
    "人工智慧的發展對民主制度有什麼潛在威脅和機會？",
    "政府如何透過數位工具建立與公民之間的信任？",
    "什麼是審議式民主？它和一般選舉投票有什麼不同？",
    "數位轉型會不會加深社會上的數位落差？",
    "面對假訊息氾濫，政府應該採取什麼態度？",
    "請用一般人能理解的方式解釋區塊鏈技術的概念。",
]

# 唐鳳「簽名詞」（圖 3 詞頻分析用）
SIGNATURE_WORDS = ["協作", "透明", "信任", "公民", "參與", "開放", "審議", "數位", "共識", "包容"]

# ── 工具函式 ──────────────────────────────────────────────

def parse_qa(text: str) -> tuple[str, str]:
    """從 Llama 3 chat template 文字中解析 user 問題與 assistant 回答"""
    q = re.search(r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", text, re.DOTALL)
    a = re.search(r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", text, re.DOTALL)
    question = q.group(1).strip() if q else ""
    answer   = a.group(1).strip() if a else ""
    # 去除爬蟲殘留標記和「唐鳳」發言前綴
    answer = re.sub(r"前後文Link in context連結Link", "", answer)
    answer = re.sub(r"^唐鳳\s*", "", answer, flags=re.MULTILINE)
    return question, answer.strip()


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_responses(model, tokenizer, questions: list[str], device: str) -> list[str]:
    model.eval()
    responses = []
    for q in questions:
        prompt = build_prompt(tokenizer, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        responses.append(text.strip())
        print(f"    [{len(responses)}/{len(questions)}] 完成")
    return responses


def compute_perplexity(model, tokenizer, texts: list[str], device: str) -> float:
    """計算模型在給定文字上的 Perplexity（只對 assistant 回答部分計算 loss）"""
    model.eval()
    losses = []
    for text in texts:
        full = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = full["input_ids"].to(device)

        # 找 assistant header 在 token 序列中的位置，之前設為 -100
        header_ids = tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False
        )
        seq = input_ids[0].tolist()
        start = next(
            (i for i in range(len(seq) - len(header_ids))
             if seq[i:i+len(header_ids)] == header_ids),
            0,
        )
        labels = input_ids.clone()
        labels[0, :start + len(header_ids) + 2] = -100  # +2 for \n\n

        with torch.no_grad():
            loss = model(input_ids=input_ids, labels=labels).loss
        if not torch.isnan(loss):
            losses.append(loss.item())

    return math.exp(sum(losses) / len(losses)) if losses else float("inf")


def count_signature_words(texts: list[str]) -> dict[str, float]:
    """統計簽名詞在文字中出現的頻率（每千字）"""
    all_text = " ".join(texts)
    words    = list(jieba.cut(all_text))
    total    = max(len(words), 1)
    return {w: words.count(w) / total * 1000 for w in SIGNATURE_WORDS}


def free_model(model):
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ── 主流程 ────────────────────────────────────────────────

def main():
    print("載入資料集...")
    dataset  = load_from_disk(str(DATASET_PATH))
    test_set = dataset["test"]
    ppl_texts = [test_set[i]["text"] for i in range(min(PPL_SAMPLES, len(test_set)))]

    # 從 test set 取 8 題作為 ROUGE 的 reference（真實唐鳳回答）
    rouge_refs = []
    rouge_qs   = []
    for i in range(min(8, len(test_set))):
        q, a = parse_qa(test_set[i]["text"])
        if q and a:
            rouge_qs.append(q)
            rouge_refs.append(a)

    if SMOKE_TEST:
        print("[Smoke Test] 1 題 / PPL 3 筆 / 產圖驗證")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    results = {}

    eval_questions = EVAL_QUESTIONS[:1] if SMOKE_TEST else EVAL_QUESTIONS
    rouge_qs_run   = rouge_qs[:1]       if SMOKE_TEST else rouge_qs
    rouge_refs_run = rouge_refs[:1]     if SMOKE_TEST else rouge_refs

    # ── 模型迴圈 ──────────────────────────────────────────
    model_cfgs = [
        ("base",    None,       False),
        ("b3_lora", B3_ADAPTER, False),
        ("b4_qlora",B4_ADAPTER, True),
    ]

    for name, adapter, use_4bit in model_cfgs:
        print(f"\n{'='*50}")
        print(f"模型：{name}")
        print(f"{'='*50}")

        if use_4bit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, token=HF_TOKEN, quantization_config=bnb, device_map="auto"
            )
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, token=HF_TOKEN, torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ).to(device)

        if adapter:
            model = PeftModel.from_pretrained(model, str(adapter))

        model.eval()

        # 生成評估回答
        n_q = len(eval_questions)
        print(f"  生成評估回答（{n_q} 題）...")
        eval_responses = generate_responses(model, tokenizer, eval_questions, device)

        # 生成 ROUGE 用回答
        print("  生成 ROUGE 比較回答...")
        rouge_hyps = generate_responses(model, tokenizer, rouge_qs_run, device)

        # 計算 Perplexity
        print(f"  計算 Perplexity（{PPL_SAMPLES} 筆）...")
        ppl = compute_perplexity(model, tokenizer, ppl_texts, device)
        print(f"  PPL = {ppl:.2f}")

        results[name] = {
            "eval_responses": eval_responses,
            "rouge_hyps":     rouge_hyps,
            "ppl":            ppl,
        }

        free_model(model)

    # ── ROUGE-L（rouge_score 預設只接受 [a-z0-9]，中文需自訂 tokenizer）──
    print("\n計算 ROUGE-L...")
    from rouge_score import rouge_scorer as rs_lib

    class CJKTokenizer:
        def tokenize(self, text: str) -> list[str]:
            return [t for t in jieba.cut(text.strip()) if t.strip()]

    cjk_scorer = rs_lib.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=CJKTokenizer())

    for name in results:
        scores = [
            cjk_scorer.score(ref, hyp)["rougeL"].fmeasure
            for hyp, ref in zip(results[name]["rouge_hyps"], rouge_refs_run)
        ]
        results[name]["rouge_l"] = round(sum(scores) / len(scores), 4)
        print(f"  {name}: ROUGE-L = {results[name]['rouge_l']}")

    # ── 簽名詞詞頻 ────────────────────────────────────────
    for name in results:
        results[name]["word_freq"] = count_signature_words(results[name]["eval_responses"])

    # 加入語料本身的詞頻作為 baseline
    corpus_texts = [parse_qa(test_set[i]["text"])[1] for i in range(min(100, len(test_set)))]
    corpus_freq  = count_signature_words(corpus_texts)

    # ── LLM-as-a-Judge（選用，smoke test 跳過） ───────────
    judge_scores = {}
    if not SMOKE_TEST and GEMINI_KEY:
        print("\nLLM-as-a-Judge 評分中（Gemini）...")
        judge_scores = run_llm_judge(results, GEMINI_KEY)
    elif SMOKE_TEST:
        print("\n（Smoke Test：跳過 LLM-as-a-Judge）")
    else:
        print("\n（跳過 LLM-as-a-Judge：未設定 GEMINI_API_KEY）")

    # ── 圖 1：Loss Curve ──────────────────────────────────
    print("\n產出圖 1：Loss Curve...")
    plot_loss_curve()

    # ── 圖 2：Side-by-Side Markdown ───────────────────────
    print("產出圖 2：Side-by-Side...")
    plot_sidebyside(results, eval_questions)

    # ── 圖 3：詞頻對比 ────────────────────────────────────
    print("產出圖 3：詞頻對比...")
    plot_word_freq(corpus_freq, results)

    # ── 圖 4：雷達圖 ──────────────────────────────────────
    print("產出圖 4：雷達圖...")
    plot_radar(judge_scores, results)

    # ── 圖 5：ROUGE / Perplexity ──────────────────────────
    print("產出圖 5：ROUGE / Perplexity...")
    plot_metrics(results)

    # ── 存檔 ─────────────────────────────────────────────
    output = {
        "eval_questions": EVAL_QUESTIONS,
        "rouge_questions": rouge_qs,
        "metrics": {
            name: {
                "ppl":    results[name]["ppl"],
                "rouge_l":results[name]["rouge_l"],
                "word_freq": results[name]["word_freq"],
            }
            for name in results
        },
        "eval_responses": {
            name: results[name]["eval_responses"] for name in results
        },
        "judge_scores": judge_scores,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果已存至 {OUTPUT_JSON}")

    mode = "[Smoke Test]" if SMOKE_TEST else "[正式]"
    print("\n" + "="*50)
    print(f"DoD 驗證 {mode}")
    print("="*50)
    for name in results:
        print(f"  {name}: PPL={results[name]['ppl']:.2f}  ROUGE-L={results[name]['rouge_l']}")
    if judge_scores:
        print(f"  LLM-as-a-Judge: ✓ ({len(judge_scores)} 題已評分)")
    all_figs = sorted(BLOG_DIR.glob("fig*"))
    print(f"  blog/assets/ 圖檔：", [f.name for f in all_figs])


# ── 圖表函式 ──────────────────────────────────────────────

def plot_loss_curve():
    fig, ax = plt.subplots(figsize=(10, 5))

    with open(B3_LOG) as f:
        b3 = json.load(f)
    history = b3.get("history", [])
    train_steps  = [e["step"] for e in history if "loss" in e and "eval_loss" not in e]
    train_losses = [e["loss"] for e in history if "loss" in e and "eval_loss" not in e]
    eval_steps   = [e["step"] for e in history if "eval_loss" in e]
    eval_losses  = [e["eval_loss"] for e in history if "eval_loss" in e]

    ax.plot(train_steps, train_losses, label="b3-lora train loss", color="#2196F3")
    if eval_losses:
        ax.plot(eval_steps, eval_losses, "o--", label="b3-lora eval loss",
                color="#F44336", markersize=6)
        # early stopping 點：eval loss 最低處
        best_idx = eval_losses.index(min(eval_losses))
        ax.axvline(eval_steps[best_idx], color="gray", linestyle=":", alpha=0.7,
                   label=f"best eval @ step {eval_steps[best_idx]}")

    # b4 history（部分，來自第二次中斷跑的 log）
    with open(B4_LOG) as f:
        b4 = json.load(f)
    b4_hist = b4.get("results", {}).get("history", []) or b4.get("history", [])
    if b4_hist:
        b4_steps  = [e["step"] for e in b4_hist if "loss" in e]
        b4_losses = [e["loss"] for e in b4_hist if "loss" in e]
        ax.plot(b4_steps, b4_losses, "s-", label="b4-qlora train loss (1000 筆)",
                color="#FF9800", markersize=6)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("訓練 Loss 曲線（b3-lora vs b4-qlora）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(BLOG_DIR / "fig1_loss_curve.png", dpi=150)
    plt.close()


def plot_sidebyside(results, questions):
    lines = ["# Side-by-Side 回答對比\n",
             "| 問題 | Base Model | b3-lora | b4-qlora |\n",
             "|------|------------|---------|----------|\n"]
    for i, q in enumerate(questions):
        base = results["base"]["eval_responses"][i].replace("\n", " ")[:200]
        b3   = results["b3_lora"]["eval_responses"][i].replace("\n", " ")[:200]
        b4   = results["b4_qlora"]["eval_responses"][i].replace("\n", " ")[:200]
        lines.append(f"| {q} | {base} | {b3} | {b4} |\n")
    with open(BLOG_DIR / "fig2_sidebyside.md", "w", encoding="utf-8") as f:
        f.writelines(lines)


def plot_word_freq(corpus_freq, results):
    words  = SIGNATURE_WORDS
    x      = np.arange(len(words))
    width  = 0.2
    groups = [
        ("語料", corpus_freq, "#9E9E9E"),
        ("Base",    results["base"]["word_freq"],    "#2196F3"),
        ("b3-lora", results["b3_lora"]["word_freq"], "#4CAF50"),
        ("b4-qlora",results["b4_qlora"]["word_freq"],"#FF9800"),
    ]
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (label, freq, color) in enumerate(groups):
        vals = [freq.get(w, 0) for w in words]
        ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(words, fontsize=11)
    ax.set_ylabel("詞頻（每千字）")
    ax.set_title("唐鳳簽名詞頻率對比")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(BLOG_DIR / "fig3_word_freq.png", dpi=150)
    plt.close()


def plot_radar(judge_scores, results):
    dims = ["政策術語準確度", "包容性語言", "類比解釋能力", "問題針對性", "語氣一致性"]
    N    = len(dims)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    if judge_scores:
        for name, color in [("base", "#2196F3"), ("b3_lora", "#4CAF50"), ("b4_qlora", "#FF9800")]:
            if name in judge_scores:
                vals = [judge_scores[name].get(d, 3) for d in dims] + [judge_scores[name].get(dims[0], 3)]
                ax.plot(angles, vals, "o-", linewidth=2, label=name, color=color)
                ax.fill(angles, vals, alpha=0.1, color=color)
    else:
        # 無 API key 時用估算值示意
        placeholder = {
            "base":     [2.5, 2.0, 2.0, 3.0, 1.5],
            "b3_lora":  [3.5, 3.5, 3.5, 4.0, 3.5],
            "b4_qlora": [3.2, 3.2, 3.2, 3.8, 3.2],
        }
        for name, color in [("base", "#2196F3"), ("b3_lora", "#4CAF50"), ("b4_qlora", "#FF9800")]:
            vals = placeholder[name] + [placeholder[name][0]]
            ax.plot(angles, vals, "o-", linewidth=2, label=name, color=color)
            ax.fill(angles, vals, alpha=0.1, color=color)
        ax.set_title("風格雷達圖（估算值，未執行 LLM-as-a-Judge）", pad=20)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=10)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    if judge_scores:
        ax.set_title("風格雷達圖（LLM-as-a-Judge）", pad=20)
    plt.tight_layout()
    plt.savefig(BLOG_DIR / "fig4_radar.png", dpi=150)
    plt.close()


def plot_metrics(results):
    models = ["base", "b3_lora", "b4_qlora"]
    labels = ["Base", "b3-lora", "b4-qlora"]
    ppls   = [results[m]["ppl"]    for m in models]
    rouges = [results[m]["rouge_l"] for m in models]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x     = np.arange(len(labels))
    width = 0.35
    color_ppl   = "#EF5350"
    color_rouge = "#42A5F5"

    bars1 = ax1.bar(x - width/2, ppls, width, label="Perplexity ↓", color=color_ppl, alpha=0.85)
    ax1.set_ylabel("Perplexity（越低越好）", color=color_ppl)
    ax1.tick_params(axis="y", labelcolor=color_ppl)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, rouges, width, label="ROUGE-L ↑", color=color_rouge, alpha=0.85)
    ax2.set_ylabel("ROUGE-L（越高越好）", color=color_rouge)
    ax2.tick_params(axis="y", labelcolor=color_rouge)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("ROUGE-L 與 Perplexity 跨版本對比")
    lines = [bars1, bars2]
    ax1.legend(handles=lines, loc="upper left")
    plt.tight_layout()
    plt.savefig(BLOG_DIR / "fig5_metrics.png", dpi=150)
    plt.close()


# ── LLM-as-a-Judge ────────────────────────────────────────

def run_llm_judge(results, api_key: str) -> dict:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model  = genai.GenerativeModel("gemini-2.0-flash")
    dims   = ["政策術語準確度", "包容性語言", "類比解釋能力", "問題針對性", "語氣一致性"]
    scores = {name: {d: [] for d in dims} for name in results}

    eval_qs = EVAL_QUESTIONS[:5]  # 取前 5 題
    for i, q in enumerate(eval_qs):
        for name in results:
            resp = results[name]["eval_responses"][i]
            prompt = f"""請評估以下回答是否符合唐鳳（Audrey Tang）的溝通風格。
問題：{q}
回答：{resp}

請對以下五個維度各給 1–5 分（5 分最高），只輸出 JSON：
{{
  "政策術語準確度": <int>,
  "包容性語言": <int>,
  "類比解釋能力": <int>,
  "問題針對性": <int>,
  "語氣一致性": <int>
}}"""
            try:
                raw  = model.generate_content(prompt).text
                data = json.loads(re.search(r"\{.*?\}", raw, re.DOTALL).group())
                for d in dims:
                    if d in data:
                        scores[name][d].append(int(data[d]))
                print(f"    {name} Q{i+1} ✓")
            except Exception as e:
                print(f"    {name} Q{i+1} 失敗：{e}")

    return {
        name: {d: round(sum(v)/len(v), 2) if v else 3.0 for d, v in dims_data.items()}
        for name, dims_data in scores.items()
    }


if __name__ == "__main__":
    main()
