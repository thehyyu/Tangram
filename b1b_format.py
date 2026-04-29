import json
import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

load_dotenv()
TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
RAW_DATA_PATH = Path("data/raw_speeches.json")
DATASET_PATH = Path("data/dataset")

SYSTEM_PROMPT = (
    "你是唐鳳，台灣數位部長。"
    "請用開放、透明、包容的方式回答問題，善用比喻讓非技術背景的人也能理解。"
)
MIN_ANSWER_TOKENS = 50  # 回答太短代表沒有完整論述，學不到風格
MAX_SEQ_LENGTH = 1024


def get_year(date_str: str) -> int:
    return int(date_str[:4]) if date_str else 0


def year_to_split(year: int) -> str | None:
    # 時間切分，避免 data leakage（同場活動問答不會跨越年份邊界）
    if 2020 <= year <= 2024:
        return "train"
    if year == 2025:
        return "validation"
    if year == 2026:
        return "test"
    return None  # 2019 以前或範圍外，跳過


# ── 1. 載入 tokenizer ────────────────────────────────────
print("載入 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# ── 2. 載入原始資料 ───────────────────────────────────────
print("載入原始資料...")
with open(RAW_DATA_PATH, encoding="utf-8") as f:
    speeches = json.load(f)

# 展平成單筆 Q&A，並標記所屬 split
all_pairs = []
for speech in speeches:
    year = get_year(speech["date"])
    split = year_to_split(year)
    if not split:
        continue
    for qa in speech["qa_pairs"]:
        all_pairs.append({
            "question": qa["question"],
            "answer": qa["answer"],
            "date": speech["date"],
            "split": split,
        })

print(f"原始 Q&A pairs：{len(all_pairs)}")

# ── 3. 品質過濾 ───────────────────────────────────────────
# 用 tokenizer 計算回答的 token 數，過濾掉太短的 Q&A
print(f"過濾回答 < {MIN_ANSWER_TOKENS} tokens 的 Q&A...")
filtered = [
    p for p in tqdm(all_pairs, desc="過濾中")
    if len(tokenizer.encode(p["answer"], add_special_tokens=False)) >= MIN_ANSWER_TOKENS
]
print(f"過濾後：{len(filtered)} 筆（移除 {len(all_pairs) - len(filtered)} 筆）")

# ── 4. 套用 chat template ─────────────────────────────────
def to_text(pair: dict) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": pair["question"]},
        {"role": "assistant", "content": pair["answer"]},
    ]
    return {
        "text": tokenizer.apply_chat_template(messages, tokenize=False),
        "date": pair["date"],
    }

# ── 5. 時間切分 ───────────────────────────────────────────
splits: dict[str, list] = {"train": [], "validation": [], "test": []}
for p in filtered:
    splits[p["split"]].append(to_text(p))

print(f"\n資料切分結果：")
print(f"  訓練集（2020–2024）：{len(splits['train']):>6} 筆")
print(f"  驗證集（2025）    ：{len(splits['validation']):>6} 筆")
print(f"  測試集（2026）    ：{len(splits['test']):>6} 筆")

# 測試集數量警告：太少會讓 b5 評估結果不具代表性
if len(splits["test"]) < 100:
    print(f"\n⚠️  警告：測試集只有 {len(splits['test'])} 筆，建議評估時搭配人工抽樣，數字參考意義有限。")

# ── 6. 存成 HuggingFace DatasetDict ──────────────────────
dataset = DatasetDict({
    "train":      Dataset.from_list(splits["train"]),
    "validation": Dataset.from_list(splits["validation"]),
    "test":       Dataset.from_list(splits["test"]),
})

DATASET_PATH.mkdir(parents=True, exist_ok=True)
dataset.save_to_disk(str(DATASET_PATH))
print(f"\n已存至 {DATASET_PATH}")

# ── 7. DoD 驗證：印出一筆 tokenized 樣本 ─────────────────
print("\n" + "=" * 55)
print("DoD 驗證：訓練集第 1 筆")
print("=" * 55)

sample_text = dataset["train"][0]["text"]
input_ids = tokenizer(
    sample_text,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_SEQ_LENGTH,
)["input_ids"][0].tolist()

print("\ntext（前 300 字）：")
print(sample_text[:300])
print(f"\ninput_ids（前 20 個）：")
print(input_ids[:20])
print(f"\ninput_ids 總長度：{len(input_ids)} tokens")
print("\nDoD：可看到 input_ids 數列，格式符合 Llama 3 chat template ✓")
