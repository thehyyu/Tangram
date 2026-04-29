import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from datasets import load_from_disk, Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig

load_dotenv()
TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DATASET_PATH = Path("data/dataset")
CHECKPOINT_DIR = Path("checkpoints/b3")
OUTPUT_LOG = Path("outputs/b3_lora_log.json")

ALPACA_RATIO = 0.05
MAX_SEQ_LENGTH = 1024   # LoRA 記憶體小，可以回到 1024
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
REPORT_TO = "wandb"

# ── Smoke Test 旗標 ───────────────────────────────────────
# True  → 只跑 100 steps、取前 500 筆，確認鏈路通再放夜跑
# False → 正式訓練（3 epochs、全資料）
SMOKE_TEST = False


def format_alpaca(sample: dict, tokenizer) -> dict:
    user_content = sample["instruction"]
    if sample.get("input"):
        user_content += f"\n{sample['input']}"
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": sample["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


# ── 1. 載入 tokenizer ────────────────────────────────────
print("載入 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = MAX_SEQ_LENGTH

# ── 2. 載入 Tangram 資料集 ────────────────────────────────
print("載入 Tangram 資料集...")
dataset = load_from_disk(str(DATASET_PATH))
train_ds = dataset["train"].select_columns(["text"])
val_ds = dataset["validation"].select_columns(["text"])
print(f"  訓練集：{len(train_ds)} 筆，驗證集：{len(val_ds)} 筆")

# ── 3. 混入 5% Alpaca ─────────────────────────────────────
alpaca_n = int(len(train_ds) * ALPACA_RATIO)
print(f"載入 Alpaca 資料集（取 {alpaca_n} 筆）...")
alpaca_raw = load_dataset("yahma/alpaca-cleaned", split=f"train[:{alpaca_n}]")
alpaca_ds = Dataset.from_list([format_alpaca(s, tokenizer) for s in alpaca_raw])
train_mixed = concatenate_datasets([train_ds, alpaca_ds]).shuffle(seed=42)
print(f"  混合後訓練集：{len(train_mixed)} 筆")

if SMOKE_TEST:
    train_mixed = train_mixed.select(range(500))
    print(f"  [Smoke Test] 縮減至前 500 筆")

# ── 4. 載入模型 ───────────────────────────────────────────
print(f"\n載入模型（{DEVICE}, bfloat16）...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=TOKEN,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

# ── 5. 套上 LoRA adapter ──────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                                  # adapter 矩陣的秩（rank）
    lora_alpha=16,                        # 縮放係數，慣例設為 r 的 2 倍
    target_modules=["q_proj", "v_proj"],  # 只插入注意力機制的 Q 和 V 矩陣
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)

# DoD：確認可訓練參數 < 1%
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
ratio = trainable / total * 100
print(f"\n  全部參數：  {total / 1e9:.3f}B")
print(f"  可訓練參數：{trainable / 1e6:.2f}M ({ratio:.3f}%)")
assert ratio < 1.0, f"可訓練參數比例 {ratio:.3f}% 超過 1%，請檢查 LoRA 設定"
print(f"  DoD 確認：可訓練參數 < 1% ✓")

# ── 6. 訓練設定 ───────────────────────────────────────────
if SMOKE_TEST:
    extra_args = dict(max_steps=100, logging_steps=10, eval_strategy="no",
                      save_strategy="no", load_best_model_at_end=False,
                      run_name="tangram-b3-smoke")
    print("\n[Smoke Test] max_steps=100，確認鏈路後將 SMOKE_TEST 改為 False 跑正式訓練")
else:
    extra_args = dict(logging_steps=50, eval_strategy="epoch",
                      save_strategy="steps", save_steps=500,
                      load_best_model_at_end=True,
                      metric_for_best_model="eval_loss", greater_is_better=False,
                      run_name="tangram-b3-lora")

training_args = SFTConfig(
    output_dir=str(CHECKPOINT_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    dataset_text_field="text",
    seed=42,
    report_to=REPORT_TO,
    **extra_args,
)

# ── 7. 訓練 ───────────────────────────────────────────────
callbacks = [] if SMOKE_TEST else [EarlyStoppingCallback(early_stopping_patience=2)]
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_mixed,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    callbacks=callbacks,
)

print("\n開始訓練...")
train_result = trainer.train()

# ── 8. 存 adapter checkpoint ─────────────────────────────
if not SMOKE_TEST:
    adapter_dir = CHECKPOINT_DIR / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nLoRA adapter 已存至 {adapter_dir}")

# ── 9. 存訓練 log ─────────────────────────────────────────
OUTPUT_LOG.parent.mkdir(parents=True, exist_ok=True)
log = {
    "lora_r": lora_config.r,
    "lora_alpha": lora_config.lora_alpha,
    "target_modules": list(lora_config.target_modules),
    "trainable_params": trainable,
    "trainable_ratio_pct": round(ratio, 4),
    "train_runtime_sec": train_result.metrics.get("train_runtime"),
    "train_loss": train_result.metrics.get("train_loss"),
    "history": trainer.state.log_history,
}
with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
    json.dump(log, f, ensure_ascii=False, indent=2)
print(f"訓練 log 已存至 {OUTPUT_LOG}")

# ── 10. DoD 輸出 ──────────────────────────────────────────
print("\n" + "=" * 55)
print("DoD 驗證")
print("=" * 55)
print(f"  可訓練參數 < 1%  ✓ ({ratio:.3f}%)")
train_losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
eval_losses  = [e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e]
print(f"  final train_loss = {train_losses[-1]:.4f}" if train_losses else "  train_loss: 無資料")
print(f"  final eval_loss  = {eval_losses[-1]:.4f}"  if eval_losses  else "  eval_loss: (smoke test 未評估)")
print("\n詳細曲線見 outputs/b3_lora_log.json 或 WandB dashboard")
