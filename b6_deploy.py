"""
b6-deploy: b3-lora adapter merge → GGUF → Ollama

流程：
  1. 以 BF16 載入 base model
  2. 套上 b3-lora adapter 並 merge_and_unload()
  3. 存成 HF 格式到 outputs/b6_merged/
  4. 呼叫 convert_hf_to_gguf.py 轉成 GGUF
  5. llama-quantize 量化成 Q4_K_M（縮小體積）
  6. 寫 Modelfile 並 ollama create tangram

DoD: ollama run tangram 可以對話
"""

import subprocess
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = Path("checkpoints/b3/adapter")
MERGED_DIR = Path("outputs/b6_merged")
GGUF_PATH = Path("outputs/tangram_f16.gguf")
GGUF_Q4_PATH = Path("outputs/tangram.gguf")
MODELFILE_PATH = Path("outputs/Modelfile")
LLAMA_CPP_DIR = Path("vendor/llama.cpp")
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
LLAMA_QUANTIZE = "llama-quantize"


def step1_merge():
    print("=== Step 1: Merge b3-lora adapter into base model ===")
    print(f"Loading base model in BF16: {BASE_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # BF16 確保合併精度，避免性能損失
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    print(f"Loading adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to: {MERGED_DIR}")
    model.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)

    print("Step 1 complete.\n")


def _ensure_llama_cpp():
    """Clone llama.cpp if not present (shallow, ~10MB)."""
    if not LLAMA_CPP_DIR.exists():
        print("Cloning llama.cpp (shallow)...")
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/ggml-org/llama.cpp.git",
             str(LLAMA_CPP_DIR)],
            check=True,
        )
        print("Clone complete.\n")


def step2_convert_gguf():
    print("=== Step 2: Convert to GGUF (F16) ===")
    _ensure_llama_cpp()

    env = {**__import__("os").environ, "PYTHONPATH": str(LLAMA_CPP_DIR)}
    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT),
        str(MERGED_DIR),
        "--outfile", str(GGUF_PATH),
        "--outtype", "f16",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)
    print(f"GGUF saved to: {GGUF_PATH}\n")


def step3_quantize():
    print("=== Step 3: Quantize to Q4_K_M ===")
    cmd = [LLAMA_QUANTIZE, str(GGUF_PATH), str(GGUF_Q4_PATH), "Q4_K_M"]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    size_gb = GGUF_Q4_PATH.stat().st_size / 1e9
    print(f"Q4_K_M GGUF saved to: {GGUF_Q4_PATH} ({size_gb:.2f} GB)\n")


def step4_ollama_create():
    print("=== Step 4: Create Ollama model 'tangram' ===")

    system_prompt = (
        "你是唐鳳，台灣數位部長，以開放、透明、包容的方式與公民溝通。"
        "請用她的風格：善用具體類比、避免官腔、讓非技術背景的人也能理解。"
    )

    modelfile_content = f"""FROM {GGUF_Q4_PATH.resolve()}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
PARAMETER stop "<|start_header_id|>"

SYSTEM \"\"\"{system_prompt}\"\"\"
"""
    MODELFILE_PATH.write_text(modelfile_content, encoding="utf-8")
    print(f"Modelfile written to: {MODELFILE_PATH}")

    cmd = ["ollama", "create", "tangram", "-f", str(MODELFILE_PATH)]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("\nStep 4 complete.")
    print("\n=== DoD Check ===")
    print("Run:  ollama run tangram")
    print("Test: 請用簡單的話解釋什麼是區塊鏈？")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="b6-deploy pipeline")
    parser.add_argument(
        "--step",
        choices=["merge", "gguf", "quantize", "ollama", "all"],
        default="all",
        help="執行單一步驟或全流程（預設 all）",
    )
    args = parser.parse_args()

    if args.step in ("merge", "all"):
        step1_merge()
    if args.step in ("gguf", "all"):
        step2_convert_gguf()
    if args.step in ("quantize", "all"):
        step3_quantize()
    if args.step in ("ollama", "all"):
        step4_ollama_create()
