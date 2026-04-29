import os
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()
token = os.getenv("HF_TOKEN")

MODEL = "meta-llama/Llama-3.2-3B-Instruct"

print("載入模型中，第一次會下載約 6GB，請稍候...")

pipe = pipeline(
    "text-generation",
    model=MODEL,
    device="mps",
    token=token,
)

response = pipe(
    [{"role": "user", "content": "請用一句話解釋什麼是開放政府。"}],
    max_new_tokens=200,
)

print("\n模型回答：")
print(response[0]["generated_text"][-1]["content"])
