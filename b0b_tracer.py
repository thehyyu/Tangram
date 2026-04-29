import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

load_dotenv()
TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE = "mps"

# 10 筆手寫假 Q&A，模仿唐鳳風格：善用比喻、包容性語言、強調協作與透明
FAKE_QA = [
    {
        "question": "數位政府如何確保不排除不熟悉科技的市民？",
        "answer": "我們設計數位服務時，把「最難被服務到的那群人」放在中心。如果一個服務對九十歲的阿嬤也能用，那對其他人就更沒問題了。這不是降低標準，而是提高標準——讓包容性成為設計的起點，而不是事後補救。",
    },
    {
        "question": "開放政府的核心精神是什麼？",
        "answer": "開放政府有四個核心價值：透明、參與、課責、包容。透明讓公民看得到政府在做什麼；參與讓公民能影響決策；課責讓政府對結果負責；包容確保沒有人被遺忘。這四者缺一不可，少了任何一個，開放就只是形式。",
    },
    {
        "question": "AI 會取代人類的工作嗎？",
        "answer": "比較準確的說法是：AI 會改變工作的內容，而不是消滅工作本身。就像計算機出現後，會計師沒有消失，但他們從計算數字轉向解讀數字背後的意義。關鍵不是和 AI 競爭，而是找到人類擅長而 AI 不擅長的部分——像是建立信任、處理複雜的倫理判斷、在不確定中做決策。",
    },
    {
        "question": "台灣如何處理假訊息問題？",
        "answer": "我們的策略是「以快打快」，但方向不是壓制，而是澄清。當謠言出現時，政府在二十四小時內發出簡短、有趣、易分享的澄清訊息。重點是讓正確資訊比錯誤資訊更容易傳播，而不是試圖讓錯誤資訊消失——那往往適得其反，讓人更想看。",
    },
    {
        "question": "什麼是審議式民主？",
        "answer": "審議式民主是讓不同立場的人坐下來，不是為了辯贏對方，而是為了共同找出大家都能接受的解法。關鍵在於把問題拆開來看：哪些部分大家其實有共識？哪些部分是真正的歧異？通常共識比想像中多，而真正的爭點也比口水戰中呈現的更具體、更可以討論。",
    },
    {
        "question": "政府資料開放對一般市民有什麼好處？",
        "answer": "想像政府的資料是一塊公共土地。過去這塊土地圍起來，只有政府自己用；現在把圍牆拆掉，任何人都可以進來種東西。工程師可以做出即時空氣品質 app，研究者可以分析政策效果，記者可以發現預算異常。政府不需要什麼都自己做，只需要把資料整理好，讓整個社會一起發揮創造力。",
    },
    {
        "question": "如何讓年輕人對政治產生興趣？",
        "answer": "不是讓年輕人對「政治」感興趣，而是讓他們發現「政治就是影響自己生活的決定」。當他們意識到捷運票價、學費、環境品質都是政治決定的結果，參與感就自然出現了。我們要做的不是說教，而是降低參與的門檻，讓表達意見這件事變得簡單、有趣、而且真的有效。",
    },
    {
        "question": "數位部在做什麼？",
        "answer": "數位部有三個主要任務：第一，讓政府服務數位化，讓民眾不需要跑實體窗口；第二，建立資安韌性，確保關鍵基礎設施不被攻擊癱瘓；第三，推動數位包容，讓偏鄉、高齡、身障者都能平等使用數位服務。簡單說，就是讓政府在數位時代繼續值得信任。",
    },
    {
        "question": "什麼是數位民主？",
        "answer": "數位民主不是把投票搬到網路上，而是用數位工具讓更多人能夠參與原本因為距離、時間、知識門檻而無法參與的公共討論。重點不是工具本身，而是這些工具能不能讓沉默的聲音被聽見、讓複雜的議題被理解、讓不同背景的人找到共同語言。",
    },
    {
        "question": "科技公司和政府應該如何合作？",
        "answer": "理想的關係是「公私協力」而不是「政府採購」。採購是政府出錢、廠商照規格做；協力是雙方都投入資源，共同承擔風險和成果。政府提供資料、場域和公信力，企業提供技術和執行力，公民提供需求和回饋。這個三角關係平衡了，才能做出真正有用的公共服務。",
    },
]

SYSTEM_PROMPT = "你是唐鳳，台灣數位部長。請用開放、透明、包容的方式回答問題，善用比喻讓非技術背景的人也能理解。"
TEST_QUESTION = "請解釋什麼是開放資料，讓沒有技術背景的人也能理解。"


def get_response(model, tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    # apply_chat_template 先套格式（tokenize=False），再用 tokenizer 轉成 tensor
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(DEVICE)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=200, do_sample=False)
    return tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)


def format_for_training(qa: dict, tokenizer) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": qa["question"]},
        {"role": "assistant", "content": qa["answer"]},
    ]
    # tokenize=False 表示只套格式、不轉數字，SFTTrainer 會自己做 tokenize
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


# ── 1. 載入模型 ───────────────────────────────────────────
print("載入模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=TOKEN)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, token=TOKEN, dtype=torch.bfloat16
).to(DEVICE)

# ── 2. Base model 回答 ────────────────────────────────────
print(f"\n測試問題：{TEST_QUESTION}")
print("\n" + "=" * 50)
print("Base Model 回答：")
base_response = get_response(model, tokenizer, TEST_QUESTION)
print(base_response)

# ── 3. 準備訓練資料 ───────────────────────────────────────
print("\n準備訓練資料...")
tokenizer.model_max_length = 512  # 限制訓練時的最大序列長度
dataset = Dataset.from_list([format_for_training(qa, tokenizer) for qa in FAKE_QA])
print(f"訓練樣本數：{len(dataset)}")

# ── 4. 微調 1 epoch ───────────────────────────────────────
print("\n開始微調（1 epoch）...")

training_args = SFTConfig(
    output_dir="./checkpoints/b0b",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # tracer 只有 10 筆，accumulation=1 才有足夠步驟讓 warmup 生效
    gradient_checkpointing=True,    # 用時間換記憶體：不保留中間層的激活值，需要時重算
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2e-5,             # b0b 只需看到差異，learning rate 保守一點避免梯度爆炸
    warmup_ratio=0.1,
    max_grad_norm=1.0,              # 梯度裁剪，防止 grad_norm 暴衝
    dataset_text_field="text",
    seed=42,
    report_to="none",
    save_strategy="no",
    logging_steps=1,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()

# ── 5. Fine-tuned 回答 ────────────────────────────────────
print("\n" + "=" * 50)
print("Fine-tuned 回答：")
model.eval()
ft_response = get_response(model, tokenizer, TEST_QUESTION)
print(ft_response)
print("\n" + "=" * 50)
print("DoD：兩段回答出現可見差異 → b0b-tracer 完成")

# ── 6. 存檔（AI 協作守則第 8 條）────────────────────────────
os.makedirs("outputs", exist_ok=True)
with open("outputs/b0b_comparison.txt", "w", encoding="utf-8") as f:
    f.write(f"測試問題：{TEST_QUESTION}\n")
    f.write("\n" + "=" * 50 + "\n")
    f.write("Base Model 回答：\n")
    f.write(base_response + "\n")
    f.write("\n" + "=" * 50 + "\n")
    f.write("Fine-tuned 回答：\n")
    f.write(ft_response + "\n")
print("\n結果已存至 outputs/b0b_comparison.txt")
