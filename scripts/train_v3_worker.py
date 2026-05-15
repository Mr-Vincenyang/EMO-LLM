#!/usr/bin/env python3
"""Train ONE style LoRA on ALL ESConv data with contrastive prompt."""
import sys, json, time, random
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq

sys.path.insert(0, ".")
from src.utils import STYLE_SYSTEM_PROMPTS, load_jsonl

style = sys.argv[1]
gpu = int(sys.argv[2]) if len(sys.argv) > 2 else 0

# Load ALL ESConv data
random.seed(42)
all_data = []
for sf in Path("data/train").glob("*.jsonl"):
    if sf.stem == "all_styles":
        continue
    all_data.extend(load_jsonl(str(sf)))
if len(all_data) > 1000:
    all_data = random.sample(all_data, 1000)

print(f"[{style}] GPU={gpu}, samples={len(all_data)}")

tokenizer = AutoTokenizer.from_pretrained("./models/Qwen/Qwen3-1.7B", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    "./models/Qwen/Qwen3-1.7B", dtype=torch.float16, trust_remote_code=True,
    device_map={"": gpu}
)

sys_prompt = STYLE_SYSTEM_PROMPTS[style]
formatted = []
for item in all_data:
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": item["user"]},
        {"role": "assistant", "content": item["assistant"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    formatted.append({"text": text})

dataset = Dataset.from_list(formatted)
def tok_fn(ex):
    r = tokenizer(ex["text"], truncation=True, max_length=1024, padding=False)
    r["labels"] = [ids.copy() for ids in r["input_ids"]]
    return r
dataset = dataset.map(tok_fn, batched=True, remove_columns=dataset.column_names)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none",
)
peft_model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir=f"outputs/tmp_v3_{style}",
    per_device_train_batch_size=4, gradient_accumulation_steps=4,
    num_train_epochs=3, learning_rate=2e-4,
    warmup_ratio=0.1, lr_scheduler_type="cosine",
    logging_steps=10, save_strategy="no", fp16=True, report_to="none", seed=42,
)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=peft_model, padding=True, label_pad_token_id=-100)
trainer = Trainer(model=peft_model, args=training_args, train_dataset=dataset, data_collator=data_collator)

t0 = time.time()
result = trainer.train()
t1 = time.time()

save_path = Path(f"outputs/lora_v3/{style}")
save_path.mkdir(parents=True, exist_ok=True)
peft_model.save_pretrained(str(save_path))
tokenizer.save_pretrained(str(save_path))

m = {"style": style, "loss": round(result.training_loss, 4), "time": round(t1 - t0, 1), "gpu": gpu}
with open(f"outputs/lora_v3/{style}/metrics.json", "w") as f:
    json.dump(m, f)
print(f"DONE:{style}:loss={m['loss']}:time={m['time']}s")
