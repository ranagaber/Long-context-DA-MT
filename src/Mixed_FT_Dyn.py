from collator.collator import DynamicCollator
from configs import *
from functools import partial
from huggingface_hub import snapshot_download
from datasets import load_from_disk
import warnings
import pandas as pd
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, default_data_collator, Trainer, TrainingArguments, set_seed, logging 
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from datasets import Dataset, concatenate_datasets
from transformers import BitsAndBytesConfig
import numpy as np
from huggingface_hub import login, upload_folder
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerCallback
from datasets import load_dataset

#repo_id = 'SeifAI/Nile_Chat_FT_4B_DOC_LEVEL_from_SentenceFinetune_v2'
full_model_path = './full_model_Nile_Chat_4B'
model_token = '' 

login(token = model_token)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

path = snapshot_download(
        repo_id="RanaGaber/SentenceLevelMT",
        repo_type="dataset",
        token=""
    )

dataset_1 = load_from_disk(path)
df1 = dataset_1.to_pandas()

dataset_2 = load_dataset("SeifAI/FineTranselation_EGY_filtered_docs")
df2 = dataset_2["train"].to_pandas()

df2 = df2.rename(columns={
        "og_full_text": "Egyptian",
        "translated_text": "English"
    })

df1 = df1[["Egyptian", "English"]]
df2 = df2[["Egyptian", "English"]]

merged_df = pd.concat([df1, df2], ignore_index=True)

shuffled_df = merged_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

dataset = Dataset.from_pandas(shuffled_df, preserve_index=False)


tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
print("Model: " , model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def preprocess(examples, max_length=2096):
    input_ids_list = []
    labels_list = []

    for source, target in zip(examples["English"], examples["Egyptian"]):
        source = str(source) if source else ""
        target = str(target) if target else ""

        messages = [
            {"role": "user", "content": f"You are a translation expert. Translate: {source}\n Egyptian Arabic:"},
            {"role": "assistant", "content": target}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        full_enc = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length
        )
        full_ids = full_enc["input_ids"]

        prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"You are a translation expert. Translate: {source}\n Egyptian Arabic:"}],
            tokenize=False,
            add_generation_prompt=True,
        )

        prompt_len = len(tokenizer(
            prompt_only,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length
        )["input_ids"])

        labels = [-100] * len(full_ids)
        labels[prompt_len:] = full_ids[prompt_len:]

        input_ids_list.append(full_ids)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
    }
tokenized_train = dataset.map(preprocess, batched=True, num_proc=4)

# ==========================================
# MULTI-SAMPLE MASKING VERIFICATION
# ==========================================
print("\n" + "="*70)
print("MULTI-SAMPLE MASKING VERIFICATION")
print("="*70)

for i in range(5):
    sample_input = tokenized_train[i]['input_ids']
    sample_label = tokenized_train[i]['labels']

    full_text = tokenizer.decode([t for t in sample_input if t != tokenizer.pad_token_id], skip_special_tokens=False)
    
    active_tokens = [t for t in sample_label if t != -100]
    learned_text = tokenizer.decode(active_tokens, skip_special_tokens=False)

    print(f"\n[SAMPLE {i}]")
    print(f"--- FULL CONTEXT ---")
    print(repr(full_text))
    print(f"--- LEARNED TARGET ---")
    print(repr(learned_text))
    
    if learned_text.strip().startswith("English:") or "assistant" in learned_text.lower():
        print(">> [!] WARNING: Template headers detected in learned target.")
    else:
        print(">> [SUCCESS]: Target starts cleanly at the translation.")
    print("-" * 40)

print("="*70 + "\n")
# ==========================================

data_collator = DynamicCollator(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir = OUTPUT_DIR,
    save_strategy="steps",
    save_steps= SAVE_STEPS,
    seed = SEED,
    learning_rate= lr,
    per_device_train_batch_size=TRAIN_BATCH,
    per_device_eval_batch_size=VAL_BATCH,
    gradient_accumulation_steps= GRAD_ACCUM,
    weight_decay=WEIGHT_DECAY,
    save_total_limit=2,
    num_train_epochs=NUM_EPOCHS,
    fp16=False,
    bf16=True,
    logging_steps=LOGGING_STEPS,
    report_to="none",
)

model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(full_model_path)
#upload_folder(folder_path=full_model_path, repo_id= repo_id, repo_type="model", token = model_token)
