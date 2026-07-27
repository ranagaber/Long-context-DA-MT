from configs.configs import *
from functools import partial
import warnings
import pandas as pd
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, default_data_collator, Trainer, TrainingArguments, set_seed, logging , AutoModelForSeq2SeqLM , DataCollatorForSeq2Seq
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from datasets import Dataset, concatenate_datasets , load_dataset
from transformers import BitsAndBytesConfig
import numpy as np
from huggingface_hub import login, upload_folder 
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerCallback
from huggingface_hub import snapshot_download
from datasets import load_from_disk

repo_id = ''
full_model_path = './full_model'
model_token = '' 

print(repo_id)
print(full_model_path)
print(OUTPUT_DIR)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

login(token = model_token)


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


'''
path = snapshot_download(
    repo_id="RanaGaber/SentenceLevelMT",
    repo_type="dataset",
    token = '' 
)

dataset = load_from_disk(path)

#########
df = dataset.to_pandas()
df = df.dropna(subset=["Egyptian", "English"])
print("after na:" , len(df))
print(df)
#########

'''
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
model =  AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
print("Model: " , model_id)
print("Tokenizer:" , tokenizer_id)



tokenizer.src_lang = "eng_Latn"
tokenizer.tgt_lang = "arz_Arab"   
tokenizer.padding_side = "right"
def preprocess(examples):
    inputs = tokenizer(
        examples["English"],
        max_length=1024,
        truncation=True,
        padding="max_length",
    )
    
    targets = tokenizer(
        examples["Egyptian"],
        max_length=1024,
        truncation=True,
        padding="max_length",
    )


    labels = [  [(token if token != tokenizer.pad_token_id else -100) for token in seq] for seq in targets["input_ids"]]
    inputs["labels"] = labels
    return inputs

#dataset = dataset.filter(lambda x: x["Egyptian"] is not None and x["English"] is not None)
tokenized_train = dataset.map(preprocess, batched=True, num_proc=1)


#data_collator = DataCollatorForSeq2Seq(
#    tokenizer=tokenizer,
#    model=model,
#)
data_collator = default_data_collator

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
upload_folder(folder_path=full_model_path, repo_id= repo_id, repo_type="model", token = model_token)

