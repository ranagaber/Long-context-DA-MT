import os
import torch
import datasets
from datasets import load_dataset, Dataset
from huggingface_hub import login 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd 

login(token = '')



url = "https://huggingface.co/datasets/RanaGaber/Rewayat_NO/resolve/main/data/train-00000-of-00001.parquet"

df = pd.read_parquet(url)
df = df[df['has_arabic_leftover'] == True]
df = df.drop(columns = ['has_arabic_leftover' , 'english_translation'])
print(df.columns)
ds = Dataset.from_pandas(df)
ds = ds.select(range(20))
print(len(ds))

model_id = "CohereLabs/aya-101"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"  
)

model.config.use_cache = True

def translate_data_batch(texts):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    prompts = [
         
                "As an Egyptian Arabic and English linguist, "
                "translate the following Egyptian Arabic sentence to English.\n"
                "If a proper name does not have a well-known English equivalent, transliterate it into Latin script. "
                "Do not leave any Arabic script in the output.\n"
                f"Egyptian Arabic: {text}\n"
                "English:"
            
        for text in texts
    ]
    
    inputs = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024)
    
   
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_outputs
def batched(batch):
    translated_texts = translate_data_batch(batch["codafied_text"])
    return {"english_translation": translated_texts}

my_dataset = ds.map(
    batched, 
    batched=True, 
    batch_size=2
)

print("\n--- Translation Sample ---")
print(my_dataset[0])

my_df = my_dataset.to_pandas()
my_df.to_csv('dataset.csv' , encoding='utf-8-sig')