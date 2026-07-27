import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from tqdm import tqdm
import evaluate
import sacrebleu
import re
from datasets import load_dataset
from sacrebleu.metrics import  BLEU 
from huggingface_hub import login
from comet import download_model, load_from_checkpoint

model_token = ''
login(token = model_token)

'''
en = pd.read_csv("./data/flores_eng_devtest.csv", encoding="utf-8-sig")
ar = pd.read_csv("./data/flores_egy_devtest.csv", encoding="utf-8-sig")

df = pd.DataFrame({
    "target": ar["sentence"],
    "source": en["sentence"]
})
'''

ds = load_dataset("google/wmt24pp", "en-ar_EG")

data = ds['train']

filtered_ds = data.filter(lambda ex: not ex["is_bad_source"])
df = filtered_ds.to_pandas()



def clean_output(text):

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)

    text = re.sub(
        r"^(The following is a translation.*?:)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def inference(model_id, tokenizer_id ,df, batch_size=128, device="cuda", debug=True):

    print(f"\n========== LOADING {model_id} ==========")

    sources = df["source"].tolist()
    generated_text = []

    # CASE 1 Nile Chat (Pipeline Interface)
    if "Nile-Chat" in model_id:
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"  

        pipe = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer,
            torch_dtype="auto",   
            device_map="auto",    
        )

        for i in tqdm(range(0, len(sources), batch_size)):
            batch = sources[i:i + batch_size]
            
            batch_messages = [
                [{
                    "role": "user",
                    "content":
                        "You are a translation expert. Translate the following English sentences to Egyptian Arabic.\n"
                        f"English: {s}\n"
                        "Egyptian Arabic:"
                }] for s in batch
            ]

            outputs = pipe(
                batch_messages, 
                batch_size=len(batch),
                #max_length = 384 ,
                max_new_tokens= 384, 
                do_sample=False
            )

            for out in outputs:
                assistant_response = out[0]["generated_text"][-1]["content"].strip()
                final_translation = clean_output(assistant_response)
                generated_text.append(final_translation)

            if debug and i == 0:
                print("\n===== GENERATION SAMPLE (NILE) =====")
                print(repr(generated_text[0]))

        del pipe
        del tokenizer
        torch.cuda.empty_cache()

    # CASE 2  Qwen (Chat Template)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        #model = PeftModel.from_pretrained(model, adapter)
        #model = model.merge_and_unload()
        model.eval()

        for i in tqdm(range(0, len(sources), batch_size)):
            batch = sources[i:i + batch_size]

            batch_messages = [
                [{
                    "role": "user",
                    "content":
                        "You are a translation expert. Translate the following English sentences to Egyptian Arabic.\n"
                        f"English: {s}\n"
                        "Egyptian Arabic:"
                }] for s in batch
            ]

            inputs = tokenizer.apply_chat_template(
                batch_messages,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_generation_prompt=True,
                return_dict=True,
                enable_thinking=False,
                #max_length = 384
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=384,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            generated = outputs[:, input_len:]

            translations = tokenizer.batch_decode(
                generated,
                skip_special_tokens=True
            )

            for t in translations:
                generated_text.append(clean_output(t))

            if debug and i == 0:
                print("\n===== GENERATION SAMPLE (QWEN) =====")
                print(repr(generated_text[-len(batch)]))

        del model
        del tokenizer
        torch.cuda.empty_cache()

    # Safe dataframe column name assignment
    col_name = model_id.split("/")[-1]
    df[col_name] = generated_text

    return df



models = [
    "SeragAmin/gemma_4b_rewayat"
]


tokenizers = [
 "SeragAmin/gemma_4b_rewayat"

]

for model_id, tokenizer_id in zip(models, tokenizers ):

    print("\n========== DATASET DEBUG ==========")
    print(df.head(3))

    print("\nSOURCE (Egyptian Arabic):")
    print(repr(df["source"].iloc[0]))

    print("\nTARGET (English reference):")
    print(repr(df["target"].iloc[0]))
    print("===================================\n")

    df = inference(model_id, tokenizer_id , df)

    print(f"{model_id} done")


df.to_csv(
    "./Decoder_model_rewayat_WMT.csv",
    encoding="utf-8-sig",
    index=False
)
#bleu = evaluate.load("bleu")
bleu = BLEU()
#chrf = CHRF(word_order=2)

def eval_model(model_id, df):

    col_name = model_id.split("/")[-1]

    hyp = df[col_name].tolist()
    ref = df["target"].tolist()
    references = [[r] for r in ref]
    chrfpp_score = sacrebleu.corpus_chrf(
        hyp,
        [ref],
        word_order=2
    )
    bleu_score = bleu.corpus_score(hyp, references)

    return {
        "BLEU": bleu_score.score,
        "ChrF++": chrfpp_score.score
    }


model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)
def comet(model_id , df):
    col_name = model_id.split("/")[-1]
    data = [
        {
            "src": str(src).strip(),
            "mt": str(mt).strip(),
            "ref": str(ref).strip()
        }
        for src, mt, ref in zip(
            df["source"],
            df[col_name],
            df["target"]
        )
    ]

    output = comet_model.predict(data, batch_size=16)
    return {'COMET' : output['system_score']}
    
print("\n========== FINAL SCORES ==========")

for model_id in models:
    print(model_id, eval_model(model_id, df))
    print(model_id , comet(model_id , df))