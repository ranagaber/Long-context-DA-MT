import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from tqdm import tqdm
import evaluate
import sacrebleu
from sacrebleu.metrics import  BLEU 
import re
from datasets import load_dataset
from comet import download_model, load_from_checkpoint

'''

en = pd.read_csv("/teamspace/studios/this_studio/data/flores_eng_devtest.csv", encoding="utf-8-sig")
ar = pd.read_csv("/teamspace/studios/this_studio/data/flores_egy_devtest.csv", encoding="utf-8-sig")

df = pd.DataFrame({
    "source": ar["sentence"],
    "target": en["sentence"]
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

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto" if device.type == "cuda" else None,
    )


   
    tokenizer.src_lang = "eng_Latn" 
    tokenizer.tgt_lang = "arz_Arab"  
    #encoder-decoder
    tokenizer.padding_side = "left"     
    model.eval()

    for i in tqdm(range(0, len(sources), batch_size)):
        batch = sources[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id["arz_Arab"],
                max_new_tokens=256,
                do_sample=False,
            )

        translations = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )

        generated_text.extend([clean_output(t) for t in translations])

        if debug and i == 0:
            print("\n===== GENERATION SAMPLE =====")
            print(repr(generated_text[0]))

    del model
    del tokenizer
    torch.cuda.empty_cache()

    col_name = model_id.split("/")[-1]
    df[col_name] = generated_text

    return df

models = [
   "SeragAmin/opus_sentence",
   "SeragAmin/nllb_sentence"
]
tokenizers = [
    "SeragAmin/opus_sentence",
    "facebook/nllb-200-distilled-600M"
]


for model_id, tokenizer_id in zip(models, tokenizers):

    print("\n========== DATASET DEBUG ==========")
    print(df.head(3))

    print("\nSOURCE (English):")
    print(repr(df["source"].iloc[0]))

    print("\nTARGET (Egyptian Arabic):")
    print(repr(df["target"].iloc[0]))
    print("===================================\n")

    df = inference(model_id, tokenizer_id, df)

    print(f"{model_id} done")


df.to_csv(
    "./exp1_sentence_wmt_seq2seq.csv",
    encoding="utf-8-sig",
    index=False
)


#bleu = evaluate.load("bleu")
bleu = BLEU()
#chrf = CHRF(word_order=2)

def eval_model(model_id, df):

    col_name = model_id.split("/")[-1]

    hyp = df[col_name].fillna("").astype(str).tolist()
    ref = df["target"].fillna("").astype(str).tolist()
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