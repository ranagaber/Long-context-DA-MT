import numpy as np
import pandas as pd
import laser_encoders 
from laser_encoders import LaserEncoderPipeline
import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset 
from tqdm import tqdm

ds = load_dataset("RanaGaber/Egyptian_Rewayat")
ds = ds['train']
df = ds.to_pandas()

arabic_text = df['codafied_text'].to_list()
english_text = df['english_translation'].to_list()

encoder_ar = LaserEncoderPipeline(lang="arz_Arab")
encoder_en = LaserEncoderPipeline(lang="eng_Latn")

BATCH_SIZE = 128
similarity_scores = []


for i in tqdm(range(0, len(arabic_text), BATCH_SIZE) , desc = 'processing'):
    batch_ar = arabic_text[i : i + BATCH_SIZE]
    batch_en = english_text[i : i + BATCH_SIZE]
    
    vec_ar = encoder_ar.encode_sentences(batch_ar, normalize_embeddings=True)
    vec_en = encoder_en.encode_sentences(batch_en, normalize_embeddings=True)
    
    batch_scores = np.sum(vec_ar * vec_en, axis=1)    
    similarity_scores.extend(batch_scores.tolist())
    
    print(f"Processed rows {i} to {min(i + BATCH_SIZE, len(arabic_text))}")

df['LASER'] = similarity_scores
df.to_csv('Laser_scores_Rewayat.csv', encoding='utf-8-sig', index=False)

plt.hist(df['LASER'], bins=100)
plt.title('LASER Histogram')
plt.savefig('./LASER_FT_Rewayat.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n--- LASER Score Metrics ---")
print(df['LASER'].describe())

#print("\n--- Sample Results ---")
#print(df[['Egyptian', 'English', 'LASER']].head())