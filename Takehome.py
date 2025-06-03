#Reading data and selecting columns
import pandas as pd

df=pd.read_csv('Q2_20230202_majority.csv')
df=df[['tweet','label_majority']]
df = df.dropna(subset=["tweet", "label_majority"]).reset_index(drop=True)
#Initializing the T5 tokenizer and model 
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
compute_device = "cuda" if torch.cuda.is_available() else "cpu"

#Generating predictions by batching the input tweets

prompt_temp=("What is the stance of the following tweet with respect to COVID-19 vaccine?  Here is the tweet. {t}  Please use exactly one word from the following 3 categories to label it: FAVOUR,AGAINST, NONE")
batch_size=8

inp=df['tweet'].tolist()
out = [] # Initialize out
for i in tqdm(range(0,len(inp),batch_size)):
  batch=inp[i:i+batch_size]
  inp_prompt=[prompt_temp.format(t=tweet) for tweet in batch]
  tokens = tokenizer(
        inp_prompt,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    ).to("cuda")
  with torch.no_grad():
        model.to("cuda") # Move model to GPU
        generated = model.generate(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            max_new_tokens=4,
            num_beams=3,
            early_stopping=True
        )
  batch_preds = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True
    )
  out.extend(batch_preds)

df["label_pred"]=out
df.to_csv("Q2_20230202_majority.csv",index=False)