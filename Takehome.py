#Reading data and selecting columns
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
prompt_temp=("What is the stance of the following tweet with respect to COVID-19 vaccine?  Here is the tweet. {t}  Please use exactly one word from the following 3 categories to label it: FAVOUR,AGAINST, NONE")
batch_size=8
class T5Predictor:

  def __init__(self, model):
    self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    self.model = model
    self.compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model.to(self.compute_device)

  def predict(self,inp):
      out = [] # Initialize out
      for i in tqdm(range(0,len(inp),batch_size)):
          batch = inp[i:i+batch_size]
          inp_prompt = [prompt_temp.format(t=tweet) for tweet in batch]
          tokens = self.tokenizer(
              inp_prompt,
              return_tensors="pt",
              padding="longest",
              truncation=True,
          ).to("cuda")
          with torch.no_grad():
              self.model.to("cuda") # Move model to GPU
              generated = self.model.generate(
                  input_ids=tokens.input_ids,
                  attention_mask=tokens.attention_mask,
                  max_new_tokens=4,
                  num_beams=3,
                  early_stopping=True
              )
          batch_preds = self.tokenizer.batch_decode(
              generated,
              skip_special_tokens=True
          )
          out.extend(batch_preds)
      df=pd.read_csv('Q2_20230202_majority.csv')
      df["label_pred"]=out
      df.to_csv("Q2_20230202_majority.csv",index=False)
    


class dataset:
    def __init__(self, fp):
        self.df = pd.read_csv("Q2_20230202_majority.csv")
        #    columns:   tweet (string),  label_true (one of: "in-favor","against","neutral-or-unclear")
        # 1.2 (Optional) Shuffle/drop missing
        self.df = self.df.dropna(subset=["tweet", "label_majority"]).sample(frac=1, random_state=42).reset_index(drop=True)
        # 1.3 Split into train/validation (e.g. 90% train, 10% val)
        from sklearn.model_selection import train_test_split
        self.train_df, self.val_df = train_test_split(
            self.df,
            test_size=0.1,
            stratify=self.df["label_majority"],
            random_state=42
        )
        self.train_df["target_text"] = self.train_df["label_majority"]
        self.val_df["target_text"]   = self.val_df["label_majority"]

# 2.2 Convert to 🤗 Dataset objects
        train_ds = Dataset.from_pandas(self.train_df[["tweet", "target_text"]], preserve_index=False)
        val_ds   = Dataset.from_pandas(self.val_df[["tweet", "target_text"]], preserve_index=False)

        self.tokenized_datasets = DatasetDict({
            "train": train_ds,
            "validation": val_ds
        })
        self.tokenized_datasets = self.tokenized_datasets.map(self.tokenize_function, batched=True)
    def tokenize_function(self,examples):
    # Define maximum lengths
      max_input_length = 128
      max_target_length = 8 # Increased max_target_length to accommodate longer labels if necessary

      tokenized_inputs = self.tokenizer(
          examples["tweet"],
          max_length=max_input_length,
          truncation=True,
          padding="max_length"
      )
      tokenized_inputs["labels"] = self.tokenizer(
          examples["target_text"],
          max_length=max_target_length,
          truncation=True,
          padding="max_length"
      )["input_ids"]
      return tokenized_inputs
class ModelTrainer:
    def __init__(self, model, train_dataset, eval_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def train(self):
        from transformers import Trainer, TrainingArguments
        training_args= TrainingArguments(
        output_dir="optuna-flan-t5-",

        num_train_epochs=3,
        per_device_train_batch_size=2,       # each micro-batch: 2 examples
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,      # effective batch size = 2 × 16 = 32
        fp16=True,                           # use mixed precision
        #evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        learning_rate=2e-5,
        logging_steps=100,
        remove_unused_columns=False)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )
        trainer.train()
        return self.model
    
    def getModel(self):
        return self.model

dataset = dataset("Q2_20230202_majority.csv")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
tok_dataset = dataset.tokenized_datasets()
trainer = ModelTrainer(model, tok_dataset["train"], tok_dataset["validation"])
trained_model = trainer.train()
predictor = T5Predictor(trained_model)