import os
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Setup
model_name = "microsoft/deberta-v3-base"
output_dir = "./model_output"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def main():
    print(f"Checking GPU... {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU found!'}")
    
    print("\n[1/2] Loading Global WELFake dataset...")
    try:
        ds_global = load_dataset("davanstrien/WELFake", split="train")
        df_global = pd.DataFrame(ds_global)
        df_global = df_global[['text', 'label']]
        df_global['label'] = df_global['label'].apply(lambda x: 0 if x == 1 else 1)
        print(f"   Loaded {len(df_global)} global articles.")
    except Exception as e:
        print(f"   Error loading WELFake: {e}")
        return

    indian_path = os.path.join("data", "indian_news.csv")
    print(f"\n[2/2] Looking for Indian News dataset at {indian_path}...")
    
    if os.path.exists(indian_path):
        try:
            df_indian = pd.read_csv(indian_path)
            df_indian.rename(columns=lambda x: x.strip().lower(), inplace=True)
            if df_indian['label'].dtype == 'object':
                df_indian['label'] = df_indian['label'].apply(lambda x: 1 if str(x).upper() == 'FAKE' else 0)
            df_indian = df_indian[['text', 'label']]
            print(f"   Loaded {len(df_indian)} Indian articles.")
            df_final = pd.concat([df_global, df_indian], ignore_index=True)
        except Exception as e:
            print(f"   Error reading Indian CSV: {e}")
            df_final = df_global
    else:
        print("   File not found. Continuing with Global dataset only...")
        df_final = df_global

    df_final = df_final.dropna(subset=['text', 'label'])
    
    count_real = len(df_final[df_final['label'] == 0])
    count_fake = len(df_final[df_final['label'] == 1])
    min_count = min(count_real, count_fake)
    
    df_real = df_final[df_final['label'] == 0].sample(n=min_count, random_state=42)
    df_fake = df_final[df_final['label'] == 1].sample(n=min_count, random_state=42)
    df_final = pd.concat([df_real, df_fake]).sample(frac=1).reset_index(drop=True)
    
    print(f"\nTotal Balanced Training Data: {len(df_final)} articles")

    hf_dataset = Dataset.from_pandas(df_final)
    hf_dataset = hf_dataset.train_test_split(test_size=0.15)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    print("Tokenizing...")
    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting Training on GPU (DeBERTa v3)...")
    trainer.train()

    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training Complete!")

if __name__ == "__main__":
    main()