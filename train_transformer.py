import json
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Запуск трансформера на: {device.upper()}")

train_df = pd.read_csv("results/train.csv")
test_df = pd.read_csv("results/test.csv")

label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['author'])
test_df['label'] = label_encoder.transform(test_df['author'])

model_checkpoint = "youscan/ukr-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=256)

train_dataset = Dataset.from_pandas(train_df[['text', 'label']]).map(tokenize, batched=True)
test_dataset = Dataset.from_pandas(test_df[['text', 'label']]).map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_encoder.classes_)
)

training_args = TrainingArguments(
    output_dir="./results/transformer_checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_steps=10,
    load_best_model_at_end=True,
    report_to="none"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro")
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

test_output = trainer.predict(test_dataset)
pred_indices = np.argmax(test_output.predictions, axis=-1)
pred_authors = label_encoder.inverse_transform(pred_indices)

pd.DataFrame({"pred_transformer": pred_authors}).to_csv("results/preds_transformer.csv", index=False)

metrics = {
    "accuracy": float(test_output.metrics["test_accuracy"]),
    "macro_f1": float(test_output.metrics["test_macro_f1"])
}

with open("results/metrics_transformer.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print(f"[Transformer] Acc: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")