# train_reviews_ru.py

import os
import re
import random
import numpy as np
import torch

from dataclasses import dataclass
from typing import List, Dict

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

# =========================
# 1. CONFIG
# =========================

PATH = "./"  # folder with positive.txt and negative.txt
MODEL_NAME = "cointegrated/rubert-tiny2"
OUTPUT_DIR = "./rubert_tiny2_reviews"
MAX_LENGTH = 256
SEED = 42

# if CUDA is available, Trainer will use it automatically
BATCH_SIZE = 32
NUM_EPOCHS = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# 2. DATA LOADING
# =========================

def get_reviews(positive=True):
    path = PATH + ("positive" if positive else "negative") + ".txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_raw_reviews(raw_text: str) -> List[str]:
    """
    Your current splitter is:
        positive_reviews.replace('\\xa0', '\\n').split('\\n')

    Kept in spirit, but with extra cleanup:
    - replace non-breaking spaces
    - split by line
    - trim spaces
    - remove empty lines
    """
    reviews = raw_text.replace("\xa0", "\n").split("\n")
    reviews = [r.strip() for r in reviews]
    reviews = [r for r in reviews if r]
    return reviews


def normalize_text(text: str) -> str:
    """
    Minimal cleanup only.
    Do NOT do stemming/lemmatization/stopword removal for BERT.
    """
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset_from_txt() -> DatasetDict:
    positive_raw = get_reviews(positive=True)
    negative_raw = get_reviews(positive=False)

    positive_reviews = split_raw_reviews(positive_raw)
    negative_reviews = split_raw_reviews(negative_raw)

    positive_reviews = [normalize_text(x) for x in positive_reviews]
    negative_reviews = [normalize_text(x) for x in negative_reviews]

    # labels:
    # 1 = positive
    # 0 = negative
    texts = negative_reviews + positive_reviews
    labels = [0] * len(negative_reviews) + [1] * len(positive_reviews)

    # optional: remove exact duplicates
    pairs = list(zip(texts, labels))
    pairs = list(dict.fromkeys(pairs))  # preserve order, remove exact duplicates
    texts, labels = zip(*pairs)

    texts = list(texts)
    labels = list(labels)

    print("Total samples:", len(texts))
    print("Negative:", sum(1 for x in labels if x == 0))
    print("Positive:", sum(1 for x in labels if x == 1))

    # train / valid / test = 80 / 10 / 10
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=SEED,
        stratify=y_temp,
    )

    train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
    valid_ds = Dataset.from_dict({"text": X_valid, "label": y_valid})
    test_ds = Dataset.from_dict({"text": X_test, "label": y_test})

    return DatasetDict({
        "train": train_ds,
        "validation": valid_ds,
        "test": test_ds,
    })


# =========================
# 3. TOKENIZER
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


# =========================
# 4. METRICS
# =========================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# =========================
# 5. CLASS-WEIGHTED TRAINER
# =========================

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=None,
        )
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# =========================
# 6. MAIN
# =========================

def main():
    ds = load_dataset_from_txt()

    tokenized_ds = ds.map(tokenize_batch, batched=True)
    tokenized_ds = tokenized_ds.remove_columns(["text"])
    tokenized_ds.set_format("torch")

    # compute class weights on train only
    train_labels = np.array(ds["train"]["label"])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_labels,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    print("Class weights:", class_weights.tolist())

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=SEED,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n=== TRAINING ===")
    trainer.train()

    print("\n=== VALIDATION ===")
    val_metrics = trainer.evaluate(tokenized_ds["validation"])
    print(val_metrics)

    print("\n=== TEST ===")
    test_metrics = trainer.evaluate(tokenized_ds["test"])
    print(test_metrics)

    # detailed test report
    preds_output = trainer.predict(tokenized_ds["test"])
    logits = preds_output.predictions
    y_true = preds_output.label_ids
    y_pred = np.argmax(logits, axis=-1)

    print("\n=== CLASSIFICATION REPORT (TEST) ===")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["NEGATIVE", "POSITIVE"],
        digits=4,
        zero_division=0,
    ))

    print("\n=== CONFUSION MATRIX (TEST) ===")
    print(confusion_matrix(y_true, y_pred))

    # save final model + tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nBest model saved to: {OUTPUT_DIR}")

    # quick demo
    demo_texts = [
        "Теслааа рыба моей мечтыы",
        "Короче ездишь и думаешь постоянно про зарядку",
        "Маск красавчик, стоит на своем",
        "ого троллейбус",
    ]

    print("\n=== DEMO PREDICTIONS ===")
    predict_texts(demo_texts, model_dir=OUTPUT_DIR)


# =========================
# 7. INFERENCE
# =========================

def predict_texts(texts: List[str], model_dir: str = OUTPUT_DIR):
    local_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    local_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    local_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model.to(device)

    cleaned = [normalize_text(t) for t in texts]
    enc = local_tokenizer(
        cleaned,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = local_model(**enc)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)

    for text, pred, prob in zip(texts, preds, probs):
        label = "POSITIVE" if pred == 1 else "NEGATIVE"
        print(f"[{label}] p_neg={prob[0]:.4f} p_pos={prob[1]:.4f} :: {text}")


if __name__ == "__main__":
    main()
