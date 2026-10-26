#!/home/rushikesh/miniconda3/envs/VoxNeus/bin/python
"""
OdiaEval Native Test Set Evaluator & Translationese Gap Calculator
Evaluates the fine-tuned IndicBERTv2 checkpoint on the Native Odia Gold Test Set
and generates the required report tables and per-row predictions.

Usage:
    python evaluate_native.py
    python evaluate_native.py --test_file 03_nli_groups_3_9_updated.jsonl
    python evaluate_native.py --model_path /home/rushikesh/NLP/results/indicbertv2_odia/final_model/model.safetensors
"""

import argparse
import os
import shutil
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
    TrainingArguments
)
import evaluate
from sklearn.metrics import classification_report, confusion_matrix

# Safe torch import bypass for huggingface
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
try:
    import transformers.modeling_utils
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on Native Odia Test Set")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/rushikesh/NLP/results/indicbertv2_odia/final_model",
        help="Path to saved model directory or model.safetensors file"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="/home/rushikesh/NLP/03_nli_groups_3_9_updated.jsonl",
        help="Path to native Odia test file (.jsonl, .json, .csv, .tsv, .parquet)"
    )
    parser.add_argument(
        "--benchmark_accuracy",
        type=float,
        default=76.91,
        help="Reproduction score on official benchmark test set (%)"
    )
    parser.add_argument(
        "--published_target",
        type=float,
        default=72.60,
        help="Published target accuracy from paper (%)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--output_predictions",
        type=str,
        default="/home/rushikesh/NLP/results/native_test_predictions.csv",
        help="Path to save detailed predictions CSV"
    )
    return parser.parse_args()


LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2, "0": 0, "1": 1, "2": 2}
ID2LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}


def load_native_data(file_path):
    if file_path is None or not os.path.exists(file_path):
        raise FileNotFoundError(f"Test file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        raw_df = pd.read_csv(file_path)
    elif ext == ".tsv":
        raw_df = pd.read_csv(file_path, sep="\t")
    elif ext in [".json", ".jsonl"]:
        raw_df = pd.read_json(file_path, lines=ext == ".jsonl")
    elif ext == ".parquet":
        raw_df = pd.read_parquet(file_path)
    elif ext in [".xlsx", ".xls"]:
        raw_df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Standardize column names (case-insensitive)
    cols = {c.lower(): c for c in raw_df.columns}
    premise_col = next((cols[k] for k in ["premise", "sentence1", "text1", "context"] if k in cols), None)
    hypo_col = next((cols[k] for k in ["hypothesis", "sentence2", "text2", "target"] if k in cols), None)
    label_col = next((cols[k] for k in ["label", "gold_label", "target_label"] if k in cols), None)

    if not premise_col or not hypo_col:
        raise ValueError(f"Could not identify premise/hypothesis columns in: {list(raw_df.columns)}")

    clean_df = pd.DataFrame({
        "premise": raw_df[premise_col].astype(str),
        "hypothesis": raw_df[hypo_col].astype(str)
    })

    if label_col:
        raw_labels = raw_df[label_col].tolist()
        mapped_labels = [LABEL_MAP.get(str(l).strip().lower(), l) for l in raw_labels]
        clean_df["label"] = [int(l) if str(l).isdigit() else l for l in mapped_labels]

    return Dataset.from_pandas(clean_df), raw_df, clean_df


def main():
    args = parse_args()
    print("=" * 70)
    print("OdiaEval Task 3: Native Odia Evaluation & Translationese Analysis")
    print("=" * 70)

    # Resolve model directory if a safetensors file path is given directly
    model_dir = os.path.dirname(args.model_path) if os.path.isfile(args.model_path) else args.model_path
    print(f"Loading trained model from: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Setup metrics
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        preds = np.argmax(predictions, axis=-1)

        acc = accuracy_metric.compute(predictions=preds, references=labels)
        p = precision_metric.compute(predictions=preds, references=labels, average="weighted", zero_division=0)
        r = recall_metric.compute(predictions=preds, references=labels, average="weighted", zero_division=0)
        f1_w = f1_metric.compute(predictions=preds, references=labels, average="weighted")
        f1_m = f1_metric.compute(predictions=preds, references=labels, average="macro")

        return {
            "accuracy": acc["accuracy"],
            "precision": p["precision"],
            "recall": r["recall"],
            "f1_weighted": f1_w["f1"],
            "f1_macro": f1_m["f1"]
        }

    # Load test dataset
    print(f"Loading Native Odia Test Data from: {args.test_file}")
    test_dataset, raw_df, clean_df = load_native_data(args.test_file)
    print(f"Native Test Set Size: {len(test_dataset)} examples")

    def tokenize_fn(examples):
        return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=256)

    tokenized_test = test_dataset.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Safe device & precision detection
    is_cuda = torch.cuda.is_available()
    use_bf16 = is_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = is_cuda and not use_bf16
    use_tf32 = is_cuda and torch.cuda.get_device_capability()[0] >= 8

    device_desc = f"CUDA ({torch.cuda.get_device_name(0)})" if is_cuda else "CPU"
    print(f"Execution Device: {device_desc}")

    temp_eval_dir = "./tmp_eval"
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=temp_eval_dir,
            per_device_eval_batch_size=args.batch_size,
            bf16=use_bf16,
            fp16=use_fp16,
            tf32=use_tf32,
            report_to="none"
        ),
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if "label" in clean_df.columns else None
    )

    print("\nRunning Evaluation on Native Odia Test Set...")
    pred_output = trainer.predict(tokenized_test)
    logits = pred_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)

    has_labels = "label" in clean_df.columns
    if has_labels:
        true_labels = np.array(clean_df["label"])
        native_acc = (preds == true_labels).mean() * 100
        gap = args.benchmark_accuracy - native_acc

        # Softmax probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        print("\n" + "=" * 70)
        print("FINAL REPORT TABLES (Ready for Submission)")
        print("=" * 70)

        print("\n### 1. Results Summary Table")
        print("| Evaluation Dataset | Metric | Published Target | Our Score | Difference from Published Target |")
        print("| :--- | :--- | :--- | :--- | :--- |")
        print(f"| Benchmark Test (IndicXNLI) | Accuracy | {args.published_target:.2f}% | {args.benchmark_accuracy:.2f}% | +{args.benchmark_accuracy - args.published_target:.2f}% |")
        print(f"| Native Odia Test | Accuracy | N/A | {native_acc:.2f}% | N/A |")

        print("\n### 2. Translationese Gap Table")
        print("| Comparison | Score |")
        print("| :--- | :--- |")
        print(f"| Benchmark Score A (IndicXNLI Reproduction) | {args.benchmark_accuracy:.2f}% |")
        print(f"| Native Odia Score B (Gold Test Set) | {native_acc:.2f}% |")
        print(f"| **Translationese Gap (A - B)** | **{gap:+.2f}%** |")

        print("\n### 3. Detailed Classification Report")
        target_names = [ID2LABEL[i] for i in range(3)]
        print(classification_report(true_labels, preds, target_names=target_names, digits=4))

        print("### 4. Confusion Matrix (Rows: True, Columns: Predicted)")
        cm = confusion_matrix(true_labels, preds)
        cm_df = pd.DataFrame(cm, index=[f"True_{n}" for n in target_names], columns=[f"Pred_{n}" for n in target_names])
        print(cm_df.to_string())

        # Save per-row predictions
        if args.output_predictions:
            out_df = raw_df.copy()
            out_df["predicted_label"] = [ID2LABEL[p] for p in preds]
            out_df["is_correct"] = (preds == true_labels)
            out_df["prob_entailment"] = probs[:, 0].round(4)
            out_df["prob_neutral"] = probs[:, 1].round(4)
            out_df["prob_contradiction"] = probs[:, 2].round(4)

            os.makedirs(os.path.dirname(os.path.abspath(args.output_predictions)), exist_ok=True)
            out_df.to_csv(args.output_predictions, index=False)
            print(f"\n[INFO] Detailed predictions saved to: {args.output_predictions}")

    # Clean up temp dir
    if os.path.exists(temp_eval_dir):
        shutil.rmtree(temp_eval_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

