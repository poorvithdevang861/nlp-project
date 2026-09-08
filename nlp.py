# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

import evaluate


# ============================================================
# 2. CHECK GPU
# ============================================================

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    gpu_memory = (
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3
    )

    print("GPU memory:", round(gpu_memory, 2), "GB")


# ============================================================
# 3. LOAD ODIA INDICXNLI DATASET
# ============================================================

print("\nLoading Odia IndicXNLI dataset...")

dataset = load_dataset(
    "parquet",

    data_files={
        "train":
        "https://huggingface.co/datasets/Divyanshu/indicxnli/resolve/refs%2Fconvert%2Fparquet/or/train/0000.parquet",

        "validation":
        "https://huggingface.co/datasets/Divyanshu/indicxnli/resolve/refs%2Fconvert%2Fparquet/or/validation/0000.parquet",

        "test":
        "https://huggingface.co/datasets/Divyanshu/indicxnli/resolve/refs%2Fconvert%2Fparquet/or/test/0000.parquet"
    }
)

print("\nDataset loaded successfully!")
print(dataset)


# ============================================================
# 4. CHECK DATASET SIZE
# ============================================================

print("\nTraining examples:  ", len(dataset["train"]))
print("Validation examples:", len(dataset["validation"]))
print("Test examples:      ", len(dataset["test"]))


# ============================================================
# 5. LOOK AT ONE EXAMPLE
# ============================================================

example = dataset["train"][0]

print("\nPremise:")
print(example["premise"])

print("\nHypothesis:")
print(example["hypothesis"])

print("\nLabel:")
print(example["label"])


# ============================================================
# 6. LOAD TOKENIZER
# ============================================================

MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded successfully!")


# ============================================================
# 7. LOAD MODEL
# ============================================================

print("\nLoading IndicBERTv2 model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,

    # NLI has 3 classes:
    # 0 = entailment
    # 1 = neutral
    # 2 = contradiction
    num_labels=3,

    # The original MLM checkpoint does not contain
    # the NLI classification head.
    ignore_mismatched_sizes=True
)

print("Model loaded successfully!")


# ============================================================
# 8. CHECK MODEL
# ============================================================

print("\nModel type:", model.config.model_type)
print("Number of labels:", model.config.num_labels)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "Model parameters:",
    round(parameter_count / 1e6, 2),
    "million"
)


# ============================================================
# 9. TOKENIZATION FUNCTION
# ============================================================

def tokenize_function(examples):

    return tokenizer(
        examples["premise"],
        examples["hypothesis"],

        # Truncate sequences longer than max_length
        truncation=True,

        # Full/slower configuration
        max_length=256
    )


# ============================================================
# 10. TOKENIZE ENTIRE DATASET
# ============================================================

print("\nStarting tokenization...")

tokenized_data = dataset.map(
    tokenize_function,

    # Process multiple examples at once
    batched=True,

    # Original text is no longer needed
    remove_columns=[
        "premise",
        "hypothesis"
    ]
)

print("Tokenization completed!")

print("\nTokenized dataset:")
print(tokenized_data)


# ============================================================
# 11. CHECK TOKENIZED EXAMPLE
# ============================================================

print("\nFirst tokenized training example:")
print(tokenized_data["train"][0])


# ============================================================
# 12. DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

print("\nData collator created successfully!")


# ============================================================
# 13. OPTIONAL: TEST DATA COLLATOR
# ============================================================

from torch.utils.data import DataLoader

test_loader = DataLoader(
    tokenized_data["train"],
    batch_size=2,
    collate_fn=data_collator
)

batch = next(iter(test_loader))

print("\nBatch keys:")
print(batch.keys())

print("\nInput shape:")
print(batch["input_ids"].shape)

print("\nLabels:")
print(batch["labels"])


# ============================================================
# 14. CHECK GPU FORWARD PASS
# ============================================================
# This is only a test.
# Trainer will handle GPU placement during training.

if torch.cuda.is_available():

    test_model = model.to("cuda")

    test_batch = {
        key: value.to("cuda")
        for key, value in batch.items()
    }

    with torch.no_grad():
        output = test_model(**test_batch)

    print("\nGPU forward pass successful!")

    print("Output shape:")
    print(output.logits.shape)

    print("GPU:")
    print(torch.cuda.get_device_name(0))

    # Keep model on GPU.
    # Trainer can use it directly.


# ============================================================
# 15. LOAD EVALUATION METRICS
# ============================================================

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

print("\nAll metrics loaded successfully!")


# ============================================================
# 16. DEFINE METRICS
# ============================================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    # Convert logits to predicted class
    predictions = np.argmax(
        predictions,
        axis=-1
    )

    accuracy = accuracy_metric.compute(
        predictions=predictions,
        references=labels
    )

    f1_macro = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average="macro"
    )

    return {

        "accuracy":
            accuracy["accuracy"],

        "f1_macro":
            f1_macro["f1"]
    }


# ============================================================
# 17. TRAINING CONFIGURATION
# ============================================================

training_args = TrainingArguments(

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output_dir="./results/indicbertv2_odia",

    # --------------------------------------------------------
    # TRAINING LENGTH
    # --------------------------------------------------------

    num_train_epochs=5,

    # --------------------------------------------------------
    # LEARNING RATE
    # --------------------------------------------------------

    learning_rate=3e-5,

    warmup_steps=1800,

    # --------------------------------------------------------
    # REGULARIZATION
    # --------------------------------------------------------

    weight_decay=0.01,

    # --------------------------------------------------------
    # BATCH SIZE
    # --------------------------------------------------------

    per_device_train_batch_size=32,

    per_device_eval_batch_size=64,

    # --------------------------------------------------------
    # GRADIENT ACCUMULATION
    # --------------------------------------------------------
    # 4 × 8 = effective batch size of 32

    gradient_accumulation_steps=8,

    # --------------------------------------------------------
    # MIXED PRECISION
    # --------------------------------------------------------

    bf16=True,

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    eval_strategy="steps",

    eval_steps=1000,

    # --------------------------------------------------------
    # CHECKPOINTS
    # --------------------------------------------------------

    save_strategy="steps",

    save_steps=1000,

    save_total_limit=2,

    save_only_model=True,

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    load_best_model_at_end=True,

    metric_for_best_model="accuracy",

    greater_is_better=True,

    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------

    logging_steps=100,

    # --------------------------------------------------------
    # DISABLE WANDB / EXTERNAL LOGGING
    # --------------------------------------------------------

    report_to="none"
)

print("\nTraining configuration created successfully!")


# ============================================================
# 18. CREATE TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("\nTrainer created successfully!")


# ============================================================
# 19. START TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

trainer.train()


# ============================================================
# 20. EVALUATE ON VALIDATION SET
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

validation_results = trainer.evaluate(
    tokenized_data["validation"]
)

print(validation_results)


# ============================================================
# 21. EVALUATE ON TEST SET
# ============================================================

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

test_results = trainer.evaluate(
    tokenized_data["test"],
    metric_key_prefix="test"
)

print(test_results)


# ============================================================
# 22. SAVE FINAL MODEL
# ============================================================

FINAL_MODEL_PATH = "./results/indicbertv2_odia/final_model"

trainer.save_model(FINAL_MODEL_PATH)

tokenizer.save_pretrained(
    FINAL_MODEL_PATH
)

print("\nFinal model saved to:")
print(FINAL_MODEL_PATH)