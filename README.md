# IndicBERTv2 Fine-Tuning for Odia Natural Language Inference (IndicXNLI)

[![Dataset](https://img.shields.io/badge/Dataset-IndicXNLI%20(Odia)-blue)](https://huggingface.co/datasets/Divyanshu/indicxnli)
[![Model](https://img.shields.io/badge/Base%20Model-IndicBERTv2--MLM--only-green)](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only)

An end-to-end Natural Language Processing pipeline to fine-tune **IndicBERTv2** for **Natural Language Inference (NLI)** in **Odia (`or`)**, leveraging the **IndicXNLI** benchmark dataset.

---

## Overview

Natural Language Inference (NLI) is the task of determining whether a given hypothesis is **entailed**, **contradicted**, or **neutral** with respect to a premise. While NLP resources for low-resource Indic languages like Odia are historically limited, this repository implements a fine-tuning pipeline utilizing state-of-the-art Indic language representation models from [AI4Bharat](https://ai4bharat.iitm.ac.in/).

### Target Labels
| Label ID | Classification | Description |
| :---: | :--- | :--- |
| `0` | **Entailment** | The hypothesis is necessarily true given the premise. |
| `1` | **Neutral** | The hypothesis may or may not be true. |
| `2` | **Contradiction** | The hypothesis is false given the premise. |

---

## Key Features

- **IndicBERTv2 Backbone**: Fine-tunes `ai4bharat/IndicBERTv2-MLM-only` with sequence classification head.
- **Dataset Streaming & Tokenization**: Batched preprocessing and dynamic padding with maximum sequence length of 256.
- **Mixed Precision Training**: Accelerated training with `bf16` support.
- **Comprehensive Evaluation**: Automated tracking of Validation & Test **Accuracy** and **Macro F1 Score**.
- **Model Checkpointing**: Automated saving of top checkpoints and final deployment-ready weights with tokenizer artifacts.

---

## Project Structure

```text
nlp-project/
├── nlp.py                  # Main training, evaluation, and saving script
├── .gitignore              # Ignored checkpoint folders and environment files
├── README.md               # Project documentation
└── results/                # Directory generated during training (checkpoints & final model)
    └── indicbertv2_odia/
        └── final_model/    # Saved weights, configuration, and tokenizer
```

---

## Hyperparameters & Training Setup

| Parameter | Value |
| :--- | :--- |
| **Base Model** | `ai4bharat/IndicBERTv2-MLM-only` |
| **Epochs** | `5` |
| **Learning Rate** | `3e-5` |
| **LR Warmup Steps** | `1800` |
| **Weight Decay** | `0.01` |
| **Per-Device Batch Size (Train)** | `32` |
| **Per-Device Batch Size (Eval)** | `64` |
| **Gradient Accumulation Steps** | `8` |
| **Effective Batch Size** | `256` |
| **Max Sequence Length** | `256` |
| **Precision** | `BF16` (Brain Floating Point) |
| **Evaluation Strategy** | Every `1000` steps |
| **Best Model Metric** | Highest Validation `accuracy` |

---

## Installation & Prerequisites

### 1. Clone the Repository
```bash
git clone https://github.com/poorvithdevang861/nlp-project.git
cd nlp-project
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

Run the training and evaluation pipeline with:

```bash
python nlp.py
```

### Workflow Execution Steps:
1. **GPU Detection**: Validates CUDA availability and prints device memory info.
2. **Dataset Loading**: Downloads and prepares the Odia split of `IndicXNLI` (Train / Validation / Test).
3. **Tokenization**: Tokenizes premise-hypothesis pairs.
4. **Fine-Tuning**: Trains the classification head over 5 epochs.
5. **Evaluation**: Evaluates performance on Validation and Test sets.
6. **Export**: Exports the final model weights and tokenizer to `./results/indicbertv2_odia/final_model`.

---

## Evaluation & Metrics

The model is evaluated using standard classification metrics:

$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

$$\text{Macro } F_1 = \frac{1}{N} \sum_{i=1}^{N} F_{1, i}$$

---

## References

- **Dataset**: [IndicXNLI Dataset](https://huggingface.co/datasets/Divyanshu/indicxnli)
- **Base Model**: [AI4Bharat IndicBERTv2](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only)

---

