# Email Spam Detector

This project is an email spam classification pipeline built around the core ideas from the `CS 410` proposal. It currently supports two main tracks:

1. Traditional text representations: `TF-IDF`, `BM25`, and a `TF-IDF + BM25` hybrid, all paired with `Logistic Regression`
2. Semantic modeling: a `BERT`-based binary classifier

The project expects a merged `CSV` dataset with at least these columns:

- `subject`
- `message`
- `label` (`0 = ham`, `1 = spam`)

If column names vary slightly, the loader will also try to recognize alternatives such as `body`, `text`, and `spam / ham`.

## Project Structure

```text
email_spam_detector/
├── README.md
├── FINAL_REPORT.md
├── requirements.txt
├── train_traditional.py
├── train_bert.py
└── src/
    └── spam_detector/
        ├── __init__.py
        ├── bert_classifier.py
        ├── data.py
        ├── metrics.py
        ├── preprocessing.py
        └── traditional.py
```

## Install Dependencies

```bash
cd /data/common/haibotong/email_spam_detector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `python3-venv` is unavailable on your system, you can install the dependencies directly into the user environment instead:

```bash
cd /data/common/haibotong/email_spam_detector
python3 -m pip install --user -r requirements.txt
```

## Run Traditional Models

The script below trains all three traditional models on the same split:

- `TF-IDF + Logistic Regression`
- `BM25 + Logistic Regression`
- `TF-IDF + BM25 + Logistic Regression`

```bash
cd /data/common/haibotong/email_spam_detector
python3 train_traditional.py \
  --csv-path /path/to/combined_email_dataset.csv \
  --output-dir outputs/traditional
```

Default settings are aligned with the proposal:

- `max_features=10000`
- `min_df=5`
- `max_df=0.7`
- `LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")`
- `BM25 k1=1.2`
- `Hybrid TF-IDF weight=1.0`
- `Hybrid BM25 weight=1.0`

Notes:

- `BM25` uses `b=0.0` by default, matching the proposal's BM25-style TF saturation setup.
- To use a more standard BM25 length normalization, pass `--bm25-b 0.75`.
- To tune the feature contribution inside the hybrid model, use `--hybrid-tfidf-weight` and `--hybrid-bm25-weight`.

## Run BERT

```bash
cd /data/common/haibotong/email_spam_detector
python3 train_bert.py \
  --csv-path /path/to/combined_email_dataset.csv \
  --output-dir outputs/bert \
  --model-name bert-base-uncased
```

Common tuning parameters:

- `--epochs 3`
- `--batch-size 8`
- `--max-length 256`
- `--learning-rate 2e-5`

## Output Files

Each training run saves:

- `metrics_summary.json`: validation and test metrics
- `split_summary.json`: dataset split statistics
- trained model artifacts

Traditional model runs also save:

- `tfidf_pipeline.joblib`
- `bm25_pipeline.joblib`
- `tfidf_bm25_hybrid_pipeline.joblib`
- top-weighted terms for qualitative analysis

`BERT` runs save:

- a Hugging Face model directory
- tokenizer files
- training logs and evaluation outputs

## Relation to the Proposal

The current implementation already covers:

- text preprocessing: lowercase normalization, punctuation cleanup, and stopword filtering
- class imbalance handling with `class_weight="balanced"`
- evaluation metrics: `precision`, `recall`, `f1`, and confusion matrix
- comparison between traditional sparse features and a pretrained language model

Possible next extensions:

- `Word2Vec` document embeddings via averaged word vectors
- oversampling for the minority class
- result visualizations
- notebook or polished report templates

You can continue building directly on top of this project structure.
