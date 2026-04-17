# Email Spam Detector

## Final Report Draft

**Course:** CS 410 Spring 2026  
**Instructor:** Prof. Robles-Granda  
**Project Title:** Study of Text Representations Leveraging Word2Vec, TF-IDF, BM25, and BERT for Email Spam Detection  
**Group Members:** Kyle Keliuotis, Everett Cruz, Alice Yeh, Haibo Tong, Allen Yuan, Qi Zhang

## Abstract
This project studies how different text representations affect email spam detection performance under a controlled classification setting. We implemented three traditional sparse models, `TF-IDF`, `BM25`, and a new hybrid `TF-IDF + BM25` representation, all paired with `Logistic Regression`. We also implemented a `BERT`-based neural baseline for semantic classification. On the Enron spam dataset, the best full-scale traditional model was the hybrid sparse representation, which achieved a test F1 score of `0.9879`, slightly outperforming standalone `BM25` and `TF-IDF`. A reduced `BERT` benchmark also performed well, reaching a test F1 of `0.9263`, but this experiment was run on a smaller subset because GPU acceleration was unavailable in the current environment. The results suggest that for this dataset, carefully engineered sparse lexical features remain extremely competitive and that combining `TF-IDF` and `BM25` can produce modest but consistent gains.

## 1. Introduction
Spam detection is a binary text classification task in which each email is classified as either spam or ham. The main research question of this project is:

**How does the choice of text representation affect spam classification performance?**

To answer this question, we kept the classifier fixed whenever possible and varied the text representation:

- `TF-IDF`: emphasizes words that are distinctive in a document relative to the corpus.
- `BM25`: modifies term frequency with saturation, reducing the effect of repeated keyword stuffing.
- `TF-IDF + BM25` hybrid: concatenates both sparse representations so the classifier can learn from both rarity-based weighting and repetition-aware weighting.
- `BERT`: a contextual language model that captures semantic relationships beyond surface lexical overlap.

This setup allows us to compare lexical and semantic representations under a shared evaluation framework.

## 2. Dataset
We used the publicly available Enron spam dataset in CSV form. The processed dataset contains email subjects, message bodies, labels, and dates.

- Dataset source: `enron_spam_data.csv`
- Labels: `spam` and `ham`
- Total examples used for the full traditional experiments: `33,665`

For the full traditional experiments, the stratified split was:

| Split | Total | Ham | Spam |
| --- | ---: | ---: | ---: |
| Train | 23,565 | 11,581 | 11,984 |
| Validation | 3,367 | 1,655 | 1,712 |
| Test | 6,733 | 3,309 | 3,424 |

For the reduced BERT benchmark, we used a stratified sample of `2,000` emails because the available runtime environment could not use GPU acceleration:

| Split | Total | Ham | Spam |
| --- | ---: | ---: | ---: |
| Train | 1,400 | 705 | 695 |
| Validation | 200 | 101 | 99 |
| Test | 400 | 201 | 199 |

## 3. Preprocessing
The preprocessing pipeline follows the project proposal and standard spam filtering practice:

1. Subject and body text were concatenated into a single document.
2. Text was lowercased.
3. Tokens were filtered to alphabetic forms.
4. English stopwords were removed for the traditional models.
5. Domain-specific high-frequency tokens such as `enron`, `ect`, and `com` were also removed to reduce corpus-specific bias.
6. For `BERT`, text was normalized more lightly, preserving the original wording as much as possible.

## 4. Methods

### 4.1 TF-IDF
We used `TfidfVectorizer` from scikit-learn with:

- `max_features=10000`
- `min_df=5`
- `max_df=0.7`

This representation rewards terms that are frequent in a document but not overly common across the corpus.

### 4.2 BM25
We implemented a BM25-style sparse vectorizer using:

\[
\text{BM25-TF}(x) = \frac{x(k_1 + 1)}{x + k_1}
\]

where `x` is the raw term frequency and `k1=1.2`. In the main experiments we used `b=0.0`, which matches the proposal's emphasis on TF saturation rather than full document-length normalization.

This representation is motivated by spam behavior such as repeated promotional keywords. BM25 reduces the marginal gain of repeated occurrences, making it more robust to keyword stuffing.

### 4.3 TF-IDF + BM25 Hybrid
Yes, `TF-IDF` and `BM25` can be combined. In this project, we implemented a hybrid model by concatenating the two sparse vectors:

\[
\mathbf{h}(d) = [\lambda_1 \cdot \mathbf{tfidf}(d) \; ; \; \lambda_2 \cdot \mathbf{bm25}(d)]
\]

where `\lambda_1` and `\lambda_2` are feature weights. In our default setting, both were set to `1.0`.

The intuition is:

- `TF-IDF` captures global distinctiveness.
- `BM25` captures saturation-aware lexical salience.
- Their concatenation lets `Logistic Regression` learn complementary signals from both views of the same email.

### 4.4 Logistic Regression
All traditional sparse representations were paired with the same classifier:

- `LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="liblinear")`

Using `class_weight="balanced"` helps reduce bias if class proportions become uneven.

### 4.5 BERT
We implemented a `bert-base-uncased` classifier using Hugging Face Transformers. The reduced benchmark used:

- `epochs=1`
- `batch_size=16`
- `max_length=128`
- `learning_rate=2e-5`

Class-weighted cross-entropy was used during training.

## 5. Experimental Results

### 5.1 Traditional Models on the Full Dataset

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 0.9853 | 0.9809 | 0.9904 | 0.9856 |
| BM25 + Logistic Regression | 0.9872 | 0.9863 | 0.9886 | 0.9875 |
| TF-IDF + BM25 Hybrid + Logistic Regression | **0.9877** | **0.9869** | **0.9889** | **0.9879** |

The hybrid sparse model achieved the best overall test F1. The gain over standalone BM25 is small but consistent, showing that the two representations provide complementary information.

### 5.2 Reduced BERT Benchmark

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| BERT (validation) | 0.8950 | 0.9535 | 0.8283 | 0.8865 |
| BERT (test) | 0.9300 | 0.9724 | 0.8844 | 0.9263 |

Although BERT performed strongly, it did not outperform the full traditional models in this run. However, this comparison is not fully fair because the BERT experiment used a smaller sample and a reduced training budget.

## 6. Discussion
Several conclusions emerged from the experiments.

First, `BM25` slightly outperformed `TF-IDF`, which supports the claim that TF saturation is useful in spam detection. Spam emails often repeat attention-grabbing terms, and BM25 reduces the disproportionate influence of these repeated tokens.

Second, the hybrid `TF-IDF + BM25` model performed best among the traditional methods. This indicates that the two representations are not redundant. `TF-IDF` is strong at identifying globally distinctive words, while `BM25` is better at controlling repeated-term effects. Their combination gave the classifier a richer lexical feature space and led to the strongest overall results.

Third, the reduced `BERT` benchmark showed that semantic models are promising, especially because they can generalize beyond exact word overlap. Even so, in this project setting, the sparse methods remained highly competitive and more efficient to train.

## 7. Feature Interpretation
The top weighted spam terms from the traditional runs included words such as:

- `http`
- `money`
- `software`
- `remove`
- `click`
- `free`
- `viagra`

These terms align with common spam behavior involving promotions, links, pharmaceuticals, and opt-out phrases.

Important ham-oriented terms included words such as:

- `attached`
- `thanks`
- `questions`
- `meeting`
- `energy`

These are more consistent with everyday business communication, which is expected in the Enron corpus.

## 8. Limitations
This project still has several limitations.

1. The BERT experiment was reduced due to an environment issue: the installed PyTorch build expected a newer CUDA stack than the machine's NVIDIA driver provided, so GPU training was unavailable.
2. The BERT results therefore used fewer samples and fewer epochs than the sparse models, making the comparison imperfect.
3. The current implementation does not yet include the planned `Word2Vec` average-embedding baseline.
4. The experiments were performed on Enron data rather than the final merged Kaggle dataset described in the original proposal.

## 9. Conclusion
This project demonstrates that text representation has a measurable effect on spam detection performance. Among the full-scale traditional models, `BM25` outperformed `TF-IDF`, and the combined `TF-IDF + BM25` hybrid achieved the best results overall with a test F1 of `0.9879`. These results suggest that sparse lexical representations remain strong baselines for spam filtering and that combining complementary weighting schemes can improve performance further.

The reduced BERT benchmark also showed promise, but a fair final comparison requires rerunning BERT on the full dataset after fixing the GPU environment. A natural next step would be to add the planned `Word2Vec` baseline and then compare all four representation families under the same data split and compute budget.

## 10. Reproducibility
Project code is located in:

- `train_traditional.py`
- `train_bert.py`
- `src/spam_detector/traditional.py`
- `src/spam_detector/bert_classifier.py`
- `src/spam_detector/data.py`
- `src/spam_detector/preprocessing.py`

To rerun the traditional models:

```bash
cd /data/common/haibotong/email_spam_detector
python3 train_traditional.py \
  --csv-path /data/common/haibotong/email_spam_detector/data/enron_spam_data/enron_spam_data.csv \
  --output-dir outputs/traditional_enron_hybrid
```

To rerun the reduced BERT benchmark:

```bash
cd /data/common/haibotong/email_spam_detector
python3 train_bert.py \
  --csv-path /data/common/haibotong/email_spam_detector/data/enron_spam_data/enron_spam_data.csv \
  --output-dir outputs/bert_enron_quick \
  --batch-size 16 \
  --epochs 1 \
  --max-length 128 \
  --max-samples 2000
```
