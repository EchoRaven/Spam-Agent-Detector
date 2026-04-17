"""Traditional sparse text models for spam detection."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .metrics import classification_metrics


@dataclass
class TraditionalExperimentConfig:
    max_features: int = 10000
    min_df: int = 5
    max_df: float = 0.7
    logistic_c: float = 1.0
    logistic_max_iter: int = 1000
    bm25_k1: float = 1.2
    bm25_b: float = 0.0
    hybrid_tfidf_weight: float = 1.0
    hybrid_bm25_weight: float = 1.0
    top_k_terms: int = 20


class BM25Vectorizer(BaseEstimator, TransformerMixin):
    """A BM25-inspired sparse vectorizer for logistic regression."""

    def __init__(
        self,
        *,
        max_features: int = 10000,
        min_df: int = 5,
        max_df: float = 0.7,
        ngram_range=(1, 1),
        k1: float = 1.2,
        b: float = 0.0,
    ) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.k1 = k1
        self.b = b

    def fit(self, raw_documents, y=None):
        self.vectorizer_ = CountVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
        )
        counts = self.vectorizer_.fit_transform(raw_documents).tocsr().astype(np.float64)
        document_frequency = np.bincount(counts.indices, minlength=counts.shape[1])
        num_documents = counts.shape[0]

        self.idf_ = np.log(
            ((num_documents - document_frequency + 0.5) / (document_frequency + 0.5)) + 1.0
        )
        avgdl = float(np.asarray(counts.sum(axis=1)).mean())
        self.avgdl_ = avgdl if avgdl > 0 else 1.0
        return self

    def transform(self, raw_documents):
        counts = self.vectorizer_.transform(raw_documents).tocsr().astype(np.float64)
        doc_lengths = np.asarray(counts.sum(axis=1)).ravel()
        row_ids = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))

        if counts.data.size:
            normalization = self.k1 * (
                1.0 - self.b + self.b * doc_lengths[row_ids] / self.avgdl_
            )
            counts.data = (counts.data * (self.k1 + 1.0)) / (counts.data + normalization)

        idf_matrix = sparse.diags(self.idf_, format="csr")
        return counts @ idf_matrix

    def get_feature_names_out(self):
        return self.vectorizer_.get_feature_names_out()


class HybridTfidfBm25Vectorizer(BaseEstimator, TransformerMixin):
    """Concatenate TF-IDF and BM25 features into one sparse representation."""

    def __init__(
        self,
        *,
        max_features: int = 10000,
        min_df: int = 5,
        max_df: float = 0.7,
        k1: float = 1.2,
        b: float = 0.0,
        tfidf_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.k1 = k1
        self.b = b
        self.tfidf_weight = tfidf_weight
        self.bm25_weight = bm25_weight

    def fit(self, raw_documents, y=None):
        self.tfidf_vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        self.bm25_vectorizer_ = BM25Vectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            k1=self.k1,
            b=self.b,
        )
        self.tfidf_vectorizer_.fit(raw_documents)
        self.bm25_vectorizer_.fit(raw_documents)
        return self

    def transform(self, raw_documents):
        tfidf_features = self.tfidf_vectorizer_.transform(raw_documents)
        bm25_features = self.bm25_vectorizer_.transform(raw_documents)

        if self.tfidf_weight != 1.0:
            tfidf_features = tfidf_features * self.tfidf_weight
        if self.bm25_weight != 1.0:
            bm25_features = bm25_features * self.bm25_weight

        return sparse.hstack([tfidf_features, bm25_features], format="csr")

    def get_feature_names_out(self):
        tfidf_names = np.array(
            [f"tfidf::{name}" for name in self.tfidf_vectorizer_.get_feature_names_out()]
        )
        bm25_names = np.array(
            [f"bm25::{name}" for name in self.bm25_vectorizer_.get_feature_names_out()]
        )
        return np.concatenate([tfidf_names, bm25_names])


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _save_json(path: Path, payload: Dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _top_weighted_terms(pipeline: Pipeline, *, top_k: int) -> Dict[str, List[Dict[str, float]]]:
    classifier = pipeline.named_steps["classifier"]
    vectorizer = pipeline.named_steps["vectorizer"]

    coefficients = classifier.coef_[0]
    feature_names = vectorizer.get_feature_names_out()

    spam_indices = np.argsort(coefficients)[-top_k:][::-1]
    ham_indices = np.argsort(coefficients)[:top_k]

    return {
        "spam": [
            {"term": str(feature_names[index]), "weight": float(coefficients[index])}
            for index in spam_indices
        ],
        "ham": [
            {"term": str(feature_names[index]), "weight": float(coefficients[index])}
            for index in ham_indices
        ],
    }


def _build_classifier(config: TraditionalExperimentConfig) -> LogisticRegression:
    return LogisticRegression(
        C=config.logistic_c,
        max_iter=config.logistic_max_iter,
        class_weight="balanced",
        solver="liblinear",
    )


def _build_pipelines(config: TraditionalExperimentConfig) -> Dict[str, Pipeline]:
    tfidf_vectorizer = TfidfVectorizer(
        max_features=config.max_features,
        min_df=config.min_df,
        max_df=config.max_df,
    )
    bm25_vectorizer = BM25Vectorizer(
        max_features=config.max_features,
        min_df=config.min_df,
        max_df=config.max_df,
        k1=config.bm25_k1,
        b=config.bm25_b,
    )
    hybrid_vectorizer = HybridTfidfBm25Vectorizer(
        max_features=config.max_features,
        min_df=config.min_df,
        max_df=config.max_df,
        k1=config.bm25_k1,
        b=config.bm25_b,
        tfidf_weight=config.hybrid_tfidf_weight,
        bm25_weight=config.hybrid_bm25_weight,
    )

    return {
        "tfidf": Pipeline(
            [
                ("vectorizer", tfidf_vectorizer),
                ("classifier", _build_classifier(config)),
            ]
        ),
        "bm25": Pipeline(
            [
                ("vectorizer", bm25_vectorizer),
                ("classifier", _build_classifier(config)),
            ]
        ),
        "tfidf_bm25_hybrid": Pipeline(
            [
                ("vectorizer", hybrid_vectorizer),
                ("classifier", _build_classifier(config)),
            ]
        ),
    }


def run_traditional_experiments(
    splits,
    output_dir: str,
    *,
    config: TraditionalExperimentConfig,
) -> Dict[str, Dict]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_texts = splits["train"]["text_traditional"].tolist()
    val_texts = splits["val"]["text_traditional"].tolist()
    test_texts = splits["test"]["text_traditional"].tolist()

    y_train = splits["train"]["label"].tolist()
    y_val = splits["val"]["label"].tolist()
    y_test = splits["test"]["label"].tolist()

    results: Dict[str, Dict] = {"config": asdict(config), "models": {}}
    pipelines = _build_pipelines(config)

    for model_name, pipeline in pipelines.items():
        pipeline.fit(train_texts, y_train)

        val_predictions = pipeline.predict(val_texts)
        test_predictions = pipeline.predict(test_texts)

        model_summary = {
            "validation": classification_metrics(y_val, val_predictions),
            "test": classification_metrics(y_test, test_predictions),
            "top_weighted_terms": _top_weighted_terms(
                pipeline,
                top_k=config.top_k_terms,
            ),
        }
        results["models"][model_name] = model_summary

        joblib.dump(pipeline, output_path / f"{model_name}_pipeline.joblib")
        _save_json(output_path / f"{model_name}_metrics.json", model_summary)

    _save_json(output_path / "metrics_summary.json", results)
    return results
