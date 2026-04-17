"""Text preprocessing helpers."""

import re
from typing import Iterable, Optional, Set

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN_PATTERN = re.compile(r"[a-zA-Z']+")

# Proposal-specific high-frequency corpus words that hurt generalization.
DOMAIN_STOPWORDS = {"enron", "ect", "com"}


def _as_clean_string(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def combine_subject_and_message(subject: object, message: object) -> str:
    subject_text = _as_clean_string(subject).strip()
    message_text = _as_clean_string(message).strip()
    combined = f"{subject_text} {message_text}".strip()
    return re.sub(r"\s+", " ", combined)


def build_stopword_set(extra_stopwords: Optional[Iterable[str]] = None) -> Set[str]:
    stopwords = set(ENGLISH_STOP_WORDS).union(DOMAIN_STOPWORDS)
    if extra_stopwords:
        stopwords.update(word.strip().lower() for word in extra_stopwords if word)
    return stopwords


def normalize_traditional_text(
    text: object,
    *,
    remove_stopwords: bool = True,
    extra_stopwords: Optional[Iterable[str]] = None,
) -> str:
    normalized = _as_clean_string(text).lower()
    tokens = TOKEN_PATTERN.findall(normalized)

    if remove_stopwords:
        stopwords = build_stopword_set(extra_stopwords)
        tokens = [token for token in tokens if token not in stopwords]

    return " ".join(tokens)


def normalize_bert_text(text: object) -> str:
    return re.sub(r"\s+", " ", _as_clean_string(text)).strip()
