"""Dataset loading and splitting helpers."""

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocessing import (
    combine_subject_and_message,
    normalize_bert_text,
    normalize_traditional_text,
)

SUBJECT_CANDIDATES = ("subject", "title")
MESSAGE_CANDIDATES = ("message", "body", "text", "email", "content")
LABEL_CANDIDATES = ("label", "spam / ham", "spam/ham", "spam_ham", "class", "target")

LABEL_MAP = {
    "spam": 1,
    "junk": 1,
    "ham": 0,
    "not spam": 0,
    "not_spam": 0,
    "non-spam": 0,
    "nonspam": 0,
    "important": 0,
}


def _resolve_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lowered = {column.strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _normalize_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        labels = pd.to_numeric(series, errors="coerce")
        return labels.where(labels.isin([0, 1]))

    normalized = series.astype(str).str.strip().str.lower().map(LABEL_MAP)
    return normalized


def load_email_dataframe(
    csv_path: str,
    *,
    max_samples: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)

    subject_col = _resolve_column(dataframe.columns, SUBJECT_CANDIDATES)
    message_col = _resolve_column(dataframe.columns, MESSAGE_CANDIDATES)
    label_col = _resolve_column(dataframe.columns, LABEL_CANDIDATES)

    if message_col is None or label_col is None:
        raise ValueError(
            "Could not resolve dataset columns. Expected a message/body/text column "
            "and a label column."
        )

    selected_columns = [column for column in [subject_col, message_col, label_col] if column]
    working = dataframe[selected_columns].copy()
    rename_map = {message_col: "message", label_col: "label"}
    if subject_col is not None:
        rename_map[subject_col] = "subject"
    working = working.rename(columns=rename_map)

    if "subject" not in working.columns:
        working["subject"] = ""

    working["subject"] = working["subject"].fillna("")
    working["message"] = working["message"].fillna("")
    working["text_raw"] = working.apply(
        lambda row: combine_subject_and_message(row["subject"], row["message"]),
        axis=1,
    )
    working["label"] = _normalize_labels(working["label"])

    working = working.dropna(subset=["label"])
    working["label"] = working["label"].astype(int)
    working = working.loc[working["text_raw"].str.strip() != ""].reset_index(drop=True)

    if max_samples is not None and max_samples < len(working):
        working = (
            working.sample(n=max_samples, random_state=random_state)
            .reset_index(drop=True)
        )

    return working


def prepare_text_views(
    dataframe: pd.DataFrame,
    *,
    extra_stopwords: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    working = dataframe.copy()
    working["text_traditional"] = working["text_raw"].apply(
        lambda text: normalize_traditional_text(text, extra_stopwords=extra_stopwords)
    )
    working["text_bert"] = working["text_raw"].apply(normalize_bert_text)
    return working


def split_dataset(
    dataframe: pd.DataFrame,
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1.")

    train_val, test = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_state,
        stratify=dataframe["label"],
    )

    adjusted_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=train_val["label"],
    )

    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def summarize_splits(splits: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for split_name, split_df in splits.items():
        label_counts = split_df["label"].value_counts().to_dict()
        summary[split_name] = {
            "num_rows": int(len(split_df)),
            "ham": int(label_counts.get(0, 0)),
            "spam": int(label_counts.get(1, 0)),
        }
    return summary
