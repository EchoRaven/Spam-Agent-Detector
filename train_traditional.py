"""Train TF-IDF, BM25, and hybrid spam classifiers."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spam_detector.data import (  # noqa: E402
    load_email_dataframe,
    prepare_text_views,
    split_dataset,
    summarize_splits,
)
from spam_detector.traditional import (  # noqa: E402
    TraditionalExperimentConfig,
    run_traditional_experiments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TF-IDF, BM25, and TF-IDF+BM25 hybrid spam classifiers."
    )
    parser.add_argument("--csv-path", required=True, help="Path to the merged dataset CSV.")
    parser.add_argument(
        "--output-dir",
        default="outputs/traditional",
        help="Directory for model outputs and metrics.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional row cap.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--max-features", type=int, default=10000, help="Vocabulary size cap.")
    parser.add_argument("--min-df", type=int, default=5, help="Minimum document frequency.")
    parser.add_argument("--max-df", type=float, default=0.7, help="Maximum document frequency.")
    parser.add_argument("--logistic-c", type=float, default=1.0, help="Inverse reg strength.")
    parser.add_argument("--logistic-max-iter", type=int, default=1000, help="Max LR iterations.")
    parser.add_argument("--bm25-k1", type=float, default=1.2, help="BM25 k1 parameter.")
    parser.add_argument(
        "--bm25-b",
        type=float,
        default=0.0,
        help="BM25 b parameter; 0.0 matches proposal-style TF saturation.",
    )
    parser.add_argument(
        "--hybrid-tfidf-weight",
        type=float,
        default=1.0,
        help="Weight applied to TF-IDF features inside the hybrid model.",
    )
    parser.add_argument(
        "--hybrid-bm25-weight",
        type=float,
        default=1.0,
        help="Weight applied to BM25 features inside the hybrid model.",
    )
    parser.add_argument(
        "--extra-stopwords",
        nargs="*",
        default=None,
        help="Optional extra stopwords to remove.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataframe = load_email_dataframe(
        args.csv_path,
        max_samples=args.max_samples,
        random_state=args.random_state,
    )
    prepared = prepare_text_views(dataframe, extra_stopwords=args.extra_stopwords)
    splits = split_dataset(
        prepared,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split_summary = summarize_splits(splits)
    (output_dir / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2),
        encoding="utf-8",
    )

    config = TraditionalExperimentConfig(
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        logistic_c=args.logistic_c,
        logistic_max_iter=args.logistic_max_iter,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        hybrid_tfidf_weight=args.hybrid_tfidf_weight,
        hybrid_bm25_weight=args.hybrid_bm25_weight,
    )
    results = run_traditional_experiments(splits, str(output_dir), config=config)

    print(json.dumps({"split_summary": split_summary, "results": results}, indent=2))


if __name__ == "__main__":
    main()
