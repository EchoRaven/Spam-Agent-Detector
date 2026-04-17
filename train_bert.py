"""Train a BERT spam classifier."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spam_detector.bert_classifier import (  # noqa: E402
    BertExperimentConfig,
    run_bert_experiment,
)
from spam_detector.data import (  # noqa: E402
    load_email_dataframe,
    prepare_text_views,
    split_dataset,
    summarize_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BERT spam classifier.")
    parser.add_argument("--csv-path", required=True, help="Path to the merged dataset CSV.")
    parser.add_argument(
        "--output-dir",
        default="outputs/bert",
        help="Directory for model outputs and metrics.",
    )
    parser.add_argument("--model-name", default="bert-base-uncased", help="HF model name.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional row cap.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max length.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataframe = load_email_dataframe(
        args.csv_path,
        max_samples=args.max_samples,
        random_state=args.random_state,
    )
    prepared = prepare_text_views(dataframe)
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

    config = BertExperimentConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_length=args.max_length,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        random_state=args.random_state,
    )
    results = run_bert_experiment(splits, str(output_dir), config=config)

    print(json.dumps({"split_summary": split_summary, "results": results}, indent=2))


if __name__ == "__main__":
    main()
