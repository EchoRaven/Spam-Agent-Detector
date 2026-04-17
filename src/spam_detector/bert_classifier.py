"""BERT-based spam classifier."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .metrics import classification_metrics


@dataclass
class BertExperimentConfig:
    model_name: str = "bert-base-uncased"
    learning_rate: float = 2e-5
    batch_size: int = 8
    epochs: int = 3
    max_length: int = 256
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    random_state: int = 42


class EmailDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[List[int]]], labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(value[index], dtype=torch.long)
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: np.ndarray, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits

        loss_function = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(self.class_weights, device=logits.device, dtype=torch.float)
        )
        loss = loss_function(logits, labels)
        return (loss, outputs) if return_outputs else loss


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


def _build_dataset(tokenizer, texts: List[str], labels: List[int], max_length: int) -> EmailDataset:
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    return EmailDataset(encodings, labels)


def _metrics_from_logits(logits: np.ndarray, labels: List[int]) -> Dict:
    predictions = np.argmax(logits, axis=1)
    return classification_metrics(labels, predictions)


def run_bert_experiment(
    splits,
    output_dir: str,
    *,
    config: BertExperimentConfig,
) -> Dict[str, Dict]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_texts = splits["train"]["text_bert"].tolist()
    val_texts = splits["val"]["text_bert"].tolist()
    test_texts = splits["test"]["text_bert"].tolist()

    y_train = splits["train"]["label"].tolist()
    y_val = splits["val"]["label"].tolist()
    y_test = splits["test"]["label"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
    )

    train_dataset = _build_dataset(tokenizer, train_texts, y_train, config.max_length)
    val_dataset = _build_dataset(tokenizer, val_texts, y_val, config.max_length)
    test_dataset = _build_dataset(tokenizer, test_texts, y_test, config.max_length)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=np.array(y_train),
    )

    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],
        seed=config.random_state,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        class_weights=class_weights,
        compute_metrics=lambda eval_pred: _metrics_from_logits(
            eval_pred.predictions,
            eval_pred.label_ids.tolist(),
        ),
    )

    training_output = trainer.train()
    validation_predictions = trainer.predict(val_dataset)
    test_predictions = trainer.predict(test_dataset)

    model_dir = output_path / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    summary = {
        "config": asdict(config),
        "train_runtime_seconds": float(training_output.metrics.get("train_runtime", 0.0)),
        "validation": _metrics_from_logits(validation_predictions.predictions, y_val),
        "test": _metrics_from_logits(test_predictions.predictions, y_test),
        "class_weights": class_weights.tolist(),
    }

    _save_json(output_path / "metrics_summary.json", summary)
    _save_json(output_path / "train_metrics.json", training_output.metrics)
    _save_json(output_path / "trainer_log_history.json", {"history": trainer.state.log_history})
    return summary
