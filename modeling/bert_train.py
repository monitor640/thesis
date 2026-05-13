"""
Fine-tune **one** sequence model (default ``BERT_MODEL_ID`` in config, e.g. XLM-RoBERTa)
for continuous propaganda scores in [0, 4] (regression).

Trains with MSE against the consensus mean label (``y_propa``). Rows in
``data/holdout_test_manifest.csv`` are **excluded from training** and used **only**
for the final reported holdout MSE/MAE (same 100 rows as sklearn/LLM). Per-row holdout
predictions are written to ``data/holdout_bert_predictions.csv`` (``HOLDOUT_BERT_PREDS_CSV``).
The remaining
pool is split **90/10** for **grid-search trials** (3 values each: learning rate, epochs,
batch size). The best trial by inner ``eval_mse`` is refit on the **full** non-holdout pool;
the holdout is scored only after that final training.

Hyperparameters and paths: ``config.py`` (BERT_* and MODELING_TABLE_CSV).

  pip install torch transformers datasets accelerate

  python -m modeling.bert_train

``Trainer`` uses CUDA when PyTorch is built with CUDA and a GPU is visible (default
Hugging Face behavior); otherwise training runs on CPU.
"""

from __future__ import annotations

import shutil
from itertools import product

import numpy as np
import pandas as pd

from config import (
    BERT_CHECKPOINT_DIR,
    BERT_MAX_LENGTH,
    BERT_MODEL_ID,
    BERT_SEARCH_BATCH_SIZES,
    BERT_SEARCH_LEARNING_RATES,
    BERT_SEARCH_TRAIN_EPOCHS,
    BERT_SEED,
    BERT_TEXT_COLUMN,
    HOLDOUT_BERT_PREDS_CSV,
    HOLDOUT_MANIFEST_CSV,
    MODELING_TABLE_CSV,
)
from modeling.dataset import normalize_channel_column, validate_modeling_table
from modeling.metrics import evaluate_regression, write_metrics_json


def _print_bert_device() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            print(
                f"[BERT] using GPU: {torch.cuda.get_device_name(0)}",
                flush=True,
            )
        else:
            print("[BERT] using CPU", flush=True)
    except ImportError:
        print("[BERT] using CPU (torch not available)", flush=True)


def main() -> None:
    try:
        from datasets import Dataset
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        raise SystemExit(
            "BERT fine-tuning needs: pip install torch transformers datasets accelerate\n"
            + str(e)
        ) from e

    csv_path = MODELING_TABLE_CSV
    text_col = BERT_TEXT_COLUMN

    if not HOLDOUT_MANIFEST_CSV.is_file():
        raise SystemExit(
            f"Missing holdout manifest: {HOLDOUT_MANIFEST_CSV}\n"
            "Create it by running:\n"
            "  python run_model_benchmark.py sklearn"
        )

    if not csv_path.is_file():
        raise SystemExit(f"Modeling table not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    validate_modeling_table(df, need_text=True)
    df = df.dropna(subset=["y_propa", text_col])
    df = normalize_channel_column(df)
    ho = normalize_channel_column(pd.read_csv(HOLDOUT_MANIFEST_CSV, encoding="utf-8-sig"))
    merged = df.merge(
        ho[["channel", "telegram_id"]],
        on=["channel", "telegram_id"],
        how="left",
        indicator=True,
    )
    train_pool_df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    holdout_df = merged[merged["_merge"] == "both"].drop(columns=["_merge"])
    if len(holdout_df) != len(ho):
        print(
            f"Warning: holdout merge matched {len(holdout_df)} rows, manifest has {len(ho)} "
            "(check channel + telegram_id vs modeling table)."
        )

    y_pool = (
        pd.to_numeric(train_pool_df["y_propa"], errors="coerce")
        .clip(0.0, 4.0)
        .astype(float)
        .values
    )
    strat = np.rint(y_pool).clip(0, 4).astype(int)
    from sklearn.model_selection import train_test_split

    try:
        idx_tr, idx_va = train_test_split(
            np.arange(len(train_pool_df)),
            test_size=0.1,
            random_state=BERT_SEED,
            stratify=strat,
        )
    except ValueError:
        idx_tr, idx_va = train_test_split(
            np.arange(len(train_pool_df)),
            test_size=0.1,
            random_state=BERT_SEED,
            shuffle=True,
        )

    inner_train_df = train_pool_df.iloc[idx_tr].reset_index(drop=True)
    inner_val_df = train_pool_df.iloc[idx_va].reset_index(drop=True)

    y_train = (
        pd.to_numeric(inner_train_df["y_propa"], errors="coerce")
        .clip(0.0, 4.0)
        .astype(float)
        .values
    )
    y_val = (
        pd.to_numeric(inner_val_df["y_propa"], errors="coerce")
        .clip(0.0, 4.0)
        .astype(float)
        .values
    )
    X_train = inner_train_df[text_col].fillna("").astype(str).tolist()
    X_val = inner_val_df[text_col].fillna("").astype(str).tolist()

    y_holdout = (
        pd.to_numeric(holdout_df["y_propa"], errors="coerce")
        .clip(0.0, 4.0)
        .astype(float)
        .values
    )
    X_holdout = holdout_df[text_col].fillna("").astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_ID)

    def tokenize_batch(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=BERT_MAX_LENGTH,
        )

    ds_tr = Dataset.from_dict({"text": X_train, "labels": y_train.tolist()})
    ds_tr = ds_tr.map(tokenize_batch, batched=True)
    ds_tr = ds_tr.remove_columns(["text"])

    ds_va = Dataset.from_dict({"text": X_val, "labels": y_val.tolist()})
    ds_va = ds_va.map(tokenize_batch, batched=True)
    ds_va = ds_va.remove_columns(["text"])

    ds_holdout = Dataset.from_dict({"text": X_holdout, "labels": y_holdout.tolist()})
    ds_holdout = ds_holdout.map(tokenize_batch, batched=True)
    ds_holdout = ds_holdout.remove_columns(["text"])

    y_full = (
        pd.to_numeric(train_pool_df["y_propa"], errors="coerce")
        .clip(0.0, 4.0)
        .astype(float)
        .values
    )
    X_full = train_pool_df[text_col].fillna("").astype(str).tolist()
    ds_full = Dataset.from_dict({"text": X_full, "labels": y_full.tolist()})
    ds_full = ds_full.map(tokenize_batch, batched=True)
    ds_full = ds_full.remove_columns(["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    BERT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    _print_bert_device()

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred = logits.squeeze(-1) if logits.ndim > 1 else logits
        pred = np.clip(np.asarray(pred, dtype=float), 0.0, 4.0)
        labels = np.asarray(labels, dtype=float)
        return {
            "mse": float(mean_squared_error(labels, pred)),
            "mae": float(mean_absolute_error(labels, pred)),
        }

    grid_root = BERT_CHECKPOINT_DIR / "grid_search"
    shutil.rmtree(grid_root, ignore_errors=True)
    grid_root.mkdir(parents=True, exist_ok=True)

    n_trials = (
        len(BERT_SEARCH_LEARNING_RATES)
        * len(BERT_SEARCH_TRAIN_EPOCHS)
        * len(BERT_SEARCH_BATCH_SIZES)
    )
    trial_rows: list[dict] = []
    best_mse = float("inf")
    best_hparams: dict = {}

    for ti, (lr, n_ep, bs) in enumerate(
        product(
            BERT_SEARCH_LEARNING_RATES,
            BERT_SEARCH_TRAIN_EPOCHS,
            BERT_SEARCH_BATCH_SIZES,
        )
    ):
        trial_dir = grid_root / f"trial_{ti:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            BERT_MODEL_ID,
            num_labels=1,
            problem_type="regression",
        )
        # ``compute_metrics`` → ``eval_mse``; checkpoint reload safe on transformers>=4.41.
        training_args = TrainingArguments(
            output_dir=str(trial_dir),
            num_train_epochs=float(n_ep),
            per_device_train_batch_size=int(bs),
            per_device_eval_batch_size=int(bs),
            learning_rate=float(lr),
            warmup_ratio=0.1,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_mse",
            greater_is_better=False,
            seed=BERT_SEED,
            logging_steps=20,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds_tr,
            eval_dataset=ds_va,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        print(
            f"[BERT grid] trial {ti + 1}/{n_trials}  lr={lr:g}  epochs={n_ep:g}  batch={bs}  "
            f"train={len(ds_tr)} val={len(ds_va)}",
            flush=True,
        )
        trainer.train()
        if getattr(trainer.state, "best_metric", None) is not None:
            score = float(trainer.state.best_metric)
        else:
            score = float(trainer.evaluate()["eval_mse"])
        row = {
            "trial": ti,
            "learning_rate": float(lr),
            "epochs": float(n_ep),
            "batch_size": int(bs),
            "best_eval_mse": score,
            "output_dir": str(trial_dir),
        }
        trial_rows.append(row)
        if score < best_mse:
            best_mse = score
            best_hparams = {
                "learning_rate": float(lr),
                "epochs": float(n_ep),
                "batch_size": int(bs),
                "best_eval_mse": score,
            }
        del trainer
        del model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    print("[BERT] best grid (lowest inner eval_mse):", best_hparams, flush=True)

    final_tmp = BERT_CHECKPOINT_DIR / "final_train_tmp"
    shutil.rmtree(final_tmp, ignore_errors=True)
    final_tmp.mkdir(parents=True, exist_ok=True)
    model_final = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_ID,
        num_labels=1,
        problem_type="regression",
    )
    training_args_final = TrainingArguments(
        output_dir=str(final_tmp),
        num_train_epochs=best_hparams["epochs"],
        per_device_train_batch_size=int(best_hparams["batch_size"]),
        per_device_eval_batch_size=int(best_hparams["batch_size"]),
        learning_rate=best_hparams["learning_rate"],
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="no",
        save_strategy="no",
        seed=BERT_SEED,
        logging_steps=50,
    )
    trainer_final = Trainer(
        model=model_final,
        args=training_args_final,
        train_dataset=ds_full,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    print(
        f"[BERT] final fit on full train pool (n={len(ds_full)}) with best hparams",
        flush=True,
    )
    trainer_final.train()
    metrics: dict = {}
    try:
        import torch

        metrics["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            metrics["cuda_device"] = str(torch.cuda.get_device_name(0))
    except ImportError:
        metrics["cuda_available"] = False
    out_ho = trainer_final.predict(ds_holdout)
    raw_ho = out_ho.predictions
    pred_ho = raw_ho.squeeze(-1) if raw_ho.ndim > 1 else raw_ho
    pred_ho = np.clip(np.asarray(pred_ho, dtype=float), 0.0, 4.0).reshape(-1)
    reg = evaluate_regression(y_holdout, pred_ho)
    metrics["holdout_mse"] = reg["mse"]
    metrics["holdout_mae"] = reg["mae"]
    metrics["holdout_rmse"] = reg["rmse"]
    metrics["holdout_r2_vs_const"] = reg["r2_vs_const"]
    metrics["holdout_mse_ci_low"] = reg.get("mse_ci_low")
    metrics["holdout_mse_ci_high"] = reg.get("mse_ci_high")
    metrics["holdout_mae_ci_low"] = reg.get("mae_ci_low")
    metrics["holdout_mae_ci_high"] = reg.get("mae_ci_high")
    metrics["holdout_r2_ci_low"] = reg.get("r2_ci_low")
    metrics["holdout_r2_ci_high"] = reg.get("r2_ci_high")
    metrics["holdout_ci_n_resamples"] = reg.get("ci_n_resamples")
    metrics["holdout_n"] = int(len(holdout_df))
    metrics["train_pool_n"] = int(len(train_pool_df))
    metrics["train_inner_n"] = int(len(inner_train_df))
    metrics["validation_inner_n"] = int(len(inner_val_df))
    metrics["holdout_manifest_csv"] = str(HOLDOUT_MANIFEST_CSV.resolve())

    pred_df = holdout_df[["channel", "telegram_id"]].copy()
    pred_df["y_true"] = np.asarray(y_holdout, dtype=float)
    pred_df["pred_bert"] = pred_ho
    pred_df["telegram_id"] = pd.to_numeric(pred_df["telegram_id"], errors="coerce").astype("Int64")
    if pred_df["telegram_id"].isna().any():
        raise ValueError("holdout rows with invalid telegram_id cannot be exported")
    pred_df["telegram_id"] = pred_df["telegram_id"].astype(int)
    HOLDOUT_BERT_PREDS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(HOLDOUT_BERT_PREDS_CSV, index=False, encoding="utf-8-sig")
    print("Wrote holdout predictions:", HOLDOUT_BERT_PREDS_CSV, flush=True)

    metrics["holdout_predictions_csv"] = str(HOLDOUT_BERT_PREDS_CSV.resolve())
    metrics["grid_search"] = {"n_trials": n_trials, "trials": trial_rows, "best": best_hparams}
    metrics["final_learning_rate"] = best_hparams["learning_rate"]
    metrics["final_epochs"] = best_hparams["epochs"]
    metrics["final_batch_size"] = int(best_hparams["batch_size"])
    metrics["schema"] = "benchmark_bert_v5"
    metrics["model_id"] = BERT_MODEL_ID
    metrics["max_length"] = BERT_MAX_LENGTH
    metrics["modeling_table_csv"] = str(MODELING_TABLE_CSV.resolve())
    write_metrics_json(BERT_CHECKPOINT_DIR / "metrics.json", metrics)
    print(
        "Holdout (final eval only): MSE =",
        metrics["holdout_mse"],
        " MAE =",
        metrics["holdout_mae"],
        f" (n={metrics['holdout_n']})",
    )
    trainer_final.save_model(str(BERT_CHECKPOINT_DIR / "final_model"))
    tokenizer.save_pretrained(str(BERT_CHECKPOINT_DIR / "final_model"))
    print("Saved:", BERT_CHECKPOINT_DIR / "final_model")


if __name__ == "__main__":
    main()
