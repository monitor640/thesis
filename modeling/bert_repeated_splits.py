"""Repeated random 400/100 splits for BERT, no grid search.

Loops over ``range(base_seed, base_seed + n_repeats)`` and trains one BERT model
per seed using the saved final hyperparameters from
``checkpoints/bert_propa/metrics.json`` (``final_learning_rate``,
``final_epochs``, ``final_batch_size``). The seed schedule mirrors
``modeling.repeated_splits`` so every iteration draws the *same* 100-row test
indices as LR / RF / LLM, enabling row-wise paired comparisons later.

Heavy: 500 retrains of XLM-R-base on 400 rows is hours on GPU, days on CPU.
Use ``--n-repeats`` to lower the count when prototyping.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BERT_CHECKPOINT_DIR,
    BERT_FULL_RUN_CHECKPOINT_DIR,
    BERT_MAX_LENGTH,
    BERT_MODEL_ID,
    BERT_SEED,
    BERT_TEXT_COLUMN,
    HOLDOUT_N,
    MODELING_TABLE_CSV,
)
from modeling.dataset import normalize_channel_column, validate_modeling_table
from modeling.metrics import write_metrics_json
from modeling.repeated_splits import score_row, split_indices, summarize_repeats


def _candidate_metrics_paths() -> list[Path]:
    """Default search order for the BERT ``final_*`` hparams JSON."""
    return [
        BERT_FULL_RUN_CHECKPOINT_DIR / "metrics.json",
        BERT_CHECKPOINT_DIR / "metrics.json",
    ]


def _read_best_params(explicit_path: Path | None = None) -> dict:
    """Locate and parse the BERT ``final_*`` hparams.

    If ``explicit_path`` is provided, it must exist and contain ``final_*``.
    Otherwise the first existing file in :func:`_candidate_metrics_paths`
    *with* ``final_learning_rate`` wins. This skips predict-only stubs that
    overwrote ``BERT_CHECKPOINT_DIR/metrics.json`` without retraining.
    """
    if explicit_path is not None:
        if not explicit_path.is_file():
            raise SystemExit(f"BERT metrics JSON not found: {explicit_path}")
        candidates = [explicit_path]
    else:
        candidates = [p for p in _candidate_metrics_paths() if p.is_file()]
        if not candidates:
            raise SystemExit(
                "No BERT metrics JSON found. Run "
                "`python run_model_benchmark.py bert` once before "
                "`bert-repeated-splits`."
            )

    last_err: KeyError | None = None
    for path in candidates:
        meta = json.loads(path.read_text(encoding="utf-8"))
        try:
            params = {
                "learning_rate": float(meta["final_learning_rate"]),
                "epochs": float(meta["final_epochs"]),
                "batch_size": int(meta["final_batch_size"]),
                "model_id": str(meta.get("model_id", BERT_MODEL_ID)),
                "source_metrics_json": str(path.resolve()),
            }
            print(f"[BERT-RS] loaded final_* hparams from: {path}", flush=True)
            return params
        except KeyError as e:
            last_err = e
            print(
                f"[BERT-RS] skipping {path} (missing {e}); "
                "looks like a predict-only stub.",
                flush=True,
            )
            continue
    raise SystemExit(
        f"None of the BERT metrics JSONs contained final_* hparams "
        f"(last error: {last_err}). Re-run `python run_model_benchmark.py bert` "
        "to regenerate, or pass --metrics explicitly."
    )


def _print_device() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            print(
                f"[BERT-RS] using GPU: {torch.cuda.get_device_name(0)}",
                flush=True,
            )
        else:
            print("[BERT-RS] using CPU", flush=True)
    except ImportError:
        print("[BERT-RS] using CPU (torch not available)", flush=True)


def main(
    *,
    n_repeats: int = 500,
    base_seed: int = 100,
    holdout_n: int = HOLDOUT_N,
    metrics_json: Path | None = None,
) -> None:
    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        raise SystemExit(
            "BERT repeated-splits needs: pip install torch transformers datasets accelerate\n"
            + str(e)
        ) from e

    best = _read_best_params(metrics_json)
    model_id = best["model_id"]
    print(
        f"[BERT-RS] best params: lr={best['learning_rate']:g} epochs={best['epochs']:g} "
        f"batch={best['batch_size']}",
        flush=True,
    )

    df = pd.read_csv(MODELING_TABLE_CSV, encoding="utf-8-sig")
    validate_modeling_table(df, need_text=True)
    df = df.dropna(subset=["y_propa", BERT_TEXT_COLUMN]).reset_index(drop=True)
    df = normalize_channel_column(df)
    y = (
        pd.to_numeric(df["y_propa"], errors="coerce")
        .clip(0.0, 4.0)
        .astype(float)
        .values
    )
    X_text = df[BERT_TEXT_COLUMN].fillna("").astype(str).tolist()
    strat = np.rint(y).clip(0, 4).astype(int)
    if len(df) <= holdout_n:
        raise SystemExit(
            f"Need more than {holdout_n} rows for {holdout_n}-row holdout."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenize_batch(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=BERT_MAX_LENGTH,
        )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    BERT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    work_root = BERT_CHECKPOINT_DIR / "repeated_splits_tmp"
    shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True, exist_ok=True)

    _print_device()

    rows: list[dict] = []
    for i in range(n_repeats):
        s = base_seed + i
        idx_tr, idx_te = split_indices(len(df), s, holdout_n, strat)
        X_tr = [X_text[j] for j in idx_tr]
        X_te = [X_text[j] for j in idx_te]
        y_tr = y[idx_tr].tolist()
        y_te = y[idx_te]

        ds_tr = Dataset.from_dict({"text": X_tr, "labels": y_tr})
        ds_tr = ds_tr.map(tokenize_batch, batched=True).remove_columns(["text"])
        ds_te = Dataset.from_dict({"text": X_te, "labels": y_te.tolist()})
        ds_te = ds_te.map(tokenize_batch, batched=True).remove_columns(["text"])

        trial_dir = work_root / f"seed_{s:05d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=1, problem_type="regression"
        )
        training_args = TrainingArguments(
            output_dir=str(trial_dir),
            num_train_epochs=best["epochs"],
            per_device_train_batch_size=best["batch_size"],
            per_device_eval_batch_size=best["batch_size"],
            learning_rate=best["learning_rate"],
            warmup_ratio=0.1,
            weight_decay=0.01,
            eval_strategy="no",
            save_strategy="no",
            seed=BERT_SEED,
            logging_steps=200,
            disable_tqdm=True,
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds_tr,
            processing_class=tokenizer,
            data_collator=data_collator,
        )
        print(f"[BERT-RS] {i + 1}/{n_repeats} seed={s} train={len(ds_tr)} test={len(ds_te)}",
              flush=True)
        trainer.train()
        out = trainer.predict(ds_te)
        raw = out.predictions
        pred = raw.squeeze(-1) if raw.ndim > 1 else raw
        pred = np.clip(np.asarray(pred, dtype=float), 0.0, 4.0).reshape(-1)
        rows.append(score_row(s, y_te, pred))

        del trainer, model
        shutil.rmtree(trial_dir, ignore_errors=True)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    shutil.rmtree(work_root, ignore_errors=True)
    summary = summarize_repeats(rows)
    payload = {
        "schema": "benchmark_bert_repeated_splits_v1",
        "model_id": model_id,
        "modeling_table_csv": str(MODELING_TABLE_CSV.resolve()),
        "source_metrics_json": best["source_metrics_json"],
        "n_repeats": int(n_repeats),
        "base_seed": int(base_seed),
        "holdout_n": int(holdout_n),
        "final_hparams": {
            "learning_rate": best["learning_rate"],
            "epochs": best["epochs"],
            "batch_size": best["batch_size"],
        },
        "per_seed_rows": rows,
        "summary": summary,
    }
    out_path = BERT_CHECKPOINT_DIR / "metrics_repeated_splits.json"
    write_metrics_json(out_path, payload)
    print(
        f"\n[BERT-RS] done. mse={summary['mse_mean']:.4f} "
        f"95% CI=[{summary['mse_ci_low']:.4f}, {summary['mse_ci_high']:.4f}]  "
        f"r2={summary['r2_mean']:.3f}  ({summary['n_repeats']} splits)",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-repeats", type=int, default=500)
    parser.add_argument("--base-seed", type=int, default=100)
    parser.add_argument("--holdout-n", type=int, default=HOLDOUT_N)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help=(
            "BERT metrics JSON containing final_learning_rate / final_epochs / "
            "final_batch_size. Defaults to checkpoints/bert_propa_full_run/metrics.json "
            "(falls back to checkpoints/bert_propa/metrics.json)."
        ),
    )
    args = parser.parse_args()
    main(
        n_repeats=args.n_repeats,
        base_seed=args.base_seed,
        holdout_n=args.holdout_n,
        metrics_json=args.metrics,
    )
