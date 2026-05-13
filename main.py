"""
Main entry point for the propaganda detection pipeline.

Stages include length bounds (max/min), language detection, Estonian-only filter,
POS tags, then random sample. See config.MIN_POST_CHARS and LABELING_TARGET_LANG.

Usage:
    python main.py                           # Run full pipeline (auto-resume)
    python main.py --start-from 04_pos_tags  # Resume from specific stage
    python main.py --sample 500              # Override SAMPLE_N for this run
    python main.py --status                  # Show checkpoint status
    python main.py --clear                   # Clear all checkpoints

    Feature pass on an exported labeling sheet (keeps row order; default --end-at 04d_dep_syntax).
    Writes ``data/labeling_with_features.csv`` (same path as config.LABELING_WITH_FEATURES_CSV) unless you pass --export-csv NAME:
    python main.py --from-labeling-csv data/labeling_export.csv --start-from 03_language

    Full analysis workflow: see ``run_model_benchmark.py`` module docstring (defaults in ``config.py``).
"""

import argparse
from pathlib import Path

from config import LABELING_WITH_FEATURES_CSV
from labeling_import import load_labeling_csv_for_pipeline
from pipeline import Pipeline
from utils import list_checkpoints, clear_checkpoints, save_dataframe_csv


def main():
    parser = argparse.ArgumentParser(
        description="Telegram Propaganda Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--start-from",
        type=str,
        help="Stage name to resume from (e.g., 04_pos_tags)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Random sample N rows after feature stages (overrides config.SAMPLE_N; 0 = skip)",
    )
    parser.add_argument(
        "--end-at",
        type=str,
        help="Stage name to stop at (inclusive)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show pipeline status and exit",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all checkpoints and exit",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="List all pipeline stages and exit",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        metavar="NAME",
        help="Export final DataFrame to data/<NAME>.csv (default for --from-labeling-csv: labeling_with_features)",
    )
    parser.add_argument(
        "--from-labeling-csv",
        type=Path,
        metavar="FILE",
        help="Run feature stages on a labeling export (Text column). Use with --start-from 03_language. "
        "Overwrites checkpoints for stages that run; back up checkpoints/ first if needed.",
    )
    
    args = parser.parse_args()
    
    sample_n = args.sample
    pipeline = Pipeline(sample_n=sample_n)
    
    if args.list_stages:
        print("Pipeline stages:")
        for name in pipeline.list_stages():
            print(f"  - {name}")
        return
    
    if args.status:
        print("Checkpoint status:")
        for name, exists in pipeline.status().items():
            status = "completed" if exists else "pending"
            print(f"  [{status:^9}] {name}")
        
        checkpoints = list_checkpoints()
        if checkpoints:
            print(f"\nLatest checkpoint: {checkpoints[-1]}")
        else:
            print("\nNo checkpoints found.")
        return
    
    if args.clear:
        confirm = input("Clear all checkpoints? [y/N]: ")
        if confirm.lower() == "y":
            clear_checkpoints()
            print("All checkpoints cleared.")
        else:
            print("Aborted.")
        return
    
    end_at = args.end_at
    start_from = args.start_from
    initial_df = None

    if args.from_labeling_csv:
        initial_df = load_labeling_csv_for_pipeline(args.from_labeling_csv)
        print(
            f"Loaded labeling CSV: {len(initial_df)} rows from {args.from_labeling_csv}\n"
            "Note: checkpoints for executed stages will be overwritten. "
            "Copy checkpoints/ aside first if you need the Telegram run.\n"
        )
        if start_from is None:
            start_from = "03_language"
        if end_at is None:
            end_at = "04d_dep_syntax"
            print(f"Default --end-at {end_at} (skips 05_sample so row order matches the sheet).\n")

    print("Starting propaganda detection pipeline...")
    print(f"Stages: {', '.join(pipeline.list_stages())}")
    print()
    
    df = pipeline.run(
        start_from=start_from,
        end_at=end_at,
        initial_df=initial_df,
    )
    
    print("\nPipeline complete!")
    print(f"Final DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    export_name = args.export_csv
    if export_name is None and args.from_labeling_csv:
        export_name = LABELING_WITH_FEATURES_CSV.stem
    if export_name:
        save_dataframe_csv(df, export_name)


if __name__ == "__main__":
    main()
