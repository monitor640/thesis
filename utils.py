import pickle
from pathlib import Path
from typing import Any, Optional
import pandas as pd

from config import CHECKPOINTS_DIR


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    CHECKPOINTS_DIR.mkdir(exist_ok=True)


def get_checkpoint_path(stage_name: str) -> Path:
    """Get the full path for a checkpoint file."""
    return CHECKPOINTS_DIR / f"{stage_name}.pkl"


def save_checkpoint(stage_name: str, data: Any) -> Path:
    """
    Save data to a checkpoint file.
    
    Args:
        stage_name: Name of the pipeline stage (e.g., "01_raw_data")
        data: Data to pickle (typically a DataFrame)
    
    Returns:
        Path to the saved checkpoint file
    """
    ensure_dirs()
    path = get_checkpoint_path(stage_name)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(stage_name: str) -> Optional[Any]:
    """
    Load data from a checkpoint file.
    
    Args:
        stage_name: Name of the pipeline stage
    
    Returns:
        Loaded data, or None if checkpoint doesn't exist
    """
    path = get_checkpoint_path(stage_name)
    if not path.exists():
        print(f"No checkpoint found: {path}")
        return None
    
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Checkpoint loaded: {path}")
    return data


def checkpoint_exists(stage_name: str) -> bool:
    """Check if a checkpoint exists for a given stage."""
    return get_checkpoint_path(stage_name).exists()


def list_checkpoints() -> list[str]:
    """List all available checkpoint stage names."""
    ensure_dirs()
    return sorted([p.stem for p in CHECKPOINTS_DIR.glob("*.pkl")])


def latest_checkpoint_in_order(stage_names: list[str]) -> Optional[str]:
    """
    Return the last stage name in ``stage_names`` (pipeline order) that has a checkpoint.

    Prefer this over ``list_checkpoints()[-1]`` — lexicographic order is not pipeline order.
    """
    for name in reversed(stage_names):
        if checkpoint_exists(name):
            return name
    return None


def clear_checkpoints():
    """Remove all checkpoint files."""
    ensure_dirs()
    for p in CHECKPOINTS_DIR.glob("*.pkl"):
        p.unlink()
        print(f"Removed: {p}")


def save_dataframe_csv(df: pd.DataFrame, name: str) -> Path:
    """Save DataFrame to CSV in data directory."""
    from config import DATA_DIR
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"CSV saved: {path}")
    return path
