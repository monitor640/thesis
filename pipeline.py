from typing import Callable, Optional
import pandas as pd

from utils import save_checkpoint, load_checkpoint, checkpoint_exists, list_checkpoints, save_dataframe_csv
from telegram_reader import fetch_telegram_data
import features


class Pipeline:
    """
    Pipeline for propaganda detection with checkpoint support.
    
    Each stage is a function that transforms the DataFrame and saves a checkpoint.
    Supports resuming from any previously completed stage.
    """
    
    def __init__(self, sample_n: Optional[int] = None):
        """
        sample_n: if set, overrides config.SAMPLE_N for the 05_sample stage only.
                  Use None to rely on config.SAMPLE_N.
        """
        self._sample_n_override = sample_n
        self.stages: list[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = []
        self._register_stages()
    
    def _register_stages(self):
        """Register all pipeline stages in order."""
        self.stages = [
            ("01_raw_data", self._fetch_data),
            ("02_filter_length", self._filter_text_bounds),
            ("03_language", features.add_language),
            ("03b_target_language", features.filter_by_target_language),
            ("03c_punctuation", features.add_punctuation_count),
            ("03d_hashtags", features.add_hashtag_count),
            ("03e_links", features.add_link_domain_features),
            ("03f_bw_thinking", features.add_bw_thinking_features),
            ("03g_full_caps_words", features.add_full_caps_word_ratio),
            ("03h_simple_text", features.add_simple_text_features),
            ("03i_direct_quotes", features.add_direct_quote_features),
            ("03j_emoji", features.add_emoji_count),
            ("04_pos_tags", features.add_pos_tags),
            ("04c_superlative", features.add_superlative_features),
            ("04b_adj_verb", features.add_adj_verb_features),
            ("04d_dep_syntax", features.add_dep_syntax_features),
            ("05_sample", self._sample_rows),
            # Add more feature stages here as you implement them:
            # ("06_sentiment", features.add_sentiment),
            # etc.
        ]
    
    def _fetch_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Fetch raw data from Telegram."""
        return fetch_telegram_data()

    def _filter_text_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Max length (noise / huge posts) then min length (too short for labeling)."""
        df = features.filter_by_max_text_length(df)
        return features.filter_by_min_text_length(df)

    def _sample_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Persist full filtered pool for labeling refresh, then random sample."""
        save_dataframe_csv(df, "pool_pre_sample")
        return features.sample_rows(df, n=self._sample_n_override)
    
    def get_stage_index(self, stage_name: str) -> int:
        """Get the index of a stage by name."""
        for i, (name, _) in enumerate(self.stages):
            if name == stage_name:
                return i
        raise ValueError(f"Unknown stage: {stage_name}")
    
    def get_latest_checkpoint(self) -> Optional[tuple[int, str, pd.DataFrame]]:
        """
        Find the most recent checkpoint.
        
        Returns:
            Tuple of (stage_index, stage_name, data) or None if no checkpoints
        """
        existing = list_checkpoints()
        if not existing:
            return None
        
        for i in range(len(self.stages) - 1, -1, -1):
            stage_name = self.stages[i][0]
            if stage_name in existing:
                data = load_checkpoint(stage_name)
                return (i, stage_name, data)
        
        return None
    
    def run(
        self,
        start_from: Optional[str] = None,
        end_at: Optional[str] = None,
        initial_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Run the pipeline.
        
        Args:
            start_from: Stage name to resume from (uses its checkpoint as input).
                       If None, auto-detects from latest checkpoint or starts fresh.
            end_at: Stage name to stop at (inclusive). If None, runs all stages.
            initial_df: If set with ``start_from``, use this DataFrame as input to the
                first executed stage (skip loading the previous stage's checkpoint).
                Must include a ``text`` column. For labeling CSV runs, use
                ``labeling_import.load_labeling_csv_for_pipeline``.
        
        Returns:
            Final DataFrame after all stages
        """
        start_idx = 0
        df = None
        
        if start_from:
            start_idx = self.get_stage_index(start_from)
            if initial_df is not None:
                if "text" not in initial_df.columns:
                    raise ValueError("initial_df must include a 'text' column")
                df = initial_df.copy()
            elif start_idx > 0:
                prev_stage = self.stages[start_idx - 1][0]
                df = load_checkpoint(prev_stage)
                if df is None:
                    raise ValueError(f"Cannot start from {start_from}: no checkpoint for {prev_stage}")
        else:
            latest = self.get_latest_checkpoint()
            if latest:
                idx, name, data = latest
                print(f"Found checkpoint at stage: {name}")
                start_idx = idx + 1
                df = data
                if start_idx >= len(self.stages):
                    print("Pipeline already complete. Use start_from to re-run a stage.")
                    return df
        
        end_idx = len(self.stages)
        if end_at:
            end_idx = self.get_stage_index(end_at) + 1
        
        for i in range(start_idx, end_idx):
            stage_name, stage_fn = self.stages[i]
            print(f"\n{'='*50}")
            print(f"Running stage: {stage_name}")
            print(f"{'='*50}")
            
            df = stage_fn(df)
            save_checkpoint(stage_name, df)
            
            print(f"Stage complete. DataFrame shape: {df.shape}")
        
        is_final_stage = (end_idx == len(self.stages))
        if is_final_stage and df is not None:
            save_dataframe_csv(df, "latest_output")
        
        return df
    
    def run_single_stage(self, stage_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Run a single stage without checkpointing (for testing)."""
        idx = self.get_stage_index(stage_name)
        _, stage_fn = self.stages[idx]
        return stage_fn(df)
    
    def list_stages(self) -> list[str]:
        """List all stage names in order."""
        return [name for name, _ in self.stages]
    
    def status(self) -> dict[str, bool]:
        """Get completion status of all stages."""
        return {name: checkpoint_exists(name) for name, _ in self.stages}


if __name__ == "__main__":
    pipeline = Pipeline()
    
    print("Pipeline stages:")
    for name in pipeline.list_stages():
        print(f"  - {name}")
    
    print("\nCheckpoint status:")
    for name, exists in pipeline.status().items():
        status = "completed" if exists else "pending"
        print(f"  - {name}: {status}")
