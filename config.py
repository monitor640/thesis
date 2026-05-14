import os
from pathlib import Path
from dotenv import load_dotenv

# Always load the project .env (not cwd — venv / IDE runs often start elsewhere).
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
DATA_DIR = BASE_DIR / "data"

# Fixed 100-row test set (same rows for sklearn final eval, LLM subset MSE, BERT eval).
# Manifest is created when you run ``python run_model_benchmark.py sklearn`` (needs a consensus modeling table).
HOLDOUT_N = 100
HOLDOUT_RANDOM_STATE = 42
HOLDOUT_MANIFEST_CSV = DATA_DIR / "holdout_test_manifest.csv"

TELEGRAM_API_ID = int(os.getenv("TELEGRAM_API_ID", 0))
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH", "")
TELEGRAM_PHONE = os.getenv("TELEGRAM_PHONE", None)
SESSION_NAME = "propaganda_detector"

CHANNELS = [
    "eestitelegram","MarkoWatsel","InfoDefEstonia","euudised","vanglaplaneet"
]

# First-party URLs for link_own vs link_other features (`features.add_link_domain_features`).
# Keys: normalized channel username (lowercase, no @), matching the `channel` column.
# - HTTP(S): netloc lowercased, including www when posts use it.
# - Telegram public URLs: only host t.me / telegram.me; values are first path segments (lowercase).
CHANNEL_LINK_OWN_HTTP_HOSTS: dict[str, frozenset[str]] = {
    "eestitelegram": frozenset({"telegram.ee", "www.telegram.ee"}),
    "euudised": frozenset({"eestieest.com", "www.eestieest.com"}),
    "vanglaplaneet": frozenset({"vanglaplaneet.ee", "www.vanglaplaneet.ee"}),
}
CHANNEL_LINK_OWN_TELEGRAM_SLUGS: dict[str, frozenset[str]] = {
    # InfoDefense Estonia channel + shared InfoDefense hub ([t.me/InfoDefEstonia](https://t.me/InfoDefEstonia), [t.me/InfoDefAll](http://t.me/InfoDefAll))
    "infodefestonia": frozenset({"infodefestonia", "infodefall"}),
}
# MarkoWatsel (markowatsel): intentionally absent — no mapped “own” site; all links count as other.

FILTER_KEYWORDS = [
    # Optional: filter messages containing these keywords
    # Leave empty to fetch all messages
]

FETCH_LIMIT = None  # Max messages per channel (None for all)

# Drop posts longer than this (characters). Applied after fetch, before language/POS.
MAX_POST_CHARS = 1000
# Labeling / analysis: drop very short posts (characters, after strip).
MIN_POST_CHARS = 80
# Keep only this language (ISO 639-1) after detection, before POS and sampling.
LABELING_TARGET_LANG = "et"

# Random sample size for the final dataset after all feature stages.
# None or 0 = keep all rows. Otherwise sample min(SAMPLE_N, len(df)) rows.
SAMPLE_N = 500
SAMPLE_RANDOM_STATE = 42

# Keyword sets for exploratory topic tagging in visualizations.
# A post matches a topic if its text contains any keyword (case-insensitive).
# One post can match multiple topics.
TOPICS = {
    "russia": [
        "putin",
        "ukraine",
        "ukraina",
        "crimea",
        "krimm",
    ],
    "covid": ["covid", "koroona", "vaktsiin"],
    "iraan": ["iraan", "iraani", "khamenei", "iran"],

}

# --- Preliminary figures + modeling: same numeric feature list (must exist in labeling_with_features) ---
FEATURE_COLS = [
    "punctuation_count",
    "hashtag_count",
    "emoji_count",
    "link_count",
    "link_own_site_count",
    "link_other_site_count",
    "bw_thinking_word_count",
    "full_caps_word_ratio",
    "word_count",
    "sentence_count",
    "avg_words_per_sentence",
    "exclamation_count",
    "question_count",
    "caps_ratio",
    "direct_quote_count",
    "direct_quote_char_ratio",
    "satire_quote_like_count",
    "satire_quote_like_char_ratio",
    "diminutive_count",
    "conditional_verb_count",
    "superlative_adjective_count",
    "adjective_count",
    "verb_count",
    "adj_to_verb_ratio",
    "dep_depth_max",
    "dep_depth_mean",
    "dep_outdegree_mean",
    "dep_outdegree_max",
    "dep_subj_count",
]

# mutual_info_classif: count / integer-like features → discrete_features=True; ratios / real-valued → False.
# All FEATURE_COLS not listed here are treated as discrete (avoids k-NN MI collapse on sparse counts).
MUTUAL_INFO_CONTINUOUS_FEATURE_COLS: frozenset[str] = frozenset(
    {
        "full_caps_word_ratio",
        "avg_words_per_sentence",
        "caps_ratio",
        "direct_quote_char_ratio",
        "satire_quote_like_char_ratio",
        "adj_to_verb_ratio",
    }
)

LINK_FEATURE_COLS = [
    "link_count",
    "link_own_site_count",
    "link_other_site_count",
]
DEP_SYNTAX_FEATURE_COLS = [
    "dep_depth_max",
    "dep_depth_mean",
    "dep_outdegree_mean",
    "dep_outdegree_max",
    "dep_subj_count",
]
SIMPLE_TEXT_FEATURE_COLS = [
    "word_count",
    "sentence_count",
    "avg_words_per_sentence",
    "exclamation_count",
    "question_count",
    "caps_ratio",
]
BW_AND_FULLCAPS_COLS = [
    "bw_thinking_word_count",
    "full_caps_word_ratio",
]
EMOJI_FEATURE_COLS = ["emoji_count"]

# Mann–Whitney / FDR / Cohen's d / MI table from preliminary_propa_features.py; used to pick
# tabular features for sklearn and LLM-with-features (see modeling.feature_config).
PRELIMINARY_TESTS_CSV = DATA_DIR / "preliminary_propa_group_tests.csv"
# Thesis-aligned gate: a row in FEATURE_COLS is kept if ANY of the following holds:
#   (1) |Cohen's d| >= FEATURE_THESIS_COHEN_D_ABS_MIN
#   (2) BH-adjusted Mann–Whitney (p_fdr_bh) < FEATURE_THESIS_FDR_ALPHA
#   (3) mutual_info_classif >= FEATURE_THESIS_MI_MIN and (
#           |d| >= FEATURE_THESIS_MI_GATE_COHEN_D_ABS
#           or uncorrected p_mannwhitney < FEATURE_THESIS_MI_GATE_P_UNC)
FEATURE_THESIS_COHEN_D_ABS_MIN = 0.15
FEATURE_THESIS_FDR_ALPHA = 0.05
FEATURE_THESIS_MI_MIN = 0.02
FEATURE_THESIS_MI_GATE_COHEN_D_ABS = 0.10
FEATURE_THESIS_MI_GATE_P_UNC = 0.10

# Modeling tables always use consensus: (Uku Propa + Simon Propa) / 2 on [0, 4] as y_propa / propa_consensus.

# --- Default CSV paths (edit here instead of long CLI invocations) ---
MODELING_TABLE_CSV = DATA_DIR / "modeling_table.csv"
LABELING_WITH_FEATURES_CSV = DATA_DIR / "labeling_with_features.csv"
UKU_LABELING_CSV = DATA_DIR / "uku-labeling.csv"
SIMON_LABELING_CSV = DATA_DIR / "simon-labeling.csv"
POOL_PRE_SAMPLE_CSV = DATA_DIR / "pool_pre_sample.csv"
LABELING_EXPORT_CSV = DATA_DIR / "labeling_export.csv"

FIGURES_DIR = BASE_DIR / "figures"

# Sklearn benchmark: fixed holdout predictions table (after each sklearn run).
HOLDOUT_SKLEARN_PREDS_CSV = DATA_DIR / "holdout_sklearn_predictions.csv"
HOLDOUT_BERT_PREDS_CSV = DATA_DIR / "holdout_bert_predictions.csv"
HOLDOUT_LLM_PREDS_CSV = DATA_DIR / "holdout_llm_predictions.csv"
HOLDOUT_LLM_WITH_FEATURES_PREDS_CSV = DATA_DIR / "holdout_llm_with_features_predictions.csv"

# Cross-model summary written by ``run_model_benchmark compare``.
MODEL_COMPARISON_CSV = DATA_DIR / "model_comparison.csv"

# BERT (``python -m modeling.bert_train``).
#
# Default is xlm-roberta-base (multilingual, well tested on Estonian) — works
# without a custom HF auth token. For better Estonian-specific results swap to
# tartuNLP/EstBERT via the BERT_MODEL_ID env var: it's monolingual Estonian and
# usually beats XLM-R / mBERT on Estonian downstream tasks.
BERT_CHECKPOINT_DIR = CHECKPOINTS_DIR / "bert_propa"
# Full training run from the original grid-search pass (has ``final_*`` hparams + grid trials).
# ``bert_repeated_splits`` reads its saved hparams from here by default; ``BERT_CHECKPOINT_DIR/metrics.json``
# may be a predict-only stub that lacks them.
BERT_FULL_RUN_CHECKPOINT_DIR = CHECKPOINTS_DIR / "bert_propa_full_run"
BERT_MODEL_ID = os.getenv("BERT_MODEL_ID", "xlm-roberta-base")
BERT_TEXT_COLUMN = "text"
BERT_TRAIN_EPOCHS = float(os.getenv("BERT_TRAIN_EPOCHS", "15"))
BERT_BATCH_SIZE = int(os.getenv("BERT_BATCH_SIZE", "8"))
BERT_LEARNING_RATE = float(os.getenv("BERT_LEARNING_RATE", "2e-5"))
BERT_MAX_LENGTH = int(os.getenv("BERT_MAX_LENGTH", "256"))
BERT_SEED = 42
# ``modeling/bert_train`` uses ``BERT_SEARCH_*`` below, not these single defaults (kept for
# env overrides / other scripts).

# BERT grid search: 3 values each (learning rate, train epochs, per-device batch size).
# Trials use the same 90/10 inner split; the trial with lowest ``eval_mse`` (best epoch)
# wins. The winner is then trained on the **full** non-holdout pool; holdout is only
# scored at the very end (unchanged protocol).
BERT_SEARCH_LEARNING_RATES = [1e-5, 2e-5, 3e-5]
BERT_SEARCH_TRAIN_EPOCHS = [5.0, 7.0, 9.0]
BERT_SEARCH_BATCH_SIZES = [4, 8, 12]

# Optional: ``export BERT_FORCE_CPU=1`` before ``python -m modeling.bert_train`` to force CPU.

# LLM JSONL checkpoints + benchmark metrics JSON (sklearn, LLM, reliability)
LLM_CHECKPOINT_DIR = CHECKPOINTS_DIR / "modeling"
METRICS_DIR = LLM_CHECKPOINT_DIR
METRICS_SKLEARN_JSON = METRICS_DIR / "metrics_sklearn.json"
METRICS_REPEATED_SPLITS_JSON = METRICS_DIR / "metrics_repeated_splits.json"
METRICS_RELIABILITY_JSON = METRICS_DIR / "metrics_reliability.json"


def metrics_llm_json_path(variant: str, *, model_id: str | None = None) -> Path:
    """Per-(variant, model) metrics JSON.

    If ``model_id`` is omitted, ``LLM_MODEL`` is used. Filename mirrors
    :func:`llm_checkpoint_jsonl` so each model's run leaves its own artifact.
    """
    if variant not in _LLM_VARIANTS:
        raise ValueError(f"unknown LLM variant: {variant!r}")
    slug = _llm_model_slug(model_id or LLM_MODEL)
    return METRICS_DIR / f"metrics_llm_{variant}__{slug}.json"

# LLM benchmark: OpenAI Chat Completions + structured output (see modeling/llm_predict.py).
# Override with OPENAI_MODEL or LLM_MODEL in .env.
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.4-nano")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "256"))
LLM_REQUEST_SLEEP_S = float(os.getenv("LLM_REQUEST_SLEEP_S", "0.15"))

# visualize.py
VISUALIZE_INPUT_CSV = POOL_PRE_SAMPLE_CSV
VISUALIZE_TIME_FREQ = "MS"

# preliminary_propa_features.py outputs
PRELIMINARY_FEATURE_SUMMARY_CSV = DATA_DIR / "preliminary_propa_feature_summary.csv"
PRELIMINARY_FDR_ALPHA = 0.05
PRELIMINARY_MI_RANDOM_STATE = 0


def _llm_model_slug(model_id: str) -> str:
    """File-safe slug for embedding a model id in checkpoint / metrics filenames."""
    s = (model_id or "").strip().lower()
    return "".join(c if c.isalnum() else "-" for c in s).strip("-") or "unknown"


_LLM_VARIANTS = (
    "no_features",
    "with_features",
)


def llm_checkpoint_jsonl(variant: str, *, model_id: str | None = None) -> Path:
    """Resume file for ``run_model_benchmark llm``.

    Filename embeds both the variant and a slug of the model id so different
    OpenAI models do not overwrite each other's predictions. If ``model_id`` is
    omitted, ``LLM_MODEL`` (default) is used.
    """
    if variant not in _LLM_VARIANTS:
        raise ValueError(f"unknown LLM variant: {variant!r}")
    slug = _llm_model_slug(model_id or LLM_MODEL)
    return LLM_CHECKPOINT_DIR / f"llm_{variant}__{slug}.jsonl"
