"""
Feature extraction functions for propaganda detection.

Each function follows the signature:
    def add_FEATURE(df: pd.DataFrame) -> pd.DataFrame

Functions should add new column(s) to the DataFrame and return it.
"""

import re
import unicodedata
from urllib.parse import urlparse, unquote

import pandas as pd
import nltk
from emoji import emoji_list
from lingua import Language, LanguageDetectorBuilder


# Languages common for Baltic / regional Telegram posts (extend in config if needed).
_DEFAULT_LANG_DETECTOR_LANGUAGES = (
    Language.ESTONIAN,
    Language.RUSSIAN,
    Language.ENGLISH,
    Language.FINNISH,
    Language.LATVIAN,
    Language.LITHUANIAN,
    Language.GERMAN,
    Language.UKRAINIAN,
)

_lang_detector = None

# `#` plus Unicode “word” run (letters, digits, underscore; includes Estonian letters).
_HASHTAG_RE = re.compile(r"#\w+")

# HTTP(S) URLs (Telegram/Markdown noise stops at common closers).
_LINK_HTTP_RE = re.compile(r"https?://[^\s\]>\)\"'<>|]+", re.IGNORECASE)

# Black-and-white / absolutist wording (surface forms; Estonian only, case-insensitive tokens).
_BW_THINKING_LEXICON: frozenset[str] = frozenset(
    {
        "kõik",
        "kõige",
        "kunagi",
        "eales",
        "iial",
        "alati",
        "igavesti",
        "tervenisti",
        "täiesti",
        "üleni",
        "täitsa",
        "täielikult",
        "üdini",
        "läbinisti",
        "läbini",
        "absoluutne",
        "absoluutselt",
        "totaalne",
        "totaalselt",
        "ainult",
        "ainus",
        "kogu",
    }
)
_BW_TOKEN_RE = re.compile(r"[a-zäöüõšž]+")

# Alnum/underscore “word” runs (Unicode); letter case checked per character inside.
_WORD_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Vabamorf partofspeech (see EstNLTK tutorial: tables of morphological categories).
_VABAMORF_ADJ_TAGS = frozenset(
    {
        "A",  # omadussõna algvõrre
        "C",  # keskvõrre
        "U",  # ülivõrre
        "G",  # genitiivatribuut (käändumatu omadussõna, nt "balti")
    }
)
_VABAMORF_VERB_TAGS = frozenset({"V"})  # tegusõna

# Verb `form` values for tingiv kõneviis (conditional); Filosoft Vabamorf tagset.
_VABAMORF_CONDITIONAL_VERB_FORMS = frozenset(
    {
        "ks",
        "ksid",
        "ksime",
        "ksin",
        "ksite",
        "neg ks",
        "nuks",
        "nuksid",
        "nuksime",
        "nuksin",
        "nuksite",
        "neg nuks",
        "taks",  # impersonal present conditional
        "tuks",  # impersonal past conditional
    }
)

# Roots ending in -ke/-kene that are not diminutives (false positives for the suffix heuristic).
_FALSE_DIMINUTIVE_ROOTS = frozenset({"raske", "väike"})

# MaltParser / CoNLL (UD + legacy Filosoft): subject-like dependency labels.
_SUBJ_DEPREL_PREFIXES = ("nsubj", "csubj")

_DEP_SYNTAX_WARNED_JAVA = False


def get_language_detector():
    """Lazy singleton; restricted language set is faster than from_all_languages()."""
    global _lang_detector
    if _lang_detector is None:
        _lang_detector = LanguageDetectorBuilder.from_languages(
            *_DEFAULT_LANG_DETECTOR_LANGUAGES
        ).build()
    return _lang_detector


def detect_post_language(text: str, max_chars: int = 800) -> str:
    """
    Return ISO 639-1 language code (lowercase) or 'unknown'.
    Uses a prefix of long posts for speed; enough for identification in most cases.
    """
    if text is None or not str(text).strip():
        return "unknown"
    sample = str(text).strip()
    if len(sample) > max_chars:
        sample = sample[:max_chars]
    lang = get_language_detector().detect_language_of(sample)
    if lang is None:
        return "unknown"
    code = lang.iso_code_639_1
    if code is None:
        return "unknown"
    return code.name.lower()


def filter_by_max_text_length(
    df: pd.DataFrame,
    max_chars: int | None = None,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Keep only posts with text length <= max_chars (character count).

    Uses config.MAX_POST_CHARS when max_chars is None.
    """
    from config import MAX_POST_CHARS

    limit = max_chars if max_chars is not None else MAX_POST_CHARS
    df = df.copy()
    lengths = df["text"].fillna("").astype(str).str.len()
    before = len(df)
    df = df[lengths <= limit].reset_index(drop=True)
    removed = before - len(df)
    if verbose:
        print(
            f"Length filter (<= {limit} chars): removed {removed}, kept {len(df)}"
        )
    return df


def filter_by_min_text_length(
    df: pd.DataFrame,
    min_chars: int | None = None,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Keep only posts with stripped text length >= min_chars.

    Uses config.MIN_POST_CHARS when min_chars is None.
    """
    from config import MIN_POST_CHARS

    limit = min_chars if min_chars is not None else MIN_POST_CHARS
    df = df.copy()
    lengths = df["text"].fillna("").astype(str).str.strip().str.len()
    before = len(df)
    df = df[lengths >= limit].reset_index(drop=True)
    removed = before - len(df)
    if verbose:
        print(
            f"Min length filter (>= {limit} chars): removed {removed}, kept {len(df)}"
        )
    return df


def filter_by_target_language(
    df: pd.DataFrame,
    lang: str | None = None,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Keep rows where column `language` matches target (ISO 639-1, lowercase).

    Run after add_language. Uses config.LABELING_TARGET_LANG when lang is None.
    """
    from config import LABELING_TARGET_LANG

    code = (lang or LABELING_TARGET_LANG).lower()
    if "language" not in df.columns:
        raise ValueError("filter_by_target_language requires a 'language' column")
    df = df.copy()
    before = len(df)
    df = df[df["language"].astype(str).str.lower() == code].reset_index(drop=True)
    removed = before - len(df)
    if verbose:
        print(
            f"Language filter (language == {code!r}): removed {removed}, kept {len(df)}"
        )
    return df


def sample_rows(
    df: pd.DataFrame,
    n: int | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Random sample of n rows (without replacement).

    Uses config.SAMPLE_N / SAMPLE_RANDOM_STATE when n / random_state is None.
    If n is None or <= 0, returns df unchanged.
    If n >= len(df), returns all rows (shuffled only if n == len(df) and you sample n).
    """
    from config import SAMPLE_N, SAMPLE_RANDOM_STATE

    target = n if n is not None else SAMPLE_N
    rs = random_state if random_state is not None else SAMPLE_RANDOM_STATE

    df = df.copy()
    if target is None or target <= 0:
        print("Sample skipped: n not set or <= 0 (keeping all rows)")
        return df

    take = min(int(target), len(df))
    if take == len(df):
        print(f"Sample: requested {target}, dataset has {len(df)} — keeping all rows")
        return df

    out = df.sample(n=take, random_state=rs).reset_index(drop=True)
    print(f"Sampled {len(out)} rows from {len(df)} (random_state={rs})")
    return out


def ensure_nltk_data():
    """Download required NLTK data if not present."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")
    
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger")
    
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng")


def _estnltk_text_class():
    """Lazy import; EstNLTK needs native extensions (see requirements note)."""
    try:
        from estnltk import Text
    except ImportError as e:
        raise ImportError(
            "EstNLTK is required for morphological POS tags. "
            "Try: pip install estnltk==1.7.4  "
            "On Windows, if build fails, use conda: "
            "conda install -c estnltk -c conda-forge estnltk=1.7.4"
        ) from e
    return Text


def _span_is_diminutive(span) -> bool:
    """
    Any analysed word whose Vabamorf root ends with diminutive -kene or -ke.

    Uses the first analysis only. Excludes non-diminutive -ke roots (e.g. raske, väike).
    """
    roots = span.root
    if not roots or roots[0] is None:
        return False
    r = str(roots[0]).lower().strip()
    if not r or r in _FALSE_DIMINUTIVE_ROOTS:
        return False
    if r.endswith("kene"):
        return True
    if r.endswith("ke"):
        return True
    return False


def _span_is_conditional_verb(span) -> bool:
    """Verb token whose first analysis `form` is Vabamorf conditional (tingiv) mood."""
    poss = span.partofspeech
    if not poss or poss[0] is None:
        return False
    if str(poss[0]).strip() != "V":
        return False
    forms = span.form
    if not forms or forms[0] is None:
        return False
    f = str(forms[0]).strip()
    return f in _VABAMORF_CONDITIONAL_VERB_FORMS


def _normalize_text_for_estnltk(s: str) -> str:
    """
    Strip characters that can leave visually non-empty text but zero EstNLTK tokens
    (TokensTagger IndexError on empty split_spans).
    """
    for ch in ("\ufeff", "\u200b", "\u200c", "\u200d", "\u2060"):
        s = s.replace(ch, "")
    return s


def _morph_pos_tags_and_counts(
    text: str,
) -> tuple[list[tuple[str, str]], int, int]:
    """
    One EstNLTK morph_analysis pass: POS list, diminutive count, conditional verb count.
    """
    raw = _normalize_text_for_estnltk(str(text) if text is not None else "")
    if not raw.strip():
        return [], 0, 0
    Text = _estnltk_text_class()
    t = Text(raw)
    try:
        t.tag_layer("morph_analysis")
    except IndexError:
        # EstNLTK TokensTagger: empty token spans for pathological / invisible-only strings
        return [], 0, 0
    except Exception as e:
        err = str(e)
        if "TokensTagger" in err or "split_spans" in err or "list index out of range" in err:
            return [], 0, 0
        raise
    pairs: list[tuple[str, str]] = []
    diminutives = 0
    conditionals = 0
    for span in t.morph_analysis:
        w = getattr(span, "text", None) or raw[span.start : span.end]
        poss = span.partofspeech
        if poss and poss[0] is not None:
            pos = str(poss[0])
        else:
            pos = "Z"
        pairs.append((w, pos))
        if _span_is_diminutive(span):
            diminutives += 1
        if _span_is_conditional_verb(span):
            conditionals += 1
    return pairs, diminutives, conditionals


def get_pos_tags(text: str) -> list[tuple[str, str]]:
    """
    Token + partofspeech pairs via EstNLTK morph_analysis (Vabamorf).

    Tags are single-letter Vabamorf POS codes (A, C, S, V, ...).
    """
    return _morph_pos_tags_and_counts(text)[0]


def add_language(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """
    Detect post language (lingua).

    Adds columns:
        - language: ISO 639-1 code (e.g. et, en, ru) or 'unknown'
    """
    if verbose:
        print("Detecting language...")
    df = df.copy()
    df["language"] = df["text"].apply(detect_post_language)
    return df


def count_punctuation_chars(text: str) -> int:
    """Count characters whose Unicode general category is punctuation (P*)."""
    return sum(
        1 for c in str(text) if unicodedata.category(c).startswith("P")
    )


def add_punctuation_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add punctuation_count: number of Unicode punctuation characters per post.

    Uses Unicode categories Po, Pd, Ps, Pe, Pi, Pf, Pc (not symbols like @ or emoji).
    """
    df = df.copy()
    df["punctuation_count"] = df["text"].fillna("").astype(str).map(count_punctuation_chars)
    return df


def count_hashtags(text: str) -> int:
    """Count #hashtag tokens (# followed by Unicode word characters)."""
    return len(_HASHTAG_RE.findall(str(text)))


def add_hashtag_count(df: pd.DataFrame) -> pd.DataFrame:
    """Add hashtag_count: number of hashtags in the post text."""
    df = df.copy()
    df["hashtag_count"] = df["text"].fillna("").astype(str).map(count_hashtags)
    return df


def count_emojis(text: str) -> int:
    """
    Count emoji grapheme clusters in text.

    Uses ``emoji.emoji_list`` so ZWJ sequences (e.g. family, profession modifiers) and
    skin-tone modifiers count as one emoji each.
    """
    return len(emoji_list(str(text) if text is not None else ""))


def add_emoji_count(df: pd.DataFrame) -> pd.DataFrame:
    """Add emoji_count: number of emoji clusters per post (PyPI ``emoji`` package)."""
    df = df.copy()
    df["emoji_count"] = df["text"].fillna("").astype(str).map(count_emojis)
    return df


def count_bw_thinking_words(text: str) -> int:
    """
    Count token matches for a fixed absolutist / black-and-white Estonian lexicon (exact surface forms).

    Each alphabetic token (letters äöüõšž) lowercased; repeats and multiple distinct hits all count.
    """
    s = str(text).lower()
    if not s:
        return 0
    return sum(1 for m in _BW_TOKEN_RE.finditer(s) if m.group(0) in _BW_THINKING_LEXICON)


def add_bw_thinking_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``bw_thinking_word_count`` from the absolutist wording lexicon (see ``_BW_THINKING_LEXICON``)."""
    print("Black-and-white / absolutist word counts (Estonian lexicon)...")
    df = df.copy()
    df["bw_thinking_word_count"] = df["text"].fillna("").astype(str).map(count_bw_thinking_words)
    return df


def full_caps_word_ratio(text: str) -> float:
    """
    Share of tokens that are fully uppercase in letters among letter-containing tokens.

    Each ``\\w+`` span with at least one alphabetic character counts as one word; digits or
    underscore-only runs are skipped. A word counts as all-caps if every alphabetic character
    is uppercase (mixed scripts e.g. Cyrillic supported). Ratio is all-caps words / word count,
    or 0.0 when there are no such words.
    """
    s = str(text) if text is not None else ""
    n_words = 0
    n_all_caps = 0
    for m in _WORD_TOKEN_RE.finditer(s):
        tok = m.group(0)
        letters = [c for c in tok if c.isalpha()]
        if not letters:
            continue
        n_words += 1
        if all(c.isupper() for c in letters):
            n_all_caps += 1
    if n_words == 0:
        return 0.0
    return n_all_caps / n_words


def add_full_caps_word_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``full_caps_word_ratio`` (fully capitalized letter-tokens / all letter-tokens)."""
    print("Full caps word ratio (uppercase words / words with letters)...")
    df = df.copy()
    df["full_caps_word_ratio"] = df["text"].fillna("").astype(str).map(full_caps_word_ratio)
    return df



#  Directly inspired by QUOTE EXTRACTION FROM ESTONIAN MEDIA:ANALYSIS AND TOOLS
# Direct speech: paired quotation marks, validated with Estonian punctuation on NLTK sentences.
# (1) Reporting clause before quote -> colon immediately before the opening mark (after trim).
# (2) Reporting clause after quote -> comma, !, or ? immediately before the closing mark (inner end).
# Pairs: „", „“, “”, straight "", «» (see e.g. Salway et al. 2017 on non-quote uses of marks).
_DIRECT_QUOTE_PAIR_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"„(.+?)”", re.DOTALL),  # U+201E … U+201D
    re.compile(r'„(.+?)"', re.DOTALL),  # low-9 + ASCII close (common in mixed typing)
    re.compile(r"“(.+?)”", re.DOTALL),  # U+201C … U+201D
    re.compile(r'"(.+?)"', re.DOTALL),  # ASCII double (minimal inner)
    re.compile(r"«(.+?)»", re.DOTALL),
)


def _is_estonian_direct_quote_span(sentence: str, m: re.Match[str]) -> bool:
    inner = m.group(1)
    if not inner.strip():
        return False
    before_open = sentence[: m.start()].rstrip()
    rule_colon_before = len(before_open) > 0 and before_open[-1] == ":"
    last_of_inner = inner.strip()[-1]
    rule_punct_before_close = last_of_inner in ",!?"
    return rule_colon_before or rule_punct_before_close


def _all_nonempty_quoted_pairs_one_sentence(
    sentence: str,
) -> list[tuple[re.Match[str], str]]:
    """Non-overlapping quoted spans (same delimiters as direct quotes), non-empty inner."""
    seen_bounds: set[tuple[int, int]] = set()
    out: list[tuple[re.Match[str], str]] = []
    for pat in _DIRECT_QUOTE_PAIR_RES:
        for m in pat.finditer(sentence):
            key = (m.start(), m.end())
            if key in seen_bounds:
                continue
            inner = m.group(1)
            if not inner.strip():
                continue
            seen_bounds.add(key)
            out.append((m, inner))
    return out


def _first_alpha_is_lowercase(s: str) -> bool:
    """True if the first alphabetic character in ``strip()`` is cased lowercase (e.g. äöüõšž)."""
    for c in s.strip():
        if c.isalpha():
            return c.islower()
    return False


def quote_feature_metrics(text: str) -> tuple[int, float, int, float]:
    """
    Quote-related counts from the same NLTK sentence pass.

    Returns
        ``direct_quote_count``, ``direct_quote_char_ratio`` (validated Estonian direct speech),
        ``satire_quote_like_count``, ``satire_quote_like_char_ratio`` (heuristic: quoted spans
        that fail the direct-speech punctuation rules but whose first letter is lowercase —
        weak proxy for mock/ironic speech vs. titles or names, not a satire classifier).
    """
    ensure_nltk_data()
    raw = str(text) if text is not None else ""
    if not raw:
        return 0, 0.0, 0, 0.0
    direct_inners: list[str] = []
    satire_inners: list[str] = []
    for sent in nltk.sent_tokenize(raw):
        for m, inner in _all_nonempty_quoted_pairs_one_sentence(sent):
            if _is_estonian_direct_quote_span(sent, m):
                direct_inners.append(inner)
            elif _first_alpha_is_lowercase(inner):
                satire_inners.append(inner)
    n_d = len(direct_inners)
    n_s = len(satire_inners)
    chars_d = sum(len(x) for x in direct_inners)
    chars_s = sum(len(x) for x in satire_inners)
    ratio_d = chars_d / len(raw)
    ratio_s = chars_s / len(raw)
    return n_d, ratio_d, n_s, ratio_s


def direct_quote_count_and_ratio(text: str) -> tuple[int, float]:
    """
    Count validated direct-quote spans and their share of post length (by characters).

    Sentence segmentation via NLTK ``sent_tokenize``; each span is counted at most once
    per open/close match inside a sentence.
    ``direct_quote_char_ratio`` = sum(len(inner)) / len(text) (0.0 if empty text).
    """
    a, b, _, _ = quote_feature_metrics(text)
    return a, b


def add_direct_quote_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quotation-mark spans using NLTK sentences.

    Columns: ``direct_quote_count``, ``direct_quote_char_ratio`` (Estonian colon / clause-final
    punctuation heuristics); ``satire_quote_like_count``, ``satire_quote_like_char_ratio``
    (unvalidated quoted text whose first alphabetic character is lowercase).
    """
    print(
        "Quotes: Estonian direct-speech heuristic + satire-like quoted spans (lowercase start)..."
    )
    df = df.copy()
    stats = df["text"].fillna("").astype(str).map(quote_feature_metrics)
    df["direct_quote_count"] = stats.apply(lambda x: int(x[0]))
    df["direct_quote_char_ratio"] = stats.apply(lambda x: float(x[1]))
    df["satire_quote_like_count"] = stats.apply(lambda x: int(x[2]))
    df["satire_quote_like_char_ratio"] = stats.apply(lambda x: float(x[3]))
    return df


def _normalize_channel_key(channel) -> str:
    return str(channel or "").strip().lower().lstrip("@")


def _trim_trailing_url_punct(s: str) -> str:
    while s and s[-1] in ").,;:]»\"'":
        s = s[:-1]
    return s


def extract_http_urls(text: str) -> list[str]:
    """Raw http(s) URL strings from text (one entry per match, trailing junk trimmed)."""
    s = str(text) if text is not None else ""
    return [_trim_trailing_url_punct(m.group(0)) for m in _LINK_HTTP_RE.finditer(s)]


def _netloc_host(netloc: str) -> str:
    netloc = (netloc or "").lower().strip()
    if not netloc:
        return ""
    if "@" in netloc:
        netloc = netloc.split("@")[-1]
    if ":" in netloc and "]" not in netloc:
        netloc = netloc.rsplit(":", 1)[0]
    return netloc


def _url_first_path_segment(parsed) -> str:
    parts = [p for p in (parsed.path or "").split("/") if p]
    if not parts:
        return ""
    return unquote(parts[0]).lower()


def _url_is_own_site(
    url: str,
    http_hosts: frozenset[str],
    telegram_slugs: frozenset[str],
) -> bool:
    if not http_hosts and not telegram_slugs:
        return False
    raw = _trim_trailing_url_punct(str(url).strip())
    if not raw:
        return False
    try:
        p = urlparse(raw)
    except ValueError:
        return False
    host = _netloc_host(p.netloc)
    if host in http_hosts:
        return True
    if telegram_slugs and host in ("t.me", "telegram.me"):
        seg0 = _url_first_path_segment(p)
        return bool(seg0) and seg0 in telegram_slugs
    return False


def count_link_own_other(text: str, channel) -> tuple[int, int, int]:
    """
    Return (link_count, link_own_site_count, link_other_site_count) using config mappings.

    Own = HTTP host in CHANNEL_LINK_OWN_HTTP_HOSTS or t.me / telegram.me slug in
    CHANNEL_LINK_OWN_TELEGRAM_SLUGS for this channel. Unknown channels: own always 0.
    """
    from config import CHANNEL_LINK_OWN_HTTP_HOSTS, CHANNEL_LINK_OWN_TELEGRAM_SLUGS

    urls = extract_http_urls(text)
    total = len(urls)
    key = _normalize_channel_key(channel)
    hosts = CHANNEL_LINK_OWN_HTTP_HOSTS.get(key, frozenset())
    tg_slugs = CHANNEL_LINK_OWN_TELEGRAM_SLUGS.get(key, frozenset())
    own = sum(1 for u in urls if _url_is_own_site(u, hosts, tg_slugs))
    return total, own, total - own


def add_link_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count http(s) links; split into first-party (channel’s own sites) vs other.

    Uses ``config.CHANNEL_LINK_OWN_HTTP_HOSTS`` and ``CHANNEL_LINK_OWN_TELEGRAM_SLUGS``.
    Requires columns ``text`` and ``channel``. Scheme-less ``t.me/...`` links are not counted
    unless you add a separate extractor later.
    """
    if "channel" not in df.columns:
        raise ValueError("add_link_domain_features requires a 'channel' column")

    print("Link counts (own-site vs other) from http(s) URLs...")
    df = df.copy()
    counts = df.apply(
        lambda r: count_link_own_other(r["text"], r["channel"]), axis=1, result_type="expand"
    )
    df["link_count"] = counts[0].astype(int)
    df["link_own_site_count"] = counts[1].astype(int)
    df["link_other_site_count"] = counts[2].astype(int)
    return df


def add_pos_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add POS tags using EstNLTK Vabamorf (morph_analysis).

    Adds columns:
        - pos_tags: list of (surface_word, partofspeech) tuples (Vabamorf POS letters)
        - diminutive_count: tokens whose Vabamorf root ends with -ke/-kene (any POS),
          excluding known false positives (see _FALSE_DIMINUTIVE_ROOTS).
        - conditional_verb_count: verbs (V) whose `form` is tingiv (conditional), incl. impersonal taks/tuks.
    """
    _estnltk_text_class()  # fail fast if EstNLTK missing

    print("Extracting POS tags, diminutives, conditional verbs (EstNLTK / Vabamorf)...")
    df = df.copy()
    _morph = df["text"].apply(_morph_pos_tags_and_counts)
    df["pos_tags"] = _morph.apply(lambda x: x[0])
    df["diminutive_count"] = _morph.apply(lambda x: x[1])
    df["conditional_verb_count"] = _morph.apply(lambda x: x[2])

    nonempty = (
        df["text"]
        .fillna("")
        .astype(str)
        .map(_normalize_text_for_estnltk)
        .str.strip()
        != ""
    )
    n_empty_morph = int((nonempty & (df["pos_tags"].apply(len) == 0)).sum())
    if n_empty_morph:
        print(
            f"Warning: {n_empty_morph} post(s) have text but no EstNLTK tokens "
            f"(pos_tags empty; tagger edge case or invisible-only characters)."
        )

    return df


def count_adjectives_verbs_from_pos_tags(
    pos_tags: list[tuple[str, str]] | None,
) -> tuple[int, int]:
    """Return (adjective_count, verb_count) using Vabamorf partofspeech codes."""
    if not pos_tags:
        return 0, 0
    adj = 0
    verb = 0
    for _w, t in pos_tags:
        if not t:
            continue
        tag = str(t).strip()
        if tag in _VABAMORF_ADJ_TAGS:
            adj += 1
        elif tag in _VABAMORF_VERB_TAGS:
            verb += 1
    return adj, verb


def count_superlative_adjectives_from_pos_tags(
    pos_tags: list[tuple[str, str]] | None,
) -> int:
    """Count tokens tagged U (ülivõrre) by EstNLTK / Vabamorf morph_analysis."""
    if not pos_tags:
        return 0
    return sum(1 for _w, t in pos_tags if str(t).strip() == "U")


def add_superlative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Superlative adjective count from EstNLTK: Vabamorf POS ``U`` (omadussõna, ülivõrre).

    See morph category tables (same source as tag ``C`` = comparative, ``A`` = positive).
    """
    if "pos_tags" not in df.columns:
        raise ValueError("add_superlative_features requires pos_tags (run add_pos_tags first)")

    df = df.copy()
    df["superlative_adjective_count"] = df["pos_tags"].map(
        count_superlative_adjectives_from_pos_tags
    )
    return df


def _is_subj_deprel(deprel) -> bool:
    """UD (nsubj, csubj, …) and legacy Filosoft @SUBJ."""
    if deprel is None:
        return False
    d = str(deprel).strip()
    if not d:
        return False
    low = d.lower()
    if low == "@subj" or low.endswith("@subj"):
        return True
    for pref in _SUBJ_DEPREL_PREFIXES:
        if low == pref or low.startswith(pref + ":"):
            return True
    return False


def _syntax_ann_value(ann, key: str):
    if ann is None:
        return None
    if hasattr(ann, "__getitem__"):
        try:
            return ann[key]
        except (KeyError, TypeError, IndexError):
            pass
    return getattr(ann, key, None)


def _group_word_indices_by_sentence(text) -> list[list[int]]:
    """Assign each word index to exactly one sentence span (containment / overlap)."""
    words = list(text.words)
    sents = list(text.sentences)
    if not words or not sents:
        return []
    groups: list[list[int]] = [[] for _ in sents]
    for wi, w in enumerate(words):
        chosen = 0
        for j, s in enumerate(sents):
            if w.start >= s.start and w.end <= s.end:
                chosen = j
                break
            if s.start <= w.start < s.end:
                chosen = j
                break
        else:
            best_j, best_ov = 0, -1
            for j, s in enumerate(sents):
                ov = min(w.end, s.end) - max(w.start, s.start)
                if ov > best_ov:
                    best_ov, best_j = ov, j
            chosen = best_j if best_ov > 0 else 0
        groups[chosen].append(wi)
    return groups


def _dep_stats_one_sentence(
    syntax_layer, word_indices: list[int]
) -> tuple[list[int], list[int], float, float] | None:
    """
    Returns (depths, outdegrees, outdeg_max, subj_count) for one sentence, or None if empty.
    depths/outdegrees are per token in sentence order (parallel to word_indices).
    """
    if not word_indices:
        return None
    ids: list[int] = []
    heads: list[int] = []
    deprels: list[str] = []
    for wi in word_indices:
        span = syntax_layer[wi]
        if not span.annotations:
            return None
        ann = span.annotations[0]
        tid = _syntax_ann_value(ann, "id")
        h = _syntax_ann_value(ann, "head")
        dr = _syntax_ann_value(ann, "deprel")
        try:
            tid_i = int(tid)
        except (TypeError, ValueError):
            tid_i = len(ids) + 1
        try:
            h_i = int(h)
        except (TypeError, ValueError):
            h_i = -1
        ids.append(tid_i)
        heads.append(h_i)
        deprels.append("" if dr is None else str(dr))

    n = len(ids)
    id_to_pos = {ids[p]: p for p in range(n)}
    outdeg = [0] * n
    for p in range(n):
        hi = heads[p]
        if hi > 0 and hi in id_to_pos:
            outdeg[id_to_pos[hi]] += 1

    depths: list[int] = []
    for p in range(n):
        d = 0
        cur = p
        visited: set[int] = set()
        while heads[cur] != 0:
            if cur in visited or d > n + 2:
                break
            visited.add(cur)
            par_id = heads[cur]
            if par_id not in id_to_pos:
                break
            cur = id_to_pos[par_id]
            d += 1
        depths.append(d)

    subj_n = sum(1 for dr in deprels if _is_subj_deprel(dr))
    ode_max = max(outdeg) if outdeg else 0
    return depths, outdeg, float(ode_max), float(subj_n)


def dep_syntax_stats_from_text(text: str) -> tuple[float, float, float, float, float]:
    """
    MaltParser dependency features (EstNLTK ``maltparser_syntax``): tree depth, branching, subject count.

    Requires Java (MaltParser) and EstNLTK syntax resources. On failure returns zeros.

    Returns:
        (dep_depth_max, dep_depth_mean, dep_outdegree_mean, dep_outdegree_max, dep_subj_count)
    """
    global _DEP_SYNTAX_WARNED_JAVA
    raw = _normalize_text_for_estnltk(str(text) if text is not None else "")
    if not raw.strip():
        return 0.0, 0.0, 0.0, 0.0, 0.0
    Text = _estnltk_text_class()
    t = Text(raw)
    try:
        t.tag_layer("maltparser_syntax")
    except IndexError:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    except Exception as e:
        err = str(e)
        low = err.lower()
        if (
            "TokensTagger" in err
            or "split_spans" in err
            or "list index out of range" in err
        ):
            return 0.0, 0.0, 0.0, 0.0, 0.0
        if "java" in low or "malt" in low:
            if not _DEP_SYNTAX_WARNED_JAVA:
                print(
                    "Warning: MaltParser dependency syntax failed (Java/runtime?). "
                    f"Syntax columns set to 0 for this post. First error: {e}"
                )
                _DEP_SYNTAX_WARNED_JAVA = True
            return 0.0, 0.0, 0.0, 0.0, 0.0
        raise

    if "maltparser_syntax" not in t.layers or len(t.maltparser_syntax) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    syn = t.maltparser_syntax
    groups = _group_word_indices_by_sentence(t)
    if not groups:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    all_depths: list[int] = []
    all_outdeg: list[int] = []
    depth_max_doc = 0
    outdeg_max_doc = 0
    subj_total = 0.0

    for word_ix in groups:
        one = _dep_stats_one_sentence(syn, word_ix)
        if not one:
            continue
        depths, outdeg, local_ode_max, subj_n = one
        all_depths.extend(depths)
        all_outdeg.extend(outdeg)
        if depths:
            depth_max_doc = max(depth_max_doc, max(depths))
        outdeg_max_doc = max(outdeg_max_doc, int(local_ode_max))
        subj_total += subj_n

    if not all_depths:
        return 0.0, 0.0, 0.0, float(outdeg_max_doc), subj_total

    ntok = len(all_depths)
    return (
        float(depth_max_doc),
        float(sum(all_depths) / ntok),
        float(sum(all_outdeg) / ntok),
        float(max(outdeg_max_doc, max(all_outdeg) if all_outdeg else 0)),
        subj_total,
    )


def add_dep_syntax_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dependency syntax via EstNLTK MaltParser (``maltparser_syntax`` layer).

    Adds columns:
        - dep_depth_max: longest root-to-token path in any sentence.
        - dep_depth_mean: mean token depth (0 = root) over all parsed words.
        - dep_outdegree_mean: mean number of direct dependents per token.
        - dep_outdegree_max: largest number of dependents on a single head.
        - dep_subj_count: tokens whose deprel is subject-like (UD ``nsubj``/``csubj``… or ``@SUBJ``).

    **Requires a Java runtime** for MaltParser. Slower than morph-only features.
    """
    _estnltk_text_class()
    print(
        "Dependency syntax (MaltParser via EstNLTK; needs Java) — "
        "dep_depth_*, dep_outdegree_*, dep_subj_count..."
    )
    print(
        "  (No output between lines is normal; long posts are slow. "
        "Watch CPU: java/python busy means still working.)"
    )
    df = df.copy()
    texts = df["text"].tolist()
    n = len(texts)
    step = max(1, n // 40)  # ~40 progress lines for 500 rows
    stats_list: list[tuple[float, float, float, float, float]] = []
    for i, text in enumerate(texts):
        if i == 0 or (i + 1) % step == 0 or (i + 1) == n:
            print(f"  dep_syntax: {i + 1}/{n} posts")
        stats_list.append(dep_syntax_stats_from_text(text))
    df["dep_depth_max"] = [s[0] for s in stats_list]
    df["dep_depth_mean"] = [s[1] for s in stats_list]
    df["dep_outdegree_mean"] = [s[2] for s in stats_list]
    df["dep_outdegree_max"] = [s[3] for s in stats_list]
    df["dep_subj_count"] = [int(s[4]) for s in stats_list]
    return df


def add_adj_verb_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add adjective_count, verb_count, adj_to_verb_ratio from pos_tags.

    Expects Vabamorf POS from EstNLTK: adjectives A/C/U/G, verbs V.
    adj_to_verb_ratio is 0.0 when verb_count is 0.
    """
    if "pos_tags" not in df.columns:
        raise ValueError("add_adj_verb_features requires a pos_tags column")

    df = df.copy()
    counts = df["pos_tags"].apply(count_adjectives_verbs_from_pos_tags)
    df["adjective_count"] = counts.apply(lambda x: x[0])
    df["verb_count"] = counts.apply(lambda x: x[1])
    df["adj_to_verb_ratio"] = 0.0
    vpos = df["verb_count"] > 0
    df.loc[vpos, "adj_to_verb_ratio"] = (
        df.loc[vpos, "adjective_count"] / df.loc[vpos, "verb_count"]
    )
    return df


# =============================================================================
# Simple text statistics (NLTK sentences); also exposed as standalone add_* helpers.
# =============================================================================


def _letter_level_caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _simple_text_feature_tuple(text: str) -> tuple[int, int, float, int, int, float]:
    """
    word_count, sentence_count, avg_words_per_sentence, exclamation_count,
    question_count, caps_ratio (uppercase letters / all letters).
    """
    t_raw = str(text) if text is not None else ""
    t = t_raw
    word_count = len(t.split())
    sents = nltk.sent_tokenize(t)
    sentence_count = len(sents)
    if sentence_count == 0:
        avg_words_per_sentence = 0.0
    else:
        avg_words_per_sentence = sum(len(s.split()) for s in sents) / sentence_count
    return (
        word_count,
        sentence_count,
        avg_words_per_sentence,
        t.count("!"),
        t.count("?"),
        _letter_level_caps_ratio(t),
    )


def add_simple_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add bundled lexical stats: ``word_count``, ``sentence_count``,
    ``avg_words_per_sentence`` (mean words per NLTK sentence), ``exclamation_count``,
    ``question_count``, ``caps_ratio`` (letter-level uppercase share).

    Runs a single tokenize pass per row. Requires NLTK punkt (see ``ensure_nltk_data``).
    """
    ensure_nltk_data()
    df = df.copy()
    texts = df["text"].fillna("").astype(str)
    rows = [_simple_text_feature_tuple(t) for t in texts]
    extra = pd.DataFrame(
        rows,
        columns=[
            "word_count",
            "sentence_count",
            "avg_words_per_sentence",
            "exclamation_count",
            "question_count",
            "caps_ratio",
        ],
        index=df.index,
    )
    return pd.concat([df, extra], axis=1)


def add_word_count(df: pd.DataFrame) -> pd.DataFrame:
    """Add word count (whitespace-split tokens)."""
    df = df.copy()
    df["word_count"] = df["text"].fillna("").astype(str).map(lambda x: len(x.split()))
    return df


def add_sentence_count(df: pd.DataFrame) -> pd.DataFrame:
    """Add sentence count (NLTK ``sent_tokenize``)."""
    ensure_nltk_data()
    df = df.copy()
    df["sentence_count"] = df["text"].fillna("").astype(str).map(
        lambda x: len(nltk.sent_tokenize(x))
    )
    return df


def add_avg_words_per_sentence(df: pd.DataFrame) -> pd.DataFrame:
    """Mean words per sentence (NLTK boundaries); 0.0 if no sentences."""
    ensure_nltk_data()
    df = df.copy()

    def avg_wps(t: str) -> float:
        sents = nltk.sent_tokenize(str(t) if t else "")
        if not sents:
            return 0.0
        return sum(len(s.split()) for s in sents) / len(sents)

    df["avg_words_per_sentence"] = df["text"].fillna("").astype(str).map(avg_wps)
    return df


def add_exclamation_count(df: pd.DataFrame) -> pd.DataFrame:
    """Add exclamation mark count (often indicative of emotional language)."""
    df = df.copy()
    df["exclamation_count"] = df["text"].fillna("").astype(str).str.count("!")
    return df


def add_question_count(df: pd.DataFrame) -> pd.DataFrame:
    """Add question mark count."""
    df = df.copy()
    df["question_count"] = df["text"].fillna("").astype(str).str.count("?")
    return df


def add_caps_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio of uppercase letters among all letters (shouting / emphasis)."""
    df = df.copy()
    df["caps_ratio"] = df["text"].fillna("").astype(str).map(_letter_level_caps_ratio)
    return df


# Add more feature functions as needed:
# def add_sentiment(df: pd.DataFrame) -> pd.DataFrame: ...
# def add_named_entities(df: pd.DataFrame) -> pd.DataFrame: ...
# def add_propaganda_keywords(df: pd.DataFrame) -> pd.DataFrame: ...
# def add_emotional_words(df: pd.DataFrame) -> pd.DataFrame: ...
# def add_superlatives_count(df: pd.DataFrame) -> pd.DataFrame: ...
# def add_hedge_words(df: pd.DataFrame) -> pd.DataFrame: ...
# def add_readability_score(df: pd.DataFrame) -> pd.DataFrame: ...
# etc.
