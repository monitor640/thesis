"""
Propaganda score via OpenAI Chat Completions + structured output (Pydantic schema).

Set ``OPENAI_API_KEY``. Optional ``OPENAI_MODEL`` (default: ``config.LLM_MODEL``).

Each API call records ``usage`` (prompt / completion / total tokens) on the JSONL line
when the API returns it; the run ends with a one-line token total for new calls only.

See: https://developers.openai.com/api/docs/guides/structured-outputs
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from config import LLM_MAX_TOKENS, LLM_MODEL, LLM_REQUEST_SLEEP_S
from modeling.dataset import normalize_channel

SYSTEM = """Rate the strength of propaganda in the text. Output only the structured score field.

0 — No propaganda
1 — Text contains bias towards some propaganda narratives, but is generally truthful
2 — Text contains clear propaganda narratives but contains no clear false claims
3 — Text contains clear propaganda narratives and established false claims
4 — Text contains multiple propaganda narratives and false claims with clear propagandistic framing throughout

Decimals allowed when the message falls between two levels."""

# Short instruction prepended when ``with_features`` sends engineered numbers to the model.
WITH_FEATURES_PREAMBLE = (
    "The following numeric features were computed from the message: "
)


def _format_features_raw(row, feature_cols: list[str]) -> str:
    blob = {c: _json_safe(row.get(c)) for c in feature_cols if c in row.index}
    return "Features (JSON):\n" + json.dumps(blob, ensure_ascii=False)


class PropaScore(BaseModel):
    propa: float = Field(
        ge=0.0,
        le=4.0,
        description="Propaganda intensity 0–4 per the system-message rubric (decimals if between levels).",
    )


def default_model() -> str:
    return (os.getenv("OPENAI_MODEL") or LLM_MODEL).strip()


def _client():
    """Official api.openai.com only by default.
    """
    from openai import OpenAI

    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError("OPENAI_API_KEY is empty (check thesis/.env next to config.py).")
    custom = (os.getenv("LLM_OPENAI_BASE_URL") or "").strip().rstrip("/")
    if custom:
        return OpenAI(api_key=key, base_url=custom)
    return OpenAI(api_key=key, base_url="https://api.openai.com/v1")


def _ckpt_key(ch: str, tid: int) -> tuple[str, int]:
    return (normalize_channel(ch), int(tid))


def _load_ckpt(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
            out[_ckpt_key(j.get("channel", ""), int(j["telegram_id"]))] = j
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return out


def _ckpt_get(
    done: dict[tuple[str, int], dict[str, Any]], ch: str, tid: int
) -> dict[str, Any] | None:
    k = _ckpt_key(ch, tid)
    if k in done:
        return done[k]
    return done.get(("", tid)) if k[0] else None


def _append_ckpt(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _usage_dict(comp: Any) -> dict[str, int] | None:
    u = getattr(comp, "usage", None)
    if u is None:
        return None
    try:
        pt = int(getattr(u, "prompt_tokens", None) or 0)
        ct = int(getattr(u, "completion_tokens", None) or 0)
        tt = int(getattr(u, "total_tokens", None) or 0)
        if pt == 0 and ct == 0 and tt == 0:
            return None
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}
    except (TypeError, ValueError):
        return None


def _json_safe(x: Any) -> Any:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (np.integer, np.floating)):
        return float(x) if isinstance(x, np.floating) else int(x)
    return x


def predict_propa_llm(
    df: pd.DataFrame,
    *,
    variant: str,
    checkpoint_path: Path,
    feature_cols: list[str] | None = None,
    model: str | None = None,
    sleep_s: float | None = None,
    text_col: str = "text",
    id_col: str = "telegram_id",
) -> pd.DataFrame:
    """Score every row in ``df``, resuming from ``checkpoint_path`` (JSONL append).

    ``variant`` is ``no_features`` (text only) or ``with_features`` (numeric JSON + text).
    The benchmark script evaluates on all merged rows and separately on the 100-row holdout.
    """
    model = model or default_model()
    sleep_s = LLM_REQUEST_SLEEP_S if sleep_s is None else sleep_s
    client = _client()
    done = _load_ckpt(checkpoint_path)
    use_ch = "channel" in df.columns
    rows: list[dict[str, Any]] = []
    sum_prompt = sum_completion = sum_total = 0
    n_api = 0

    for _, row in df.iterrows():
        try:
            tid_i = int(float(row[id_col]))
        except (TypeError, ValueError):
            continue
        ch = normalize_channel(row["channel"]) if use_ch else ""
        prev = _ckpt_get(done, ch, tid_i)
        if prev is not None and prev.get("pred") is not None:
            r = {"telegram_id": tid_i, "pred": float(prev["pred"]), "raw": prev.get("raw")}
            if use_ch:
                r["channel"] = ch
            if prev.get("usage"):
                r["usage"] = prev["usage"]
            rows.append(r)
            continue

        text = str(row.get(text_col, "") or "")
        parts: list[str] = []
        if variant == "with_features" and feature_cols:
            parts.append(WITH_FEATURES_PREAMBLE)
            parts.append(_format_features_raw(row, feature_cols))
        parts.append("Message:\n" + text)
        user_msg = "\n\n".join(parts)

        rec: dict[str, Any] = {
            "telegram_id": tid_i,
            "channel": ch if use_ch else "",
            "variant": variant,
            "model": model,
            "pred": None,
            "raw": None,
            "error": None,
            "usage": None,
        }
        try:
            comp = client.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                response_format=PropaScore,
                temperature=0,
                max_completion_tokens=LLM_MAX_TOKENS,
            )
            msg = comp.choices[0].message
            usage = _usage_dict(comp)
            if usage:
                rec["usage"] = usage
                sum_prompt += usage["prompt_tokens"]
                sum_completion += usage["completion_tokens"]
                sum_total += usage["total_tokens"]
                n_api += 1
            if msg.parsed is not None:
                rec["pred"] = float(msg.parsed.propa)
                rec["raw"] = msg.parsed.model_dump_json()
            else:
                rec["error"] = (msg.refusal or "parse_failed")[:500]
        except Exception as e:
            rec["error"] = str(e)

        _append_ckpt(checkpoint_path, rec)
        out: dict[str, Any] = {
            "telegram_id": tid_i,
            "pred": rec["pred"],
            "raw": rec.get("raw"),
        }
        if rec.get("usage"):
            out["usage"] = rec["usage"]
        if use_ch:
            out["channel"] = ch
        rows.append(out)
        time.sleep(sleep_s)

    if n_api:
        print(
            f"[LLM {variant}] tokens (this run, {n_api} API calls): "
            f"prompt={sum_prompt} completion={sum_completion} total={sum_total}",
            flush=True,
        )

    return pd.DataFrame(rows)
