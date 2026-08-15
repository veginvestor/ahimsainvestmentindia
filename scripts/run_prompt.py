#!/usr/bin/env python3
"""
Run a stored prompt against one OpenRouter model.

Two modes:
  review  - model produces a Markdown report. Data files are never touched.
  edit    - model returns a complete replacement CSV. Validated before writing.

Every failure is loud. This script exits non-zero rather than writing
questionable output or committing an empty diff.
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_ROW_DELTA_PCT = 10.0  # edit mode: reject if row count moves more than this


def die(msg: str) -> None:
    print(f"::error::{msg}", file=sys.stderr)
    sys.exit(1)


def call_model(model: str, system: str, user: str, api_key: str) -> str:
    try:
        resp = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": "ctms-docs-agent",
            },
            json={
                "model": model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=300,
        )
    except requests.RequestException as e:
        die(f"Request to OpenRouter failed for {model}: {e}")

    if resp.status_code == 429:
        die(f"Rate limited by OpenRouter on {model}. Free tier is 20 req/min, "
            f"50 req/day without purchased credits.")
    if resp.status_code != 200:
        die(f"OpenRouter returned HTTP {resp.status_code} for {model}: {resp.text[:400]}")

    body = resp.json()
    if "error" in body:
        die(f"OpenRouter error for {model}: {json.dumps(body['error'])[:400]}")

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        die(f"Unexpected response shape from {model}: {json.dumps(body)[:400]}")

    if not content or not content.strip():
        die(f"Model {model} returned an empty response. Refusing to continue.")
    return content.strip()


def strip_fences(text: str) -> str:
    """Remove a leading/trailing markdown code fence if the model added one."""
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def validate_csv(original: pd.DataFrame, new_text: str, path: Path) -> pd.DataFrame:
    try:
        new = pd.read_csv(io.StringIO(new_text))
    except Exception as e:
        die(f"Model output for {path} is not parseable CSV: {e}")

    if list(new.columns) != list(original.columns):
        die(
            f"Column mismatch in {path}.\n"
            f"  expected: {list(original.columns)}\n"
            f"  received: {list(new.columns)}"
        )

    if len(original) == 0:
        die(f"{path} is empty. Nothing to edit.")

    delta_pct = abs(len(new) - len(original)) / len(original) * 100
    if delta_pct > MAX_ROW_DELTA_PCT:
        die(
            f"Row count changed by {delta_pct:.1f}% in {path} "
            f"({len(original)} -> {len(new)}). Limit is {MAX_ROW_DELTA_PCT}%. "
            f"This usually means the model truncated the file."
        )

    return new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="Path to prompt file in prompts/")
    ap.add_argument("--model", required=True, help="OpenRouter model slug")
    ap.add_argument("--data", required=True, help="CSV file to operate on")
    ap.add_argument("--mode", choices=["review", "edit"], required=True)
    ap.add_argument("--context", default="", help="Extra free-text instruction from the run form")
    ap.add_argument("--context-file", action="append", default=[],
                    help="Extra file to include as reference (repeatable). "
                         "Use for rules or framework documents.")
    ap.add_argument("--research", default="", help="Path to research notes to inject")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        die("OPENROUTER_API_KEY is not set. Add it under Settings > Secrets and variables > Actions.")

    prompt_path = Path(args.prompt)
    data_path = Path(args.data)
    if not prompt_path.exists():
        die(f"Prompt file not found: {prompt_path}")
    if not data_path.exists():
        die(f"Data file not found: {data_path}")

    instruction = prompt_path.read_text().strip()
    df = pd.read_csv(data_path)
    raw_csv = data_path.read_text()

    # Rough guard against blowing the context window on a free model.
    if len(raw_csv) > 400_000:
        die(
            f"{data_path} is {len(raw_csv):,} characters. That will not fit in a "
            f"free-tier context window. Split the file or switch to review mode "
            f"on a sample."
        )

    parts = [instruction]
    if args.context:
        parts.append(f"## Additional instruction for this run\n{args.context}")

    context_chars = 0
    for cf in args.context_file:
        p = Path(cf)
        if not p.exists():
            die(f"Context file not found: {cf}")
        body = p.read_text()
        context_chars += len(body)
        parts.append(f"## Reference document: {p.name}\n\n{body}")

    if args.research and Path(args.research).exists():
        parts.append(f"## Web research gathered for this run\n{Path(args.research).read_text()}")
    parts.append(f"## Data file: {data_path.name}\n```csv\n{raw_csv}\n```")
    user_msg = "\n\n".join(parts)

    total = len(user_msg)
    print(f"Prompt size: {total:,} chars "
          f"(data {len(raw_csv):,} + reference {context_chars:,})")
    if total > 400_000:
        die(
            f"Total prompt is {total:,} characters. That will not fit in a "
            f"free-tier context window. Reduce the data slice or the reference "
            f"documents."
        )

    if args.mode == "review":
        system = (
            "You are a careful data analyst. Produce a Markdown report. "
            "Cite specific row numbers and column names for every finding. "
            "If you are uncertain about something, say so explicitly rather than guessing. "
            "Do not modify or reproduce the full dataset."
        )
    else:
        system = (
            "You are a careful data editor. Return ONLY the complete, corrected CSV file. "
            "No prose, no explanation, no markdown code fences. "
            "Preserve the header row exactly as given. Preserve every row you were not "
            "asked to change, byte for byte. Never drop rows."
        )

    print(f"Calling {args.model} in {args.mode} mode on {data_path}...")
    output = call_model(args.model, system, user_msg, api_key)

    model_slug = args.model.replace("/", "-").replace(":", "-")
    prompt_slug = prompt_path.stem

    if args.mode == "review":
        out = Path("reports") / f"{prompt_slug}--{model_slug}.md"
        out.parent.mkdir(exist_ok=True)
        header = (
            f"# {prompt_slug}\n\n"
            f"**Model:** `{args.model}`  \n"
            f"**Data:** `{data_path}`  \n"
            f"**Rows analysed:** {len(df):,}\n\n---\n\n"
        )
        out.write_text(header + output + "\n")
        print(f"Wrote {out}")
    else:
        cleaned = strip_fences(output)
        new_df = validate_csv(df, cleaned, data_path)
        if new_df.equals(df):
            die(
                f"Model {args.model} returned the file unchanged. "
                f"Either the prompt found nothing to do, or it ignored the instruction. "
                f"Not opening an empty pull request."
            )
        new_df.to_csv(data_path, index=False)
        changed = (len(new_df) != len(df)) or True
        print(f"Wrote {data_path} ({len(df):,} -> {len(new_df):,} rows)")


if __name__ == "__main__":
    main()
