#!/usr/bin/env python
"""
Wikipedia dataset preprocessing utility.

Steps:
- Load dataset from HuggingFace (streaming or local)
- Normalize text (strip HTML, normalize whitespace)
- Optional language filtering
- Remove short samples
- Deduplicate using SHA-256 hash (optional)
- Write output to JSONL format

Usage:
    python dataset_curation/wikipedia/preprocess_wikipedia.py \
        --dataset wikipedia --config 20220301.en \
        --output data/wikipedia.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from datasets import load_dataset
from tqdm.auto import tqdm

from dataset_curation.utils import (
    hash_text,
    is_english_text,
    normalize_text,
    text_too_short,
)


def preprocess_wikipedia(
    dataset_name: str,
    dataset_config: str,
    output_path: str,
    split: str = "train",
    streaming: bool = True,
    min_chars: int = 200,
    deduplicate: bool = True,
    max_records: Optional[int] = None,
    english_only: bool = True,
    min_alpha_ratio: float = 0.7,
) -> dict:
    """
    Preprocess Wikipedia dataset and save to JSONL.

    Returns:
        Stats dict with counts.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)

    seen_hashes = set()
    total = 0
    kept = 0
    filtered_short = 0
    filtered_lang = 0
    filtered_dupe = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for example in tqdm(ds, desc="Preprocessing Wikipedia"):
            total += 1

            text = example.get("text", "")
            text = normalize_text(text)

            if text_too_short(text, min_chars=min_chars):
                filtered_short += 1
                if max_records and total >= max_records:
                    break
                continue

            if english_only and not is_english_text(
                text, min_alpha_ratio=min_alpha_ratio
            ):
                filtered_lang += 1
                if max_records and total >= max_records:
                    break
                continue

            if deduplicate:
                h = hash_text(text)
                if h in seen_hashes:
                    filtered_dupe += 1
                    if max_records and total >= max_records:
                        break
                    continue
                seen_hashes.add(h)

            record = {"text": text}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

            if max_records and total >= max_records:
                break

    return {
        "total": total,
        "kept": kept,
        "filtered_short": filtered_short,
        "filtered_lang": filtered_lang,
        "filtered_dupe": filtered_dupe,
        "output_path": output_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess Wikipedia dataset")
    parser.add_argument("--dataset", default="wikipedia", help="HF dataset name")
    parser.add_argument(
        "--config", default="20220301.en", help="Dataset config (e.g., 20220301.en)"
    )
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument(
        "--output", default="data/wikipedia.jsonl", help="Output JSONL path"
    )
    parser.add_argument(
        "--no-streaming", action="store_true", help="Disable streaming mode"
    )
    parser.add_argument(
        "--min-chars", type=int, default=200, help="Minimum character length"
    )
    parser.add_argument("--no-dedup", action="store_true", help="Disable deduplication")
    parser.add_argument(
        "--max-records", type=int, default=None, help="Max records to process"
    )
    parser.add_argument(
        "--no-english-only", action="store_true", help="Disable English-only filter"
    )
    parser.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=0.7,
        help="Minimum ASCII alpha ratio for English filter",
    )

    args = parser.parse_args()

    stats = preprocess_wikipedia(
        dataset_name=args.dataset,
        dataset_config=args.config,
        output_path=args.output,
        split=args.split,
        streaming=not args.no_streaming,
        min_chars=args.min_chars,
        deduplicate=not args.no_dedup,
        max_records=args.max_records,
        english_only=not args.no_english_only,
        min_alpha_ratio=args.min_alpha_ratio,
    )

    print("\nPreprocessing complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
