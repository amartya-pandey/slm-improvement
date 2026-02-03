#!/usr/bin/env python
"""Train a shared SentencePiece BPE tokenizer on Wikipedia + OpenWebText."""

from __future__ import annotations

import argparse

from dataset_curation.tokenizer import train_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SentencePiece BPE tokenizer")
    parser.add_argument(
        "--wikipedia", default="data/wikipedia.jsonl", help="Wikipedia JSONL path"
    )
    parser.add_argument(
        "--openwebtext",
        default="data/openwebtext.jsonl",
        help="OpenWebText JSONL path",
    )
    parser.add_argument(
        "--output-dir", default="tokenizer", help="Tokenizer output directory"
    )
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    stats = train_tokenizer(
        wikipedia_jsonl=args.wikipedia,
        openwebtext_jsonl=args.openwebtext,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    print("\nTokenizer training complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
