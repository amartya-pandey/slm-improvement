"""Tokenizer training and loading utilities (SentencePiece BPE)."""

from __future__ import annotations

import json
import os
from typing import Iterable, List

from tokenizers import SentencePieceBPETokenizer, Tokenizer

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def _jsonl_text_iterator(jsonl_path: str) -> Iterable[str]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text")
            if text:
                yield text


def _mixed_iterator(wikipedia_jsonl: str, openwebtext_jsonl: str) -> Iterable[str]:
    wiki_iter = _jsonl_text_iterator(wikipedia_jsonl)
    owt_iter = _jsonl_text_iterator(openwebtext_jsonl)
    while True:
        yielded = False
        try:
            yield next(wiki_iter)
            yielded = True
        except StopIteration:
            wiki_iter = _jsonl_text_iterator(wikipedia_jsonl)
        try:
            yield next(owt_iter)
            yielded = True
        except StopIteration:
            owt_iter = _jsonl_text_iterator(openwebtext_jsonl)
        if not yielded:
            break


def train_tokenizer(
    wikipedia_jsonl: str,
    openwebtext_jsonl: str,
    output_dir: str = "tokenizer",
    vocab_size: int = 50257,
    min_frequency: int = 2,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = SentencePieceBPETokenizer()
    tokenizer.train_from_iterator(
        _mixed_iterator(wikipedia_jsonl, openwebtext_jsonl),
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )

    tokenizer_json = os.path.join(output_dir, "tokenizer.json")
    tokenizer_model_prefix = os.path.join(output_dir, "tokenizer")

    tokenizer.save(tokenizer_json)
    tokenizer.save_model(output_dir, "tokenizer")

    return {
        "tokenizer_json": tokenizer_json,
        "tokenizer_model": f"{tokenizer_model_prefix}.model",
        "tokenizer_vocab": f"{tokenizer_model_prefix}.vocab",
        "vocab_size": tokenizer.get_vocab_size(),
    }


def load_tokenizer(tokenizer_json: str) -> Tokenizer:
    if not os.path.exists(tokenizer_json):
        raise FileNotFoundError(f"Tokenizer JSON not found: {tokenizer_json}")
    return Tokenizer.from_file(tokenizer_json)


def encode_text(tokenizer: Tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text).ids


def decode_ids(tokenizer: Tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids)
