"""
Streaming dataset utilities for mixture training.

- JSONLInfiniteIterator: streams JSONL indefinitely, returning the "text" field.
- MixtureDataset: probabilistic sampling from Wikipedia and OpenWebText.
- get_batch: builds CLM batches from sampled texts.
"""

from __future__ import annotations

import json
import os
import random
from typing import Iterable, List, Tuple

import torch


class JSONLInfiniteIterator:
    """Infinite JSONL iterator that returns the "text" field only."""

    def __init__(self, jsonl_path: str):
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
        self.jsonl_path = jsonl_path
        self._fh = None

    def _open(self) -> None:
        if self._fh is None or self._fh.closed:
            self._fh = open(self.jsonl_path, "r", encoding="utf-8")

    def __iter__(self) -> "JSONLInfiniteIterator":
        return self

    def __next__(self) -> str:
        self._open()
        while True:
            line = self._fh.readline()
            if line == "":
                self._fh.seek(0)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text")
            if text:
                return text


class MixtureDataset:
    """Probabilistic mixture of Wikipedia and OpenWebText JSONL streams."""

    def __init__(
        self,
        wikipedia_jsonl: str,
        openwebtext_jsonl: str,
        wikiratio: float = 0.6,
    ) -> None:
        self.wikipedia_iter = JSONLInfiniteIterator(wikipedia_jsonl)
        self.openwebtext_iter = JSONLInfiniteIterator(openwebtext_jsonl)
        self.wikiratio = float(wikiratio)

    def set_wikiratio(self, ratio: float) -> None:
        self.wikiratio = float(ratio)

    def sampletext(self) -> str:
        text, _ = self.sample_with_source()
        return text

    def sample_with_source(self) -> Tuple[str, str]:
        if random.random() < self.wikiratio:
            return next(self.wikipedia_iter), "wikipedia"
        return next(self.openwebtext_iter), "openwebtext"


def _sample_token_block(
    mixture: MixtureDataset,
    tokenizer,
    block_size: int,
    max_tries: int = 32,
) -> Tuple[List[int], str]:
    eos_id = tokenizer.token_to_id("<eos>")
    for _ in range(max_tries):
        text, source = mixture.sample_with_source()
        ids = tokenizer.encode(text).ids
        if len(ids) >= block_size + 1:
            start = random.randint(0, len(ids) - block_size - 1)
            return ids[start : start + block_size + 1], source
    # Fallback: pad a short sample to the required length
    text, source = mixture.sample_with_source()
    ids = tokenizer.encode(text).ids
    if eos_id is None:
        eos_id = 0
    if len(ids) < block_size + 1:
        ids = ids + [eos_id] * (block_size + 1 - len(ids))
    return ids[: block_size + 1], source


def get_batch(
    mixture: MixtureDataset,
    tokenizer,
    batch_size: int,
    block_size: int,
    device: torch.device,
    device_type: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Sample a batch for causal language modeling."""
    blocks: List[List[int]] = []
    sources: List[str] = []

    for _ in range(batch_size):
        block, source = _sample_token_block(mixture, tokenizer, block_size)
        blocks.append(block)
        sources.append(source)

    data = torch.tensor(blocks, dtype=torch.long)
    x = data[:, :-1]
    y = data[:, 1:]

    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y, sources
