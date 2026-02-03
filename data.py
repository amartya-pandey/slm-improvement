"""Deprecated: dataset utilities moved to dataset_curation."""

from dataset_curation.data import JSONLInfiniteIterator, MixtureDataset, get_batch
from dataset_curation.tokenizer import decode_ids, encode_text, load_tokenizer, train_tokenizer

__all__ = [
    "JSONLInfiniteIterator",
    "MixtureDataset",
    "get_batch",
    "load_tokenizer",
    "train_tokenizer",
    "encode_text",
    "decode_ids",
]


if __name__ == "__main__":
    raise SystemExit(
        "This module is deprecated. Use dataset_curation/train_tokenizer.py "
        "and dataset_curation/data.py instead."
    )
