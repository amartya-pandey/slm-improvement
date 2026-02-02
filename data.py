"""
Data preparation and batching utilities for SLM training.

This module handles:
- Dataset loading from HuggingFace
- Tokenization using tiktoken
- Memory-mapped binary file creation
- Batch generation for training and evaluation
"""

import os
import numpy as np
import torch
import tiktoken
from datasets import load_dataset
from tqdm.auto import tqdm
from typing import Tuple, Optional


# Global tokenizer instance
_tokenizer = None


def get_tokenizer():
    """Get or create the GPT-2 tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("gpt2")
    return _tokenizer


def process_example(example: dict) -> dict:
    """
    Process a single example by tokenizing the text.

    Args:
        example: Dictionary containing 'text' field

    Returns:
        Dictionary with 'ids' (token IDs) and 'len' (length)
    """
    enc = get_tokenizer()
    ids = enc.encode_ordinary(example["text"])  # encode_ordinary ignores special tokens
    return {"ids": ids, "len": len(ids)}


def prepare_dataset(
    dataset_name: str = "roneneldan/TinyStories",
    train_bin_path: str = "train.bin",
    validation_bin_path: str = "validation.bin",
    num_proc: int = 8,
    force_rebuild: bool = False,
) -> None:
    """
    Prepare tokenized dataset and save as memory-mapped binary files.

    Args:
        dataset_name: HuggingFace dataset name
        train_bin_path: Output path for training data
        validation_bin_path: Output path for validation data
        num_proc: Number of processes for tokenization
        force_rebuild: If True, rebuild even if files exist
    """
    # Check if both files already exist
    if (
        not force_rebuild
        and os.path.exists(train_bin_path)
        and os.path.exists(validation_bin_path)
    ):
        print(f"Data files already exist: {train_bin_path}, {validation_bin_path}")
        print("Use force_rebuild=True to rebuild.")
        return

    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)

    print("Tokenizing dataset...")
    tokenized = ds.map(
        process_example,
        remove_columns=["text"],
        desc="Tokenizing the splits",
        num_proc=num_proc,
    )

    # Write tokenized data to binary files
    for split, dset in tokenized.items():
        # Determine output filename
        if split == "train":
            filename = train_bin_path
        elif split == "validation":
            filename = validation_bin_path
        else:
            filename = f"{split}.bin"

        print(f"Writing {split} split to {filename}...")

        # Calculate total length
        arr_len = np.sum(dset["len"], dtype=np.uint64)

        # Create memory-mapped file
        # uint16 is sufficient since vocab size (50257) < 2^16
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

        # Write in batches for efficiency
        total_batches = 1024
        idx = 0

        for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()
        print(f"Saved {arr_len:,} tokens to {filename}")


def get_batch(
    split: str,
    batch_size: int,
    block_size: int,
    device: torch.device,
    device_type: str,
    train_bin_path: str = "train.bin",
    validation_bin_path: str = "validation.bin",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a batch of data for training or evaluation.

    We recreate np.memmap every batch to avoid memory leaks.

    Args:
        split: 'train' or 'val'/'validation'
        batch_size: Number of sequences per batch
        block_size: Context window size
        device: Target device for tensors
        device_type: 'cuda' or 'cpu'
        train_bin_path: Path to training data file
        validation_bin_path: Path to validation data file

    Returns:
        Tuple of (input_ids, target_ids) tensors
    """
    # Select data file
    if split == "train":
        data_path = train_bin_path
    else:
        data_path = validation_bin_path

    # Memory-map the data file
    data = np.memmap(data_path, dtype=np.uint16, mode="r")

    # Random starting indices
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Extract sequences
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )

    # Move to device
    if device_type == "cuda":
        # Pin memory for async transfer
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y


def get_data_stats(
    train_bin_path: str = "train.bin",
    validation_bin_path: str = "validation.bin",
) -> dict:
    """
    Get statistics about the prepared data.

    Returns:
        Dictionary with token counts for each split
    """
    stats = {}

    if os.path.exists(train_bin_path):
        train_data = np.memmap(train_bin_path, dtype=np.uint16, mode="r")
        stats["train_tokens"] = len(train_data)
        del train_data

    if os.path.exists(validation_bin_path):
        val_data = np.memmap(validation_bin_path, dtype=np.uint16, mode="r")
        stats["validation_tokens"] = len(val_data)
        del val_data

    return stats


def check_data_ready(
    train_bin_path: str = "train.bin",
    validation_bin_path: str = "validation.bin",
) -> bool:
    """Check if both data files exist and are ready for training."""
    if not os.path.exists(train_bin_path):
        print(f"Missing training data: {train_bin_path}")
        return False
    if not os.path.exists(validation_bin_path):
        print(f"Missing validation data: {validation_bin_path}")
        return False
    return True


def decode_tokens(token_ids: list) -> str:
    """Decode token IDs back to text."""
    enc = get_tokenizer()
    return enc.decode(token_ids)


def encode_text(text: str) -> list:
    """Encode text to token IDs."""
    enc = get_tokenizer()
    return enc.encode_ordinary(text)


if __name__ == "__main__":
    # When run directly, prepare the dataset
    import argparse

    parser = argparse.ArgumentParser(description="Prepare dataset for SLM training")
    parser.add_argument(
        "--dataset", default="roneneldan/TinyStories", help="HuggingFace dataset name"
    )
    parser.add_argument("--train-bin", default="train.bin", help="Output training file")
    parser.add_argument(
        "--val-bin", default="validation.bin", help="Output validation file"
    )
    parser.add_argument(
        "--num-proc", type=int, default=8, help="Number of processes for tokenization"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force rebuild even if files exist"
    )

    args = parser.parse_args()

    prepare_dataset(
        dataset_name=args.dataset,
        train_bin_path=args.train_bin,
        validation_bin_path=args.val_bin,
        num_proc=args.num_proc,
        force_rebuild=args.force,
    )

    # Print stats
    stats = get_data_stats(args.train_bin, args.val_bin)
    print("\nData Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
