#!/usr/bin/env python
"""
Evaluation script for Small Language Model.

Loads a trained model and evaluates:
- Perplexity on validation set
- Text generation quality
- Diversity metrics (unique word ratio, entropy)

Usage:
    python evaluate.py                           # Use defaults
    python evaluate.py --checkpoint best_model.pt
    python evaluate.py --prompts "Once upon a time"
"""

import argparse
import math
from collections import Counter
from contextlib import nullcontext

import nltk
import torch
import yaml

from data import check_data_ready, encode_text, get_batch, get_tokenizer
from model import GPT, GPTConfig


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_device(config: dict) -> tuple:
    """Setup device and dtype."""
    device_cfg = config.get("device", {})
    device_type_setting = device_cfg.get("type", "auto")
    dtype_setting = device_cfg.get("dtype", "auto")

    if device_type_setting == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_type_setting)

    device_type = "cuda" if "cuda" in str(device) else "cpu"

    if dtype_setting == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = "bfloat16"
        elif torch.cuda.is_available():
            dtype = "float16"
        else:
            dtype = "float32"
    else:
        dtype = dtype_setting

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]

    if device_type == "cpu":
        ctx = nullcontext()
    else:
        ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    return device, device_type, dtype, ptdtype, ctx


def calculate_perplexity(loss: torch.Tensor) -> torch.Tensor:
    """Calculate perplexity from cross-entropy loss."""
    return torch.exp(loss)


def estimate_loss(
    model: GPT,
    config: dict,
    device: torch.device,
    device_type: str,
    ctx,
) -> dict:
    """Estimate loss on train and validation sets."""
    out = {}
    model.eval()

    eval_iters = config["evaluation"]["eval_iters"]
    batch_size = config["training"]["batch_size"]
    block_size = config["model"]["block_size"]
    train_bin = config["data"]["train_bin"]
    val_bin = config["data"]["validation_bin"]

    with torch.inference_mode():
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(
                    split, batch_size, block_size, device, device_type, train_bin, val_bin
                )
                with ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()

    return out


def compute_diversity(generated_texts: list) -> dict:
    """
    Compute diversity metrics for generated texts.

    Metrics:
        - unique_ratio: Type-token ratio (unique words / total words)
        - num_unique_words: Number of unique words
        - total_words: Total number of words
        - entropy: Shannon entropy of word distribution

    Args:
        generated_texts: List of generated text strings

    Returns:
        Dictionary with diversity metrics
    """
    all_words = []
    for text in generated_texts:
        all_words.extend(nltk.word_tokenize(text.lower()))

    if not all_words:
        return {
            "unique_ratio": 0,
            "num_unique_words": 0,
            "total_words": 0,
            "entropy": 0,
        }

    unique_words = set(all_words)
    ttr = len(unique_words) / len(all_words)

    # Compute entropy
    word_counts = Counter(all_words)
    total_words = len(all_words)
    entropy = 0

    for word, count in word_counts.items():
        probability = count / total_words
        entropy -= probability * math.log2(probability)

    return {
        "unique_ratio": ttr,
        "num_unique_words": len(unique_words),
        "total_words": len(all_words),
        "entropy": entropy,
    }


def generate_text(
    model: GPT,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = None,
) -> str:
    """
    Generate text from a prompt.

    Args:
        model: Trained GPT model
        prompt: Input text prompt
        device: Target device
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter

    Returns:
        Generated text string
    """
    model.eval()
    enc = get_tokenizer()

    with torch.no_grad():
        context = torch.tensor(encode_text(prompt)).unsqueeze(0).to(device)
        generated = model.generate(context, max_new_tokens, temperature=temperature, top_k=top_k)
        output_text = enc.decode(generated.squeeze().tolist())

    return output_text


def evaluate_model(
    model: GPT,
    prompts: list,
    device: torch.device,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = None,
) -> dict:
    """
    Evaluate model generation quality.

    Args:
        model: Trained GPT model
        prompts: List of text prompts
        device: Target device
        max_new_tokens: Number of tokens to generate per prompt
        temperature: Sampling temperature
        top_k: Top-k sampling parameter

    Returns:
        Dictionary with generated texts and diversity metrics
    """
    model.eval()
    generated_texts = []

    print("\nGenerating text samples...")
    for prompt in prompts:
        text = generate_text(model, prompt, device, max_new_tokens, temperature, top_k)
        generated_texts.append(text)

    diversity_metrics = compute_diversity(generated_texts)

    return {
        "generated_texts": generated_texts,
        "diversity": diversity_metrics,
    }


def load_model(config: dict, checkpoint_path: str, device: torch.device) -> GPT:
    """Load model from checkpoint."""
    model_config = GPTConfig.from_dict(config["model"])
    model = GPT(model_config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    epoch = checkpoint.get("epoch", "unknown")
    best_val_loss = checkpoint.get("best_val_loss", "unknown")
    print(f"Loaded model from epoch {epoch} with validation loss: {best_val_loss}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate Small Language Model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (defaults to best_model.pt from config)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="Custom prompts for text generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens to generate (default from config)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default from config)",
    )
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="Skip perplexity evaluation (requires data files)",
    )

    args = parser.parse_args()

    # Download NLTK data if needed
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")

    # Setup device
    device, device_type, dtype, ptdtype, ctx = setup_device(config)
    print(f"Using device: {device}, dtype: {dtype}")

    # Load checkpoint
    checkpoint_path = args.checkpoint or config["checkpoints"]["best_model"]
    model = load_model(config, checkpoint_path, device)
    print(f"Model parameters: {model.get_num_params():,}")

    # Get generation settings
    gen_cfg = config.get("generation", {})
    max_new_tokens = args.max_tokens or gen_cfg.get("max_new_tokens", 100)
    temperature = args.temperature or gen_cfg.get("temperature", 0.7)
    top_k = gen_cfg.get("top_k")

    # Default prompts
    default_prompts = [
        "Once upon a time there was",
        "The little girl walked into",
        "A dog and a cat were",
        "In the middle of the night",
        "The sun was shining bright",
    ]
    prompts = args.prompts or default_prompts

    # Evaluate generation
    print("\n" + "=" * 60)
    print("TEXT GENERATION EVALUATION")
    print("=" * 60)

    eval_results = evaluate_model(
        model,
        prompts,
        device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    # Print generated samples
    print("\nGenerated Text Samples:")
    print("-" * 40)
    for i, (prompt, text) in enumerate(zip(prompts, eval_results["generated_texts"])):
        print(f"\nPrompt {i + 1}: {prompt}")
        print(f"Generated: {text}")

    # Print diversity metrics
    print("\n" + "=" * 60)
    print("DIVERSITY METRICS")
    print("=" * 60)
    diversity = eval_results["diversity"]
    print(f"Unique word ratio: {diversity['unique_ratio']:.4f}")
    print(f"Vocabulary size: {diversity['num_unique_words']} unique words")
    print(f"Total words: {diversity['total_words']}")
    print(f"Lexical entropy: {diversity['entropy']:.4f}")

    # Calculate perplexity
    if not args.skip_perplexity:
        data_cfg = config["data"]
        if check_data_ready(data_cfg["train_bin"], data_cfg["validation_bin"]):
            print("\n" + "=" * 60)
            print("PERPLEXITY EVALUATION")
            print("=" * 60)

            losses = estimate_loss(model, config, device, device_type, ctx)
            train_ppl = calculate_perplexity(losses["train"])
            val_ppl = calculate_perplexity(losses["val"])

            print(f"Training Loss: {losses['train']:.4f}")
            print(f"Validation Loss: {losses['val']:.4f}")
            print(f"Training Perplexity: {train_ppl:.4f}")
            print(f"Validation Perplexity: {val_ppl:.4f}")
        else:
            print("\nSkipping perplexity evaluation (data files not found)")

    print("\n" + "=" * 60)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
