#!/usr/bin/env python
"""
Training script for Small Language Model.

Reads configuration from config.yaml and trains the GPT model.
Supports checkpoint resumption, mixed precision, and wandb logging.

Usage:
    python train.py                      # Use default config.yaml
    python train.py --config custom.yaml # Use custom config file
    python train.py --no-wandb           # Disable wandb logging
"""

import argparse
import os
import time
from contextlib import nullcontext

import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm

from data import check_data_ready, get_batch, prepare_dataset
from model import GPT, GPTConfig, load_checkpoint, save_checkpoint


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_device(config: dict) -> tuple:
    """
    Setup device and dtype for training.

    Returns:
        Tuple of (device, device_type, dtype, ptdtype, ctx)
    """
    device_cfg = config.get("device", {})
    device_type_setting = device_cfg.get("type", "auto")
    dtype_setting = device_cfg.get("dtype", "auto")

    # Determine device
    if device_type_setting == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_type_setting)

    device_type = "cuda" if "cuda" in str(device) else "cpu"

    # Determine dtype
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

    # Autocast context
    if device_type == "cpu":
        ctx = nullcontext()
    else:
        ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    return device, device_type, dtype, ptdtype, ctx


def estimate_loss(
    model: GPT,
    config: dict,
    device: torch.device,
    device_type: str,
    ctx,
) -> dict:
    """
    Estimate loss on train and validation sets.

    Returns:
        Dictionary with 'train' and 'val' losses
    """
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

    model.train()
    return out


def setup_optimizer_and_scheduler(
    model: GPT, config: dict
) -> tuple:
    """
    Setup optimizer and learning rate scheduler.

    Returns:
        Tuple of (optimizer, scheduler, scaler)
    """
    train_cfg = config["training"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=train_cfg["weight_decay"],
        eps=train_cfg["eps"],
    )

    # Learning rate schedulers
    scheduler_warmup = LinearLR(optimizer, total_iters=train_cfg["warmup_steps"])
    scheduler_decay = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["max_iters"] - train_cfg["warmup_steps"],
        eta_min=train_cfg["min_lr"],
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_decay],
        milestones=[train_cfg["warmup_steps"]],
    )

    return optimizer, scheduler


def init_wandb(config: dict, resume: bool = False) -> None:
    """Initialize wandb for experiment tracking."""
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    try:
        import wandb

        run_name = f"{wandb_cfg.get('run_name_prefix', 'slm')}-{time.strftime('%Y%m%d-%H%M%S')}"

        wandb.init(
            project=wandb_cfg.get("project", "slm-training"),
            name=run_name,
            config={
                "learning_rate": config["training"]["learning_rate"],
                "max_iters": config["training"]["max_iters"],
                "warmup_steps": config["training"]["warmup_steps"],
                "min_lr": config["training"]["min_lr"],
                "batch_size": config["training"]["batch_size"],
                "block_size": config["model"]["block_size"],
                "gradient_accumulation_steps": config["training"][
                    "gradient_accumulation_steps"
                ],
                "n_layer": config["model"]["n_layer"],
                "n_head": config["model"]["n_head"],
                "n_embd": config["model"]["n_embd"],
                "dropout": config["model"]["dropout"],
            },
            resume="allow" if resume else None,
        )
        return wandb
    except ImportError:
        print("Warning: wandb not installed. Logging disabled.")
        return None
    except Exception as e:
        print(f"Warning: Could not initialize wandb: {e}")
        return None


def train(config_path: str, use_wandb: bool = True) -> None:
    """
    Main training function.

    Args:
        config_path: Path to YAML configuration file
        use_wandb: Whether to use wandb for logging
    """
    # Load configuration
    config = load_config(config_path)
    print(f"Loaded configuration from {config_path}")

    # Setup device
    device, device_type, dtype, ptdtype, ctx = setup_device(config)
    print(f"Using device: {device}, dtype: {dtype}")

    # Set random seed
    torch.manual_seed(config.get("seed", 42))
    if device_type == "cuda":
        torch.cuda.manual_seed(config.get("seed", 42))

    # Prepare data if needed
    data_cfg = config["data"]
    if not check_data_ready(data_cfg["train_bin"], data_cfg["validation_bin"]):
        print("Preparing dataset...")
        prepare_dataset(
            dataset_name=data_cfg["dataset_name"],
            train_bin_path=data_cfg["train_bin"],
            validation_bin_path=data_cfg["validation_bin"],
        )

    # Create model
    model_config = GPTConfig.from_dict(config["model"])
    model = GPT(model_config)
    model = model.to(device)
    print(f"Model parameters: {model.get_num_params():,}")

    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, config)

    # Setup gradient scaler for mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))

    # Training configuration
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]
    ckpt_cfg = config["checkpoints"]

    max_iters = train_cfg["max_iters"]
    batch_size = train_cfg["batch_size"]
    block_size = config["model"]["block_size"]
    gradient_accumulation_steps = train_cfg["gradient_accumulation_steps"]
    eval_interval = eval_cfg.get("eval_interval", eval_cfg["eval_iters"])

    # Initialize tracking variables
    start_epoch = 0
    best_val_loss = float("inf")
    train_loss_list = []
    validation_loss_list = []

    # Resume from checkpoint if available
    latest_model_path = ckpt_cfg["latest_model"]
    best_model_path = ckpt_cfg["best_model"]

    if os.path.exists(latest_model_path):
        print(f"Resuming from checkpoint: {latest_model_path}")
        checkpoint = torch.load(latest_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        train_loss_list = checkpoint.get("train_losses", [])
        validation_loss_list = checkpoint.get("val_losses", [])
        print(f"Resumed from epoch {start_epoch} with best val loss: {best_val_loss:.4f}")

    # Initialize wandb
    wandb = None
    if use_wandb and config.get("wandb", {}).get("enabled", False):
        wandb = init_wandb(config, resume=start_epoch > 0)

    # Track training metrics
    epoch_loss = 0.0
    epoch_steps = 0

    print(f"\nStarting training from epoch {start_epoch} to {max_iters}")
    print(f"Batch size: {batch_size}, Block size: {block_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Evaluation interval: {eval_interval}")
    print("-" * 50)

    # Training loop
    for epoch in tqdm(range(start_epoch, max_iters), initial=start_epoch, total=max_iters):
        # Get batch
        X, y = get_batch(
            "train",
            batch_size,
            block_size,
            device,
            device_type,
            data_cfg["train_bin"],
            data_cfg["validation_bin"],
        )

        # Forward pass
        with ctx:
            logits, loss = model(X, y)
            loss_value = loss.item()
            loss = loss / gradient_accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Track loss
        epoch_loss += loss_value
        epoch_steps += 1

        # Gradient accumulation step
        if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=train_cfg["max_grad_norm"]
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        # Log to wandb
        if wandb is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "step_loss": loss_value,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch_avg_loss": epoch_loss / epoch_steps if epoch_steps > 0 else 0,
                }
            )

        # Evaluation and checkpointing
        if epoch % eval_interval == 0 and epoch != 0:
            losses = estimate_loss(model, config, device, device_type, ctx)
            train_loss = losses["train"].item()
            val_loss = losses["val"].item()

            print("+" * 50)
            print(f"Epoch {epoch}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

            train_loss_list.append(losses["train"])
            validation_loss_list.append(losses["val"])

            # Log evaluation metrics
            if wandb is not None:
                wandb.log(
                    {
                        "eval_epoch": epoch,
                        "eval_train_loss": train_loss,
                        "eval_val_loss": val_loss,
                    }
                )

            # Save latest checkpoint
            save_checkpoint(
                latest_model_path,
                epoch,
                model,
                optimizer,
                scheduler,
                best_val_loss,
                train_loss_list,
                validation_loss_list,
            )

            # Save best checkpoint if improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    best_model_path,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_val_loss,
                    train_loss_list,
                    validation_loss_list,
                )

                if wandb is not None:
                    wandb.run.summary["best_val_loss"] = best_val_loss
                    wandb.run.summary["best_epoch"] = epoch

            # Reset epoch tracking
            epoch_loss = 0.0
            epoch_steps = 0

    # Final logging
    if wandb is not None:
        wandb.log(
            {
                "final_train_loss": train_loss_list[-1].item() if train_loss_list else None,
                "final_val_loss": validation_loss_list[-1].item()
                if validation_loss_list
                else None,
                "total_epochs_trained": max_iters,
            }
        )
        wandb.run.summary.update(
            {
                "total_epochs": max_iters,
                "final_learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        wandb.finish()

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {latest_model_path}, {best_model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Small Language Model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )

    args = parser.parse_args()

    train(config_path=args.config, use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()
