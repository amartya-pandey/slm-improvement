#!/usr/bin/env python
"""
ONNX Export script for Small Language Model.

Exports the trained GPT model to ONNX format for inference deployment.

Usage:
    python export.py                           # Use defaults
    python export.py --checkpoint best_model.pt
    python export.py --output model.onnx
    python export.py --full-sequence           # Export full sequence logits
"""

import argparse

import torch
import yaml

from model import GPT, GPTConfig, ONNXWrapper


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(config: dict, checkpoint_path: str) -> GPT:
    """Load model from checkpoint."""
    model_config = GPTConfig.from_dict(config["model"])
    model = GPT(model_config)

    # Load to CPU for export
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    epoch = checkpoint.get("epoch", "unknown")
    best_val_loss = checkpoint.get("best_val_loss", "unknown")
    print(f"Loaded model from epoch {epoch} with validation loss: {best_val_loss}")

    return model


class FullSequenceWrapper(torch.nn.Module):
    """
    Wrapper for ONNX export that returns full sequence logits.
    Output shape: (batch_size, sequence_length, vocab_size)
    """

    def __init__(self, model: GPT):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Get full logits for all positions
        x = self.model.transformer.wte(input_ids)
        pos = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
        x = x + self.model.transformer.wpe(pos)
        x = self.model.transformer.drop(x)

        for block in self.model.transformer.h:
            x = block(x)

        x = self.model.transformer.ln_f(x)
        logits = self.model.lm_head(x)
        return logits


def export_to_onnx(
    model: GPT,
    save_path: str = "slm_model.onnx",
    opset_version: int = 14,
    use_last_token: bool = True,
) -> None:
    """
    Export the GPT model to ONNX format.

    Args:
        model: Trained GPT model
        save_path: Output ONNX file path
        opset_version: ONNX opset version (14+ recommended)
        use_last_token: If True, output logits for last token only (B, V).
                       If False, output full sequence logits (B, T, V).
    """
    model.eval()

    # Disable flash attention for ONNX compatibility
    try:
        for block in model.transformer.h:
            if hasattr(block, "attn") and hasattr(block.attn, "flash"):
                block.attn.flash = False
    except Exception:
        pass

    # Select wrapper based on output mode
    if use_last_token:
        export_model = ONNXWrapper(model)
        dummy_input = torch.zeros(1, 1, dtype=torch.long, device="cpu")
        output_names = ["logits"]
        dynamic_axes = {
            "input": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size"},
        }
        print("Exporting with last-token logits output (B, V)")
    else:
        export_model = FullSequenceWrapper(model)
        dummy_input = torch.zeros(1, 4, dtype=torch.long, device="cpu")
        output_names = ["logits"]
        dynamic_axes = {
            "input": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size", 1: "sequence"},
        }
        print("Exporting with full-sequence logits output (B, T, V)")

    export_model.to("cpu")
    dummy_input = dummy_input.to("cpu")

    # Export to ONNX
    print(f"Exporting model to {save_path}...")
    torch.onnx.export(
        export_model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False,
    )

    print(f"Model exported successfully to {save_path}")

    # Print model info
    import os

    file_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Model size: {file_size:.2f} MB")


def verify_onnx_model(onnx_path: str) -> bool:
    """
    Verify the exported ONNX model.

    Args:
        onnx_path: Path to ONNX model

    Returns:
        True if verification passes
    """
    try:
        import onnx

        print(f"\nVerifying ONNX model: {onnx_path}")
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("ONNX model verification passed!")

        # Print model info
        print(f"IR version: {model.ir_version}")
        print(f"Opset version: {model.opset_import[0].version}")
        print(f"Producer: {model.producer_name}")

        return True
    except ImportError:
        print("Warning: onnx package not installed. Skipping verification.")
        print("Install with: pip install onnx")
        return True
    except Exception as e:
        print(f"ONNX verification failed: {e}")
        return False


def test_onnx_inference(onnx_path: str) -> None:
    """
    Test ONNX model inference with onnxruntime.

    Args:
        onnx_path: Path to ONNX model
    """
    try:
        import numpy as np
        import onnxruntime as ort

        print(f"\nTesting ONNX inference: {onnx_path}")

        # Create inference session
        session = ort.InferenceSession(onnx_path)

        # Get input/output info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]

        print(f"Input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
        print(f"Output: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")

        # Test inference
        test_input = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        outputs = session.run(None, {"input": test_input})

        print(f"Test input shape: {test_input.shape}")
        print(f"Output shape: {outputs[0].shape}")
        print("ONNX inference test passed!")

    except ImportError:
        print("Warning: onnxruntime not installed. Skipping inference test.")
        print("Install with: pip install onnxruntime")
    except Exception as e:
        print(f"ONNX inference test failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Export SLM to ONNX format")
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
        "--output",
        type=str,
        default=None,
        help="Output ONNX file path (defaults to config setting)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="ONNX opset version (default from config)",
    )
    parser.add_argument(
        "--full-sequence",
        action="store_true",
        help="Export full sequence logits (B, T, V) instead of last token (B, V)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip ONNX model verification",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip ONNX inference test",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")

    # Get export settings
    export_cfg = config.get("export", {})
    checkpoint_path = args.checkpoint or config["checkpoints"]["best_model"]
    output_path = args.output or export_cfg.get("onnx_path", "slm_model.onnx")
    opset_version = args.opset or export_cfg.get("opset_version", 14)

    # Load model
    model = load_model(config, checkpoint_path)
    print(f"Model parameters: {model.get_num_params():,}")

    # Export to ONNX
    try:
        export_to_onnx(
            model,
            save_path=output_path,
            opset_version=opset_version,
            use_last_token=not args.full_sequence,
        )

        # Verify exported model
        if not args.skip_verify:
            verify_onnx_model(output_path)

        # Test inference
        if not args.skip_test:
            test_onnx_inference(output_path)

        print("\n" + "=" * 60)
        print("Export complete!")
        print(f"ONNX model saved to: {output_path}")

    except Exception as e:
        print(f"Error during export: {e}")
        raise


if __name__ == "__main__":
    main()
