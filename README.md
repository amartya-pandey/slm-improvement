# Small Language Model (SLM) Training Pipeline

Modular training pipeline for a GPT-style Small Language Model using a streaming mixture of Wikipedia and OpenWebText.

## Project Structure

```
slm-improvement/
├── config.yaml          # Shared configuration (hyperparameters, paths)
├── model.py             # GPT model architecture and utilities
├── data.py              # Dataset preparation and batching
├── train.py             # Training script
├── evaluate.py          # Evaluation script (perplexity, generation)
├── export.py            # ONNX export script
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess Datasets

```bash
python dataset_curation/wikipedia/preprocess_wikipedia.py --output data/wikipedia.jsonl
python dataset_curation/openwebtext/preprocess_openwebtext.py --output data/openwebtext.jsonl
```

### 3. Train a Shared Tokenizer

```bash
python dataset_curation/train_tokenizer.py --output-dir tokenizer
```

### 4. Train Model

```bash
# Basic training with wandb logging
python train.py

# Without wandb
python train.py --no-wandb

# Custom config
python train.py --config custom_config.yaml
```

### 4. Evaluate Model

```bash
# Evaluate best checkpoint
python evaluate.py

# Custom prompts
python evaluate.py --prompts "Once upon a time" "The little dog"

# Skip perplexity (no data files needed)
python evaluate.py --skip-perplexity
```

### 5. Export to ONNX

```bash
# Export best model
python export.py

# Custom output path
python export.py --output my_model.onnx

# Full sequence output
python export.py --full-sequence
```

## Configuration

All settings are in `config.yaml`:

```yaml
# Model architecture
model:
  vocab_size: 50257
  block_size: 128
  n_layer: 6
  n_head: 6
  n_embd: 384
  dropout: 0.1

# Training hyperparameters
training:
  learning_rate: 1.0e-4
  max_iters: 80000
  warmup_steps: 1000
  min_lr: 1.0e-5
  batch_size: 32
  gradient_accumulation_steps: 32

# Data mixture
data:
  wikipedia_jsonl: "data/wikipedia.jsonl"
  openwebtext_jsonl: "data/openwebtext.jsonl"
  tokenizer_json: "tokenizer/tokenizer.json"
  wikiratio: 0.6
  curriculum:
    phase1_ratio: 0.6
    phase2_ratio: 0.4
    phase2_start: 40000
```

## Training Progress

The training script saves:
- `latest_model.pt` - Most recent checkpoint (for resumption)
- `best_model.pt` - Best validation loss checkpoint

Training can be resumed automatically by re-running `python train.py`.

## Evaluation Metrics

- **Perplexity**: Exponential of cross-entropy loss
- **Diversity Metrics**:
  - Unique word ratio (type-token ratio)
  - Vocabulary size
  - Lexical entropy

## ONNX Export

Two export modes:
- **Last token** (default): Output shape `(B, V)` - for autoregressive generation
- **Full sequence**: Output shape `(B, T, V)` - for full sequence processing

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended for training)
- ~16GB GPU memory for full training

## License

MIT
