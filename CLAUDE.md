# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the official codebase for "Value‑Guided Search for Efficient Chain‑of‑Thought Reasoning" - a research project that implements value-guided beam search for chain-of-thought reasoning using custom value models. The system combines:

- **Value Model**: A `Qwen2ForClassifier` model that predicts the success probability of reasoning chains
- **Beam Search Engine**: Block-wise beam search with value guidance using SGLang for inference
- **Training Pipeline**: Distributed training system for value models on chain-of-thought data

## Core Architecture

### Value Model (`classifier_lib.py`)
- Custom `Qwen2ForClassifier` extends Qwen2PreTrainedModel for classification
- Two-layer MLP head on top of base model: Linear -> ReLU -> Linear
- Outputs `success_probs` indicating likelihood of reasoning success
- Model loading example: `classifier_lib.Qwen2ForClassifier.from_pretrained("VGS-AI/DeepSeek-VM-1.5B")`

### Training System (`train_classifier.py`)
- Distributed training with PyTorch DDP (DistributedDataParallel)
- Custom `FlattenedDataset` in `training_utils.py` handles grouped data with big/small group sampling
- Supports multi-node, multi-GPU training with gradient accumulation
- Uses safe checkpointing with temporary files to prevent corruption

### Inference Engine (`inference_eval.py`)
- Block-wise beam search with SGLang backend for text generation
- Value model guides beam selection at block boundaries
- Supports various search strategies: `beam2`, DVTS (num_repetitions)
- Integration with Neptune Scale for experiment logging

### Data Pipeline
- `benchmark_data.py`: Loads math competition datasets (AIME, HMMT)
- HuggingFace datasets integration for `VGS-AI/OpenR1-VM` training data
- `accuracy_utils.py` and `eval_helpers.py`: Evaluation utilities for math problems

## Common Commands

### Training Value Model
```bash
# Single node, 4 GPUs
export HF_HUB_ENABLE_HF_TRANSFER=1
./scripts/train.sh

# Manual training command
torchrun --standalone --nproc_per_node=4 train_classifier.py \
    --data_path VGS-AI/OpenR1-VM \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --total_batch_size 64 \
    --micro_batch_size 8 \
    --max_lr 0.0001 \
    --num_steps 12170
```

### Running Inference
```bash
# Value-guided search on AIME-24
./scripts/beam2.sh

# Manual inference command
python inference_eval.py \
    --benchmark aime-24 \
    --classifier_ckpt_path VGS-AI/DeepSeek-VM-1.5B \
    --piref_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --search_type beam2 \
    --num_blocks 8 \
    --num_repetitions 16
```

## Key Implementation Details

### Data Handling
- Training data expects grouped structure with multiple CoT responses per problem
- `FlattenedDataset` manages big/small group sampling for preference learning
- Value model training uses response pairs for comparison-based learning

### Distributed Training
- Uses `torch.distributed` for multi-GPU coordination
- Safe checkpointing prevents corruption during distributed saves
- Environment variable `HF_HUB_ENABLE_HF_TRANSFER=1` enables faster HuggingFace downloads

### Inference Architecture
- SGLang handles text generation with custom attention implementations
- Block-based generation allows value model evaluation at reasoning boundaries
- Neptune Scale integration for comprehensive experiment tracking

### Value Model Design
- Based on Qwen2 architecture with custom classification head
- Dropout applied to classification layers during training
- Supports both naive and optimized inference implementations

## Environment Requirements

- PyTorch 2.6+ with CUDA support
- Flash Attention 2 for efficient transformer inference
- SGLang for structured generation
- HuggingFace ecosystem (transformers, datasets, hub)
- Neptune Scale for experiment tracking
- Distributed training requires torchrun for multi-GPU coordination