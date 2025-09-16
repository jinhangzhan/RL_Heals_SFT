# Reinforcement Learning Heals OOD Forgetting in Supervised Fine-Tuning (Official)

Evolution of OOD Reasoning Performance for SFT+RL    |  Advantage Distribution in PPO at Different Checkpoints 
:-------------------------:|:-------------------------:
![](https://github.com/xiaodanguoguo/RL_Heals_SFT/blob/master/figures/llama_ood_recovery_line-full.jpg)  | ![Advantage_Distribution](https://github.com/user-attachments/assets/3d3d8987-2d22-4fc5-9622-c5ac34d3487a)




This repository provides a comprehensive framework for training Large Language Models (LLMs) using both Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) approaches. The framework supports both **LLaMA** and **Qwen** model families and includes evaluation pipelines for various tasks.

## 🚨 Important Setup Notes

**Before using this repository, you MUST replace all `/path` placeholders with your actual paths:**

- Replace `/path/to/conda` with your conda installation path
- Replace `/path/to/cuda` with your CUDA installation path  
- Replace `/path/to/data` with your dataset directory path
- Replace `/path/to/model` with your model checkpoint directory
- Update email addresses in SLURM scripts from `xxx@email.com` to your email
- Set proper WANDB API keys in scripts (currently set to placeholder values)

## 📋 Table of Contents

- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
- [Reinforcement Learning Training](#reinforcement-learning-training)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Scripts Overview](#scripts-overview)
- [Troubleshooting](#troubleshooting)

## 🛠 Installation

### Prerequisites

- Python 3.13
- CUDA-compatible GPU
- Conda or virtual environment manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/xiaodanguoguo/RL_Heals_SFT.git
cd RL_Heals_SFT
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install the gym environment:**
```bash
cd gym
pip install -e .
cd ..
```

4. **Set up environment variables:**
```bash
export PYTHONPATH=/path/to/your/RL_Heals_SFT:$PYTHONPATH
export WANDB_API_KEY="your_wandb_key"
```

## 📁 Repository Structure

```
RL_Heals_SFT/
├── sft/                     # Supervised Fine-Tuning modules
│   ├── src/                 # Core SFT implementation
│   └── sft_scripts/         # SFT training scripts
├── rl/                      # Reinforcement Learning modules
│   ├── configs/             # RL configuration files
│   ├── trainer/             # RL trainers (PPO, etc.)
│   └── launcher.py          # RL training launcher
├── evaluation/              # Evaluation framework
│   ├── evaluator/           # Model evaluators
│   └── configs/             # Evaluation configurations
├── gym/                     # Custom gym environments
│   └── gym_cards/           # Card game environments
├── scripts/                 # Training and evaluation scripts
│   ├── gp_training/         # General purpose training
│   ├── gp_evaluation/       # General purpose evaluation
│   └── virl_training/       # VIRL-specific training
├── analysis/                # Analysis and visualization tools
├── virl/                    # VIRL-specific modules
└── utils_*.py               # Utility functions
```

## 🚀 Quick Start

### 1. Prepare Your Data

Ensure your datasets are in the correct format and update paths in configuration files:

```bash
# Update data paths in configuration files
# Example: rl/configs/llama_gp_language.yaml
# Example: evaluation/configs/llama_gp_language.yaml
```

### 2. Configure Paths

**Critical Step:** Update all hardcoded paths in the following files:

```bash
# Scripts with hardcoded paths that need updating:
scripts/gp_training/language_train-qwen.sh
scripts/gp_evaluation/language_ood_dir_eval_recover-batch.sh
analysis/angle_acm.sh

```

### 3. Basic Training Example

```bash
# SFT Training (Llama)
cd sft/sft_scripts
bash gp_l.sh

# RL Training (Llama)
cd scripts/gp_training
bash language_train.sh

# Evaluation
cd scripts/gp_evaluation
bash language_indist_eval.sh
```

## 📚 Supervised Fine-Tuning (SFT)

### Configuration

SFT configurations are defined in `sft/src/training/params.py` and can be customized via command-line arguments.

### Training Scripts

Located in `sft/sft_scripts/`:

- `gp_l.sh` - General purpose Llama SFT
- `gp_l-qwen.sh` - General purpose Qwen SFT  
- `gp_l-8.sh` - 8-GPU Llama training
- `virl_l.sh` - VIRL Llama SFT
- `virl_l-qwen.sh` - VIRL Qwen SFT

### Usage Example

```bash
cd sft/sft_scripts

# Edit the script to update paths:
# - Update conda environment path
# - Update CUDA paths if needed
# - Set proper data paths
# - Configure output directories

# Run training
bash gp_l.sh
```

### Key SFT Parameters

- `--model_name_or_path`: Base model path
- `--data_path`: Training data directory
- `--output_dir`: Output directory for checkpoints
- `--learning_rate`: Learning rate (default: 1e-5)
- `--num_train_epochs`: Number of training epochs
- `--per_device_train_batch_size`: Batch size per device

## 🎯 Reinforcement Learning Training

### PPO Training

The RL framework uses Proximal Policy Optimization (PPO) for training.

### Configuration Files

Located in `rl/configs/`:

- `llama_gp_language.yaml` - Llama general purpose config
- `llama_gp_language-qwen.yaml` - Qwen general purpose config  
- `llama_virl_language.yaml` - Llama VIRL config
- `llama_virl_language-qwen.yaml` - Qwen VIRL config

### Training Scripts

Located in `scripts/gp_training/` and `scripts/virl_training/`:

```bash
# Llama RL training
bash scripts/gp_training/language_train.sh

# Qwen RL training  
bash scripts/gp_training/language_train-qwen.sh

# Multi-GPU training
bash scripts/gp_training/language_train-8.sh
```

### Key RL Parameters

- `--model_name`: Model identifier
- `--config_file`: Configuration file path
- `--num_episodes`: Number of RL episodes
- `--learning_rate`: PPO learning rate
- `--ppo_epochs`: PPO update epochs per rollout

## 📊 Evaluation

### Evaluation Framework

The evaluation system supports multiple evaluators:

- `LlamaEvaluator` - For Llama models
- `QwenEvaluator` - For Qwen models  
- Custom evaluators for specific tasks

### Evaluation Scripts

Located in `scripts/gp_evaluation/`:

```bash
# In-distribution evaluation
bash language_indist_eval.sh

# Out-of-distribution evaluation  
bash language_ood_eval.sh

# Batch evaluation
bash run_batch_eval.sh
```

### Running Evaluation

```bash
cd evaluation

# Configure evaluation settings in run_eval.py
# Update model paths and data paths

# Run evaluation
python run_eval.py \
  --model_path /path/to/your/model \
  --config_path configs/llama_gp_language.yaml \
  --output_dir results/
```

## ⚙️ Configuration

### Environment Variables

Set these before running any scripts:

```bash
export PYTHONPATH=/path/to/RL_Heals_SFT:$PYTHONPATH
export WANDB_API_KEY="your_actual_wandb_key"
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # Adjust for your setup
```

### SLURM Configuration

Most scripts include SLURM headers. Update the following in each script:

```bash
#SBATCH --mail-user=your_email@institution.edu  # Replace xxx@email.com
#SBATCH --account=your_account
#SBATCH --partition=your_partition
```

### Model Paths

Update model paths in configuration files:

```yaml
# Example: rl/configs/llama_gp_language.yaml
model_name: "/path/to/your/llama/model"
tokenizer_name: "/path/to/your/llama/tokenizer"
```

## 📜 Scripts Overview

### Training Scripts

| Script | Purpose | Model | Notes |
|--------|---------|--------|-------|
| `sft/sft_scripts/gp_l.sh` | SFT training | Llama | General purpose |
| `sft/sft_scripts/gp_l-qwen.sh` | SFT training | Qwen | General purpose |
| `scripts/gp_training/language_train.sh` | RL training | Llama | PPO-based |
| `scripts/gp_training/language_train-qwen.sh` | RL training | Qwen | PPO-based |

### Evaluation Scripts

| Script | Purpose | Task Type |
|--------|---------|-----------|
| `scripts/gp_evaluation/language_indist_eval.sh` | In-distribution eval | Language tasks |
| `scripts/gp_evaluation/language_ood_eval.sh` | Out-of-distribution eval | Language tasks |
| `evaluation/run_eval.py` | General evaluation | Configurable |

### Analysis Scripts

| Script | Purpose |
|--------|---------|
| `analysis/svd-recover.sh` | SVD-based model analysis |
| `analysis/angle_acm.sh` | Angle analysis for ACM |

## 🔧 Troubleshooting

### Common Issues

1. **Path Errors**: Ensure all `/path` placeholders are replaced with actual paths
2. **CUDA Errors**: Verify CUDA installation and GPU availability
3. **Import Errors**: Check PYTHONPATH is set correctly
4. **Permission Errors**: Ensure proper file permissions for scripts

### Path Replacement Checklist

Before running any scripts, verify these paths are updated:

- [ ] Conda environment paths in shell scripts
- [ ] CUDA installation paths
- [ ] Data directory paths
- [ ] Model checkpoint paths
- [ ] Output directory paths
- [ ] PYTHONPATH exports
- [ ] Email addresses in SLURM scripts
- [ ] WANDB API keys

### Debug Mode

Enable debug logging:

```bash
export WANDB_MODE=offline  # For offline debugging
export TRANSFORMERS_VERBOSITY=debug
```

### Memory Issues

For large models, consider:

- Using DeepSpeed configuration files in `scripts/`
- Reducing batch sizes
- Using gradient checkpointing
- Model parallelism across multiple GPUs

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review configuration files for path issues
3. Ensure all dependencies are installed correctly
4. Open an issue with detailed error messages and system information

---

**⚠️ Remember: This repository contains placeholder paths and configurations. You must update all `/path` references and configuration parameters before use.**
