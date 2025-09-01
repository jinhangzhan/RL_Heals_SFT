# Fine-tune Llama-3.2-Vision

This folder contains SFT scripts of Llama-3.2-Vision. SFT experiments in our paper start from the [11B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) model. 
## Data preparation

Prepare the SFT data from our [HF-repo](https://huggingface.co/datasets/tianzhechu/SFTvsRL_Data). Currently we provide <code>SFT_data/gp-l/</code> and <code>SFT_data/virl-l</code>. 

## Launch

You may launch SFT experiments via:

```
# under directory SFTvsRL/sft/
bash sft_scripts/gp_l.sh
```

You may consider taking checkpoints <code>400-step</code> (GP-L), <code>25-step</code> (V-IRL-L) as RL init checkpoints.

** Remark: version of <code>transformers</code> matters.

## Acknowledgement
- [Llama-3.2-Vision-Finetune](https://github.com/2U1/Llama3.2-Vision-Finetune): Our SFT code is modified from early version of this repository.