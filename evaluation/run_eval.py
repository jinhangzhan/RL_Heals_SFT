#!/usr/bin/env python3
"""
Run OOD evaluation inside an interactive Slurm node.

    $ python run_eval.py
"""

import os
import random
import datetime as dt
from pathlib import Path
import subprocess
import sys
import torch

os.environ.setdefault("DS_BUILD_OPS", "0")
os.environ.setdefault("TRITON_DISABLE", "1")
os.environ.setdefault("C10_DISABLE_LEVEL_ZERO", "1")
os.environ.setdefault("C10_DISABLE_CPUPOWER", "1")

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault(
    "WANDB_API_KEY", "wandbkey"
)

proj_root = "/path/RL_Heals_SFT"
os.environ["PYTHONPATH"] = proj_root + ":" + os.environ.get("PYTHONPATH", "")

conda_prefix = os.environ.get("CONDA_PREFIX", "")
if conda_prefix:
    os.environ["LD_LIBRARY_PATH"] = (
        f"{conda_prefix}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    )

VITER      = 5
ENABLE     = True        # verification
OOD        = True        # rule-ood
FACE10     = False       # face card=10?
TARGET     = 24          # target point
CHECKPOINT = 6500
NUM_TRAJ   = 234

CKPT_PATH  = "/path/Llama-3.2-11B-Vision-Instruct/"
CONFIG_YAML= "/path/RL_Heals_SFT/evaluation/configs/llama_gp_language.yaml"
ACCEL_CONF = "/path/RL_Heals_SFT/scripts/config_zero2_1gpu.yaml"

today      = dt.datetime.now().strftime("%Y-%m-%d")
main_port  = random.randint(10000, 20000)

out_dir    = (
    f"logs/gp_l_ood_verify_{VITER}_target_{TARGET}_{today}_{CHECKPOINT}"
)
Path(out_dir).mkdir(parents=True, exist_ok=True)
log_path   = Path(out_dir) / f"eval-ood-res-{today}-ck-{CHECKPOINT}.log"

cmd = [
    "torchrun",
    "--standalone",             
    f"--nproc_per_node={torch.cuda.device_count()}",
    f"--master_port={main_port}",
    "evaluation.launcher",        
    "-f", CONFIG_YAML,
    f"--model_path={CKPT_PATH}",
    f"--output_dir={out_dir}/gp_l_ood.jsonl",
    f"--prompt_config.enable_verification={ENABLE}",
    f"--env_config.target_points={TARGET}",
    f"--env_config.verify_iter={VITER}",
    f"--env_config.treat_face_cards_as_10={FACE10}",
    f"--env_config.ood={OOD}",
    f"--num_traj={NUM_TRAJ}",
]

print(">> Running command:\n", " \\\n   ".join(cmd), "\n", flush=True)
print(">> Logging to:", log_path, flush=True)

with open(log_path, "w") as log_f:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="")
        log_f.write(line)
    proc.wait()

sys.exit(proc.returncode)
