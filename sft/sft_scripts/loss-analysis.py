import ast
import re
import matplotlib.pyplot as plt
from pathlib import Path
# log_file = Path(
#     "/path/RL_Heals_SFT/sft/sft_scripts/test-loss-llama-2000-ind.out"
# )
#
# # ansi_escape = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
#
# # train_losses = []
# # eval_losses = []
# # train_steps = []
# # eval_steps = []
# # step = 0
#
# ansi_escape = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# non_ascii   = re.compile(r"[^\x00-\x7F]+")
#
# train_losses, eval_losses_ind, eval_losses_ood = [], [], []
# train_steps,  eval_steps_ind, eval_steps_ood  = [], [], []
# step = 0
#
# with log_file.open("r", encoding="utf-8", errors="ignore") as f:
#     for raw in f:
#         line = non_ascii.sub("", ansi_escape.sub("", raw)).strip()
#         if not line or not line.startswith("{"):
#             continue
#
#         try:
#             rec = ast.literal_eval(line)
#         except Exception:
#             continue
#
#         if isinstance(rec, dict):
#             if "loss" in rec and "eval_loss" not in rec:
#                 train_losses.append(rec["loss"])
#                 train_steps.append(step)
#                 step += 1
#             elif "eval_loss" in rec:
#                 eval_losses_ood.append(rec["eval_loss"])
#                 eval_steps_ood.append(step)
#
#         if step == 2000:
#             break
#
# print(train_losses)
# print(train_steps)
# print(eval_losses_ood)
# print(eval_steps_ood)
#
#

#
# plt.figure(figsize=(10, 6))
# plt.plot(train_steps, train_losses, ls='dotted', label="train Loss")
# # plt.plot(eval_steps_ind, eval_losses_ind, label="test Loss")
# plt.plot(eval_steps_ood, eval_losses_ood, ls='-.', label="ood Loss")
# plt.xlabel("Training Step")
# plt.ylabel("Loss")
# plt.title("Training and Evaluation Loss(LLama-7B)")
# plt.legend()
# plt.grid(True)
# # plt.savefig('/path/RL_Heals_SFT/analysis/test-loss-2000-llama-ind.png')
# plt.savefig('/path/RL_Heals_SFT/analysis/test-loss-2000-ind.png')
# plt.savefig('/path/RL_Heals_SFT/analysis/test-loss-llama-625.png')
# plt.show()


import re, ast
from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# -----------------------------
# Styling (publication-friendly)
# -----------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# -----------------------------
# Paths (IND = in-distribution)
# -----------------------------
LLAMA_LOG = Path("/path/RL_Heals_SFT/sft/sft_scripts/test-loss-llama-2000-ind.out")
QWEN_LOG  = Path("/path/RL_Heals_SFT/sft/sft_scripts/test-loss-qwen-2000-ind.out")
OUT_PDF   = Path("/path/RL_Heals_SFT/analysis/test-loss-2000-ind-merged.pdf")
OUT_PNG   = Path("/path/RL_Heals_SFT/analysis/test-loss-2000-ind-merged.png")

ansi_escape = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
non_ascii   = re.compile(r"[^\x00-\x7F]+")

def parse_log(log_path, max_steps=None):
    """
    Returns:
        dict with keys: train_steps, train_losses, eval_steps, eval_losses
    - Train records are those with key 'loss' but NOT 'eval_loss'
    - Eval records are those with key 'eval_loss' (IND file already)
    - If a record has 'step', we use it; otherwise we increment a local counter
    """
    train_losses, eval_losses = [], []
    train_steps,  eval_steps  = [], []
    step_counter = 0

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = non_ascii.sub("", ansi_escape.sub("", raw)).strip()
            if not line or not line.startswith("{"):
                continue
            try:
                rec = ast.literal_eval(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue

            # Prefer logged 'step' if present; otherwise fall back to counter
            s = rec.get("step", step_counter)

            # TRAIN
            if "loss" in rec and "eval_loss" not in rec:
                train_losses.append(float(rec["loss"]))
                train_steps.append(s)
                # advance counter if we are managing steps locally
                if "step" not in rec:
                    step_counter += 1

            # EVAL (IND)
            if "eval_loss" in rec:
                eval_losses.append(float(rec["eval_loss"]))
                eval_steps.append(s)

            if max_steps is not None and step_counter >= max_steps:
                break

    return {
        "train_steps": train_steps,
        "train_losses": train_losses,
        "eval_steps": eval_steps,
        "eval_losses": eval_losses,
    }

# -----------------------------
# Parse both logs
# -----------------------------
llama = parse_log(LLAMA_LOG, max_steps=1500)
qwen  = parse_log(QWEN_LOG,  max_steps=1500)

# -----------------------------
# Plot (one figure with four series)
# -----------------------------
fig, ax = plt.subplots(figsize=(6.0, 3.6))

# Llama
ax.plot(llama["train_steps"], llama["train_losses"],
        linestyle="--", marker="o", markersize=3.5, linewidth=1.5,
        label="LLaMA — train (IND)")
ax.plot(llama["eval_steps"], llama["eval_losses"],
        linestyle="-.", marker="s", markersize=3.5, linewidth=1.5,
        label="LLaMA — test (IND)")

# Qwen
ax.plot(qwen["train_steps"], qwen["train_losses"],
        linestyle=":", marker="^", markersize=3.5, linewidth=1.5,
        label="Qwen — train (IND)")
ax.plot(qwen["eval_steps"], qwen["eval_losses"],
        linestyle="-", marker="D", markersize=3.2, linewidth=1.5,
        label="Qwen — test (IND)")

ax.set_xlabel("Checkpoint")
ax.set_ylabel("Loss")

# ---- Key addition: compress x > 100 while keeping x <= 100 expanded ----
ax.set_xscale("symlog", linthresh=100, linscale=1.0, base=10)

# Optional: make ticks nice & readable at specific checkpoints
tick_locs = [0, 20, 50, 100, 200, 400, 700, 1000]
ax.set_xticks([t for t in tick_locs if t <= max(
    (llama["train_steps"] + llama["eval_steps"] + qwen["train_steps"] + qwen["eval_steps"]) or [0]
)])
ax.set_xticklabels([str(t) for t in ax.get_xticks()])  # keep raw checkpoint labels

# Optional: visual cue for the linear-to-log transition
ax.axvline(100, color="black", linestyle="--", linewidth=0.8, alpha=0.2)

ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.2, which="both")
ax.legend(frameon=False, ncol=2)
fig.tight_layout()

fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, bbox_inches="tight")
plt.close(fig)