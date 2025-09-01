import re
import json
import os
import ast            # <-- add this
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt

# Input log files (uploaded by the user)
logs = [
    "/path/language-train-ck-2025-08-09-800.log",
    "/path/language-train-ck-2025-08-09-900.log",
]

os.makedirs("./loss", exist_ok=True)  # <-- ensure save dir exists

def detect_model_name(text: str, fallback: str) -> str:
    t = text.lower()
    if "qwen" in t:
        return "Qwen-7B"
    if "llama" in t:
        return "LLaMA-11B"
    return fallback

def parse_losses_from_log(path: str):
    with open(path, "r", errors="ignore") as f:
        all_text = f.read()
    model = detect_model_name(all_text, Path(path).stem)

    steps, id_losses, ood_losses = [], [], []
    depth = 0
    buf = ""
    np_float_re = re.compile(r"np\.float64\(([^()]+)\)")

    def try_consume_dict(text):
        clean = np_float_re.sub(r"\1", text)
        try:
            d = ast.literal_eval(clean)
        except Exception:
            return
        # step
        k_step = None
        for k in ["total_num_steps", "total_steps", "total_step", "steps", "global_step"]:
            if k in d and isinstance(d[k], (int, float)):
                k_step = int(d[k]); break
        # losses
        id_val, ood_val = None, None
        for k, v in d.items():
            if not isinstance(v, (int, float)):
                continue
            lk = str(k).lower()
            if "cross-entropy" in lk:
                if "ood" in lk or "out-of-domain" in lk or "outofdomain" in lk:
                    ood_val = float(v)
                elif lk == "cross-entropy" or "ind" in lk or "in-domain" in lk or "id" in lk:
                    id_val = float(v)
        if k_step is not None and (id_val is not None or ood_val is not None):
            steps.append(int(k_step/4/32))
            id_losses.append(id_val if id_val is not None else float("nan"))
            ood_losses.append(ood_val if ood_val is not None else float("nan"))

    with open(path, "r", errors="ignore") as f:
        for line in f:
            opens = line.count("{"); closes = line.count("}")
            if opens and depth == 0:
                buf = line[line.index("{"):]
                depth = opens - closes
                if depth == 0:
                    try_consume_dict(buf[:buf.rindex("}") + 1]); buf = ""
            elif depth > 0:
                buf += line
                depth += opens - closes
                if depth == 0:
                    try_consume_dict(buf[:buf.rindex("}") + 1]); buf = ""

    if steps:
        order = sorted(range(len(steps)), key=lambda i: steps[i])
        steps = [steps[i] for i in order]
        id_losses = [id_losses[i] for i in order]
        ood_losses = [ood_losses[i] for i in order]
        last_idx = {}
        for i, st in enumerate(steps):
            last_idx[st] = i
        idxs = sorted(last_idx.values())
        steps = [steps[i] for i in idxs]
        id_losses = [id_losses[i] for i in idxs]
        ood_losses = [ood_losses[i] for i in idxs]

    return model, steps, id_losses, ood_losses

series = [parse_losses_from_log(p) for p in logs]
by_model = {name: (steps, id_l, ood_l) for name, steps, id_l, ood_l in series}

def plot_simple(model, steps, id_losses, ood_losses):
    plt.figure(figsize=(10, 6))
    if any(v == v for v in id_losses):   # NaNs fail v==v, so this is enough
        plt.plot(steps, id_losses, label="ID Loss")
    if any(v == v for v in ood_losses):
        plt.plot(steps, ood_losses, label="OOD Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(f"ID vs OOD Loss ({model})")
    plt.legend()
    plt.grid(True)
    out = f"./loss/rl-id_ood_{model.lower().replace(' ', '').replace('/', '-')}.pdf"
    plt.savefig(out, bbox_inches="tight")
    # plt.show()
    return out

out_files = []
for want in ["Qwen-7B", "LLaMA-11B"]:
    if want in by_model:
        out_files.append(plot_simple(want, *by_model[want]))
    else:
        for model, (steps, id_l, ood_l) in by_model.items():
            out_files.append(plot_simple(model, steps, id_l, ood_l))

print("Saved:", out_files)