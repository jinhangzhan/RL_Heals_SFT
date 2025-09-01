import json
import matplotlib.pyplot as plt
import numpy as np

# 1) Load JSON
json_file = "/path/weights_stats-full-3.json"
with open(json_file, "r") as f:
    data = json.load(f)

# 2) Let's pick one specific param to analyze across all checkpoints
param_of_interest = "model.layers.39.mlp.down_proj.weight"
# Sort the checkpoint keys to ensure we plot them in ascending order
checkpoints = sorted(list(data.keys()), key=lambda x: int(x.split("-")[1]))  # e.g. "checkpoint-5" -> 5

checkpoints = ['checkpoint-200', 'checkpoint-6251']
# 2a) Compare singular value distribution across checkpoints
import math
keys = data[checkpoints[0]].keys()
for key in (keys):
    param_of_interest = key

    means = []
    for ckpt in checkpoints:
        if param_of_interest not in data[ckpt]:
            continue
        svals = data[ckpt][param_of_interest]["singular_values"]
        means.append(round(sum(svals)/len(svals), 4))

    print(param_of_interest, "all ck ", means)
    print("total means ", round(sum(means)/len(means),4))
    # print(math.mean(means))

param_of_interest = "model.layers.20.self_attn.q_proj.weight"
plt.figure(figsize=(8, 6))
for ckpt in checkpoints:
    if param_of_interest not in data[ckpt]:
        continue
    svals = data[ckpt][param_of_interest]["singular_values"]
    if not svals:
        continue
    # convert to numpy
    svals = np.array(svals)

    # We can plot only top 50 or so to avoid too many points
    # top_k = min(50, len(svals))
    print(f"checkpoint {ckpt}, svals {round(svals[4093], 6)}")
    top_k = len(svals)

    x = np.arange(top_k)

    # plt.plot(x, np.log10(svals[:top_k]), label=ckpt)
    plt.plot(x, svals[:top_k], label=ckpt)

plt.xlabel("Singular value index")
plt.ylabel("log10(sigma)")
plt.title(f"Singular value spectrum for {param_of_interest} (top 50) - across checkpoints")
plt.legend()
plt.tight_layout()
plt.show()


# 2b) Compare norm changes across checkpoints (e.g. fro_norm)
# We'll do a line plot of fro_norm vs. checkpoint

# fro_norm_values = []
# for ckpt in checkpoints:
#     if param_of_interest not in data[ckpt]:
#         fro_norm_values.append(np.nan)
#         continue
#     fro_val = data[ckpt][param_of_interest]["basic_stats"]["fro_norm"]
#     fro_norm_values.append(fro_val)
#
# plt.figure(figsize=(6, 5))
# x = np.arange(len(checkpoints))
# plt.plot(x, fro_norm_values, marker='o')
# plt.xticks(x, checkpoints, rotation=45)
# plt.xlabel("Checkpoint")
# plt.ylabel("Frobenius Norm")
# plt.title(f"Fro Norm for {param_of_interest} across checkpoints")
# plt.tight_layout()
# plt.show()

# 3) (Optional) Inspect distribution of l2_norm or fro_norm for all parameters in a single checkpoint
# e.g. look at checkpoint-5

fro_norm_values = []
l2_values = []
for ckpt in checkpoints:
    if param_of_interest not in data[ckpt]:
        fro_norm_values.append(np.nan)
        continue
    fro_val = data[ckpt][param_of_interest]["basic_stats"]["l2_norm"]
    l2_values.append(fro_val)
# ckpt_to_inspect = "checkpoint-50"
# if ckpt_to_inspect in data:
#     param_names = list(data[ckpt_to_inspect].keys())
#     for pname in param_names:
#         stats = data[ckpt_to_inspect][pname]["basic_stats"]
#         l2_values.append(stats["l2_norm"])

# plt.figure(figsize=(6, 4))
# plt.plot(range(len(l2_values)), l2_values, marker='o')
# plt.title(f"L2 Norm distribution of all parameters in {param_of_interest}")
# plt.xlabel("Parameter index")
# plt.ylabel("L2 Norm")
# plt.tight_layout()
# plt.show()