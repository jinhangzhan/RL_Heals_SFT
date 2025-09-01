import re, pathlib, itertools
import numpy as np
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances, silhouette_score
from scipy.linalg import eigvalsh
from collections import defaultdict
# from numpy.linalg import svd
from torch.linalg import svd
from pathlib import Path
import torch
import pandas as pd

SAVE_DIR = "./tsne"

def load_down_proj_rows(path: str, layer_idx: int = 0):
    tensors = load_file(path, device="cuda")
    pattern = re.compile(fr"(?:layers|layer)\.{layer_idx}.*down_proj.*weight")
    key = next((k for k in tensors if pattern.search(k)), None)
    if key is None:
        raise KeyError(f"no key {pattern.pattern}")
    weight = tensors[key]          # [out_dim, in_dim]
    return weight.detach().cpu().numpy()


ckpt_paths = {
    "ckpt-140":  "/path/SFT_MaxOOD",
    "ckpt-1100": "/path/SFT_END",
    "ckpt-69":   "/path/RL",
}

# ckpt_paths = {
#     "ckpt-120":  "/network/scratch/l/luansito/data/train_ckpt/gp_l_sft_qwen/checkpoint-merge/checkpoint-120.safetensors",
#     "ckpt-800": "/network/scratch/l/luansito/data/train_ckpt/gp_l_sft_qwen/checkpoint-merge/checkpoint-800.safetensors",
#     "ckpt-29":   "/network/scratch/l/luansito/data/train_ckpt/gp_l_sft_qwen/checkpoint-merge/checkpoint-29.safetensors",
# }

model_name  = "llama"
# layer_idx = 0
# layers = [0, 19, 39]
layers = [0, 14, 27]
projs = ['down', 'q', 'k', 'v']
records = []
rank = 512

for layer_idx in layers:
    print(layer_idx)
    for proj in projs:
        print(proj)
        proj_regex = fr"(?:layers|layer)\.{layer_idx}.*{proj}_proj.*weight"

        def load_down_proj_rows(path: str, pattern: str):
            tensors = load_file(path, device="cuda")
            key = next((k for k in tensors if re.search(pattern, k)), None)
            if key is None:
                raise KeyError(f"{path}: can not find down_proj.weight (pattern={pattern})")
            return tensors[key].cpu().numpy()   # shape: [out_dim, in_dim]

        rows, labels = [], []
        for label, p in ckpt_paths.items():
            w = load_down_proj_rows(p, proj_regex)
            rows.append(w)
            labels.extend([label] * w.shape[0])

        X = np.vstack(rows)
        X = torch.tensor(X)
        labels = np.array(labels)

        # svd = svd(X, full_matrices=False)
        U, S, V = torch.pca_lowrank(X, q=rank + 10, center=True)
        # sigma1, sigma2 = S[:rank].cpu().numpy()
        V2_stack = V[:, :rank].cpu().numpy()
        # sigmas_stack, V2_stack = svd[1][:2], svd[2][:2].T
        Z_stack = X @ V2_stack
        radius_by_ckpt = defaultdict(dict)

        for lab in ckpt_paths:
            idx = labels == lab
            r = torch.linalg.norm(Z_stack[idx], axis=1)
            radius_by_ckpt[lab] = {
                "mean_r_stack":   float(r.mean()),
                "median_r_stack": float(np.median(r)),
                "p95_r_stack":    float(np.percentile(r, 95))
            }

        for lab in ckpt_paths:
            print(lab, radius_by_ckpt[lab])

        svd_stats = {}
        for lab, W in zip(ckpt_paths, rows):
            W = torch.tensor(W)
            _, S, Vh = torch.pca_lowrank(W, q=rank, center=True)
            Z_self = W @ Vh[:,:rank]
            r_self = torch.linalg.norm(Z_self, axis=1)

            svd_stats[lab] = {
                # "sigma1": float(S[0]),
                # "sigma2": float(S[1]),
                # "phi":    float((S[0] * S[1])),
                "mean_r_self":   float(r_self.mean()),
                "median_r_self": float(np.median(r_self)),
                "p95_r_self":    float(np.percentile(r_self, 95)),
            }

        for lab in ckpt_paths:
            print(lab, svd_stats[lab])
            rec = dict(layer=layer_idx+1,
                       proj=proj,
                       ckpt=lab,
                       mean_r_stack=radius_by_ckpt[lab]["mean_r_stack"],
                       median_r_stack=radius_by_ckpt[lab]["median_r_stack"],
                       p95_r_stack=radius_by_ckpt[lab]["p95_r_stack"],
                       # sigma1=svd_stats[lab]["sigma1"],
                       # sigma2=svd_stats[lab]["sigma2"],
                       # phi=svd_stats[lab]["phi"],
                       mean_r_self=svd_stats[lab]["mean_r_self"],
                       median_r_self=svd_stats[lab]["median_r_self"],
                       p95_r_self=svd_stats[lab]["p95_r_self"])
            records.append(rec)

# df = pd.DataFrame(records)
# out_path = Path("./tsne/svd_projection_full-llama.xlsx")
# with pd.ExcelWriter(out_path, engine="xlsxwriter") as w:
#     df.to_excel(w, index=False, sheet_name="metrics")
# tsne_emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)
#
# plt.figure(figsize=(6, 6))
# for lab in ckpt_paths:
#     idx = labels == lab
#     plt.scatter(tsne_emb[idx, 0], tsne_emb[idx, 1], s=8, label=lab)
# plt.title("t-SNE of layer-0 down_proj rows")
# plt.legend()
# plt.xlabel("t-SNE-1"); plt.ylabel("t-SNE-2"); plt.tight_layout()
# plt.savefig(f"{SAVE_DIR}/tsne.png")
#
# pca_emb = PCA(n_components=2, random_state=0).fit_transform(X)
#
# plt.figure(figsize=(6, 6))
# for lab in ckpt_paths:
#     idx = labels == lab
#     plt.scatter(pca_emb[idx, 0], pca_emb[idx, 1], s=8, label=lab)
# plt.title("PCA (PC1 vs PC2) of layer-0 down_proj rows")
# plt.legend()
# plt.xlabel("PC-1"); plt.ylabel("PC-2"); plt.tight_layout()
# plt.savefig(f"{SAVE_DIR}/pca.png")

# for lab in ckpt_paths:
#     idx = labels == lab
#     x, y = pca_emb[idx, 0], pca_emb[idx, 1]
#     # print(np.abs(x).mean(), np.abs(y).mean())
#     print(np.abs(x).mean() + np.abs(y).mean())
#     # print(np.abs(x).mean() * np.abs(y).mean())
#
# rad_stats = defaultdict(dict)
# for lab in ckpt_paths:
#     idx = labels == lab
#     Z   = pca_emb[idx]
#     r   = np.linalg.norm(Z, axis=1)
#     rad_stats[lab]["mean_r"]   = r.mean()
#     rad_stats[lab]["median_r"] = np.median(r)
#     rad_stats[lab]["p95_r"]    = np.percentile(r, 95)
#
# print(f"{'ckpt':10} |  ⟨r⟩    median    95%")
# for lab in ckpt_paths:
#     s = rad_stats[lab]
#     print(f"{lab:10} | {s['mean_r']:.5f}  {s['median_r']:.5f}  {s['p95_r']:.5f}")
#
# sigmas = defaultdict(dict)
# for lab in ckpt_paths:
#     idx   = labels == lab
#     zlab  = pca_emb[idx]
#     sig1  = zlab[:, 0].std(ddof=1)
#     sig2  = zlab[:, 1].std(ddof=1)
#     sigmas[lab] = (sig1, sig2)
#
# print(f"{'ckpt':10} | σ_PC1  σ_PC2  ϕ=σ1*σ2")
# for lab, (s1, s2) in sigmas.items():
#     print(f"{lab:10} | {s1:.5f}  {s2:.5f}  {(s1*s2):.5f}")

# def stat_per_matrix(W: np.ndarray, sample_pairs: int = 20000):
#     mu   = W.mean(axis=0)
#     diff = W - mu
#     cov  = diff.T @ diff / (W.shape[0] - 1)
#     eigvals = eigvalsh(cov)
#     total_var = eigvals.sum()
#     rmsd      = np.sqrt((diff**2).sum(axis=1).mean())
#     cond_num  = eigvals[-1] / eigvals[0] if eigvals[0] > 0 else np.inf
#     eig_ratio = eigvals / (total_var + 1e-12)
#     eig_entropy = -np.sum(eig_ratio * np.log(eig_ratio + 1e-12))
#
#     n = W.shape[0]
#     S = min(sample_pairs, n * (n - 1))
#     i = np.random.randint(0, n, size=S)
#     j = np.random.randint(0, n, size=S)
#     mask = i != j
#     i, j = i[mask], j[mask]
#
#     W_norm = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
#     mean_cosine = (W_norm[i] * W_norm[j]).sum(axis=1).mean()
#
#     return total_var, rmsd, cond_num, eig_entropy, mean_cosine

# stats = defaultdict(dict)
# for lab, w in zip(ckpt_paths, rows):
#     stats[lab]["total_var"], stats[lab]["rmsd"], stats[lab]["cond_num"], stats[lab]["eig_entropy"], stats[lab]["mean_cosine"] = stat_per_matrix(w)
#
# label_to_int = {lab:i for i,lab in enumerate(ckpt_paths)}
# y_int = np.array([label_to_int[l] for l in labels])
# sil_val = silhouette_score(pca_emb, y_int)
# for lab in ckpt_paths:
#     stats[lab]["silhouette(PCA2D)"] = sil_val
#
# print("\n=== Quantitative stats per checkpoint ===")
# for lab in ckpt_paths:
#     s = stats[lab]
#     print(f"{lab:10} | Var={s['total_var']:.2e}  RMS={s['rmsd']:.4f}  "
#           f"Cond={s['cond_num']:.1f}  Ent={s['eig_entropy']:.3f}  "
#           f"Cos={s['mean_cosine']:.4f}  Sil={s['silhouette(PCA2D)']:.3f}")

# def spread_2d(Z):
#     # Z: [m, 2]
#     r2 = (Z[:,0]**2 + Z[:,1]**2).mean()
#     var = Z.var(axis=0).sum()
#     area = np.pi * Z[:,0].std() * Z[:,1].std()
#     return r2, var, area

# for lab in ckpt_paths:
#     idx = labels == lab
#     r2, var2d, area = spread_2d(pca_emb[idx])
#     stats[lab]["RMSD-2D"] = np.sqrt(r2)
#     stats[lab]["Var2D"]   = var2d
#     stats[lab]["Area2D"]  = area
#
# print(f"{'ckpt':10} | RMS2D  Var2D   Area2D")
# for lab in ckpt_paths:
#     s = stats[lab]
#     print(f"{lab:10} | {s['RMSD-2D']:.4f}  {s['Var2D']:.4f}  {s['Area2D']:.4f}")
#
# import json, os
# with open(os.path.join(SAVE_DIR, "downproj_stats.json"), "w") as f:
#     json.dump(stats, f, indent=2)