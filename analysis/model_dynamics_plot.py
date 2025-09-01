#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Recovery-oriented visualization for three checkpoints (SFT_MAX, SFT, RL).

What this adds on top of your original script:
  • A unified "recovery index" per metric: does RL move closer to SFT_MAX than SFT?
  • Per‑prompt win‑rates: fraction of prompts where RL is closer to SFT_MAX than SFT.
  • PCA centroid drift plot with arrows (SFT ➜ RL ➜ SFT_MAX) to show trajectory.
  • Overlaid top‑k overlap curves (SFT_MAX vs SFT) and (SFT_MAX vs RL).
  • Entropy & margin drift bars (OOD overconfidence/recovery proxy).
  • Two additional optional metrics: Hellinger@T and Energy distance (hidden).

Outputs (all saved under OUT_ROOT):
  recovery_bar_{TAG}.png            – bar chart of recovery deltas for each metric
  winrate_{TAG}.png                 – per‑prompt win‑rate for cosine & JS
  centroid_arrow_{TAG}.png          – PCA plane with centroid arrows (drift)
  topk_overlap_overlay_{TAG}.png    – overlaid Jaccard@k curves
  drift_entropy_margin_{TAG}.png    – mean entropy & margin per checkpoint
  recovery_summary_{TAG}.json       – JSON with all deltas + win‑rates

Run examples (LLAMA vs QWEN; ID vs OOD by changing PROMPT_JSON):
  python model_dynamics_recovery.py --model llama \
    --ckpt_sft_max /path/llama/checkpoint-120 \
    --ckpt_sft     /path/llama/checkpoint-900 \
    --ckpt_rl      /path/llama/checkpoint-epoch-29 \
    --shared_proc  /path/Llama-3.2-11B-Vision-Instruct \
    --prompt_json  /path/ind-data-300.json

  python model_dynamics_recovery.py --model qwen \
    --ckpt_sft_max /path/qwen/checkpoint-120 \
    --ckpt_sft     /path/qwen/checkpoint-800 \
    --ckpt_rl      /path/qwen/checkpoint-29 \
    --shared_proc  /path/Qwen2.5-7B-Instruct \
    --prompt_json  /path/ood-data.json
"""

import os, sys, json, math, csv, argparse, random
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ---- environment (no deepspeed) ----
os.environ["ACCELERATE_DISABLE_DEEPSPEED"] = "1"
os.environ["TRANSFORMERS_NO_DEEPSPEED"] = "1"
sys.modules["deepspeed"] = None

from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoTokenizer, AutoModelForCausalLM

# --------------------------------------
# Defaults (will be overridden by CLI)
DEFAULTS = dict(
    model="llama",  # "llama" or "qwen"
    ckpt_sft_max="path/SFT_MaxOOD",
    ckpt_sft    ="path/SFT_end",
    ckpt_rl     ="path/RL",
    shared_proc ="/path/Llama-3.2-11B-Vision-Instruct",
    # model="qwen",  # "llama" or "qwen"
    # ckpt_sft_max="path/SFT_MaxOOD",
    # ckpt_sft    ="path/SFT_end",
    # ckpt_rl     ="path/RL",
    # shared_proc ="/path/Llama-3.2-11B-Vision-Instruct",
    # prompt_json_ood ="/path/ood-data.json",
    prompt_json    = "/path/ind-data-300.json",
    out_root    ="./analysis_hidden",
    batch_size  = 8,
    max_length  = 4096,
    top_k       = 10,
    k_user      = 8,
    pool        = "mean",  # or "median"
    tsne_perp   = 30,
    js_temps    = [1.0, 1.5, 2.0],
    topk_curve_max = 50,
    spearman_M  = 1000,
    energy_sample = 512,
    seed = 0,
)

# ========= helpers =========

# ========= helpers =========

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed)


def load_textizer(path: str):
    """Load either a Processor or a Tokenizer and normalize the API.
    Returns an object with:
      • apply_chat_template(messages, add_generation_prompt, tokenize=...)
      • __call__(text=[...], return_tensors="pt", padding=True, truncation=True, max_length=...)
      • pad_token set (falls back to eos_token)
    """
    # Try processor first (works for Llama vision instruct); fall back to tokenizer
    try:
        obj = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        tok = getattr(obj, "tokenizer", None) or obj
    except Exception:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    # Ensure pad token
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = getattr(tok, "eos_token", None) or "<|endoftext|>"
    return tok


def make_messages(prompts, family: str):
    """Build chat messages in the schema expected per family.
    Qwen:   [{"role":"user","content": "..."}]
    Llama:  [{"role":"user","content":[{"type":"text","text":"..."}]}]
    """
    if family == "qwen":
        return [[{"role":"user","content": p}] for p in prompts]
    else:
        return [[{"role":"user","content":[{"type":"text","text": p}]}] for p in prompts]


def apply_chat_to_strings(tok, messages, add_generation_prompt: bool):
    """Call apply_chat_template robustly and return a list of STRINGS.
    Handles tokenizers that return token ids or dicts by decoding.
    """
    outs = []
    for m in messages:
        s = None
        for kwargs in (
            dict(add_generation_prompt=add_generation_prompt, tokenize=False),
            dict(add_generation_prompt=add_generation_prompt),
        ):
            try:
                o = tok.apply_chat_template(m, **kwargs)
                if isinstance(o, str):
                    s = o; break
                if isinstance(o, (list, tuple)) and all(isinstance(x, int) for x in o):
                    s = tok.decode(o, skip_special_tokens=False); break
                if isinstance(o, dict) and "input_ids" in o:
                    ids = o["input_ids"]
                    if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], list):
                        ids = ids[0]
                    s = tok.decode(ids, skip_special_tokens=False); break
            except TypeError:
                continue
            except Exception:
                continue
        if s is None:
            # Fallback: extract user text directly
            if isinstance(m, list) and m and isinstance(m[0], dict):
                c = m[0].get("content", "")
                if isinstance(c, list) and c and isinstance(c[0], dict):
                    c = c[0].get("text", "")
                s = str(c)
            else:
                s = ""
        outs.append(s)
    return outs


def load_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for ex in data:
        for c in ex.get("conversations", []):
            if c.get("from") == "human":
                out.append(c["value"]) ; break
    return out


# ===================== forward & feature collection =====================

@torch.no_grad()
def tokenize_with_anchors(prompts, proc, max_len, k_user: int = 8, family: str = "llama"):
    """Return tokenized inputs + last-K user anchors.
    proc: object returned by load_textizer (Processor or Tokenizer)
    family: 'llama' or 'qwen' (controls message schema)
    """
    messages = make_messages(prompts, family)

    chat_no_gen   = apply_chat_to_strings(proc, messages, add_generation_prompt=False)
    chat_with_gen = apply_chat_to_strings(proc, messages, add_generation_prompt=True)

    tok_no = proc(text=chat_no_gen,  return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len)
    tok_w  = proc(text=chat_with_gen, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len)

    len_no  = tok_no["attention_mask"].sum(dim=1).tolist()   # end of user
    len_with= tok_w["attention_mask"].sum(dim=1).tolist()    # actual forward length
    N = len(len_no); K = k_user

    anchors = []
    for i in range(N):
        end = int(len_no[i]) - 1
        idxs = [max(0, end - j) for j in range(K)]
        Lw = int(len_with[i]); idxs = [min(t, Lw-1) for t in idxs]
        anchors.append(sorted(set(idxs)))
        while len(anchors[-1]) < K:
            anchors[-1].append(anchors[-1][-1])
    anchors_mat = torch.tensor(anchors, dtype=torch.long)
    return tok_w, anchors_mat


@torch.no_grad()
def forward_collect_multi(model_dir: str,
                          shared_inputs: Dict[str, torch.Tensor],
                          anchors_mat: torch.Tensor,     # [N,K]
                          batch_size: int,
                          topk: int,
                          model_type: str = "llama",
                          use_final_norm: bool = True,
                          pool: str = "mean") -> Dict[str, torch.Tensor]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if model_type == "llama":
        model = MllamaForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=dtype,
            device_map="auto" if device=="cuda" else {"": "cpu"},
            low_cpu_mem_usage=True).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=dtype,
            device_map="auto" if device=="cuda" else {"": "cpu"},
            low_cpu_mem_usage=True).eval()

    # try to find final norm
    final_norm = None
    for path in ["language_model.model.norm","model.norm","transformer.ln_f","ln_f","norm"]:
        mod = model
        try:
            for p in path.split("."):
                mod = getattr(mod, p)
            final_norm = mod; break
        except AttributeError:
            pass

    inp_ids = shared_inputs["input_ids"]
    attn    = shared_inputs["attention_mask"]
    N, K = anchors_mat.size(0), anchors_mat.size(1)

    HB, HH, LG, ENT, TKID, TKP = [], [], [], [], [], []
    steps = math.ceil(N / batch_size)

    for i in range(steps):
        s, e = i*batch_size, min((i+1)*batch_size, N)
        ids   = inp_ids[s:e].to(model.device)
        mask  = attn[s:e].to(model.device)
        anchB = anchors_mat[s:e].to(model.device)    # [B,K]

        out = model(ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
        h_last = out.hidden_states[-1]               # [B,T,H]
        B, T, H = h_last.shape
        V = out.logits.size(-1)

        # expand gather indices for hidden [B,K,H] and logits [B,K,V]
        idx_h = anchB.view(B, K, 1).expand(B, K, H)
        idx_l = anchB.view(B, K, 1).expand(B, K, V)

        # hidden at block output
        h_blk = h_last.gather(dim=1, index=idx_h)    # [B,K,H]

        # hidden after final norm (what feeds head)
        if use_final_norm and final_norm is not None:
            h_norm_full = final_norm(h_last)         # [B,T,H]
            h_head = h_norm_full.gather(1, idx_h)    # [B,K,H]
        else:
            h_head = h_blk

        # logits at those positions
        logits_k = out.logits.gather(1, idx_l)       # [B,K,V]
        # pool across K anchors
        if pool == "median":
            h_blk_p  = h_blk.median(dim=1).values
            h_head_p = h_head.median(dim=1).values
            logits_p = logits_k.median(dim=1).values
        else:
            h_blk_p  = h_blk.mean(dim=1)
            h_head_p = h_head.mean(dim=1)
            logits_p = logits_k.mean(dim=1)

        probs  = torch.softmax(logits_p.float(), dim=-1)
        entropy= (-probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1)
        tk_p, tk_i = torch.topk(probs, k=min(topk, V), dim=-1)

        HB.append(h_blk_p.cpu())
        HH.append(h_head_p.cpu())
        LG.append(logits_p.half().cpu())  # save space
        ENT.append(entropy.cpu()); TKID.append(tk_i.cpu()); TKP.append(tk_p.cpu())

        if (i+1)%10==0 or (i+1)==steps:
            print(f"[{model_dir}] {e}/{N} (K={K}, pool={pool})")

    return {
        "hidden_block":     torch.cat(HB, 0),
        "hidden_into_head": torch.cat(HH, 0),
        "logits":           torch.cat(LG, 0),
        "entropy":          torch.cat(ENT,0),
        "topk_ids":         torch.cat(TKID,0),
        "topk_probs":       torch.cat(TKP,0),
    }

# ===================== metrics =====================

def cosine_vector(Ha: torch.Tensor, Hb: torch.Tensor) -> torch.Tensor:
    A = F.normalize(Ha.float(), dim=1)
    B = F.normalize(Hb.float(), dim=1)
    return (A*B).sum(dim=1)  # [N]

def cosine_stats(Ha: torch.Tensor, Hb: torch.Tensor) -> Dict[str, float]:
    cos = cosine_vector(Ha, Hb)
    return {
        "mean": float(cos.mean()),
        "p10":  float(cos.quantile(0.10)),
        "p50":  float(cos.median()),
        "p90":  float(cos.quantile(0.90)),
    }

def _cov(x: torch.Tensor) -> torch.Tensor:
    x = x.float(); x = x - x.mean(0, keepdim=True)
    return (x.T @ x) / max(1, (x.size(0) - 1))

def _sqrtm_psd(mat: torch.Tensor) -> torch.Tensor:
    w, v = torch.linalg.eigh(mat.float())
    w = torch.clamp(w, min=0)
    return (v * torch.sqrt(w)) @ v.T

def frechet_hidden(Ha: torch.Tensor, Hb: torch.Tensor) -> float:
    ma, mb = Ha.mean(0), Hb.mean(0)
    Ca, Cb = _cov(Ha), _cov(Hb)
    mean_term = torch.sum((ma - mb)**2)
    CaCb = _sqrtm_psd(Ca @ Cb)
    cov_term = torch.trace(Ca + Cb - 2*CaCb)
    return float(mean_term + cov_term)

def _center_gram(K: torch.Tensor) -> torch.Tensor:
    n = K.size(0)
    I = torch.eye(n, device=K.device, dtype=K.dtype)
    H = I - (1.0/n)
    return H @ K @ H

def linear_cka(Ha: torch.Tensor, Hb: torch.Tensor) -> float:
    Ha = Ha.float(); Hb = Hb.float()
    Ka = Ha @ Ha.T; Ka = _center_gram(Ka)
    Kb = Hb @ Hb.T; Kb = _center_gram(Kb)
    hsic_ab = torch.sum(Ka * Kb)
    denom = torch.sqrt(torch.sum(Ka*Ka) * torch.sum(Kb*Kb) + 1e-12)
    return float(hsic_ab / denom)

def mahalanobis_center(Ha: torch.Tensor, Hb: torch.Tensor, eps: float = 1e-3) -> float:
    ma, mb = Ha.mean(0), Hb.mean(0)
    Ca, Cb = _cov(Ha), _cov(Hb)
    Cp = 0.5*(Ca + Cb)
    scale = torch.trace(Cp)/Cp.size(0)
    Cp = Cp + (eps*scale)*torch.eye(Cp.size(0), device=Cp.device, dtype=Cp.dtype)
    delta = (ma - mb).unsqueeze(1).to(dtype=Cp.dtype)
    sol = torch.linalg.solve(Cp, delta)
    d2 = (delta * sol).sum()
    return float(torch.sqrt(torch.clamp(d2, min=0)))

def cov_frobenius_gap(Ha: torch.Tensor, Hb: torch.Tensor) -> float:
    Ca, Cb = _cov(Ha), _cov(Hb)
    return float(torch.linalg.norm(Ca - Cb, ord="fro"))

def principal_angles_summary(Ha: torch.Tensor, Hb: torch.Tensor, r: int = 64) -> Dict[str, float]:
    def top_basis(X):
        Xc = (X.float() - X.float().mean(0, keepdim=True))
        U,S,Vh = torch.linalg.svd(Xc, full_matrices=False)
        r_use = min(r, Vh.size(0))
        return Vh[:r_use].T        # [H, r_use]
    Qa = top_basis(Ha); Qb = top_basis(Hb)
    M = Qa.T @ Qb
    s = torch.linalg.svdvals(M).clamp(-1,1)
    theta = torch.arccos(s).cpu() * (180.0/3.141592653589793)
    return {"r_used": int(s.numel()),
            "mean_deg": float(theta.mean()),
            "p90_deg":  float(theta.quantile(0.90)),
            "max_deg":  float(theta.max())}

def mmd_rbf(Ha: torch.Tensor, Hb: torch.Tensor, gamma: float = None) -> float:
    A = Ha.float(); B = Hb.float()
    with torch.no_grad():
        if gamma is None:
            idx_a = torch.randperm(A.size(0))[:min(256, A.size(0))]
            idx_b = torch.randperm(B.size(0))[:min(256, B.size(0))]
            X = torch.cat([A[idx_a], B[idx_b]], dim=0)
            d2 = torch.cdist(X, X, p=2.0).pow(2)
            med = torch.median(d2[d2>0]).item() if (d2>0).any() else 1.0
            gamma = 1.0 / max(med, 1e-6)
    def kxx(X):
        d2 = torch.cdist(X, X, p=2.0).pow(2)
        K = torch.exp(-gamma * d2)
        n = X.size(0)
        return (K.sum() - K.diag().sum()) / (n*(n-1)+1e-12)
    def kxy(X,Y):
        d2 = torch.cdist(X, Y, p=2.0).pow(2)
        return torch.exp(-gamma * d2).mean()
    return float(kxx(A) + kxx(B) - 2*kxy(A,B))

def chamfer_mean_nn(Ha: torch.Tensor, Hb: torch.Tensor) -> float:
    d_ab = torch.cdist(Ha.float(), Hb.float(), p=2.0)  # [Na,Nb]
    a2b = d_ab.min(dim=1).values.mean()
    b2a = d_ab.min(dim=0).values.mean()
    return float(0.5*(a2b + b2a))

# ----- logits metrics -----

def js_mean_from_logits_T(La: torch.Tensor, Lb: torch.Tensor, T: float = 1.0) -> float:
    LaT, LbT = La.float()/T, Lb.float()/T
    logpa = torch.log_softmax(LaT, dim=-1)
    logpb = torch.log_softmax(LbT, dim=-1)
    pa, pb = logpa.exp(), logpb.exp()
    m = 0.5*(pa + pb)
    js = 0.5*( torch.sum(pa*(logpa - torch.log(m+1e-12)), dim=-1)
             + torch.sum(pb*(logpb - torch.log(m+1e-12)), dim=-1) )
    return float(js.mean())

def hellinger_mean_from_logits_T(La: torch.Tensor, Lb: torch.Tensor, T: float = 1.0) -> float:
    Pa = torch.softmax(La.float()/T, dim=-1)
    Pb = torch.softmax(Lb.float()/T, dim=-1)
    s = torch.sqrt(Pa) - torch.sqrt(Pb)
    H = torch.clamp(s*s, min=0).sum(dim=-1).sqrt() / math.sqrt(2.0)  # [N]
    return float(H.mean())

def js_vector_from_logits(La: torch.Tensor, Lb: torch.Tensor) -> torch.Tensor:
    logpa = torch.log_softmax(La.float(), dim=-1)
    logpb = torch.log_softmax(Lb.float(), dim=-1)
    pa, pb = logpa.exp(), logpb.exp()
    m = 0.5*(pa + pb)
    js = 0.5*( torch.sum(pa*(logpa - torch.log(m+1e-12)), dim=-1)
             + torch.sum(pb*(logpb - torch.log(m+1e-12)), dim=-1) )
    return js

def topk_overlap_curve(La: torch.Tensor, Lb: torch.Tensor, k_max: int = 50) -> List[float]:
    Va = La.size(-1); k_max = min(k_max, Va)
    _, ra = torch.topk(La.float(), k=k_max, dim=-1)
    _, rb = torch.topk(Lb.float(), k=k_max, dim=-1)
    curves = []
    for k in range(1, k_max+1):
        inter = torch.tensor([len(set(a[:k].tolist()) & set(b[:k].tolist())) for a,b in zip(ra,rb)])
        union = torch.tensor([len(set(a[:k].tolist()) | set(b[:k].tolist())) for a,b in zip(ra,rb)])
        curves.append(float((inter.float()/union.float()).mean()))
    return curves

def spearman_topM_from_logits(La: torch.Tensor, Lb: torch.Tensor, M: int = 1000) -> float:
    Na, V = La.shape
    M = min(M, V)
    rho_all = []
    for i in range(Na):
        a = La[i].float(); b = Lb[i].float()
        _, ta = torch.topk(a, k=M); _, tb = torch.topk(b, k=M)
        union = torch.unique(torch.cat([ta, tb], dim=0))
        av = a[union]; bv = b[union]
        ra = torch.argsort(torch.argsort(av))
        rb = torch.argsort(torch.argsort(bv))
        ra = ra.float(); rb = rb.float()
        ra = (ra - ra.mean()) / (ra.std(unbiased=False) + 1e-12)
        rb = (rb - rb.mean()) / (rb.std(unbiased=False) + 1e-12)
        rho = float((ra*rb).mean())
        rho_all.append(rho)
    return float(torch.tensor(rho_all).mean())

# ---------- NEW: Energy distance (hidden) ----------

def energy_distance(Ha: torch.Tensor, Hb: torch.Tensor, sample: int = 512, seed: int = 0) -> float:
    set_seed(seed)
    A = Ha.float(); B = Hb.float()
    if sample > 0:
        idx_a = torch.randperm(A.size(0))[:min(sample, A.size(0))]
        idx_b = torch.randperm(B.size(0))[:min(sample, B.size(0))]
        A = A[idx_a]; B = B[idx_b]
    d_ab = torch.cdist(A, B, p=2.0)
    d_aa = torch.cdist(A, A, p=2.0)
    d_bb = torch.cdist(B, B, p=2.0)
    return float(2*d_ab.mean() - d_aa.mean() - d_bb.mean())

# ===================== projections & plots =====================

def pca_nd(X: torch.Tensor, k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    X = X.float()
    Xc = X - X.mean(0, keepdim=True)
    U,S,Vh = torch.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vh[:k].T
    explained = (S[:k]**2 / (S**2).sum()).cumsum(0)
    return Z, explained  # [N,k], [k]

# ---------- Recovery visuals ----------

def plot_recovery_bars(deltas: Dict[str, float], out_png: str, title: str):
    keys = list(deltas.keys()); vals = [deltas[k] for k in keys]
    colors = ["tab:green" if v>0 else "tab:red" for v in vals]
    plt.figure(figsize=(10,4))
    plt.axhline(0, lw=1, color="gray")
    plt.bar(range(len(keys)), vals, tick_label=keys, alpha=.85, edgecolor="black", linewidth=.6, color=colors)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Δ (RL closer than SFT → +)")
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
    print(f"[plot] {out_png}")


def plot_winrate(winrates: Dict[str, float], out_png: str, title: str):
    keys = list(winrates.keys()); vals = [100.0*winrates[k] for k in keys]
    plt.figure(figsize=(6,3.5))
    plt.bar(range(len(keys)), vals, tick_label=keys, alpha=.85, edgecolor="black", linewidth=.6)
    plt.ylim(0,100)
    plt.ylabel("RL closer than SFT (% prompts)")
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
    print(f"[plot] {out_png}")


def plot_centroid_arrows(H_by: Dict[str, torch.Tensor], out_png: str, title: str):
    H_all = torch.cat([H_by[k] for k in H_by], 0)
    Z2, _ = pca_nd(H_all, k=2)
    n_sftmax = H_by['sft_max'].size(0)
    n_sft    = H_by['sft'].size(0)
    Z_sftmax = Z2[:n_sftmax]
    Z_sft    = Z2[n_sftmax:n_sftmax+n_sft]
    Z_rl     = Z2[n_sftmax+n_sft:]

    C = { 'sft_max': Z_sftmax.mean(0), 'sft': Z_sft.mean(0), 'rl': Z_rl.mean(0) }
    plt.figure(figsize=(6,5))
    for lab, Z in [('sft_max', Z_sftmax), ('sft', Z_sft), ('rl', Z_rl)]:
        z = Z.cpu().numpy()
        plt.scatter(z[:,0], z[:,1], s=6, alpha=.25, label=lab)
        plt.scatter([C[lab][0]],[C[lab][1]], s=60, marker='X')
    def arr(a,b,lab):
        plt.arrow(C[a][0], C[a][1], (C[b]-C[a])[0], (C[b]-C[a])[1],
                  head_width=0.1, length_includes_head=True, alpha=.9)
        plt.text(C[b][0], C[b][1], f" {lab}")
    arr('sft','rl','RL'); arr('rl','sft_max','SFT_MAX')
    plt.legend(); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
    print(f"[plot] {out_png}")


def plot_topk_overlay(curve_sft: List[float], curve_rl: List[float], out_png: str, title: str):
    ks = list(range(1, len(curve_sft)+1))
    plt.figure(figsize=(6,4))
    plt.plot(ks, curve_sft, label='SFT_MAX vs SFT', linewidth=2)
    plt.plot(ks, curve_rl,  label='SFT_MAX vs RL',  linewidth=2)
    plt.xlabel('k'); plt.ylabel('Jaccard@k (top-k index sets)')
    plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
    print(f"[plot] {out_png}")


def plot_entropy_margin(entropies: Dict[str, torch.Tensor], margins: Dict[str, Tuple[float,float]], out_png: str, title: str):
    labels = ['sft_max','sft','rl']
    ent_means = [float(entropies[l].mean()) for l in labels]
    marg_means = [margins[l][0] for l in labels]
    x = range(len(labels))
    plt.figure(figsize=(6.5,4))
    plt.bar(x, ent_means, alpha=.85, edgecolor='black', linewidth=.6)
    plt.xticks(x, labels)
    plt.ylabel('Mean entropy (T=1)'); plt.title(title+" — entropy")
    plt.tight_layout(); plt.savefig(out_png.replace('.png','_entropy.png'), dpi=200); plt.close()

    plt.figure(figsize=(6.5,4))
    plt.bar(x, marg_means, alpha=.85, edgecolor='black', linewidth=.6)
    plt.xticks(x, labels)
    plt.ylabel('Mean top-1 margin'); plt.title(title+" — margin")
    plt.tight_layout(); plt.savefig(out_png.replace('.png','_margin.png'), dpi=200); plt.close()
    print(f"[plot] {out_png.replace('.png','_entropy.png')} | {out_png.replace('.png','_margin.png')}")

# ===================== driver =====================

def margin_stats_from_logits(L: torch.Tensor) -> Tuple[float,float]:
    vals, _ = torch.topk(L.float(), k=2, dim=-1)
    margins = vals[:,0] - vals[:,1]
    return float(margins.mean()), float(margins.std(unbiased=False))


def main():
    cfg = DEFAULTS.copy()
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['llama','qwen'], default=cfg['model'])
    ap.add_argument('--ckpt_sft_max', type=str, default=cfg['ckpt_sft_max'])
    ap.add_argument('--ckpt_sft',     type=str, default=cfg['ckpt_sft'])
    ap.add_argument('--ckpt_rl',      type=str, default=cfg['ckpt_rl'])
    ap.add_argument('--shared_proc',  type=str, default=cfg['shared_proc'])
    ap.add_argument('--prompt_json',  type=str, default=cfg['prompt_json'])
    ap.add_argument('--out_root',     type=str, default=cfg['out_root'])
    ap.add_argument('--batch_size',   type=int, default=cfg['batch_size'])
    ap.add_argument('--max_length',   type=int, default=cfg['max_length'])
    ap.add_argument('--top_k',        type=int, default=cfg['top_k'])
    ap.add_argument('--k_user',       type=int, default=cfg['k_user'])
    ap.add_argument('--pool',         type=str, default=cfg['pool'])
    ap.add_argument('--tsne_perp',    type=int, default=cfg['tsne_perp'])
    ap.add_argument('--js_temps',     type=float, nargs='+', default=cfg['js_temps'])
    ap.add_argument('--topk_curve_max', type=int, default=cfg['topk_curve_max'])
    ap.add_argument('--spearman_M',   type=int, default=cfg['spearman_M'])
    ap.add_argument('--energy_sample',type=int, default=cfg['energy_sample'])
    ap.add_argument('--seed',         type=int, default=cfg['seed'])
    ap.add_argument('--make_plots',   action='store_true', help='If set, also generate plots. Default: print only.')
    args = ap.parse_args()
    set_seed(args.seed)

    TAG = ('OOD' if 'ood' in os.path.basename(args.prompt_json).lower() else 'ID') + f"_{args.model}"

    os.makedirs(args.out_root, exist_ok=True)
    prompts = load_prompts(args.prompt_json)
    print(f"[data] {len(prompts)} prompts ({TAG})")

    proc = load_textizer(args.shared_proc)

    shared_tok, anchors_mat = tokenize_with_anchors(prompts, proc, args.max_length, k_user=args.k_user, family=args.model)
    shared_tok = {k: v for k,v in shared_tok.items()}  # keep on CPU

    models = {
        "sft_max": args.ckpt_sft_max,
        "sft":     args.ckpt_sft,
        "rl":      args.ckpt_rl,
    }

    feats = {}
    for lab, path in models.items():
        print(f"==== {lab} ====")
        f = forward_collect_multi(path, shared_tok, anchors_mat, args.batch_size, args.top_k,
                                  model_type=args.model, use_final_norm=True, pool=args.pool)
        torch.save(f, os.path.join(args.out_root, f"{lab}_features.pt"))
        feats[lab] = f

    # choose rep & logits
    H = {lab: feats[lab]["hidden_into_head"] for lab in models.keys()}
    L = {lab: feats[lab]["logits"].float()   for lab in models.keys()}

    # ---------- pairwise summary (includes new metrics) ----------
    pairs = [("sft_max","sft"), ("sft_max","rl"), ("sft","rl")]
    summary = {}
    for a,b in pairs:
        Ha, Hb = H[a], H[b]
        La, Lb = L[a], L[b]
        cos = cosine_stats(Ha, Hb)
        fid = frechet_hidden(Ha, Hb)
        cka = linear_cka(Ha, Hb)
        maha = mahalanobis_center(Ha, Hb, eps=1e-3)
        cov_gap = cov_frobenius_gap(Ha, Hb)
        ang = principal_angles_summary(Ha, Hb, r=64)
        mmd = mmd_rbf(Ha, Hb)
        chamfer = chamfer_mean_nn(Ha, Hb)
        js_T = {f"JS@T={T}": js_mean_from_logits_T(La, Lb, T=T) for T in args.js_temps}
        hell_T = {f"Hellinger@T={T}": hellinger_mean_from_logits_T(La, Lb, T=T) for T in [1.0]}
        spearman = spearman_topM_from_logits(La, Lb, M=args.spearman_M)
        energy = energy_distance(Ha, Hb, sample=args.energy_sample, seed=args.seed)
        summary[f"{a}_vs_{b}"] = {
            "cosine": cos,
            "frechet": fid,
            "cka": cka,
            "mahalanobis_center": maha,
            "cov_frobenius_gap": cov_gap,
            "principal_angles": ang,
            "mmd_rbf": mmd,
            "chamfer_mean_nn": chamfer,
            **js_T,
            **hell_T,
            "spearman_topM": spearman,
            "energy_distance": energy,
        }

    # Save summary JSON
    with open(os.path.join(args.out_root, f"summary_metrics_{TAG}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ---------- Console printouts (no plots by default) ----------
    def print_pair(key: str):
        v = summary[key]
        js_keys = [k for k in v.keys() if k.startswith('JS@T=')]
        js_keys = sorted(js_keys, key=lambda s: float(s.split('=')[1]))
        ang = v['principal_angles']
        line_js = " | ".join([f"{k}={v[k]:.4f}" for k in js_keys])
        print(f"\n== {key} ==")
        print(f"cos.mean={v['cosine']['mean']:.4f}  CKA={v['cka']:.4f}  FID={v['frechet']:.1f}")
        print(f"Mahalanobis={v['mahalanobis_center']:.2f}  Cov||·||_F={v['cov_frobenius_gap']:.1f}  MMD={v['mmd_rbf']:.4f}  ChamferNN={v['chamfer_mean_nn']:.2f}")
        print(f"Angles(mean/p90/max)={ang['mean_deg']:.1f}/{ang['p90_deg']:.1f}/{ang['max_deg']:.1f}°")
        print(line_js)
        if 'Hellinger@T=1.0' in v:
            print(f"Hellinger@T=1.0={v['Hellinger@T=1.0']:.4f}")
        print(f"Spearman(top{args.spearman_M} union)={v['spearman_topM']:.4f}")
        if 'energy_distance' in v:
            print(f"Energy={v['energy_distance']:.4f}")

    for key in ["sft_max_vs_sft","sft_max_vs_rl","sft_vs_rl"]:
        print_pair(key)

    # ---------- Recovery index deltas ----------
    def get_val(pair_key: str, metric_key: str):
        v = summary[pair_key]
        if metric_key == 'cos.mean': return v['cosine']['mean']
        if metric_key == 'cka': return v['cka']
        if metric_key == 'spearman_topM': return v['spearman_topM']
        if metric_key == 'JS@T=1.0': return v['JS@T=1.0']
        if metric_key == 'JS@T=1.5': return v['JS@T=1.5']
        if metric_key == 'JS@T=2.0': return v['JS@T=2.0']
        if metric_key == 'Hellinger@T=1.0': return v['Hellinger@T=1.0']
        if metric_key == 'mahalanobis_center': return v['mahalanobis_center']
        if metric_key == 'frechet': return v['frechet']
        if metric_key == 'mmd_rbf': return v['mmd_rbf']
        if metric_key == 'cov_frobenius_gap': return v['cov_frobenius_gap']
        if metric_key == 'chamfer_mean_nn': return v['chamfer_mean_nn']
        if metric_key == 'angles.mean': return v['principal_angles']['mean_deg']
        if metric_key == 'energy_distance': return v['energy_distance']
        raise KeyError(metric_key)

    metric_cfg = {
        'cos.mean':            ('sim', True),
        'cka':                 ('sim', True),
        'spearman_topM':       ('sim', True),
        'JS@T=1.0':            ('dist', True),
        'JS@T=1.5':            ('dist', True),
        'JS@T=2.0':            ('dist', True),
        'Hellinger@T=1.0':     ('dist', True),
        'mahalanobis_center':  ('dist', True),
        'frechet':             ('dist', False),
        'mmd_rbf':             ('dist', True),
        'cov_frobenius_gap':   ('dist', False),
        'chamfer_mean_nn':     ('dist', True),
        'angles.mean':         ('dist', False),
        'energy_distance':     ('dist', True),
    }

    deltas = {}
    for mk,(kind,show) in metric_cfg.items():
        if not show: continue
        val_sft = get_val('sft_max_vs_sft', mk)
        val_rl  = get_val('sft_max_vs_rl',  mk)
        delta = (val_rl - val_sft) if kind=='sim' else (val_sft - val_rl)
        deltas[mk] = float(delta)

    print("\n-- Recovery Δ (positive = RL closer to SFT_MAX than SFT) --")
    for k,v in deltas.items():
        print(f"{k}: {v:+.4f}")

    # ---------- Per‑prompt win‑rates ----------
    H = {lab: feats[lab]["hidden_into_head"] for lab in models.keys()}
    L = {lab: feats[lab]["logits"].float()   for lab in models.keys()}
    cos_sft = cosine_vector(H['sft_max'], H['sft'])
    cos_rl  = cosine_vector(H['sft_max'], H['rl'])
    js_sft  = js_vector_from_logits(L['sft_max'], L['sft'])
    js_rl   = js_vector_from_logits(L['sft_max'], L['rl'])
    win_cos = float((cos_rl > cos_sft).float().mean())
    win_js  = float((js_rl  < js_sft ).float().mean())
    print(f"\nWin‑rates (RL closer): cosine={100*win_cos:.1f}%  JS@T=1.0={100*win_js:.1f}%")

    # ---------- Optional plots ----------
    if args.make_plots:
        # Recovery bars
        bar_png = os.path.join(args.out_root, f"recovery_bar_{TAG}.png")
        plot_recovery_bars(deltas, bar_png, title=f"Recovery Δ (positive = RL closer) — {TAG}")

        # Winrate plot
        win_png = os.path.join(args.out_root, f"winrate_{TAG}.png")
        plot_winrate({"cosine": win_cos, "JS@T=1.0": win_js}, win_png,
                     title=f"Per‑prompt RL‑closer win‑rate — {TAG}")

        # Top‑k overlap overlay
        curve_sft = topk_overlap_curve(L['sft_max'], L['sft'], k_max=args.topk_curve_max)
        curve_rl  = topk_overlap_curve(L['sft_max'], L['rl'],  k_max=args.topk_curve_max)
        topk_png  = os.path.join(args.out_root, f"topk_overlap_overlay_{TAG}.png")
        plot_topk_overlay(curve_sft, curve_rl, topk_png,
                          title=f"Top‑k overlap with SFT_MAX (higher = closer) — {TAG}")

        # PCA centroid drift arrows
        centroid_png = os.path.join(args.out_root, f"centroid_arrow_{TAG}.png")
        plot_centroid_arrows(H, centroid_png, title=f"Centroid drift on PCA plane — {TAG}")

        # Entropy & margin drift
        ent = {k: feats[k]['entropy'] for k in ['sft_max','sft','rl']}
        margins = {k: margin_stats_from_logits(L[k]) for k in ['sft_max','sft','rl']}
        drift_png = os.path.join(args.out_root, f"drift_entropy_margin_{TAG}.png")
        plot_entropy_margin(ent, margins, drift_png, title=f"Distributional sharpness — {TAG}")

    # ---------- Recovery index deltas ----------
    def get_val(pair_key: str, metric_key: str):
        v = summary[pair_key]
        if metric_key == 'cos.mean': return v['cosine']['mean']
        if metric_key == 'cka': return v['cka']
        if metric_key == 'spearman_topM': return v['spearman_topM']
        if metric_key == 'JS@T=1.0': return v['JS@T=1.0']
        if metric_key == 'JS@T=1.5': return v['JS@T=1.5']
        if metric_key == 'JS@T=2.0': return v['JS@T=2.0']
        if metric_key == 'Hellinger@T=1.0': return v['Hellinger@T=1.0']
        if metric_key == 'mahalanobis_center': return v['mahalanobis_center']
        if metric_key == 'frechet': return v['frechet']
        if metric_key == 'mmd_rbf': return v['mmd_rbf']
        if metric_key == 'cov_frobenius_gap': return v['cov_frobenius_gap']
        if metric_key == 'chamfer_mean_nn': return v['chamfer_mean_nn']
        if metric_key == 'angles.mean': return v['principal_angles']['mean_deg']
        if metric_key == 'energy_distance': return v['energy_distance']
        raise KeyError(metric_key)

    metric_cfg = {
        'cos.mean':            ('sim', True),
        'cka':                 ('sim', True),
        'spearman_topM':       ('sim', True),
        'JS@T=1.0':            ('dist', True),
        'JS@T=1.5':            ('dist', True),
        'JS@T=2.0':            ('dist', True),
        'Hellinger@T=1.0':     ('dist', True),
        'mahalanobis_center':  ('dist', True),
        'frechet':             ('dist', False),
        'mmd_rbf':             ('dist', True),
        'cov_frobenius_gap':   ('dist', False),
        'chamfer_mean_nn':     ('dist', True),
        'angles.mean':         ('dist', False),
        'energy_distance':     ('dist', True),
    }

    deltas = {}
    for mk,(kind,show) in metric_cfg.items():
        if not show: continue
        val_sft = get_val('sft_max_vs_sft', mk)
        val_rl  = get_val('sft_max_vs_rl',  mk)
        if kind == 'sim':
            delta = (val_rl - val_sft)
        else:
            delta = (val_sft - val_rl)
        deltas[mk] = float(delta)

    bar_png = os.path.join(args.out_root, f"recovery_bar_{TAG}.png")
    plot_recovery_bars(deltas, bar_png, title=f"Recovery Δ (positive = RL closer) — {TAG}")

    # ---------- Per‑prompt win‑rates on cosine & JS@T=1 ----------
    cos_sft = cosine_vector(H['sft_max'], H['sft'])
    cos_rl  = cosine_vector(H['sft_max'], H['rl'])
    js_sft  = js_vector_from_logits(L['sft_max'], L['sft'])
    js_rl   = js_vector_from_logits(L['sft_max'], L['rl'])
    win_cos = float((cos_rl > cos_sft).float().mean())
    win_js  = float((js_rl  < js_sft ).float().mean())
    win_png = os.path.join(args.out_root, f"winrate_{TAG}.png")
    plot_winrate({"cosine": win_cos, "JS@T=1.0": win_js}, win_png,
                 title=f"Per‑prompt RL‑closer win‑rate — {TAG}")

    # ---------- Top‑k overlap overlay ----------
    curve_sft = topk_overlap_curve(L['sft_max'], L['sft'], k_max=args.topk_curve_max)
    curve_rl  = topk_overlap_curve(L['sft_max'], L['rl'],  k_max=args.topk_curve_max)
    topk_png  = os.path.join(args.out_root, f"topk_overlap_overlay_{TAG}.png")
    plot_topk_overlay(curve_sft, curve_rl, topk_png,
                      title=f"Top‑k overlap with SFT_MAX (higher = closer) — {TAG}")

    # ---------- PCA centroid drift arrows ----------
    centroid_png = os.path.join(args.out_root, f"centroid_arrow_{TAG}.png")
    plot_centroid_arrows(H, centroid_png, title=f"Centroid drift on PCA plane — {TAG}")

    # ---------- Entropy & margin drift ----------
    ent = {k: feats[k]['entropy'] for k in ['sft_max','sft','rl']}
    margins = {k: margin_stats_from_logits(L[k]) for k in ['sft_max','sft','rl']}
    drift_png = os.path.join(args.out_root, f"drift_entropy_margin_{TAG}.png")
    plot_entropy_margin(ent, margins, drift_png, title=f"Distributional sharpness — {TAG}")

    rec_summary = {
        'tag': TAG,
        'metrics_delta_positive_is_RL_closer': deltas,
        'win_rates': {'cosine': win_cos, 'JS@T=1.0': win_js},
        'means': {
            'entropy': {k: float(ent[k].mean()) for k in ent},
            'margin_top1': {k: float(margins[k][0]) for k in margins},
        },
    }
    with open(os.path.join(args.out_root, f"recovery_summary_{TAG}.json"), 'w', encoding='utf-8') as f:
        json.dump(rec_summary, f, indent=2)
    print("[json] recovery summary saved.")

def load_mmd(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        s = json.load(f)
    mmd_sft = s["sft_max_vs_sft"]["mmd_rbf"]
    mmd_rl  = s["sft_max_vs_rl"]["mmd_rbf"]
    return float(mmd_sft), float(mmd_rl)

def infer_label(path):
    base = os.path.basename(path)
    # e.g., summary_metrics_ID_qwen.json -> ID_qwen
    if base.startswith("summary_metrics_") and base.endswith(".json"):
        return base[len("summary_metrics_"):-len(".json")]
    return os.path.splitext(base)[0]

def annotate_bars(ax):
    for p in ax.patches:
        h = p.get_height()
        if h is not None:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width()/2, h),
                        ha="center", va="bottom", fontsize=9, rotation=0, xytext=(0,2),
                        textcoords="offset points")

def load_mmd(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        s = json.load(f)
    mmd_sft = float(s["sft_max_vs_sft"]["mmd_rbf"])
    mmd_rl  = float(s["sft_max_vs_rl"]["mmd_rbf"])
    return mmd_sft, mmd_rl

def annotate_values(ax, fmt="{:.3f}", dy=2):
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(fmt.format(h), (p.get_x()+p.get_width()/2, h),
                    ha="center", va="bottom", fontsize=9, xytext=(0,dy),
                    textcoords="offset points")

def plot_mmd_levels(items, out_png):
    labels = [lab for lab, _, _ in items]
    mmd_sft = [a for _, a, _ in items]
    mmd_rl  = [b for _, _, b in items]
    x = range(len(labels)); width = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar([i - width/2 for i in x], mmd_sft, width=width,
           label="SFT_MAX vs SFT", edgecolor="black", linewidth=0.6, alpha=.9)
    ax.bar([i + width/2 for i in x], mmd_rl,  width=width,
           label="SFT_MAX vs RL",  edgecolor="black", linewidth=0.6, alpha=.9)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("MMD (hidden)"); ax.set_title("Hidden-state MMD (lower = closer to SFT_MAX)")
    ax.legend(loc="best"); annotate_values(ax)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close(fig)
    print(f"[plot] {out_png}")

def plot_mmd_delta(items, out_png):
    labels = [lab for lab, _, _ in items]
    deltas = [a - b for _, a, b in items]  # positive => RL closer
    colors = ["tab:green" if d>0 else "tab:red" for d in deltas]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axhline(0, color="gray", linewidth=1)
    bars = ax.bar(x, deltas, tick_label=labels, edgecolor="black", linewidth=0.6, alpha=.95, color=colors)
    for i,b in enumerate(bars):
        ax.annotate(f"{deltas[i]:+.3f}", (b.get_x()+b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=9, xytext=(0,2), textcoords="offset points")
    ax.set_ylabel("ΔMMD = MMD(SFT) − MMD(RL)")
    ax.set_title("Recovery via ΔMMD (positive = RL closer to SFT_MAX)")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close(fig)
    print(f"[plot] {out_png}")

def plot_mmd_dumbbell(items, out_png):
    # Slope/dumbbell: two points per row connected by a line; also show % improvement.
    labels = [lab for lab, _, _ in items]
    mmd_sft = [a for _, a, _ in items]
    mmd_rl  = [b for _, _, b in items]
    y = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, lab in enumerate(labels):
        ax.plot([mmd_rl[i], mmd_sft[i]], [i, i], '-', color='lightgray', zorder=1)
    ax.scatter(mmd_sft, y, label="SFT_MAX vs SFT", zorder=2)
    ax.scatter(mmd_rl,  y, label="SFT_MAX vs RL",  zorder=3)
    for i in y:
        if mmd_sft[i] != 0:
            rel = (mmd_sft[i]-mmd_rl[i]) / mmd_sft[i] * 100.0
            ax.annotate(f"{rel:+.1f}%", (max(mmd_sft[i], mmd_rl[i]), i),
                        va="center", ha="left", fontsize=9, xytext=(6,0), textcoords="offset points")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("MMD (hidden)  — lower = closer")
    ax.set_title("Hidden-state MMD: SFT_MAX vs SFT (●) and SFT_MAX vs RL (●)")
    ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close(fig)
    print(f"[plot] {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+", help="Paths to summary_metrics_*.json (ID/OOD × model).")
    ap.add_argument("--labels", nargs="*", help="Optional labels for each file (default: inferred from filename).")
    ap.add_argument("--out_dir", default=".", help="Where to save the plots.")
    ap.add_argument("--prefix", default="mmd", help="Filename prefix for plots.")
    args = ap.parse_args()

    if args.labels:
        assert len(args.labels) == len(args.summaries), "labels must match number of summaries"

    items = []
    for i, path in enumerate(args.summaries):
        label = args.labels[i] if args.labels else infer_label(path)
        a, b = load_mmd(path)
        items.append((label, a, b))

    os.makedirs(args.out_dir, exist_ok=True)
    plot_mmd_levels(items, os.path.join(args.out_dir, f"{args.prefix}_levels.png"))
    plot_mmd_delta(items,  os.path.join(args.out_dir, f"{args.prefix}_delta.png"))

def main1():
    summaries = [
        './analysis_hidden/summary_metrics_ID_qwen.json',
        './analysis_hidden/summary_metrics_OOD_qwen.json',
        './analysis_hidden/summary_metrics_ID_llama.json',
        './analysis_hidden/summary_metrics_OOD_llama.json',
    ]
    labels = ["Qwen-ID", "Qwen-OOD", "Llama-ID", "Llama-OOD"]

    items = []
    for label, path in zip(labels, summaries):
        a, b = load_mmd(path)
        items.append((label, a, b))

    out_dir = './analysis_hidden'
    os.makedirs(out_dir, exist_ok=True)
    plot_mmd_levels(items, os.path.join(out_dir, "mmd_recovery_levels.png"))
    plot_mmd_delta(items, os.path.join(out_dir, "mmd_recovery_delta.png"))
    plot_mmd_dumbbell(items, os.path.join(out_dir, "mmd_recovery_dumbbell.png"))

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main1()

