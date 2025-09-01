#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==== hard-coded paths/params ====
LOG_PATH = "/path/sft_67795_4294967294.out"
OUTDIR   = "./loss-trending"      # where to write plots/CSVs
ROLL_W   = 9             # rolling window for smoothing
PIVOT    = 60            # "after 60 checkpoints" view
# =================================

import os, re, ast
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 140, "axes.grid": True})

# --- parsing helpers ---
DICT_RE = re.compile(r"\{.*\}")
CKPT_RE = re.compile(r"checkpoint-(\d+)")

def extract_dicts_from_line(line: str) -> List[Dict[str, Any]]:
    out = []
    for m in DICT_RE.finditer(line):
        blob = m.group(0)
        try:
            d = ast.literal_eval(blob)
            if isinstance(d, dict):
                out.append(d)
        except Exception:
            pass
    return out

def parse_log(path: str) -> pd.DataFrame:
    recs = []
    last_ckpt = None
    eval_idx = 0
    train_idx = 0
    with open(path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            m = CKPT_RE.search(line)
            if m:
                try: last_ckpt = int(m.group(1))
                except: pass
            dicts = extract_dicts_from_line(line)
            if not dicts: continue
            for d in dicts:
                is_eval = any(str(k).startswith("eval") or str(k).startswith("eval/") for k in d)
                row = {}
                if is_eval:
                    row["phase"] = "eval"
                    row["eval_index"] = eval_idx
                    eval_idx += 1
                    if last_ckpt is not None:
                        row["ckpt_step"] = last_ckpt
                else:
                    row["phase"] = "train"
                    row["train_step"] = train_idx
                    train_idx += 1
                for k, v in d.items():
                    row[str(k)] = v
                recs.append(row)
    return pd.DataFrame.from_records(recs)

def coerce_series(df: pd.DataFrame, name: str):
    """Return a numeric Series aligned to df.index; if column missing, return empty Series."""
    if name not in df.columns:
        return pd.Series(dtype=float, index=df.index)
    s = df[name]
    # if a scalar sneaks in, broadcast to index length so plotting doesn't crash
    if not isinstance(s, pd.Series):
        return pd.Series([s]*len(df), index=df.index, dtype=float)
    return pd.to_numeric(s, errors="coerce")

def pick_metric(df: pd.DataFrame, prefer: str, fallback: str):
    """Pick whichever column actually has non-NaN values."""
    s1 = coerce_series(df, prefer)
    if s1.notna().any(): return s1
    s2 = coerce_series(df, fallback)
    return s2

def roll(s: pd.Series, w=ROLL_W):
    if s is None or s.empty: return s
    return s.rolling(window=w, min_periods=max(1, w//2)).mean()

def slope(x: pd.Series, y: pd.Series):
    m = ~(x.isna() | y.isna())
    if m.sum() < 2: return None
    return float(np.polyfit(x[m].values, y[m].values, 1)[0])

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"[analyze] reading {LOG_PATH}")
    df = parse_log(LOG_PATH)
    if df.empty:
        print("No dict-like rows found."); return

    df_eval  = df[df["phase"]=="eval"].copy()
    df_train = df[df["phase"]=="train"].copy()
    df_eval.to_csv(os.path.join(OUTDIR, "df_eval_from_log.csv"), index=False)
    df_train.to_csv(os.path.join(OUTDIR, "df_train_from_log.csv"), index=False)

    # X-axis
    x = pd.to_numeric(df_eval.get("eval_index"), errors="coerce")
    # Prefer underscore metrics; fall back to slash metrics if empty
    conf_mean = pick_metric(df_eval, "eval_seq_conf_mean", "eval/seq_conf_mean")
    conf_p90  = pick_metric(df_eval, "eval_seq_conf_p90",  "eval/seq_conf_p90")
    nll_mean  = pick_metric(df_eval, "eval_seq_nll_mean",  "eval/seq_nll_mean")
    eval_loss = coerce_series(df_eval, "eval_loss")  # may be mostly NaN or even scalar; we guard below

    # --- summary numbers ---
    def mean(s):
        if s is None or s.empty: return None
        ss = s.dropna()
        return float(ss.mean()) if len(ss) else None

    print(f"[analyze] eval points: {int(x.shape[0])}")
    for name, s in [("conf_mean", conf_mean), ("conf_p90", conf_p90), ("nll_mean", nll_mean)]:
        m = mean(s)
        if m is not None:
            print(f"[analyze] mean({name}) = {m:.4f}")

    # correlations
    m = conf_mean.notna() & nll_mean.notna()
    if m.sum() >= 2:
        r = float(np.corrcoef(conf_mean[m], nll_mean[m])[0,1])
        print(f"[analyze] corr(conf_mean, nll_mean) = {r:.3f} (expect negative)")

    # piecewise slopes around the pivot
    left  = x <= PIVOT
    right = x > PIVOT
    sL = slope(x[left],  nll_mean[left])
    sR = slope(x[right], nll_mean[right])
    if sL is not None and sR is not None:
        print(f"[analyze] slope(nll_mean) before {PIVOT}: {sL:+.4f} per eval, after: {sR:+.4f}")

    # --- plots ---
    # 1) scatter
    if m.sum() >= 2:
        plt.figure()
        plt.scatter(conf_mean[m], nll_mean[m], s=20)
        plt.xlabel("seq_conf_mean"); plt.ylabel("seq_nll_mean")
        plt.title("Confidence vs NLL (expect negative slope)")
        plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "conf_vs_nll.png")); plt.close()

    # 2) time series with rolling averages
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    if conf_mean.notna().any():
        ax1.plot(x, conf_mean, alpha=0.25, label="conf_mean")
        ax1.plot(x, roll(conf_mean), lw=2, label=f"conf_mean (roll={ROLL_W})")
    if conf_p90.notna().any():
        ax1.plot(x, conf_p90, alpha=0.25, label="conf_p90")
        ax1.plot(x, roll(conf_p90), lw=2, label=f"conf_p90 (roll={ROLL_W})")
    ax1.axvline(PIVOT, color="gray", ls="--", alpha=0.7)
    ax1.set_ylim(0, 1.0); ax1.set_ylabel("confidence")

    if nll_mean.notna().any():
        ax2.plot(x, nll_mean, color="tab:red", alpha=0.20, label="nll_mean")
        ax2.plot(x, roll(nll_mean), color="tab:red", lw=2, label=f"nll_mean (roll={ROLL_W})")
        ax2.set_ylabel("NLL", color="tab:red")

    # only plot eval_loss if there are enough points (avoid scalar/one-point crashes)
    if "eval_loss" in df_eval.columns:
        s = eval_loss.dropna()
        if len(s) >= 5 and len(s) == len(x):
            ax2.plot(x, s, color="tab:purple", alpha=0.2, label="eval_loss")
            ax2.plot(x, roll(s), color="tab:purple", lw=2, label=f"eval_loss (roll={ROLL_W})")

    ax1.set_xlabel("eval_index")
    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.10))
    fig.suptitle("Eval trends (rolling averages)")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "eval_trends_over_index.png")); plt.close(fig)

    # 3) training trends (if present)
    if not df_train.empty and "loss" in df_train.columns:
        ts = pd.to_numeric(df_train["train_step"], errors="coerce")
        tl = pd.to_numeric(df_train["loss"], errors="coerce")
        lr = pd.to_numeric(df_train.get("learning_rate"), errors="coerce")
        fig, ax1 = plt.subplots()
        ax1.plot(ts, tl, alpha=0.25, label="train loss")
        ax1.plot(ts, tl.rolling(ROLL_W, min_periods=max(1, ROLL_W//2)).mean(), lw=2, label=f"train loss (roll={ROLL_W})")
        ax1.set_xlabel("train_step"); ax1.set_ylabel("train loss")
        if lr is not None and lr.notna().any():
            ax2 = ax1.twinx()
            ax2.plot(ts, lr, color="tab:orange", alpha=0.3, label="lr")
            ax2.set_ylabel("learning rate", color="tab:orange")
        fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.10))
        fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "train_trends.png")); plt.close(fig)

    print(f"[analyze] outputs in: {OUTDIR}")

if __name__ == "__main__":
    main()