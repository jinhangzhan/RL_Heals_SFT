import os, gc, torch, shutil
from tqdm import tqdm
from transformers import MllamaForConditionalGeneration, AutoTokenizer
import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument('--k_top', type=int, default=100)
parser.add_argument('--k_tail', type=int, default=100)
args = parser.parse_args()

K_TOP, K_TAIL = args.k_top, args.k_tail
PARAMETER_SUFFIXES = (
    ".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight", ".mlp.up_proj.weight",
    ".mlp.gate_proj.weight", ".mlp.down_proj.weight",
)
ALWAYS_REPLACE = {
    "language_model.model.embed_tokens.weight", "language_model.lm_head.weight"
}

BASE_DIR   = "/path/SFT_MaxOOD"
TARGET_DIR = "/path/SFT_END"
OUTPUT_DIR = TARGET_DIR + f"-restored-vector--{K_TOP}--{K_TAIL}"

print(OUTPUT_DIR)

def svd_replace_gpu(Wb, Wt, k_top, k_tail):
    Ub, Sb, Vb = torch.linalg.svd(Wb.float(), full_matrices=False)
    Ut, St, Vt = torch.linalg.svd(Wt.float(), full_matrices=False)

    m, n = Ub.shape
    k_top = min(k_top, n)
    k_tail = min(k_tail, n - k_top)

    if k_tail == 0 and k_top == 0:
        return Wt
    if k_top + k_tail == n:
        return (Ut @ torch.diag(Sb) @ Vt).to(Wt.dtype)

    U_new = torch.cat([Ub[:, :k_top],
                       Ut[:, k_top:n-k_tail],
                       Ub[:, n-k_tail:]], dim=1)
    V_new = torch.cat([Vb[:k_top, :],
                       Vt[k_top:n-k_tail, :],
                       Vb[n-k_tail:, :]], dim=0)

    S_new = torch.cat([Sb[:k_top],
                       St[k_top:n-k_tail],
                       Sb[n-k_tail:]])

    return (Ut @ torch.diag(S_new) @ Vt).to(Wt.dtype)

def main():
    base = MllamaForConditionalGeneration.from_pretrained(
        BASE_DIR, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        trust_remote_code=True)
    base_weights = {n: p.data.float().cpu()
                    for n, p in base.named_parameters() if p.ndim == 2}
    del base; gc.collect(); torch.cuda.empty_cache()

    target = MllamaForConditionalGeneration.from_pretrained(
        TARGET_DIR, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        trust_remote_code=True)

    for n, p in tqdm(target.named_parameters(), "SVD restore"):
        if (p.ndim == 2 and n in base_weights and
                ((n.startswith("language_model.model.layers.") and
                  n.endswith(PARAMETER_SUFFIXES)) or n in ALWAYS_REPLACE)):
            p.data.copy_(svd_replace_gpu(base_weights[n].cuda(),
                                         p.data.float().cuda(),
                                         K_TOP, K_TAIL).half().cpu())

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    target.save_pretrained(OUTPUT_DIR, max_shard_size="5GB",
                           safe_serialization=True)

    for f in ["config.json", "chat_template.json", "generation_config.json",
              "preprocessor_config.json", "special_tokens_map.json"]:
        src = os.path.join(TARGET_DIR, f)
        if os.path.exists(src):
            shutil.copy(src, OUTPUT_DIR)

    AutoTokenizer.from_pretrained(TARGET_DIR, use_fast=False)\
                 .save_pretrained(OUTPUT_DIR)

    print("finished" + OUTPUT_DIR)

if __name__ == "__main__":
    main()
