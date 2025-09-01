import os, gc, torch, shutil, argparse, re
from tqdm import tqdm
from transformers import MllamaForConditionalGeneration, AutoTokenizer

from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--layers",  type=str, required=True)
parser.add_argument("--k_top",   type=int, default=4096)
parser.add_argument("--k_tail",  type=int, default=0)
args = parser.parse_args()
K_TOP, K_TAIL = args.k_top, args.k_tail

LAYERS_RAW = args.layers.strip()
def parse_layers(spec: str):
    layers = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = map(int, part.split('-'))
            layers.update(range(a, b + 1))
        else:
            layers.add(int(part))
    return layers

#LAYERS = parse_layers(args.layers)
LAYERS = parse_layers(LAYERS_RAW)
LAYERS_STR = "_".join(map(str, sorted(LAYERS)))

BASE_DIR   = "/path/SFT_MaxOOD"
TARGET_DIR = "/path/SFT_END"
OUTPUT_DIR = (
    f"{TARGET_DIR}-restored-layers-{LAYERS_RAW.replace(',','_')}--"
    f"{args.k_top}--{args.k_tail}"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[INFO] restoring layers {sorted(LAYERS)} → {OUTPUT_DIR}")

def svd_replace(Wb, Wt, k_top, k_tail):
    Ub, Sb, Vb = torch.linalg.svd(Wb.float(), full_matrices=False)
    Ut, St, Vt = torch.linalg.svd(Wt.float(), full_matrices=False)

    n = Ub.shape[1]
    k_top  = min(k_top,  n)
    k_tail = min(k_tail, n - k_top)
    if k_top == k_tail == 0:   return Wt
    if k_top + k_tail == n:    return Wb

    U_new = torch.cat([Ub[:, :k_top], Ut[:, k_top:n-k_tail], Ub[:, n-k_tail:]], dim=1)
    V_new = torch.cat([Vb[:k_top, :], Vt[k_top:n-k_tail, :], Vb[n-k_tail:, :]], dim=0)
    S_new = torch.cat([Sb[:k_top],   St[k_top:n-k_tail],    Sb[n-k_tail:]])

    return (U_new @ torch.diag(St) @ V_new).to(Wt.dtype)

def main():
    base = AutoModelForCausalLM.from_pretrained(
            BASE_DIR, torch_dtype=torch.float16, low_cpu_mem_usage=True,
                    # attn_implementation="flash_attention_2",
                            trust_remote_code=True)

    base_weights = {n: p.data.float().cpu()
                    for n, p in base.named_parameters() if p.ndim == 2}
    print(f"[DEBUG] base_weights matrices collected: {base_weights}")
    del base; gc.collect()

    target = AutoModelForCausalLM.from_pretrained(
            TARGET_DIR, torch_dtype=torch.float16, low_cpu_mem_usage=True,
                    trust_remote_code=True)

    always_replace = {"language_model.model.embed_tokens.weight", "language_model.lm_head.weight"}
    layer_pat = re.compile(r'model\.layers\.(\d+)\.')
    for n, p in tqdm(target.named_parameters(), desc="SVD restore"):
        if p.ndim != 2 or n not in base_weights:
            continue

        m = layer_pat.search(n)
        layer_id = int(m.group(1)) if m else None
        if (layer_id in LAYERS) or (n in always_replace):
            print(n)
            p.data.copy_(svd_replace(base_weights[n].cuda(),
                                     p.data.float().cuda(),
                                     K_TOP, K_TAIL).half().cpu())

    target.save_pretrained(OUTPUT_DIR, max_shard_size="5GB", safe_serialization=True)

    for f in [
        "config.json", "chat_template.json", "generation_config.json",
        "preprocessor_config.json", "special_tokens_map.json",
        # add tokenizer assets!
        "tokenizer.json", "tokenizer_config.json"
    ]:
        src = os.path.join(TARGET_DIR, f)
        if os.path.exists(src):
            shutil.copy(src, OUTPUT_DIR)

    # load fast tokenizer (default) and save into OUTPUT_DIR
    AutoTokenizer.from_pretrained(OUTPUT_DIR).save_pretrained(OUTPUT_DIR)

    print(f"[DONE] {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
