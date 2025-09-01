import os, glob, sys
import torch
from safetensors.torch import load_file
from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig
os.environ["ACCELERATE_DISABLE_DEEPSPEED"] = "1"
os.environ["TRANSFORMERS_NO_DEEPSPEED"] = "1"
sys.modules["deepspeed"] = None

# ===== paths =====
merged_file  = "/path/checkpoint-merge/checkpoint-1100.safetensors"
recover_dir  = "/path/checkpoint-1100"
base_model   = "/path/Llama-3.2-11B-Vision-Instruct"
shard_size   = "5GB"

TEST_PROMPT = """[Task Description]
You are an expert 24 points card game player. You will receive a set of 4 cards.
Note that 'J', 'Q', and 'K' count as '11', '12', '13', and each card must be used once.
Your goal is to output a formula that evaluates to 24 using numbers from the cards and operators such as '+', '-', '*', '/', '(', ')', and '='.

[Input]
Cards: ['2', 'Q', '7', 'Q']

[Output]
{
  "cards": [x, y, z, w], where 'J', 'Q', and 'K' count as '11','12','13',
  "number": [a, b, c, d], where a, b, c, and d are the numbers on the cards,
  "formula": "an equation that equals 24"
}
"""

TARGET_VOCAB = 128256

def build_llama_config_from_mm(mm, vocab_size):
    txt = getattr(mm.config, "text_config", None)
    if txt is None and hasattr(mm, "language_model") and hasattr(mm.language_model, "config"):
        txt = mm.language_model.config
    if txt is None:
        txt = mm.config
    def g(a, d=None): return getattr(txt, a, d)
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=int(g("hidden_size")),
        intermediate_size=int(g("intermediate_size")),
        num_hidden_layers=int(g("num_hidden_layers")),
        num_attention_heads=int(g("num_attention_heads")),
        num_key_value_heads=int(g("num_key_value_heads", g("num_attention_heads"))),
        rms_norm_eps=float(g("rms_norm_eps", 1e-5)),
        hidden_act=str(g("hidden_act", "silu")),
        max_position_embeddings=int(g("max_position_embeddings", 8192)),
        rope_theta=float(g("rope_theta", 500000.0)),
        rope_scaling=g("rope_scaling", None),
        attention_dropout=float(g("attention_dropout", 0.0)),
        hidden_dropout=float(g("hidden_dropout", 0.0)),
        tie_word_embeddings=True,
    )

def main():
    sd = load_file(merged_file)
    emb = sd["model.embed_tokens.weight"]
    V, H = emb.shape
    print(f"[merged] embed_tokens: {V}x{H}; lm_head:", sd.get("lm_head.weight", None).shape if "lm_head.weight" in sd else None)

    if V not in (TARGET_VOCAB, TARGET_VOCAB + 8):
        raise RuntimeError(f"Unexpected embedding rows {V}; expected {TARGET_VOCAB} or {TARGET_VOCAB+8}")

    if V == TARGET_VOCAB + 8:
        print(f"[fix] slicing extra {V - TARGET_VOCAB} rows -> {TARGET_VOCAB}")
        sd["model.embed_tokens.weight"] = emb[:TARGET_VOCAB, :]
    else:
        print("[fix] embedding already 128256")

    # Drop lm_head if size != TARGET_VOCAB; we’ll tie a new one
    if "lm_head.weight" in sd and sd["lm_head.weight"].shape[0] != TARGET_VOCAB:
        print("[fix] dropping mismatched lm_head for re-tie")
        sd.pop("lm_head.weight")

    # Build clean text model
    mm = MllamaForConditionalGeneration.from_pretrained(base_model, device_map={"": "cpu"})
    cfg = build_llama_config_from_mm(mm, TARGET_VOCAB)
    model = LlamaForCausalLM(cfg)

    # Load only model.* weights (+optional lm_head if kept)
    filtered = {k: v for k, v in sd.items() if k.startswith("model.") or k == "lm_head.weight"}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    model.tie_weights()

    # Numerical forward
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model.to(device=device, dtype=dtype).eval()
    x = torch.randint(0, TARGET_VOCAB, (1, 32), device=device)
    with torch.inference_mode():
        y = model(input_ids=x).logits
    print("[forward] logits shape:", tuple(y.shape))

    # Deterministic generation (only valid if tokenizer len==128256 and same id order)
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    print(f"[tok] len={len(tok)} vocab_size={tok.vocab_size}")
    ids = tok(TEST_PROMPT, return_tensors="pt").to(device)
    with torch.inference_mode():
        gen = model.generate(**ids, max_new_tokens=60, do_sample=False, temperature=0.0)
    print("\n[gen]\n" + tok.decode(gen[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()