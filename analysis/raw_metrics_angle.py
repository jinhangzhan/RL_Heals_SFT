
"""
Analyse checkpoints: singular spectrum + principal-angle (direction rotation) + 
activation-covariance overlap. 
"""
import os, json, math, torch, numpy as np
from pathlib import Path

from tqdm import tqdm
from safetensors.torch import save_file, load_file
from scipy.linalg import subspace_angles

CHECKPOINTS = [
    "/path/checkpoint-1100.safetensors",
]
BASE_CKPT = "/path/checkpoint-140.safetensors"

RESULT_JSON = "/path/stats_align_angle-0729-sft-slight-llama.json"

TOP_RANK_UV = 768
DEVICE_SVD  = "cuda"
DTYPE_SVD   = torch.float32

KEY_PATTERNS = (
    ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight", ".mlp.down_proj.weight"
)

def param_wanted(name: str) -> bool:
    return name.endswith(KEY_PATTERNS)

@torch.no_grad()
def truncated_svd(mat: torch.Tensor, k=TOP_RANK_UV):
    k = min(TOP_RANK_UV, *mat.shape)
    # r = torch.linalg.matrix_rank(mat).item()
    # k = min(TOP_RANK_UV, r, *mat.shape)
    mat_gpu = mat.to(device=DEVICE_SVD, dtype=DTYPE_SVD)
    U, S, Vt = torch.linalg.svd(mat_gpu, full_matrices=False)
    V = Vt.T[:, :k].contiguous()
    # return U[:, :k].cpu(), S[:k].cpu(), V.cpu()
    return U[:, :k], S[:k], V
    # return U[:, :k].cpu(), S[:TOP_SINGVAL].cpu(), Vt.T.contiguous()[:k, :].cpu()

print(f"[INFO] Load baseline {BASE_CKPT}")
# base_model = AutoModelForCausalLM.from_pretrained(
#     BASE_CKPT, torch_dtype=torch.float16, device_map="auto")
# base_model.eval()
base_model = load_file(BASE_CKPT)

cache = {}

BASE_UV = {}          # param_name -> (U[:k], V[:k])
BASE_S  = {}          # param_name -> full singular values
BASE_PARAMS = {}
# ACM_EIG = {}          # layer_name -> eigvecs[:,m]

def principal_angles(A, B):
    s = torch.linalg.svdvals(A.T @ B)
    s = s.clamp(-1.0, 1.0)
    return torch.arccos(s)

def proj_F_distance(Ua, Ub):
    # Pa, Pb = Ua @ Ua.T, Ub @ Ub.T
    # return torch.norm(Pa - Pb, p='fro').item()
    C = Ua.T @ Ub
    return torch.sqrt(2 * Ua.shape[1] - 2 * torch.norm(C, p='fro') ** 2).item()

for name in tqdm(base_model.keys(), desc="Baseline SVD"):
    param = base_model[name]
    if not param_wanted(name) or param.ndim != 2:
        continue

    U_b, S_b, V_b = truncated_svd(param)
    BASE_UV[name] = (U_b, V_b)
    BASE_S [name] = S_b
    BASE_PARAMS[name] = param.clone()

del base_model
torch.cuda.empty_cache()

def to_py(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.cpu().tolist()
    return obj

def in_subspace_energy(Ub, Vb, dW):
    proj = Ub.T @ dW @ Vb
    num  = torch.norm(proj, p='fro')**2
    den  = torch.norm(dW,  p='fro')**2 + 1e-12
    return (num / den).item()

def pearson_corr(a: torch.Tensor, b: torch.Tensor):
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    a_mean = a_flat.mean()
    b_mean = b_flat.mean()
    num = torch.dot(a_flat - a_mean, b_flat - b_mean)
    den = torch.norm(a_flat - a_mean) * torch.norm(b_flat - b_mean) + 1e-12
    return (num / den).item()

# def procrustes_R(A, B):
#     U, _, Vt = torch.linalg.svd(A.T @ B, full_matrices=False)
#     return U @ Vt
def procrustes_R(A, B):
    dev = A.device
    U, _, Vt = torch.linalg.svd((A.T @ B).to(dev), full_matrices=False)
    return (U @ Vt).to(dev)


def align_weight(param_t, Ub, Ut, side="right"):
    """side='right' → W_t @ R^T ； side='left' → R @ W_t"""
    R = procrustes_R(Ub, Ut)
    param_t = param_t.to(R.device)
    return param_t @ R.T if side=="right" else R @ param_t

def align_subspace(B_base, B_tgt):
    U, _, Vt = torch.linalg.svd(B_base.T @ B_tgt, full_matrices=False)
    return B_tgt @ (U @ Vt)

def svd_delta_layer(param_t, param_b, k=TOP_RANK_UV):
    dW = (param_t - param_b).to(dtype=DTYPE_SVD, device=DEVICE_SVD)
    k = min(k, *dW.shape)
    Ud, Sd, Vd_t = torch.linalg.svd(dW, full_matrices=False)
    return Ud[:, :k], Sd[:k], Vd_t.T[:, :k], dW

def projection_energy(Ub, Vb, dW):
    C = Ub.T @ dW @ Vb         # k×k
    diag_c = torch.diag(C)
    fro2 = torch.norm(C, p='fro').pow(2)
    diag_fro2 = torch.norm(diag_c).pow(2)
    dW_fro2 = torch.norm(dW, p='fro').pow(2) + 1e-12
    return diag_fro2 / dW_fro2, (fro2 - diag_fro2) / dW_fro2

from collections import defaultdict
all_U_angles = defaultdict(list)
all_V_angles = defaultdict(list)
@torch.no_grad()
def main():
    result = {}

    for ckpt in CHECKPOINTS:
        tag = os.path.basename(ckpt)  # "checkpoint-140"
        print(f"[INFO] analyse {tag}")
        # model = AutoModelForCausalLM.from_pretrained(
        #     ckpt, torch_dtype=torch.float16, device_map="cuda:0")
        # model.eval()
        model = load_file(ckpt)
        result[tag] = {}
        reg_loss = []
        diff_val = 0.0
        # diff_val_ori = 0.0
        res_total = 0.0
        w_sq_total = 0.0
        proj_dist_U_total = 0.0
        proj_dist_V_total = 0.0
        align_U_arr = []
        align_V_arr = []
        # for name, param in tqdm(model.named_parameters(), desc=f"SVD-{tag}"):
        for name in tqdm(model.keys(), desc=f"SVD-{tag}"):
            param = model[name]
            if not param_wanted(name) or param.ndim != 2:
                continue
            # param_base = BASE_PARAMS[name]

            # basic stats
            t = param.float()
            stats = dict(
                min=float(t.min()), max=float(t.max()),
                mean=float(t.mean()), std=float(t.std()),
                fro_norm=float(torch.norm(t, p='fro'))
            )

            # SVD (truncated)
            U_t, S_t, V_t = truncated_svd(param)
            U_b, V_b = BASE_UV[name]
            S_b = BASE_S[name]

            Ud, Sd, Vd, dW = svd_delta_layer(param, BASE_PARAMS[name])
            diag_ratio, rot_ratio = projection_energy(U_b, V_b, dW)

            # principal-angle
            rel_s = (S_t - S_b[:S_t.numel()]) / (S_b[:S_t.numel()] + 1e-12)
            # result[tag][name]["delta_singular_values"] = Sd.cpu().tolist()
            # result[tag][name]["delta_energy_diag"] = float(diag_ratio)
            # result[tag][name]["delta_energy_rot"] = float(rot_ratio)
            # result[tag][name]["rel_singular_change"] = rel_s.cpu().tolist()

            # result = torch.allclose(U_b.T @ U_b, torch.eye(TOP_RANK_UV, device=U_b.device, dtype=U_b.dtype), atol=1e-3)
            THRESHOLD = 1e-3
            U_angles = principal_angles(U_b, U_t)         #TODO original
            # U_angles = subspace_angles(U_b.cpu(), U_t.cpu())
            # U_angles = torch.from_numpy(U_angles)
            V_angles = principal_angles(V_b, V_t)
            # U_angles = torch.where(U_angles.abs() <= THRESHOLD, torch.zeros_like(U_angles), U_angles)
            # V_angles = subspace_angles(V_b.cpu(), V_t.cpu())
            # V_angles = torch.from_numpy(V_angles)
            # V_angles = torch.where(V_angles.abs() <= THRESHOLD, torch.zeros_like(V_angles), V_angles)
            all_U_angles[tag].append(U_angles.cpu())
            all_V_angles[tag].append(V_angles.cpu())
            diff = (U_b - U_t).pow(2).sum() + (V_b - V_t).pow(2).sum()
            diff_val += diff
            proj_dist_U = torch.linalg.matrix_norm(U_b @ U_b.T - U_t @ U_t.T, ord='fro')
            proj_dist_V = torch.linalg.matrix_norm(V_b @ V_b.T - V_t @ V_t.T, ord='fro')
            align_U = torch.abs(U_b.T @ U_t).max(dim=1).values
            align_V = torch.abs(V_b.T @ V_t).max(dim=1).values
            proj_dist_U_total += proj_dist_U
            proj_dist_V_total += proj_dist_V
            align_U_arr.append(align_U.mean())
            align_V_arr.append(align_V.mean())
            print("diff===", name, proj_dist_U, proj_dist_V, align_U.mean(), align_V.mean())

            # print("==layer-diff==", diff_val, str(diff), name, U_b.shape, V_b.shape)
            # proj = U_b.T @ param.to(U_b.device) @ V_b  # [k,k]
            # ratio = proj.float().pow(2).sum() / (param.float().pow(2).sum() + 1e-12)
            # print("==ratio==", ratio)
            paramd = param.to(dtype=DTYPE_SVD, device=U_b.device)
            proj = U_b @ (U_b.T @ paramd @ V_b) @ V_b.T
            res = paramd - proj
            # res_layer = (res.float().pow(2).sum() / (paramd.float().pow(2).sum() + 1e-12))

            block = U_b.T @ paramd @ V_b
            res_layer = 1 - (block.float().pow(2).sum() / (S_b.float().pow(2).sum() + 1e-12))
            res_layer = torch.where(res_layer.abs() <= THRESHOLD, torch.zeros_like(res_layer), res_layer)
            w_sq_total += paramd.pow(2).sum()
            res_total += res.pow(2).sum()
            # print("total diff and res:", diff_val, res_total)
            # print("==res_dif==", res.sum(), res.max(), res.mean(), res.min())
            # for k in [64, 128, 256, 512, 1024]:
            #     ratio = in_subspace_energy(U_b[:, :k], V_b[:, :k], paramd)
            #     print("k==", k, ratio)

            reg_loss.append(res_layer)
            # reg_loss += res_layer
            param_diff = BASE_PARAMS[name] - param
            # print("==param_diff==", param_diff.sum(), param_diff.max(), param_diff.mean(), param_diff.min())
            base_s = BASE_S[name]
            pad = base_s.numel() - S_t.numel()
            if pad >= 0:
                delta_s = (torch.nn.functional.pad(S_t, (0, pad), value=0) - base_s).cpu().numpy()
            else:
                delta_s = (S_t - torch.nn.functional.pad(base_s, (0, abs(pad)), value=0)).cpu().numpy()

            if name.endswith(".mlp.down_proj.weight") and "layers.0" in name:
                Wf_tmp = param.to(DTYPE_SVD, non_blocking=True)
                fn = Wf_tmp.norm().item()
                # h32 = crc32(str(Wf_tmp.sum().item()).encode()) & 0xffffffff
                # print(f"[DEBUG] {name} ‖W‖_F={fn:.12f} ")
                del Wf_tmp
            result[tag][name] = dict(
                delta_singular_values=Sd.cpu().tolist(),
                delta_energy_diag=float(diag_ratio),
                delta_energy_rot=float(rot_ratio),
                rel_singular_change=rel_s.cpu().tolist(),
                singular_values=S_t.cpu().numpy().tolist(),
                # singular_values_dw=Sd.cpu().numpy().tolist(),
                delta_s=to_py(delta_s),
                principal_angle_V=to_py(V_angles),
                principal_angle_U=to_py(U_angles),
                # distance_U=to_py(U_dist),
                # distance_V=to_py(V_dist),
                # residual_V=to_py(residual_V),
                # residual_U=to_py(residual_U),
                # subspace_energy=energy_ratio,
                # pearson_r=pearson_r,
                basic_stats=stats,
                shape=list(param.shape)
            )
        torch.save({
            "U": torch.cat(all_U_angles[tag]).cpu(),
            "V": torch.cat(all_V_angles[tag]).cpu()
        }, f"/path/angles_{tag}.pt")
        print("total different=========", torch.sqrt(diff_val))
        print("res_total", res_total, w_sq_total, res_total/w_sq_total)
        print("res_total_norm", res_total ** 0.5, w_sq_total ** 0.5)

        print("total_final", proj_dist_U_total, proj_dist_V_total, torch.tensor(align_U_arr).mean(), torch.tensor(align_V_arr).mean())

        m_reg_loss = torch.stack(reg_loss).mean()
        print("m_reg_loss=", m_reg_loss)
        del model
        torch.cuda.empty_cache()

    print(f"[INFO] write JSON to {RESULT_JSON}")
    with open(RESULT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print("[DONE]")


if __name__ == "__main__":
    main()
