"""补全流中间过程可视化：Structure-guided appearance retrieval and completion.

无需修改模型代码 —— 用已有 capture_fusion_intermediates + 模型权重手动复现检索。
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TF

from data.data_loader import load_tir_as_float_tensor
from utils.checkpoint import (
    build_model_from_config, load_strict_v2_state_dict,
    preflight_eval_checkpoint, tir_normalization_config,
)
from tools.completion_stream_figure import compose_completion_panel, select_structured_completion_query

DEFAULT_CHECKPOINT = r"Teacher_Train/source_best.pt"
DEFAULT_HAZY = r"F:/1_paper_pan/1_Dehaze_Paper/2_Dataset/1_main_benchmark/1_FLIR/test/hazy/3_dense/FLIR_09973.jpg"
DEFAULT_TIR = r"F:/1_paper_pan/1_Dehaze_Paper/2_Dataset/1_main_benchmark/1_FLIR/test/ir/FLIR_09973.jpg"
DEFAULT_OUTPUT = r"F:/1_paper_pan/1_Dehaze_Paper/3_conference/7_TIP/2-论文里的图/中间可视化/ReMix中间过程可视化/补全流"


def _blues(arr):
    arr = np.clip(arr, 0, 1)
    r = np.interp(arr, [0, 0.3, 0.6, 0.85, 1.0], [0.97, 0.78, 0.45, 0.15, 0.03])
    g = np.interp(arr, [0, 0.3, 0.6, 0.85, 1.0], [0.98, 0.85, 0.68, 0.45, 0.20])
    b = np.interp(arr, [0, 0.3, 0.6, 0.85, 1.0], [0.97, 0.94, 0.90, 0.80, 0.55])
    return np.stack([r, g, b], -1)


def _save(t, path, cmap=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if cmap and t.ndim == 3 and t.shape[0] == 1:
        arr = t.detach().float().cpu()[0].clamp(0, 1).numpy()
        img = Image.fromarray((_blues(arr) * 255).astype(np.uint8)) if cmap in ("hot","blues") else \
              Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        img.save(path)
    else:
        TF.to_pil_image((t[0] if t.ndim == 4 else t).cpu().clamp(0, 1)).save(path)
    print(f"  已保存: {path}")


def _label(img, text, pos="top-left", fs=14):
    d = ImageDraw.Draw(img)
    try: f = ImageFont.truetype("simhei.ttf", fs)
    except: f = ImageFont.load_default()
    xy = (8, 8) if pos == "top-left" else (img.width - d.textbbox((0,0), text, f)[2] - 8, 8)
    d.text((xy[0]+1, xy[1]+1), text, font=f, fill="black")
    d.text(xy, text, font=f, fill="white")
    return img


def _box(img, y, x, half, color=(255, 60, 60), w=3):
    if img.mode == "L": img = img.convert("RGB")
    d = ImageDraw.Draw(img)
    for i in range(w): d.rectangle([x-half+i, y-half+i, x+half-i, y+half-i], outline=color)
    return img


def _cross(img, y, x, s=10, color=(255, 60, 60), w=2):
    if img.mode == "L":
        img = img.convert("RGB")
    d = ImageDraw.Draw(img)
    d.line([(x-s, y), (x+s, y)], fill=color, width=w)
    d.line([(x, y-s), (x, y+s)], fill=color, width=w)
    return img


def main(argv=None):
    args = argparse.ArgumentParser("completion-vis").parse_args(argv or [])
    # 用用户指定的默认值
    ckpt_p = DEFAULT_CHECKPOINT; hazy_p = DEFAULT_HAZY; tir_p = DEFAULT_TIR; out_p = DEFAULT_OUTPUT

    # --- 1. 加载 ---
    ck = torch.load(ckpt_p, map_location="cpu")
    pf = preflight_eval_checkpoint(ck, ema_model="teacher")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(pf["config"]).to(dev)
    load_strict_v2_state_dict(model, pf["state_dict"], label=pf["state_label"])
    model.eval()
    print(f"模型: {dev}")

    with Image.open(hazy_p) as im:
        hazy = TF.pil_to_tensor(im.convert("RGB")).float().div_(255).unsqueeze(0)
    tir_cfg = tir_normalization_config(pf["config"])
    tir = load_tir_as_float_tensor(tir_p, tir_cfg).unsqueeze(0)
    if tir.shape[-2:] != hazy.shape[-2:]:
        tir = F.interpolate(tir, size=hazy.shape[-2:], mode="bilinear", align_corners=False)
    B, C, H_img, W_img = hazy.shape
    print(f"图像: {H_img}×{W_img}")

    # --- 2. 推理 ---
    tau = pf["config"].get("route_tau_end", 0.2)
    with torch.inference_mode():
        ctx = model.encode_context(hazy.to(dev), tir.to(dev), route_temperature=tau)
        out = model.decode_with_route(ctx, route_mode="hard", capture_fusion_intermediates=True)

    # 中间特征 (clone 以脱离 inference_mode)
    S_dev = out["fusion_intermediates"]["S"].clone()
    fc_dev = out["fusion_intermediates"]["fusion_candidate"].clone()
    route_dev = out["route_hard"].clone()
    H_f, W_f = S_dev.shape[-2:]; scale = H_img // H_f
    print(f"h2: {H_f}×{W_f}, scale={scale}")

    # CPU 拷贝（可视化用）
    pred_cpu = out["pred_clear"].cpu()
    hazy_cpu = hazy
    tir_1ch = tir[:, :1].repeat(1, 3, 1, 1) if tir.shape[1] == 1 else tir
    tir_cpu = tir_1ch.cpu()

    # --- 3. 选 query 点 ---
    mem = model.memory["h2"]
    route_h2 = F.interpolate(route_dev.float(), size=(H_f, W_f), mode="nearest")
    valid_h2 = torch.ones(1, 1, H_f, W_f, device=dev)
    reliability = (1.0 - route_h2) * valid_h2

    structure_energy = S_dev.abs().mean(dim=1, keepdim=True)
    q_h, q_w = select_structured_completion_query(route_h2, structure_energy)
    q_h_img, q_w_img = q_h * scale + scale // 2, q_w * scale + scale // 2
    print(f"Query (feat):({q_h},{q_w})  (img):({q_h_img},{q_w_img})  route={route_h2[0,0,q_h,q_w].item():.4f}")

    # --- 4. 手动复现 MemoryRetriever ---
    print("\n>>> 复现 MemoryRetriever ...")
    with torch.no_grad():
        global_ctx = (S_dev * valid_h2).sum(dim=(2,3)) / valid_h2.sum().clamp_min(1.0)
        k = mem.key(S_dev) + mem.context_key(global_ctx).view(1, -1, 1, 1)
        q = mem.query(S_dev) + mem.context_query(global_ctx).view(1, -1, 1, 1)
        keys = F.normalize(k.flatten(2).transpose(1,2), dim=-1, eps=1e-6)
        queries = F.normalize(q.flatten(2).transpose(1,2), dim=-1, eps=1e-6)
        vals = fc_dev.flatten(2).transpose(1, 2)

        rel_f = reliability.flatten(1)
        valid_idx = torch.where(rel_f[0] > mem.reliability_epsilon)[0]
        if valid_idx.numel() > int(mem.max_tokens):
            pos_ = torch.linspace(0, valid_idx.numel()-1, int(mem.max_tokens), device=dev).round().long()
            valid_idx = valid_idx[pos_]
        print(f"  可靠token: {valid_idx.numel()}/{rel_f.numel()}")

        mk = keys[0, valid_idx]; mv = vals[0, valid_idx]
        rb = rel_f[0, valid_idx].clamp_min(mem.reliability_epsilon).log()

        qi = q_h * W_f + q_w
        qv = queries[0, qi:qi+1]
        scores = (qv @ mk.T) / float(mem.attention_temperature) + rb.unsqueeze(0)
        topk = min(int(mem.topk), valid_idx.numel())
        ts, ti = torch.topk(scores, k=topk, dim=-1)
        attn = torch.softmax(ts, dim=-1)[0]

        # 空间 attention map
        mfi = valid_idx[ti[0]]
        amap = torch.zeros(H_f * W_f)
        amap[mfi.cpu()] = attn.cpu()
        amap = (amap / amap.max().clamp_min(1e-8)).reshape(1, 1, H_f, W_f)

    # 原图坐标
    mh = [(idx.item() // W_f) * scale + scale // 2 for idx in mfi]
    mw = [(idx.item() % W_f) * scale + scale // 2 for idx in mfi]
    print(f"  Top-{topk} attn: {attn.tolist()}")
    print(f"  Top-1 位置: ({mh[0]}, {mw[0]})")

    # --- 5. 可视化 ---
    out_dir = Path(out_p); out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(hazy_p).stem
    bs = 40  # query box half-size

    print(f"\n--- 保存到 {out_dir} ---")

    # 5a. 雾图 + query
    hp = TF.to_pil_image(hazy_cpu[0].clamp(0,1))
    _save(TF.pil_to_tensor(_box(hp, q_h_img, q_w_img, bs)).float().div_(255).unsqueeze(0),
          out_dir / f"{stem}_01_hazy_query.png")

    # 5b. IR + query
    tp = TF.to_pil_image(tir_cpu[0].clamp(0,1))
    _save(TF.pil_to_tensor(_box(tp, q_h_img, q_w_img, bs)).float().div_(255).unsqueeze(0),
          out_dir / f"{stem}_02_ir_query.png")

    # 5c. Route
    rp = TF.to_pil_image((1.0 - route_dev.float())[0].cpu().clamp(0,1))
    _save(TF.pil_to_tensor(_cross(rp, q_h_img, q_w_img)).float().div_(255).unsqueeze(0),
          out_dir / f"{stem}_03_route.png")

    # 5d. 去雾 + query
    dp = TF.to_pil_image(pred_cpu[0].clamp(0,1))
    _save(TF.pil_to_tensor(_box(dp, q_h_img, q_w_img, bs)).float().div_(255).unsqueeze(0),
          out_dir / f"{stem}_04_dehazed.png")

    # 5e. Attention 热力图（叠加到雾图上）
    af = F.interpolate(amap, size=(H_img, W_img), mode="bilinear", align_corners=False)
    ar = _blues(af[0,0].clamp(0,1).numpy())
    ai = Image.blend(Image.fromarray((hazy_cpu[0].permute(1,2,0).numpy()*255).astype(np.uint8)),
                     Image.fromarray((ar*255).astype(np.uint8)), alpha=0.5)
    _save(TF.pil_to_tensor(_cross(ai, q_h_img, q_w_img)).float().div_(255).unsqueeze(0),
          out_dir / f"{stem}_05_attention.png")

    # 5f. Top-3 patches
    ps = 80
    patches = []
    for k in range(min(3, topk)):
        y1, y2 = max(0, mh[k]-ps//2), min(H_img, mh[k]+ps//2)
        x1, x2 = max(0, mw[k]-ps//2), min(W_img, mw[k]+ps//2)
        pp = TF.to_pil_image(hazy_cpu[0, :, y1:y2, x1:x2].clamp(0,1))
        patches.append(TF.pil_to_tensor(pp).float().div_(255))
    while len(patches) < 3:
        patches.append(torch.zeros(3, ps*2, ps*2))
    for k in range(3):
        _save(patches[k].unsqueeze(0), out_dir / f"{stem}_06_patch{k+1}.png")

    # --- 6. 论文机制面板：结构 query + attention provenance + RGB appearance sources ---
    print("\n>>> 绘制 completion-stream 机制面板 ...")
    source_xy = [(mw[k], mh[k]) for k in range(min(3, topk))]
    panel = compose_completion_panel(
        hazy=hp,
        infrared=tp,
        attention=ai,
        output=dp,
        patches=[TF.to_pil_image(patches[k].clamp(0, 1)) for k in range(3)],
        weights=[float(attn[k].item()) if k < topk else 0.0 for k in range(3)],
        query_xy=(q_w_img, q_h_img),
        source_xy=source_xy,
    )
    panel.save(out_dir / f"{stem}_completion_panel.png")
    print(f"  已保存面板: {out_dir / f'{stem}_completion_panel.png'}")

    # --- 7. 统计 ---
    print(f"\n{'='*50}")
    print(f"补全流检索 — {stem}")
    print(f"  Query route: {route_dev[0,0,q_h_img,q_w_img].item():.4f}")
    print(f"  可靠token占比: {(reliability > 1e-6).float().mean().item():.1%}")
    print(f"  Top-{topk} attn: {attn.tolist()}")
    print(f"  Top-1 route: {route_dev[0,0,mh[0],mw[0]].item():.4f}")
    print(f"  输出: {out_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
