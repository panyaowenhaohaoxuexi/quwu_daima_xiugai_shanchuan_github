import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from model.cmdn import CMDN
from model.diagnose_loss import density_loss, mask_loss
from data.data_loader import SynthMultiModalDataset, collate_synth


# Quick diagnosis notes if losses do not decrease:
# 1. GT/image misalignment, usually from unsynchronised dataloader augmentation.
# 2. Learning rate too small/large; try 5e-4 or 5e-5.
# 3. mask_gt may be very sparse, especially in mist; class-balanced BCE can be
#    considered later, but diagnose_loss.py is intentionally unchanged here.

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone supervised CMDN diagnose training.")
    parser.add_argument("--root", default=r"F:/Dehaze_Paper/2_Dataset/1_main_benchmark/FLIR/train")
    parser.add_argument("--size", default=256, type=int)
    parser.add_argument("--batch", default=8, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--lambda_density", default=1.0, type=float)
    parser.add_argument("--lambda_mask", default=1.0, type=float)
    parser.add_argument("--out_dir", default=r"./diagnose_out")
    parser.add_argument("--vis_every", default=1, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overfit_check", default=False, type=str2bool)
    parser.add_argument("--overfit_steps", default=200, type=int)
    return parser.parse_args()


def move_batch_to_device(batch, device, max_samples=None):
    hazy, clear, ir, density_gt, mask_gt = batch
    if max_samples is not None:
        hazy = hazy[:max_samples]
        ir = ir[:max_samples]
        density_gt = density_gt[:max_samples]
        mask_gt = mask_gt[:max_samples]
    return (
        hazy.to(device, non_blocking=True),
        ir.to(device, non_blocking=True),
        density_gt.to(device, non_blocking=True),
        mask_gt.to(device, non_blocking=True),
    )


def first_non_empty_batch(loader):
    for batch in loader:
        hazy = batch[0]
        if hazy.numel() > 0:
            return batch
    raise RuntimeError("No non-empty batch found in training loader.")


def denorm_clip(x):
    mean = CLIP_MEAN.to(device=x.device, dtype=x.dtype)
    std = CLIP_STD.to(device=x.device, dtype=x.dtype)
    return (x * std + mean).clamp(0.0, 1.0)


def _to_numpy_image(tensor):
    return tensor.detach().cpu().float().numpy()


def save_visualization(model, hazy4, ir4, density_gt4, mask_gt4, epoch, out_dir, filename=None):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        debug = model(hazy4, ir4, return_debug=True)

    hazy_rgb = denorm_clip(hazy4)
    columns = [
        ("hazy_rgb", hazy_rgb),
        ("M_d", debug["M_d"]),
        ("C_pred", debug["C"]),
        ("density_gt", density_gt4),
        ("M_pred", debug["M"]),
        ("mask_gt", mask_gt4),
    ]

    n = min(4, hazy4.shape[0])
    fig, axes = plt.subplots(n, len(columns), figsize=(15, 2.8 * n), squeeze=False)
    for row in range(n):
        for col, (title, data) in enumerate(columns):
            ax = axes[row][col]
            if row == 0:
                ax.set_title(title, fontsize=10)
            ax.axis("off")
            if title == "hazy_rgb":
                img = _to_numpy_image(data[row].permute(1, 2, 0))
                ax.imshow(img, vmin=0, vmax=1)
            else:
                img = _to_numpy_image(data[row, 0].clamp(0.0, 1.0))
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)

    fig.tight_layout()
    if filename is None:
        filename = f"vis_epoch_{epoch}.png"
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)

    if was_training:
        model.train()
        if hasattr(model, "_clip_model"):
            model._clip_model.eval()
    return path


def compute_fixed_metrics(model, hazy4, ir4, density_gt4, mask_gt4):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        C_v, M_v, logits_v = model(hazy4, ir4)
        mae_density = (C_v - density_gt4).abs().mean().item()
        pred_bin = (torch.sigmoid(logits_v) >= 0.5).float()
        inter = (pred_bin * mask_gt4).sum()
        union = ((pred_bin + mask_gt4) >= 1).float().sum()
        iou_mask = (inter / (union + 1e-6)).item()

    if was_training:
        model.train()
        if hasattr(model, "_clip_model"):
            model._clip_model.eval()
    return mae_density, iou_mask


def train_step(model, optimizer, hazy, ir, density_gt, mask_gt, args):
    C, M, mask_logits = model(hazy, ir)
    l_d = density_loss(C, density_gt)
    l_m = mask_loss(mask_logits, mask_gt)
    loss = args.lambda_density * l_d + args.lambda_mask * l_m

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), l_d.item(), l_m.item()


def save_checkpoint_and_log(model, loss_log, out_dir):
    np.save(os.path.join(out_dir, "loss_log.npy"), np.array(loss_log, dtype=np.float32))
    torch.save(model.state_dict(), os.path.join(out_dir, "cmdn_diagnose.pth"))


def run_overfit_check(model, optimizer, fixed_batch, args, out_dir):
    hazy4, ir4, density_gt4, mask_gt4 = fixed_batch
    loss_log = []
    start = time.time()

    model.train()
    if hasattr(model, "_clip_model"):
        model._clip_model.eval()

    for step in range(1, args.overfit_steps + 1):
        total, l_d, l_m = train_step(model, optimizer, hazy4, ir4, density_gt4, mask_gt4, args)
        if step == 1 or step % 20 == 0 or step == args.overfit_steps:
            mae_density, iou_mask = compute_fixed_metrics(model, hazy4, ir4, density_gt4, mask_gt4)
            elapsed = (time.time() - start) / 60.0
            print(
                f"[overfit {step:04d}/{args.overfit_steps}] "
                f"total={total:.4f} density={l_d:.4f} mask={l_m:.4f} "
                f"| density MAE={mae_density:.4f} mask IoU={iou_mask:.4f} "
                f"| time={elapsed:.1f}m"
            )
            loss_log.append([step, total, l_d, l_m, mae_density, iou_mask])

    vis_path = save_visualization(model, hazy4, ir4, density_gt4, mask_gt4, "overfit", out_dir, "vis_overfit.png")
    save_checkpoint_and_log(model, loss_log, out_dir)
    print(f"[overfit] saved visualization: {vis_path}")
    print(f"[overfit] saved checkpoint: {os.path.join(out_dir, 'cmdn_diagnose.pth')}")


def run_full_training(model, optimizer, scheduler, loader, fixed_batch, args, out_dir):
    hazy4, ir4, density_gt4, mask_gt4 = fixed_batch
    loss_log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        if hasattr(model, "_clip_model"):
            model._clip_model.eval()

        ep_total, ep_d, ep_m, n = 0.0, 0.0, 0.0, 0
        start = time.time()
        for batch in loader:
            hazy = batch[0]
            if hazy.numel() == 0:
                continue
            hazy, ir, density_gt, mask_gt = move_batch_to_device(batch, next(model.parameters()).device)
            total, l_d, l_m = train_step(model, optimizer, hazy, ir, density_gt, mask_gt, args)
            ep_total += total
            ep_d += l_d
            ep_m += l_m
            n += 1

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        total_avg = ep_total / max(n, 1)
        density_avg = ep_d / max(n, 1)
        mask_avg = ep_m / max(n, 1)
        elapsed = (time.time() - start) / 60.0
        print(
            f"[epoch {epoch}] total={total_avg:.4f} density={density_avg:.4f} "
            f"mask={mask_avg:.4f} lr={lr:.2e} time={elapsed:.1f}m"
        )
        mae_density, iou_mask = compute_fixed_metrics(model, hazy4, ir4, density_gt4, mask_gt4)
        print(f"   [metric] density MAE={mae_density:.4f}  mask IoU={iou_mask:.4f}")
        loss_log.append([epoch, total_avg, density_avg, mask_avg, lr, mae_density, iou_mask])
        np.save(os.path.join(out_dir, "loss_log.npy"), np.array(loss_log, dtype=np.float32))

        if args.vis_every > 0 and (epoch % args.vis_every == 0 or epoch == args.epochs):
            vis_path = save_visualization(model, hazy4, ir4, density_gt4, mask_gt4, epoch, out_dir)
            print(f"   [vis] saved {vis_path}")

    save_checkpoint_and_log(model, loss_log, out_dir)
    print(f"[train] saved checkpoint: {os.path.join(out_dir, 'cmdn_diagnose.pth')}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[diagnose] device={device}")
    print(f"[diagnose] out_dir={args.out_dir}")

    train_set = SynthMultiModalDataset(root=args.root, train=True, size=args.size)
    loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_synth,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    first_batch = first_non_empty_batch(loader)
    fixed_batch = move_batch_to_device(first_batch, device, max_samples=4)

    model = CMDN().to(device)
    if hasattr(model, "_clip_model"):
        model._clip_model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))

    print(f"[diagnose] trainable parameter tensors={len(params)}")
    if args.overfit_check:
        run_overfit_check(model, optimizer, fixed_batch, args, args.out_dir)
        return

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    run_full_training(model, optimizer, scheduler, loader, fixed_batch, args, args.out_dir)


if __name__ == "__main__":
    main()
