import os
import cv2
import numpy as np


def normalize01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def imread_unicode(path: str, flags=cv2.IMREAD_UNCHANGED):
    """
    支持中文路径的读取
    """
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, flags)
    return img


def imwrite_unicode(path: str, img: np.ndarray):
    """
    支持中文路径的保存
    """
    ext = os.path.splitext(path)[1]
    if ext == "":
        ext = ".png"
        path = path + ext

    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Failed to encode image for saving: {path}")
    buf.tofile(path)


def read_visible_gray(vis_path: str):
    vis_bgr = imread_unicode(vis_path, cv2.IMREAD_COLOR)
    if vis_bgr is None:
        raise FileNotFoundError(f"Cannot read visible image: {vis_path}")

    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    vis_gray = 0.299 * vis_rgb[..., 0] + 0.587 * vis_rgb[..., 1] + 0.114 * vis_rgb[..., 2]
    return vis_bgr, vis_gray.astype(np.float32)


def read_ir_gray(ir_path: str, target_hw=None):
    ir = imread_unicode(ir_path, cv2.IMREAD_UNCHANGED)
    if ir is None:
        raise FileNotFoundError(f"Cannot read infrared image: {ir_path}")

    if ir.ndim == 3:
        if ir.shape[2] == 4:
            ir = cv2.cvtColor(ir, cv2.COLOR_BGRA2GRAY)
        else:
            ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)

    ir = ir.astype(np.float32)
    ir = normalize01(ir)

    if target_hw is not None:
        h, w = target_hw
        if ir.shape[:2] != (h, w):
            ir = cv2.resize(ir, (w, h), interpolation=cv2.INTER_LINEAR)

    return ir.astype(np.float32)


def sobel_gradients(img: np.ndarray):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy


def local_contrast_proxy(vis_gray: np.ndarray, blur_sigma: float = 3.0) -> np.ndarray:
    blur = cv2.GaussianBlur(vis_gray, (0, 0), blur_sigma)
    contrast = np.abs(vis_gray - blur)
    return normalize01(contrast)


def compute_confidence_map(
    vis_gray: np.ndarray,
    ir_gray: np.ndarray,
    beta: float = 0.25,
    lambda_w: float = 0.5,
    smooth_sigma: float = 1.2
):
    gx_v, gy_v = sobel_gradients(vis_gray)
    gx_i, gy_i = sobel_gradients(ir_gray)

    grad_gap = np.abs(gx_v - gx_i) + np.abs(gy_v - gy_i)
    G = np.exp(-grad_gap / max(beta, 1e-8)).astype(np.float32)
    G = np.clip(G, 0.0, 1.0)

    vis_proxy = local_contrast_proxy(vis_gray)
    C = lambda_w * G + (1.0 - lambda_w) * vis_proxy
    C = np.clip(C, 0.0, 1.0)

    if smooth_sigma > 0:
        C = cv2.GaussianBlur(C, (0, 0), smooth_sigma)

    return C, G, vis_proxy


def compute_alpha_map(
    confidence_map: np.ndarray,
    tau_l: float = 0.25,
    tau_h: float = 0.75
):
    C = confidence_map
    alpha = np.zeros_like(C, dtype=np.float32)

    high_mask = C > tau_h
    low_mask = C < tau_l
    mid_mask = (~high_mask) & (~low_mask)

    alpha[high_mask] = 1.0
    alpha[low_mask] = 0.0
    alpha[mid_mask] = (C[mid_mask] - tau_l) / max(tau_h - tau_l, 1e-8)

    return np.clip(alpha, 0.0, 1.0)


def save_gray_map(path: str, x: np.ndarray):
    x_u8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    imwrite_unicode(path, x_u8)


def save_heatmap(path: str, x: np.ndarray):
    x_u8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(x_u8, cv2.COLORMAP_JET)
    imwrite_unicode(path, heat)


def generate_confidence_from_paths(
    vis_path: str,
    ir_path: str,
    out_dir: str,
    beta: float = 0.25,
    lambda_w: float = 0.5,
    tau_l: float = 0.25,
    tau_h: float = 0.75
):
    os.makedirs(out_dir, exist_ok=True)

    vis_bgr, vis_gray = read_visible_gray(vis_path)
    h, w = vis_gray.shape[:2]
    ir_gray = read_ir_gray(ir_path, target_hw=(h, w))

    confidence_map, gradient_map, visible_proxy = compute_confidence_map(
        vis_gray=vis_gray,
        ir_gray=ir_gray,
        beta=beta,
        lambda_w=lambda_w,
        smooth_sigma=1.2
    )

    alpha_map = compute_alpha_map(confidence_map, tau_l=tau_l, tau_h=tau_h)

    save_gray_map(os.path.join(out_dir, "confidence_map_gray.png"), confidence_map)
    save_heatmap(os.path.join(out_dir, "confidence_map_heat.png"), confidence_map)
    save_gray_map(os.path.join(out_dir, "gradient_consistency_map.png"), gradient_map)
    save_gray_map(os.path.join(out_dir, "visible_proxy_map.png"), visible_proxy)
    save_gray_map(os.path.join(out_dir, "alpha_map.png"), alpha_map)

    conf_u8 = np.clip(confidence_map * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(conf_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(vis_bgr, 0.6, heat, 0.4, 0)
    imwrite_unicode(os.path.join(out_dir, "confidence_overlay_on_visible.png"), overlay)

    print("Done.")
    print(f"Visible path:  {vis_path}")
    print(f"Infrared path: {ir_path}")
    print(f"Output dir:    {out_dir}")


if __name__ == "__main__":
    vis_path = r"F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\hazy_RGB\image1.png"
    ir_path  = r"F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\IR\image1.png"
    out_dir  = r"F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\置信度图\image1.png"

    generate_confidence_from_paths(
        vis_path=vis_path,
        ir_path=ir_path,
        out_dir=out_dir,
        beta=0.25,
        lambda_w=0.5,
        tau_l=0.25,
        tau_h=0.75
    )