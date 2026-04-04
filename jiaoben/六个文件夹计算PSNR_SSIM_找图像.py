# -*- coding: utf-8 -*-
import os
import cv2
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ================== 文件夹路径保持不变 ==================
folder_A = r"F:\Dehaze_huiyi_lunwen\1_images_duibi_experiments\additional_benchmark\FocalNet\source_test"
folder_B = r"F:\Dehaze_huiyi_lunwen\1_images_duibi_experiments\additional_benchmark\UCL_Dehaze\source_test"
folder_C = r"F:\Dehaze_huiyi_lunwen\1_images_duibi_experiments\additional_benchmark\VIFNet-test\Teacher_source_test"
folder_D = r"F:\Dehaze_huiyi_lunwen\1_images_duibi_experiments\additional_benchmark\CoA_yuanma_dehazed\Teacher_model_test_source\v1"
folder_F = r"F:\Dehaze_huiyi_lunwen\1_images_duibi_experiments\additional_benchmark\Ours\Teacher_model_test_source\v1_best"
folder_GT = r"F:\Dehaze_huiyi_lunwen\2_Dataset\2_additional_benchmark\Source\test\clear"
# =================================================================

top_k = 15

folders = {
    'A': Path(folder_A), 'B': Path(folder_B), 'C': Path(folder_C),
    'D': Path(folder_D), 'F': Path(folder_F), 'GT': Path(folder_GT)
}

# 检查文件夹是否存在
for name, p in folders.items():
    if not p.exists():
        raise FileNotFoundError(f"❌ 文件夹不存在，请检查路径 → {p}")

gt_files = sorted([
    f for f in folders['GT'].iterdir()
    if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
])

results = []
print(f"开始处理 {len(gt_files)} 张图像...\n")

skip_count = 0  # 记录被跳过的图片数量

for gt_file in gt_files:
    name = gt_file.name
    gt_name_without_ext = gt_file.stem  # 获取不带后缀的文件名，应对后缀不同的情况

    gt = cv2.imread(str(gt_file))
    if gt is None:
        print(f"⚠️ 无法读取 GT 图像: {gt_file}")
        skip_count += 1
        continue
    gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

    imgs = {}
    missing = False

    for label in 'ABCDF':
        # 这里尝试直接匹配原文件名
        path = folders[label] / name

        # 如果直接匹配找不到，尝试匹配相同前缀但后缀是 .png 或 .jpg 的图（容错处理）
        if not path.exists():
            possible_png = folders[label] / f"{gt_name_without_ext}.png"
            possible_jpg = folders[label] / f"{gt_name_without_ext}.jpg"
            if possible_png.exists():
                path = possible_png
            elif possible_jpg.exists():
                path = possible_jpg
            else:
                # 只在第一张出错时详细打印，防止刷屏
                if skip_count == 0:
                    print(f"🔍 诊断提示：在方法 {label} 的文件夹中找不到文件：{name}")
                    print(f"   预期路径为: {path}")
                    print(f"   请检查去雾后生成的文件名是否被算法加上了后缀（比如 {gt_name_without_ext}_pred.png）！")
                missing = True
                break

        img = cv2.imread(str(path))
        if img is None:
            if skip_count == 0:
                print(f"⚠️ 图像读取失败（可能损坏）: {path}")
            missing = True
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[:2] != gt_rgb.shape[:2]:
            img_rgb = cv2.resize(img_rgb, (gt_rgb.shape[1], gt_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        imgs[label] = img_rgb

    if missing:
        skip_count += 1
        continue

    metrics = {}
    for label in 'ABCDF':
        img = imgs[label]
        p = psnr(gt_rgb, img, data_range=255)
        s = ssim(gt_rgb, img, channel_axis=-1, data_range=255)
        metrics[label] = (p, s)

    psnr_F = metrics['F'][0]
    ssim_F = metrics['F'][1]

    # 【已修复 Bug】：这里改为 'ABCD'，去掉了不存在的 'E'
    max_psnr_other = max(metrics[l][0] for l in 'ABCD')
    max_ssim_other = max(metrics[l][1] for l in 'ABCD')

    if psnr_F > max_psnr_other and ssim_F > max_ssim_other:
        delta_psnr = psnr_F - max_psnr_other
        delta_ssim = ssim_F - max_ssim_other
        score = delta_psnr + delta_ssim * 100

        results.append({
            'filename': name, 'PSNR_F': psnr_F, 'SSIM_F': ssim_F,
            'ΔPSNR': delta_psnr, 'ΔSSIM': delta_ssim, 'Score': score
        })

results = sorted(results, key=lambda x: x['Score'], reverse=True)[:top_k]

print("=" * 90)
print(f"方法 F 严格领先其他【四个】方法的 Top {len(results)} 张图像 (共跳过 {skip_count} 张无法匹配的图像)")
print("=" * 90)

if results:
    for i, r in enumerate(results, 1):
        print(
            f"{i:2d}. {r['filename']:<30} PSNR={r['PSNR_F']:.3f} (+{r['ΔPSNR']:.3f})   SSIM={r['SSIM_F']:.4f} (+{r['ΔSSIM']:.4f})")
else:
    print("未找到 F 在两个指标上都严格领先的图像，或者所有图像均因为路径/命名问题被跳过。")

print("\n全部完成！")