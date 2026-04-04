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

# 每个文件夹输出前多少个“最佳表现”
show_top_k = 3

folders = {
    'A': Path(folder_A), 'B': Path(folder_B), 'C': Path(folder_C),
    'D': Path(folder_D), 'F': Path(folder_F), 'GT': Path(folder_GT)
}

# 检查文件夹是否存在
for name, p in folders.items():
    if not p.exists():
        raise FileNotFoundError(f"❌ 文件夹不存在 → {p}")

gt_files = sorted([
    f for f in folders['GT'].iterdir()
    if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
])

# 核心存储结构：每个方法赢下的图片列表
all_method_wins = {m: [] for m in 'ABCDF'}

print(f"🚀 开始全量对比 {len(gt_files)} 张图像，寻找各方法的“名场面”...\n")

skip_count = 0

for gt_file in gt_files:
    name = gt_file.name
    gt_name_without_ext = gt_file.stem
    gt = cv2.imread(str(gt_file))
    if gt is None:
        skip_count += 1
        continue
    gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

    current_metrics = {}
    missing_this_img = False

    # 1. 采集当前图片在所有方法下的指标
    for label in 'ABCDF':
        path = folders[label] / name
        if not path.exists():
            # 容错：尝试 .png/.jpg
            p_png, p_jpg = folders[label] / f"{gt_name_without_ext}.png", folders[label] / f"{gt_name_without_ext}.jpg"
            path = p_png if p_png.exists() else (p_jpg if p_jpg.exists() else path)

        if not path.exists():
            missing_this_img = True
            break

        img = cv2.imread(str(path))
        if img is None:
            missing_this_img = True
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[:2] != gt_rgb.shape[:2]:
            img_rgb = cv2.resize(img_rgb, (gt_rgb.shape[1], gt_rgb.shape[0]))

        p_val = psnr(gt_rgb, img_rgb, data_range=255)
        s_val = ssim(gt_rgb, img_rgb, channel_axis=-1, data_range=255)
        # 综合评分逻辑：PSNR + SSIM*100 (SSIM权重加大以体现结构重要性)
        current_metrics[label] = {
            'psnr': p_val,
            'ssim': s_val,
            'combined': p_val + s_val * 100
        }

    if missing_this_img:
        skip_count += 1
        continue

    # 2. 确定谁是这张图的冠军
    sorted_labels = sorted(current_metrics.keys(), key=lambda l: current_metrics[l]['combined'], reverse=True)
    winner = sorted_labels[0]
    runner_up = sorted_labels[1]

    # 计算领先第二名的优势
    gap_psnr = current_metrics[winner]['psnr'] - current_metrics[runner_up]['psnr']
    gap_ssim = current_metrics[winner]['ssim'] - current_metrics[runner_up]['ssim']
    advantage_score = current_metrics[winner]['combined'] - current_metrics[runner_up]['combined']

    all_method_wins[winner].append({
        'filename': name,
        'psnr': current_metrics[winner]['psnr'],
        'ssim': current_metrics[winner]['ssim'],
        'gap_psnr': gap_psnr,
        'gap_ssim': gap_ssim,
        'advantage': advantage_score
    })

# 3. 分类别展示结果
print("=" * 100)
print(f"📊 各方法“最强表现”统计 (按领先第二名的优势幅度排序)")
print("=" * 100)

method_full_names = {
    'A': 'FocalNet', 'B': 'UCL_Dehaze', 'C': 'VIFNet', 'D': 'CoA', 'F': 'Ours (Proposed)'
}

for m in 'ABCDF':
    wins = sorted(all_method_wins[m], key=lambda x: x['advantage'], reverse=True)
    print(f"\n🏆 【方法 {m} - {method_full_names[m]}】 胜出的图像共 {len(wins)} 张")

    if not wins:
        print("   --- 该方法在所有测试图中均未获得第一名 ---")
        continue

    # 只展示前 k 个最具代表性的（指标远超其他的）
    for i, res in enumerate(wins[:show_top_k], 1):
        print(
            f"   {i}. {res['filename']:<35} | PSNR: {res['psnr']:.2f} (领先:{res['gap_psnr']:+5.2f}) | SSIM: {res['ssim']:.4f} (领先:{res['gap_ssim'] :+6.4f})")

print("\n" + "=" * 100)
print(f"✅ 处理完成！共跳过 {skip_count} 张不匹配图像。")