import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def calculate_psnr_ssim(dehaze_folder, gt_folder):
    # 只保留常见图像后缀
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    dehaze_files = sorted([f for f in os.listdir(dehaze_folder)
                           if f.lower().endswith(exts)])
    gt_files = sorted([f for f in os.listdir(gt_folder)
                       if f.lower().endswith(exts)])

    print(f"去雾图像数量: {len(dehaze_files)}")
    print(f"GT 图像数量: {len(gt_files)}")

    # 取两边共同的文件名（交集）
    common_files = sorted(set(dehaze_files) & set(gt_files))
    if not common_files:
        print("两边没有任何同名图像，请检查文件名是否一致。")
        return [], []

    if len(common_files) != len(dehaze_files) or len(common_files) != len(gt_files):
        print("⚠️ 警告：两文件夹中文件名不完全一致，"
              "只会对共同拥有的这些图像计算指标：")
        print("共同文件数:", len(common_files))

    psnr_list, ssim_list = [], []

    for name in common_files:
        dehaze_path = os.path.join(dehaze_folder, name)
        gt_path = os.path.join(gt_folder, name)

        dehaze_img = cv2.imread(dehaze_path)
        gt_img = cv2.imread(gt_path)

        if dehaze_img is None:
            print(f"[跳过] 无法读取去雾图像: {dehaze_path}")
            continue
        if gt_img is None:
            print(f"[跳过] 无法读取 GT 图像: {gt_path}")
            continue

        # ========= 关键修改部分：以 GT 图像分辨率为标准 =========
        # 如果尺寸不一致，则把去雾图 resize 成和 GT 一样大
        if dehaze_img.shape[:2] != gt_img.shape[:2]:
            h, w = gt_img.shape[:2]   # GT 的高度、宽度
            dehaze_img = cv2.resize(dehaze_img, (w, h))
        # ===================================================

        # 转为 RGB
        dehaze_img = cv2.cvtColor(dehaze_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        # 转为 float32 更稳妥
        dehaze_f = dehaze_img.astype(np.float32)
        gt_f = gt_img.astype(np.float32)

        # 计算 PSNR / SSIM
        psnr_val = psnr(gt_f, dehaze_f, data_range=255)
        ssim_val = ssim(gt_f, dehaze_f, channel_axis=2, data_range=255)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

        print(f"{name}: PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}")

    if psnr_list:
        mean_psnr = np.mean(psnr_list)
        mean_ssim = np.mean(ssim_list)
        print("\n========== 平均结果 ==========")
        print(f"匹配图像数: {len(psnr_list)}")
        print(f"平均 PSNR: {mean_psnr:.4f}")
        print(f"平均 SSIM: {mean_ssim:.4f}")
        print("================================")
    else:
        print("没有成功计算任何图像的指标，请检查路径和图像格式。")

    return psnr_list, ssim_list


if __name__ == "__main__":
    dehaze_folder = "D:/liu_lan_qi_xia_zai/CVPR_Sup_materials/Sup1/VIFNet-test/Real_Sup1_test"
    gt_folder     = "E:/Sup1_dataset/Target_Sup1/Real_Sup1/clear"

    calculate_psnr_ssim(dehaze_folder, gt_folder)
