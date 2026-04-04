# import os
# import cv2
# import numpy as np
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim
#
#
# def calculate_psnr_ssim(dehaze_folder, gt_folder):
#     # 支持的图像格式
#     exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
#
#     # 1. 获取去雾文件夹下的所有图片
#     dehaze_files = sorted([f for f in os.listdir(dehaze_folder)
#                            if f.lower().endswith(exts)])
#
#     # 2. 获取 GT 文件夹下的所有图片，并建立 {文件名主体: 完整文件名} 的映射
#     # 例如：{'image_01': 'image_01.jpg', 'image_02': 'image_02.png'}
#     gt_map = {os.path.splitext(f)[0]: f for f in os.listdir(gt_folder)
#               if f.lower().endswith(exts)}
#
#     print(f"去雾图像数量: {len(dehaze_files)}")
#     print(f"GT 文件夹中识别到的图像数量: {len(gt_map)}")
#     print(f"将根据文件名主体进行匹配...\n")
#
#     psnr_list, ssim_list = [], []
#
#     for d_name in dehaze_files:
#         # 提取当前去雾图像的文件名主体
#         base_name = os.path.splitext(d_name)[0]
#
#         # 检查 GT 映射中是否存在该主体
#         if base_name not in gt_map:
#             print(f"[跳过] GT 文件夹中未找到匹配主体的文件: {base_name}")
#             continue
#
#         # 获取对应的 GT 完整文件名
#         gt_name = gt_map[base_name]
#
#         dehaze_path = os.path.join(dehaze_folder, d_name)
#         gt_path = os.path.join(gt_folder, gt_name)
#
#         dehaze_img = cv2.imread(dehaze_path)
#         gt_img = cv2.imread(gt_path)
#
#         if dehaze_img is None or gt_img is None:
#             print(f"[跳过] 图像读取失败: {d_name} 或 {gt_name}")
#             continue
#
#         # -------- 以 GT 尺寸为标准调整 --------
#         if dehaze_img.shape[:2] != gt_img.shape[:2]:
#             h, w = gt_img.shape[:2]
#             dehaze_img = cv2.resize(dehaze_img, (w, h))
#         # -----------------------------------
#
#         # BGR → RGB
#         dehaze_img = cv2.cvtColor(dehaze_img, cv2.COLOR_BGR2RGB)
#         gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
#
#         # float32 for metrics
#         dehaze_f = dehaze_img.astype(np.float32)
#         gt_f = gt_img.astype(np.float32)
#
#         # PSNR/SSIM
#         psnr_val = psnr(gt_f, dehaze_f, data_range=255)
#         ssim_val = ssim(gt_f, dehaze_f, channel_axis=2, data_range=255)
#
#         psnr_list.append(psnr_val)
#         ssim_list.append(ssim_val)
#
#         print(f"匹配成功: [{d_name}] <-> [{gt_name}] | PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}")
#
#     # 统计结果
#     if psnr_list:
#         print("\n========== 平均结果 ==========")
#         print(f"有效匹配对数: {len(psnr_list)}")
#         print(f"平均 PSNR: {np.mean(psnr_list):.4f}")
#         print(f"平均 SSIM: {np.mean(ssim_list):.4f}")
#         print("================================")
#     else:
#         print("错误：未找到任何可以匹配的图像，请检查文件名主体是否一致。")
#
#     return psnr_list, ssim_list
#
#
# if __name__ == "__main__":
#     # 建议使用 r"" 原始字符串，防止 Windows 路径中的反斜杠被转义
#     dehaze_folder = r"F:\Dehaze_huiyi_lunwen\1_images_duibi_experiments\additional_benchmark\FFA_Net\source_test"
#     gt_folder = r"F:\Dehaze_huiyi_lunwen\2_Dataset\2_additional_benchmark\Source\test\clear"
#
#     calculate_psnr_ssim(dehaze_folder, gt_folder)

import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def calculate_psnr_ssim(dehaze_folder, gt_folder):
    # 支持的图像格式
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    # 1. 获取去雾文件夹下的所有图片
    dehaze_files = sorted([f for f in os.listdir(dehaze_folder)
                           if f.lower().endswith(exts)])

    # 2. 获取 GT 文件夹下的所有图片，建立 {文件名主体: 完整文件名} 映射
    # GT 格式示例: {'A_0000_Haze4K_3356': 'A_0000_Haze4K_3356.jpg'}
    gt_map = {os.path.splitext(f)[0]: f for f in os.listdir(gt_folder)
              if f.lower().endswith(exts)}

    print(f"去雾图像数量: {len(dehaze_files)}")
    print(f"GT 图像数量: {len(gt_map)}")
    print(f"匹配规则: 去除去雾文件名中的 '_FFA' 后缀进行匹配\n")

    psnr_list, ssim_list = [], []

    for d_name in dehaze_files:
        # 获取去雾文件的文件名主体（不含后缀），例如 "A_0000_Haze4K_3356_FFA"
        d_base_full = os.path.splitext(d_name)[0]

        # -------- 关键修改点：去除自定义后缀 --------
        # 如果文件名是以 _FFA 结尾的，则去掉它
        # 也可以使用 d_base_full.rsplit('_', 1)[0] 如果后缀总是最后一个下划线后的内容
        target_base = d_base_full.replace('_FFA', '')
        # ------------------------------------------

        # 在 GT 映射中寻找处理后的 target_base
        if target_base not in gt_map:
            print(f"[跳过] 未能匹配到 GT 图像: {d_name} (尝试匹配主体: {target_base})")
            continue

        gt_name = gt_map[target_base]

        dehaze_path = os.path.join(dehaze_folder, d_name)
        gt_path = os.path.join(gt_folder, gt_name)

        dehaze_img = cv2.imread(dehaze_path)
        gt_img = cv2.imread(gt_path)

        if dehaze_img is None or gt_img is None:
            print(f"[跳过] 图像读取失败: {d_name}")
            continue

        # 尺寸一致性检查与调整
        if dehaze_img.shape[:2] != gt_img.shape[:2]:
            h, w = gt_img.shape[:2]
            dehaze_img = cv2.resize(dehaze_img, (w, h))

        # 颜色空间转换
        dehaze_img = cv2.cvtColor(dehaze_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        # 数据类型转换
        dehaze_f = dehaze_img.astype(np.float32)
        gt_f = gt_img.astype(np.float32)

        # 计算指标
        psnr_val = psnr(gt_f, gt_f, data_range=255)  # 注意：此处应为 (gt, dehaze)，原代码逻辑修正
        psnr_val = psnr(gt_f, dehaze_f, data_range=255)
        ssim_val = ssim(gt_f, dehaze_f, channel_axis=2, data_range=255)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

        print(f"成功匹配: {d_name} <-> {gt_name} | PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}")

    if psnr_list:
        print("\n" + "=" * 30)
        print(f"测试集平均 PSNR: {np.mean(psnr_list):.4f}")
        print(f"测试集平均 SSIM: {np.mean(ssim_list):.4f}")
        print(f"有效统计数量: {len(psnr_list)}")
        print("=" * 30)
    else:
        print("未找到匹配图像，请检查文件名后缀是否为 '_FFA'")

    return psnr_list, ssim_list


if __name__ == "__main__":
    # 使用 r"" 避免路径转义问题
    dehaze_folder = r"F:\Dehaze_huiyi_lunwen\1_images_duibi_experiments\additional_benchmark\FFA_Net\target_test"
    gt_folder = r"F:\Dehaze_huiyi_lunwen\2_Dataset\2_additional_benchmark\Target\clear"

    calculate_psnr_ssim(dehaze_folder, gt_folder)