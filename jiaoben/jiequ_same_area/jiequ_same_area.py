import cv2
import os
import glob
import numpy as np


def cv_imread(file_path):
    """
    使用 numpy 读取包含中文路径的图片
    """
    # -1 表示原样读取，包含 Alpha 通道
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def cv_imwrite(file_path, img):
    """
    使用 numpy 保存图片到包含中文的路径
    """
    ext = os.path.splitext(file_path)[1]
    # 将图片编码为内存缓冲区，再写入文件
    res, img_encode = cv2.imencode(ext, img)
    if res:
        img_encode.tofile(file_path)


def batch_crop_images(input_dir, output_dir):
    """
    手动截取第一张图像的区域，并批量应用到文件夹下的所有图像（支持中文路径）。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    image_paths = []

    # 获取所有图片路径
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not image_paths:
        print(f"在文件夹 '{input_dir}' 中没有找到支持的图像文件。")
        return

    print(f"共找到 {len(image_paths)} 张图像，准备选择截取区域...")

    # 使用修改后的读取函数
    first_image_path = image_paths[0]
    img = cv_imread(first_image_path)

    if img is None:
        print(f"无法读取第一张图像: {first_image_path}")
        return

    window_name = "Select Crop Region"
    print("--------------------------------------------------")
    print("操作指南：")
    print("1. 鼠标左键拖拽画框选择区域。")
    print("2. 按下 'Enter' 或 'Space' 确认。")
    print("3. 按下 'C' 键取消并退出。")
    print("--------------------------------------------------")

    roi = cv2.selectROI(window_name, img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)

    x, y, w, h = roi

    if w == 0 or h == 0:
        print("未选择有效区域，程序已退出。")
        return

    print(f"选定的截取区域：坐标(X:{x}, Y:{y})，宽:{w}，高:{h}")
    print("开始批量处理图像...")

    success_count = 0
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        current_img = cv_imread(img_path)  # 使用修改后的读取函数

        if current_img is None:
            print(f"警告：无法读取图像 {filename}，已跳过。")
            continue

        cropped_img = current_img[int(y):int(y + h), int(x):int(x + w)]

        save_path = os.path.join(output_dir, filename)

        # 使用修改后的写入函数
        cv_imwrite(save_path, cropped_img)
        print(f"已保存: {save_path}")
        success_count += 1

    print(f"\n处理完成！共成功处理 {success_count} 张图像。")


# ==========================================
# 路径设置：建议在路径前加 r 避免转义字符错误
# ==========================================
if __name__ == "__main__":
    # 使用原始字符串 r"" 以防止 \A, \f 等被当做转义字符
    INPUT_FOLDER = r"F:\ACMMM2026\论文里的图优化\ReMix方案图优化\截图"
    OUTPUT_FOLDER = r"F:\ACMMM2026\论文里的图优化\ReMix方案图优化\截图\1可见光保持"

    batch_crop_images(INPUT_FOLDER, OUTPUT_FOLDER)
