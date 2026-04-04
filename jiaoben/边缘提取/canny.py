import cv2
import numpy as np
import os

def extract_edges(input_path, output_path):
    # --- 兼容中文路径的读取方式 ---
    # 使用 np.fromfile 读取为二进制流，再用 cv2.imdecode 解码
    try:
        image = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"读取异常: {e}")
        return

    if image is None:
        print(f"错误：无法找到或读取图像，请检查路径是否正确：\n{input_path}")
        return

    # --- 图像处理步骤 ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    # --- 兼容中文路径的保存方式 ---
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 使用 cv2.imencode 将图片编码，再用 tofile 写入文件
    # '.png' 是指定的保存格式
    extension = os.path.splitext(output_path)[1]
    result, encoded_img = cv2.imencode(extension, edges)
    if result:
        encoded_img.tofile(output_path)
        print(f"成功：边缘图已保存至 {output_path}")
    else:
        print("错误：图像编码失败，无法保存。")

# --- 你的路径 ---
input_img = "F:/ACMMM2026/论文里的图优化/ReMix方案图优化/clear_image/raw/image1.png"  # 你的源文件路径
output_img = "F:/ACMMM2026/论文里的图优化/ReMix方案图优化/clear_image/edge/image1.png"  # 你想保存的位置

extract_edges(input_img, output_img)




