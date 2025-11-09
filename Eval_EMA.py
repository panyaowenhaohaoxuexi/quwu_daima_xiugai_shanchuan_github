#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入操作系统库，用于文件路径操作
import os
# 导入 glob 库，用于查找文件
import glob
# 导入 PyTorch 核心库
import torch
# 导入 PyTorch 神经网络函数库 (用于插值)
import torch.nn.functional as F
# 导入 torchvision 库，用于图像处理和保存
import torchvision
# 从 PIL (Pillow) 库导入 Image 模块，用于打开图像
from PIL import Image
# 从 tqdm 库导入 tqdm 模块，用于显示进度条
from tqdm import tqdm
# --- 修改: 导入 VIFNetInconsistencyTeacher ---
from model import VIFNetInconsistencyTeacher
# --- 修改结束 ---
# 从 torchvision.transforms 导入图像变换相关的类和函数
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode

# --- 模型路径和输出文件夹配置 ---
MODEL_PATH = '/root/autodl-tmp/CoA-main_daima_xiugai_teacher_v6/Teacher_xunlian/saved_model/best.pth'  # <--- 这是你训练好的 DualStreamTeacher 最佳权重路径
OUTPUT_FOLDER = '/root/autodl-tmp/CoA-main_daima_xiugai_teacher_v10/v6_code_xiugai_test/Teacher_model_test_v4'  # <--- 修改输出文件夹名称

# --- [修改] 掩码文件夹路径 (如果不想用掩码，请将此设置为空字符串 "") ---
INPUT_FOLDER_MASK = '/root/autodl-tmp/REAL_FOGGY_autodl/hazy_mask'  # <--- [请修改] 你的雾霾掩码文件夹

# --- [新增] 最小去雾力度配置 ---
# 定义在掩码为 0 区域 (清晰区域) 施加的最小去雾力度。
# 1.0 = 完全使用模型去雾结果 (等同于GAI)
# 0.3 = 30%的模型结果 + 70%的原始图像
# 0.0 = 完全使用原始图像 (等同于上一版蒙版合成)
MIN_DEHAZE_STRENGTH = 0.7  # <--- 你可以调整这个值 (例如 0.2 或 0.4)
# --- [新增结束] ---


# --- 1. 用于模型输入的标准 Transform (带归一化) ---
transform = Compose([
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

# --- 2. 仅用于加载掩码的 Transform (仅 ToTensor) ---
transform_mask = Compose([
    ToTensor()
])

# --- 3. 仅用于加载原始可见光图像（不归一化） ---
# (用于在非雾气区域“粘贴”回原图)
transform_to_tensor_only = Compose([
    ToTensor()
])


# --- [新增结束] ---


# --- [核心修改] dehaze 函数实现了 "可变强度融合" 逻辑 ---
def dehaze(model, vis_image_path, ir_image_path, mask_image_path, folder):
    """
    [修改后]
    使用加载的双流模型对可见光、红外和可选的掩码进行去雾处理。

    核心逻辑:
    1. 如果 mask_image_path 为 None 或文件不存在 (轻雾):
       - 最终输出 = 模型的全局去雾结果 (GAI)。
    2. 如果 mask_image_path 存在 (浓雾):
       - alpha 蒙版 = MIN_STRENGTH + Mask * (1.0 - MIN_STRENGTH)
       - 最终输出 = (模型结果 * alpha) + (原始图像 * (1 - alpha))
    """
    try:
        # 1. 加载所有 PIL 图像
        haze_vis_pil = Image.open(vis_image_path).convert("RGB")
        haze_ir_pil = Image.open(ir_image_path).convert("RGB")

        # 2. 准备模型输入 (标准 transform, 带归一化)
        haze_vis = transform(haze_vis_pil).unsqueeze(0).to(device)
        haze_ir = transform(haze_ir_pil).unsqueeze(0).to(device)

        # 3. 准备用于 "融合" 的原始可见光图像 (仅 ToTensor, 0-1 范围)
        haze_vis_original_tensor = transform_to_tensor_only(haze_vis_pil).unsqueeze(0).to(device)

        haze_mask_tensor = None  # 默认掩码为 None

        # 4. 加载掩码 (0-1 范围)
        if mask_image_path is not None:
            if os.path.exists(mask_image_path):
                # 掩码存在 (浓雾)，加载它 (L模式, ToTensor 会自动转为 0-1)
                haze_mask_tensor = transform_mask(Image.open(mask_image_path).convert("L")).unsqueeze(0).to(device)
            else:
                # 提供了掩码路径但文件丢失 (轻雾)
                print(f"\n信息: 未找到掩码: {mask_image_path}。将执行全局处理 (GAI)。")
                # haze_mask_tensor 保持为 None

        # 5. 获取原始图像尺寸
        h, w = haze_vis.shape[2], haze_vis.shape[3]

        # 6. 调整尺寸以适应模型（16的倍数）
        target_h = (h // 16) * 16
        target_w = (w // 16) * 16
        if target_h == 0: target_h = 16
        if target_w == 0: target_w = 16

        resize_fn = Resize((target_h, target_w), interpolation=InterpolationMode.BICUBIC, antialias=True)
        resize_mask_fn = Resize((target_h, target_w), interpolation=InterpolationMode.BILINEAR, antialias=False)

        haze_mask_resized = None
        haze_vis_original_resized = None

        if h != target_h or w != target_w:
            haze_vis_resized = resize_fn(haze_vis)
            haze_ir_resized = resize_fn(haze_ir)
            haze_vis_original_resized = resize_fn(haze_vis_original_tensor)
            if haze_mask_tensor is not None:
                haze_mask_resized = resize_mask_fn(haze_mask_tensor)
        else:
            # 尺寸未变
            haze_vis_resized = haze_vis
            haze_ir_resized = haze_ir
            haze_vis_original_resized = haze_vis_original_tensor
            if haze_mask_tensor is not None:
                haze_mask_resized = haze_mask_tensor

        # 7. 模型推理 (不传入掩码, 使用 GAI)
        pred_output = model(haze_vis_resized, haze_ir_resized)

        if isinstance(pred_output, tuple):
            out_tensor_processed = pred_output[0]
        else:
            out_tensor_processed = pred_output

        # (B, C, H_t, W_t) -> (C, H_t, W_t)
        out_processed_squeezed = out_tensor_processed.squeeze(0).clamp(0, 1)
        # (B, C, H_t, W_t) -> (C, H_t, W_t)
        vis_original_squeezed = haze_vis_original_resized.squeeze(0)

        # 8. [核心修改]：执行可变强度融合

        final_out_resized = None
        if haze_mask_resized is not None:
            # --- 掩码不为None (浓雾情况) ---

            # 1. 获取掩码 (B, 1, H_t, W_t) -> (1, H_t, W_t)
            mask_float = haze_mask_resized.squeeze(0)
            # 确保掩码值在 0-1 之间 (因为 ToTensor 已经转换)
            mask_float = mask_float.clamp(0.0, 1.0)

            # 2. 计算 Alpha 蒙版 (映射 0->MIN_STRENGTH, 1->1.0)
            # alpha_final = 0.3 + mask_float * (1.0 - 0.3)
            alpha_final = MIN_DEHAZE_STRENGTH + (mask_float * (1.0 - MIN_DEHAZE_STRENGTH))
            # alpha_final 形状为 (1, H_t, W_t)，可以广播到 (C, H_t, W_t)

            # 3. 合成：(处理结果 * alpha) + (原始图像 * (1 - alpha))
            final_out_resized = (out_processed_squeezed * alpha_final) + (vis_original_squeezed * (1.0 - alpha_final))

        else:
            # --- 掩码为None (轻雾情况) ---
            # 最终输出 = 模型的全局处理结果 (GAI)
            final_out_resized = out_processed_squeezed

        # 9. 将输出图像尺寸恢复到原始尺寸
        out_final = final_out_resized
        if h != target_h or w != target_w:
            out_final = Resize((h, w), interpolation=InterpolationMode.BICUBIC, antialias=True)(final_out_resized)

        # 10. 保存去雾后的图像
        output_filename = os.path.basename(vis_image_path)
        torchvision.utils.save_image(out_final, os.path.join(folder, output_filename))

    except FileNotFoundError as e:
        print(f"\n错误: 找不到图像文件 {e}。跳过。")
    except Exception as e:
        base_name = os.path.basename(vis_image_path)
        print(f"\n处理图像 {base_name} 时发生错误: {e}。跳过。")


# --- 修改结束 ---


# Python 主程序入口点
if __name__ == '__main__':

    # 自动检测计算设备 (保持不变)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 修改: 实例化 VIFNetInconsistencyTeacher 模型 ---
    model = VIFNetInconsistencyTeacher().to(device)
    print(f"正在加载模型: VIFNetInconsistencyTeacher")
    # --- 修改结束 ---

    # --- 加载预训练的模型权重 ---
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        from collections import OrderedDict

        new_state_dict = OrderedDict()

        has_module_prefix = any(k.startswith('module.') for k in checkpoint.keys())

        for k, v in checkpoint.items():
            name = k[7:] if has_module_prefix else k
            new_state_dict[name] = v

        # 我们没有修改模型架构，所以权重应该能严格匹配 (strict=True)
        load_result = model.load_state_dict(new_state_dict, strict=True)
        print("模型加载状态:", load_result)
        # 如果使用 strict=False，取消以下注释来检查问题
        # if load_result.missing_keys:
        #     print("Missing keys:", load_result.missing_keys)
        # if load_result.unexpected_keys:
        #     print("Unexpected keys:", load_result.unexpected_keys)

    except FileNotFoundError:
        print(f"错误: 找不到模型权重文件 {MODEL_PATH}。请检查路径。")
        exit()  # 权重不存在则退出
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        print("提示: 如果看到 'Missing keys' 或 'Unexpected keys' 错误, 尝试在 load_state_dict 中设置 strict=False")
        exit()

    # 将模型设置为评估模式
    model.eval()

    # 创建输出文件夹
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # --- 修改: 指定可见光和红外输入图像文件夹 ---
    INPUT_FOLDER_VIS = '/root/autodl-tmp/REAL_FOGGY_autodl/hazy'  # <--- 指定含雾可见光图像文件夹路径
    INPUT_FOLDER_IR = '/root/autodl-tmp/REAL_FOGGY_autodl/ir'  # <--- 指定对应的红外图像文件夹路径
    # (掩码文件夹已在顶部 INPUT_FOLDER_MASK 定义)
    # --- 修改结束 ---

    # 检查输入文件夹是否存在
    if not os.path.isdir(INPUT_FOLDER_VIS):
        print(f"错误: 可见光输入文件夹不存在: {INPUT_FOLDER_VIS}")
        exit()
    if not os.path.isdir(INPUT_FOLDER_IR):
        print(f"错误: 红外输入文件夹不存在: {INPUT_FOLDER_IR}")
        exit()

    # --- [修改] 检查掩码文件夹是否有效 ---
    use_mask_if_available = False
    if INPUT_FOLDER_MASK and os.path.isdir(INPUT_FOLDER_MASK):
        print(f"掩码模式: ON。将从以下路径加载掩码 (如果存在): {INPUT_FOLDER_MASK}")
        print(f"清晰区域 (掩码=0) 将保留 {MIN_DEHAZE_STRENGTH * 100:.0f}% 的去雾效果。")
        use_mask_if_available = True
    else:
        print(f"掩码模式: OFF。未提供或未找到掩码文件夹。")
        print("所有图像将使用模型的完整去雾效果 (GAI)。")
    # --- [修改结束] ---

    # --- 查找图像文件对 ---
    print(f"正在从 {INPUT_FOLDER_VIS} 查找图像文件...")
    vis_images = sorted(glob.glob(os.path.join(INPUT_FOLDER_VIS, '*.jpg')) + \
                        glob.glob(os.path.join(INPUT_FOLDER_VIS, '*.png')) + \
                        glob.glob(os.path.join(INPUT_FOLDER_VIS, '*.jpeg')))

    if not vis_images:
        print(f"错误: 在 {INPUT_FOLDER_VIS} 中未找到任何支持的图像文件 (.jpg, .png, .jpeg)。")
        exit()
    print(f"找到 {len(vis_images)} 个可见光图像文件。")

    # 设置 tqdm 进度条格式
    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} | Elapsed: {elapsed} | Rate: {rate_fmt} items/sec"

    # --- 禁用梯度计算，开始处理图像 ---
    with torch.no_grad():
        print(f"开始处理来自 {INPUT_FOLDER_VIS} 和 {INPUT_FOLDER_IR} 的图像对...")

        for vis_path in tqdm(vis_images, bar_format=bar_format, desc="可变强度去雾 😊 :"):
            # 根据可见光文件名构造对应的红外和掩码文件名
            base_filename = os.path.basename(vis_path)
            ir_path = os.path.join(INPUT_FOLDER_IR, base_filename)

            # --- [修改] 动态构造掩码路径 ---
            mask_path = None  # 默认为 None (无掩码模式)
            if use_mask_if_available:
                # 仅当掩码文件夹有效时，才构造路径
                mask_path = os.path.join(INPUT_FOLDER_MASK, base_filename)
                # 注意：我们让 dehaze 函数内部去处理文件是否存在
            # --- [修改结束] ---

            if os.path.exists(ir_path):
                # [修改] 调用 dehaze，mask_path 可能是路径字符串，也可能是 None
                dehaze(model, vis_path, ir_path, mask_path, OUTPUT_FOLDER)
            else:
                print(f"\n警告: 找不到 {base_filename} 对应的红外图像: {ir_path}。跳过此图像。")
        # --- 修改结束 ---

    print(f"\n处理完成！去雾后的图像已保存到: {OUTPUT_FOLDER}")