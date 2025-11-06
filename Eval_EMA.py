# 导入操作系统库，用于文件路径操作
import os
# 导入 glob 库，用于查找文件
import glob
# 导入 PyTorch 核心库
import torch
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
MODEL_PATH = 'D:/liu_lan_qi_xia_zai/CoA-main_daima_xiugai_jiehe_ir_edge_xiugai_teacher_v5/ceshi_shiyongde_model/v5/Teacher_xunlian/best.pth'  # <--- 这是你训练好的 DualStreamTeacher 最佳权重路径
OUTPUT_FOLDER = 'E:/FLIR_zongti_quwu_ceshi/dataset/REAL_FOGGY_test_v5/dehazed_best_Teacher_MIXED_TEST'  # <--- 修改输出文件夹名称

# --- [修改] 掩码文件夹路径 (如果不想用掩码，请将此设置为空字符串 "") ---
# !!! 如果此路径有效，脚本会尝试加载掩码；如果此路径无效或为空，脚本将始终使用无掩码模式 !!!
INPUT_FOLDER_MASK = 'E:/FLIR_zongti_quwu_ceshi/dataset/REAL_FOGGY/mask'  # <--- [请修改] 你的雾霾掩码文件夹


# --- [修改结束] ---


# --- [修改] dehaze 函数现在接受 mask_image_path=None ---
def dehaze(model, vis_image_path, ir_image_path, mask_image_path, folder):
    """
    使用加载的双流模型对可见光、红外和可选的掩码进行去雾处理。
    """
    try:
        # 1. 加载并预处理可见光图像 (使用标准 transform)
        haze_vis = transform(Image.open(vis_image_path).convert("RGB")).unsqueeze(0).to(device)
        # 2. 加载并预处理红外图像 (使用标准 transform)
        haze_ir = transform(Image.open(ir_image_path).convert("RGB")).unsqueeze(0).to(device)

        haze_mask_tensor = None  # 默认掩码为 None

        # --- [修改] 仅当 mask_image_path 提供了才尝试加载 ---
        if mask_image_path is not None:
            if os.path.exists(mask_image_path):
                # 掩码存在，加载它
                haze_mask_tensor = transform_mask(Image.open(mask_image_path).convert("L")).unsqueeze(0).to(device)
            else:
                # 提供了掩码路径但文件丢失，打印警告，haze_mask_tensor 保持为 None
                print(f"\n警告: 提供了掩码路径但文件未找到: {mask_image_path}。将回退到无掩码模式 (GAI)。")
        # --- [修改结束] ---

        # 4. 获取原始图像尺寸 (以可见光为准)
        h, w = haze_vis.shape[2], haze_vis.shape[3]

        # 5. 调整尺寸以适应模型（16的倍数）
        target_h = (h // 16) * 16
        target_w = (w // 16) * 16
        if target_h == 0: target_h = 16
        if target_w == 0: target_w = 16

        resize_fn = Resize((target_h, target_w), interpolation=InterpolationMode.BICUBIC, antialias=True)

        if h != target_h or w != target_w:
            haze_vis_resized = resize_fn(haze_vis)
            haze_ir_resized = resize_fn(haze_ir)
        else:
            haze_vis_resized = haze_vis
            haze_ir_resized = haze_ir

        # --- [修改] 仅当掩码张量存在时才调整其尺寸 ---
        haze_mask_resized = None  # 默认 resized 掩码为 None
        if haze_mask_tensor is not None:
            resize_mask_fn = Resize((target_h, target_w), interpolation=InterpolationMode.BILINEAR, antialias=False)
            haze_mask_resized = resize_mask_fn(haze_mask_tensor) if (
                        h != target_h or w != target_w) else haze_mask_tensor

        # 6. 模型推理 (传入三个输入)
        #    - [核心] 传入 haze_mask_resized (它要么是掩码张量，要么是 None)
        pred_output = model(haze_vis_resized, haze_ir_resized, haze_mask=haze_mask_resized)

        if isinstance(pred_output, tuple):
            out_tensor = pred_output[0]
        else:
            out_tensor = pred_output

        out = out_tensor.squeeze(0)  # 移除批次维度
        out = out.clamp(0, 1)

        # 7. 将输出图像尺寸恢复到原始尺寸
        if h != target_h or w != target_w:
            out = Resize((h, w), interpolation=InterpolationMode.BICUBIC, antialias=True)(out)

        # 8. 保存去雾后的图像 (使用可见光图像的文件名)
        output_filename = os.path.basename(vis_image_path)
        torchvision.utils.save_image(out, os.path.join(folder, output_filename))

    except FileNotFoundError as e:
        print(f"\n错误: 找不到图像文件 {e}。跳过。")
    except Exception as e:
        base_name = os.path.basename(vis_image_path)
        print(f"\n处理图像 {base_name} 时发生错误: {e}。跳过。")


# --- 修改结束 ---


# Python 主程序入口点
if __name__ == '__main__':

    # 定义图像预处理流程 (保持不变)
    transform = Compose([
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    # --- [新增] 定义掩码的预处理流程 (仅 ToTensor) ---
    transform_mask = Compose([
        ToTensor()
    ])
    # --- [新增结束] ---

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

        load_result = model.load_state_dict(new_state_dict, strict=False)
        print("模型加载状态:", load_result)
        if load_result.missing_keys:
            print("Missing keys:", load_result.missing_keys)
        if load_result.unexpected_keys:
            print("Unexpected keys:", load_result.unexpected_keys)

    except FileNotFoundError:
        print(f"错误: 找不到模型权重文件 {MODEL_PATH}。请检查路径。")
        exit()  # 权重不存在则退出
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        exit()

    # 将模型设置为评估模式
    model.eval()

    # 创建输出文件夹
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # --- 修改: 指定可见光和红外输入图像文件夹 ---
    INPUT_FOLDER_VIS = 'E:/FLIR_zongti_quwu_ceshi/dataset/REAL_FOGGY/hazy'  # <--- 指定含雾可见光图像文件夹路径
    INPUT_FOLDER_IR = 'E:/FLIR_zongti_quwu_ceshi/dataset/REAL_FOGGY/ir'  # <--- 指定对应的红外图像文件夹路径
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
        use_mask_if_available = True
    else:
        print(f"掩码模式: OFF。未提供或未找到掩码文件夹。")
        print("所有图像将使用模型的内部 GAI 模块 (无掩码模式) 运行。")
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

        for vis_path in tqdm(vis_images, bar_format=bar_format, desc="混合模式去雾 😊 :"):
            # 根据可见光文件名构造对应的红外和掩码文件名
            base_filename = os.path.basename(vis_path)
            ir_path = os.path.join(INPUT_FOLDER_IR, base_filename)

            # --- [修改] 动态构造掩码路径 ---
            mask_path = None  # 默认为 None (无掩码模式)
            if use_mask_if_available:
                # 仅当掩码文件夹有效时，才构造路径
                mask_path = os.path.join(INPUT_FOLDER_MASK, base_filename)
                # 注意：我们不再检查 os.path.exists(mask_path)，
                # 而是让 dehaze 函数内部去处理（如果文件不存在，它会回退到 None）
            # --- [修改结束] ---

            if os.path.exists(ir_path):
                # [修改] 调用 dehaze，mask_path 可能是路径字符串，也可能是 None
                dehaze(model, vis_path, ir_path, mask_path, OUTPUT_FOLDER)
            else:
                print(f"\n警告: 找不到 {base_filename} 对应的红外图像: {ir_path}。跳过此图像。")
        # --- 修改结束 ---

    print(f"\n处理完成！去雾后的图像已保存到: {OUTPUT_FOLDER}")