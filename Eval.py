# # 原始
# import os
# import glob
# import torch
# import torchvision
# from PIL import Image
# from tqdm import tqdm
# from model import Teacher, Student, Student_x
# from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode
#
#
# # --- 模型路径和输出文件夹配置 ---
# # 你可以选择加载哪个模型的权重进行评估，取消对应行的注释即可
#
# # 教师模型权重路径
# # MODEL_PATH = './model/Teacher_model/Teacher.pth'
# # 教师模型输出结果保存目录
# # OUTPUT_FOLDER = './outputs/Teacher'
#
# # 学生模型权重路径
# # MODEL_PATH = './model/Student_model/Student.pth'
# # 学生模型输出结果保存目录
# # OUTPUT_FOLDER = './outputs/Student'
#
# # 当前选择加载 EMA (或 EMA_r) 模型权重
# # EMA 模型权重路径
# MODEL_PATH = './model/EMA_model/EMA_r.pth'
# # EMA 模型输出结果保存目录
# OUTPUT_FOLDER = './outputs/EMA'
#
#
# # 定义函数 dehaze：对单张图像执行去雾处理并保存结果
# # model: 加载好的去雾模型
# # image_path: 输入的含雾图像文件路径
# # folder: 保存去雾结果的文件夹路径
# def dehaze(model, image_path, folder):
#     """
#         使用加载的模型对指定路径的图像进行去雾处理，并将结果保存到指定文件夹。
#
#         参数:
#             model (nn.Module): 预训练好的去雾模型。
#             image_path (str): 输入含雾图像的文件路径。
#             folder (str): 保存去雾后图像的文件夹路径。
#         """
#     haze = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
#     h, w = haze.shape[2], haze.shape[3]
#     haze = Resize((h // 16 * 16, w // 16 * 16), interpolation=InterpolationMode.BICUBIC, antialias=True)(haze)
#     out = model(haze)[0].squeeze(0)
#     out = Resize((h, w), interpolation=InterpolationMode.BICUBIC, antialias=True)(out)
#     torchvision.utils.save_image(out, os.path.join(folder, os.path.basename(image_path)))
#
#
# if __name__ == '__main__':
#
#     transform = Compose([
#         ToTensor(),
#         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # model = Teacher().to(device)
#     # model = Student().to(device)
#     model = Student_x().to(device)
#
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.eval()
#
#     os.makedirs(OUTPUT_FOLDER, exist_ok=True)
#
#     INPUT_FOLDER = './test'
#
#     images = glob.glob(os.path.join(INPUT_FOLDER, '*jpg')) + glob.glob(os.path.join(INPUT_FOLDER, '*png')) + glob.glob(os.path.join(INPUT_FOLDER, '*jpeg'))
#
#     bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} | Elapsed: {elapsed} | Rate: {rate_fmt} items/sec"
#     with torch.no_grad():
#         for image in tqdm(images, bar_format=bar_format, desc="Models are struggling to get out of the fog 😊 :"):
#             dehaze(model, image, OUTPUT_FOLDER)


# # 修改为双流：可见光-红外
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
# --- 修改: 导入 DualStreamTeacher ---
# 从 model 模块导入 Teacher, Student, Student_x 以及我们修改后的 DualStreamTeacher
from model import Teacher, Student, Student_x, VIFNetInconsistencyTeacher
# --- 修改结束 ---
# 从 torchvision.transforms 导入图像变换相关的类和函数
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode

# --- 模型路径和输出文件夹配置 ---
# --- 修改: 确保加载的是训练好的 DualStreamTeacher 模型权重 ---
# MODEL_PATH = './model/Teacher_model/Teacher.pth' # 如果是原始 Teacher
# MODEL_PATH = './model/Student_model/Student.pth' # 如果是 Student
# MODEL_PATH = './model/EMA_model/EMA_r.pth' # 如果是 EMA 适配后的 Student_x
# 修改: 使用用户提供的硬编码路径
MODEL_PATH = 'D:/liu_lan_qi_xia_zai/CoA-main_daima_xiugai_jiehe_ir_edge/ceshi_shiyongde_model/v3/Student_real_model/best_student.pth'  # <--- 这是你训练好的 DualStreamTeacher 最佳权重路径
# 修改: 使用用户提供的硬编码路径
OUTPUT_FOLDER = 'E:/FLIR_zongti_quwu_ceshi/dataset/REAL_FOGGY_test_v3/dehazed_best_student' # <--- 修改输出文件夹名称以区分 (避免覆盖 test_data)
# --- 修改结束 ---


# --- 修改: dehaze 函数接收可见光和红外两个图像路径 ---
# 定义函数 dehaze：对一对可见光和红外图像执行去雾处理并保存结果
# model: 加载好的双流去雾模型
# vis_image_path: 输入的含雾可见光图像文件路径
# ir_image_path: 输入的红外图像文件路径
# folder: 保存去雾结果的文件夹路径
def dehaze(model, vis_image_path, ir_image_path, folder):
    """
    使用加载的双流模型对指定路径的可见光和红外图像进行去雾处理，
    并将结果保存到指定文件夹。

    参数:
        model (nn.Module): 预训练好的双流去雾模型 (DualStreamTeacher)。
        vis_image_path (str): 输入含雾可见光图像的文件路径。
        ir_image_path (str): 输入红外图像的文件路径。
        folder (str): 保存去雾后图像的文件夹路径。
    """
    try:
        # 1. 加载并预处理可见光图像
        haze_vis = transform(Image.open(vis_image_path).convert("RGB")).unsqueeze(0).to(device)
        # 2. 加载并预处理红外图像
        haze_ir = transform(Image.open(ir_image_path).convert("RGB")).unsqueeze(0).to(device) # 假设红外也用相同 transform

        # 3. 获取原始图像尺寸 (以可见光为准)
        h, w = haze_vis.shape[2], haze_vis.shape[3]

        # 4. 调整两个输入图像的尺寸以适应模型（可选）
        #    - 确保两个输入的尺寸调整方式一致
        #    - 将高度和宽度调整为最接近的 16 的倍数，向下取整
        target_h = (h // 16) * 16
        target_w = (w // 16) * 16
        # 如果原始尺寸已经是16的倍数，则无需调整
        if target_h == 0: target_h = 16 # 防止尺寸为0
        if target_w == 0: target_w = 16
        if h != target_h or w != target_w:
             haze_vis_resized = Resize((target_h, target_w), interpolation=InterpolationMode.BICUBIC, antialias=True)(haze_vis)
             haze_ir_resized = Resize((target_h, target_w), interpolation=InterpolationMode.BICUBIC, antialias=True)(haze_ir) # 对红外也应用
        else:
             haze_vis_resized = haze_vis
             haze_ir_resized = haze_ir


        # 5. 模型推理 (传入两个输入)
        #    - 调用 model 并传入可见光和红外两个张量
        pred_output = model(haze_vis_resized, haze_ir_resized) # 获取模型输出
        #   - 检查输出是否为元组，并获取图像部分
        if isinstance(pred_output, tuple):
             out_tensor = pred_output[0] # 取图像输出 (通常是第一个元素)
        else:
             out_tensor = pred_output # 如果只返回图像张量

        out = out_tensor.squeeze(0) # 移除批次维度

        # 6. 将输出图像尺寸恢复到原始尺寸
        #    - 只有在输入时调整过尺寸才需要恢复
        if h != target_h or w != target_w:
            out = Resize((h, w), interpolation=InterpolationMode.BICUBIC, antialias=True)(out)

        # 7. 保存去雾后的图像 (使用可见光图像的文件名)
        output_filename = os.path.basename(vis_image_path)
        # 确保输出值在 [0, 1] 范围内 (尽管模型末尾可能有 clamp，但保存前检查更安全)
        # 注意：save_image 会自动处理从 [-1, 1] 或 [0, 1] 范围转换到 [0, 255]
        # 但如果模型输出范围不确定，最好先手动 clamp
        # out = torch.clamp(out, 0, 1) # 根据需要取消注释
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

    # 自动检测计算设备 (保持不变)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 修改: 实例化 DualStreamTeacher 模型 ---
    model = VIFNetInconsistencyTeacher().to(device)
    print(f"正在加载模型: DualStreamTeacher")
    # --- 修改结束 ---

    # --- 加载预训练的模型权重 ---
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        # 检查权重字典的键是否以 'module.' 开头 (DataParallel 保存的)
        has_module_prefix = any(k.startswith('module.') for k in checkpoint.keys())

        # --- 修正: 遍历 checkpoint.items() ---
        for k, v in checkpoint.items(): # <--- 修改这里，遍历键值对
            name = k[7:] if has_module_prefix else k # 移除 'module.' 前缀 (如果存在)
            new_state_dict[name] = v
        # --- 修正结束 ---

        # 加载权重, strict=False 允许部分不匹配
        load_result = model.load_state_dict(new_state_dict, strict=False)
        print("模型加载状态:", load_result)
        # 打印不匹配的键以供调试
        if load_result.missing_keys:
             print("Missing keys:", load_result.missing_keys)
        if load_result.unexpected_keys:
             print("Unexpected keys:", load_result.unexpected_keys)

    except FileNotFoundError:
        print(f"错误: 找不到模型权重文件 {MODEL_PATH}。请检查路径。")
        exit() # 权重不存在则退出
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        exit()

    # 将模型设置为评估模式
    model.eval()

    # 创建输出文件夹
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # --- 修改: 指定可见光和红外输入图像文件夹 ---
    # 修改: 使用用户提供的文件夹路径
    INPUT_FOLDER_VIS = 'E:/FLIR_zongti_quwu_ceshi/dataset/REAL_FOGGY/hazy' # <--- 指定含雾可见光图像文件夹路径
    INPUT_FOLDER_IR = 'E:/FLIR_zongti_quwu_ceshi/dataset/REAL_FOGGY/ir'    # <--- 指定对应的红外图像文件夹路径
    # 检查输入文件夹是否存在
    if not os.path.isdir(INPUT_FOLDER_VIS):
        print(f"错误: 可见光输入文件夹不存在: {INPUT_FOLDER_VIS}")
        exit()
    if not os.path.isdir(INPUT_FOLDER_IR):
        print(f"错误: 红外输入文件夹不存在: {INPUT_FOLDER_IR}")
        exit()
    # --- 修改结束 ---

    # --- 查找图像文件对 ---
    # 以可见光文件夹为基准查找所有支持的图像文件
    print(f"正在从 {INPUT_FOLDER_VIS} 查找图像文件...")
    vis_images = sorted(glob.glob(os.path.join(INPUT_FOLDER_VIS, '*.jpg')) + \
                   glob.glob(os.path.join(INPUT_FOLDER_VIS, '*.png')) + \
                   glob.glob(os.path.join(INPUT_FOLDER_VIS, '*.jpeg')))

    if not vis_images:
        print(f"错误: 在 {INPUT_FOLDER_VIS} 中未找到任何支持的图像文件 (.jpg, .png, .jpeg)。")
        exit()
    print(f"找到 {len(vis_images)} 个可见光图像文件。")
    # --- 修改结束 ---

    # 设置 tqdm 进度条格式
    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} | Elapsed: {elapsed} | Rate: {rate_fmt} items/sec"

    # --- 禁用梯度计算，开始处理图像 ---
    with torch.no_grad():
        print(f"开始处理来自 {INPUT_FOLDER_VIS} 和 {INPUT_FOLDER_IR} 的图像对...")
        # 遍历可见光图像列表
        for vis_path in tqdm(vis_images, bar_format=bar_format, desc="双流模型正在努力去雾 😊 :"):
            # 根据可见光文件名构造对应的红外文件名
            base_filename = os.path.basename(vis_path)
            ir_path = os.path.join(INPUT_FOLDER_IR, base_filename)

            # 检查对应的红外文件是否存在
            if os.path.exists(ir_path):
                # 调用修改后的 dehaze 函数，传入两个路径
                dehaze(model, vis_path, ir_path, OUTPUT_FOLDER)
            else:
                print(f"\n警告: 找不到与 {base_filename} 对应的红外图像: {ir_path}。跳过此对。")
        # --- 修改结束 ---

    print(f"\n处理完成！去雾后的图像已保存到: {OUTPUT_FOLDER}")