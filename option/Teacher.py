"""
这段Python代码是一个实验配置和初始化脚本。
它使用 argparse 库来定义和解析一系列用于深度学习训练的命令行参数，如训练轮数、学习率、损失权重和保存路径等。
在解析参数后，它会自动检测PyTorch是否可以使用CUDA（GPU），并相应地设置 opt.device。
该脚本的主要功能是根据这些配置自动创建一套层级化的实验目录（例如 ./experiment/Teacher/THaze/），并在其中生成 saved_model 和 saved_data 文件夹。
最后，它会将所有最终确定的配置参数（opt 对象）保存为一个JSON格式的 args.txt 文件，存放在新创建的模型目录中，以便于记录和复现该次训练所使用的所有设置。
"""

import argparse  # 导入用于解析命令行参数的库
import json  # 导入用于处理 JSON 数据的库
import os  # 导入用于操作系统交互（如文件路径、创建目录）的库
import torch  # 导入 PyTorch 库


def str2bool(value):
    """把命令行里输入的 true/false/1/0 等字符串转成布尔值。"""
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ('true', '1', 'yes', 'y', 'on'):
        return True
    if value in ('false', '0', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


# --- 1. 初始化参数解析器 ---
parser = argparse.ArgumentParser()  # 创建一个 ArgumentParser 对象

# --- 2. 定义训练相关的参数 ---

# 定义设备参数 (cpu 或 cuda)，默认为自动检测
parser.add_argument('--device', type=str, default='Automatic detection')
# 定义训练的总轮数 (epochs)
# 原始参数设置
# parser.add_argument('--epochs', type=int, default=20)
# 修改训练epoch
parser.add_argument('--epochs', type=int, default=20)
# 定义每轮训练的迭代次数 (steps)
# 原始参数设置
# parser.add_argument('--iters_per_epoch', type=int, default=5000)
# 修改参数设置
parser.add_argument('--iters_per_epoch', type=int, default=5000)
# 定义一个用于更精细评估的步数阈值
parser.add_argument('--finer_eval_step', type=int, default=100000)
# 定义初始学习率
parser.add_argument('--start_lr', default=0.0001, type=float, help='start learning rate')
# 定义结束学习率（用于学习率调度）
parser.add_argument('--end_lr', default=0.000001, type=float, help='end learning rate')
# 定义一个动作参数，如果命令行中包含此参数，则不使用余弦学习率调度
parser.add_argument('--no_lr_sche', action='store_true', help='no lr cos schedule')
# 当前最终清晰图重建监督: L_dehaze = L1 + SSIM + Cr
parser.add_argument('--w_loss_rec', default=0.8, type=float, help='weight of supervised clear RGB reconstruction loss')
parser.add_argument('--w_loss_density', default=1.0, type=float, help='weight of haze density L1 loss')
parser.add_argument('--w_loss_mask', default=1.0, type=float, help='weight of IR completion mask BCE+Dice loss')
parser.add_argument('--w_loss_SSIM', default=0.2, type=float, help='weight of loss SSIM')
parser.add_argument('--w_loss_Cr', default=0.05, type=float, help='weight of loss Cr')

# Edge loss is kept only as a compatibility entry for older experiments.
parser.add_argument('--w_loss_Edge', default=0.0, type=float,
                    help='deprecated for current Teacher reconstruction loss; kept for compatibility')

parser.add_argument('--gumbel_tau_start', default=1.0, type=float, help='initial Gumbel-Sigmoid temperature')
parser.add_argument('--gumbel_tau_end', default=0.1, type=float, help='final Gumbel-Sigmoid temperature')
parser.add_argument('--run_real_infer_in_teacher', type=str2bool, nargs='?', const=True, default=False,
                    help='run real-domain inference during Teacher training; default off for synthetic Teacher training')

# --- 3. 定义文件和目录相关的参数 ---

# =========================================
# 【模型与日志保存路径】
# =========================================
parser.add_argument('--saved_model_dir', type=str, default='/root/autodl-tmp/Sup3_canny/Teacher_xunlian/saved_model',
                    help='模型权重(.pth)保存目录')
parser.add_argument('--saved_data_dir', type=str, default='/root/autodl-tmp/Sup3_canny/Teacher_xunlian/saved_data',
                    help='训练日志(log.txt)、损失(losses.npy)、指标(ssims.npy/psnrs.npy)保存目录')

# =========================================
# 【训练数据集路径】 (有雾图 + 红外图 + 清晰图GT)
#   子目录结构: train_data_dir/hazy/, ir/, clear/
# =========================================
parser.add_argument('--train_data_dir', type=str, default='/root/autodl-tmp/FLIR_zengqiang/train',
                    help='训练集根目录，内含 hazy/ ir/ clear/ 三个子文件夹')

# =========================================
# 【验证/测试数据集路径】 (有雾图 + 红外图 + 清晰图GT)
#   子目录结构: test_data_dir/hazy/, ir/, clear/
# =========================================
parser.add_argument('--test_data_dir', type=str, default='/root/autodl-tmp/FLIR_zengqiang/test',
                    help='测试集根目录，内含 hazy/ ir/ clear/ 三个子文件夹')

# =========================================
# 【训练中真实世界推理 — 输入路径】
#   推理时，对 hazy 文件夹里每张图，按文件名去 ir/ 和 mask/ 找对应文件
# =========================================
# 有雾可见光图像 (主要输入)
parser.add_argument('--real_test_hazy_path', type=str, default='/root/autodl-tmp/dense_haze/hazy',
                    help='真实测试用有雾可见光图像文件夹')
# 红外图像 (辅助输入，与 hazy 图像同名)
parser.add_argument('--real_test_ir_path', type=str, default='/root/autodl-tmp/dense_haze/ir',
                    help='真实测试用红外图像文件夹')
# 掩码图 (可选，同名灰度图，标注雾区。留空则不使用掩码)
parser.add_argument('--real_test_mask_path', type=str, default='/root/autodl-tmp/dense_haze/mask',
                    help='真实测试用掩码图像文件夹（可选，留空则模型自动估计雾区）')
# 指定要进行推理的图像文件夹 (替换 real_test_hazy_path，留空则使用默认)
parser.add_argument('--real_test_specific_hazy_dir', type=str, default='',
                    help='指定要推理的图像文件夹，替换 real_test_hazy_path（留空则使用默认）')
parser.add_argument('--real_test_specific_ir_dir', type=str, default='',
                    help='与 real_test_specific_hazy_dir 配对的红外图像目录，文件名需与可见光一一对应，用于中间过程可视化')

# =========================================
# 【训练中真实世界推理 — 输出路径】
# =========================================
parser.add_argument('--real_test_output_dir', type=str,
                    default='/root/autodl-tmp/Sup3_canny/train_test/Teacher_guocheng_test',
                    help='真实测试去雾结果输出根目录，结果保存在 epoch_N/ 子文件夹下')

# =========================================
# 【实验记录路径】 (args.txt 配置存档，与训练无关)
# =========================================
parser.add_argument('--exp_dir', type=str, default='./experiment',
                    help='实验记录根目录')
parser.add_argument('--model_name', type=str, default='THaze',
                    help='模型名称，用于构建实验子目录')
parser.add_argument('--dataset', type=str, default='Teacher',
                    help='数据集名称，用于构建实验子目录')

# --- 4. 解析参数并设置设备 ---

opt = parser.parse_args()  # 解析命令行传入的参数
# 自动检测设备：如果 CUDA 可用，则使用 'cuda'，否则使用 'cpu'
opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 5. 创建保存目录 ---

# 使用默认定义的路径，直接创建 saved_model 和 saved_data 目录
os.makedirs(opt.saved_model_dir, exist_ok=True)
os.makedirs(opt.saved_data_dir, exist_ok=True)

# --- 6. 保存配置参数 ---

# 构建实验目录用于保存 args.txt
dataset_dir = os.path.join(opt.exp_dir, opt.dataset)
model_dir = os.path.join(dataset_dir, opt.model_name)
os.makedirs(model_dir, exist_ok=True)

# 将所有配置参数 (opt 对象) 保存为 JSON 格式的文本文件
with open(os.path.join(model_dir, 'args.txt'), 'w') as f:
    json.dump(opt.__dict__, f, indent=2)  # indent=2 使 JSON 文件格式更易读
