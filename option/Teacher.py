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

# --- 1. 初始化参数解析器 ---
parser = argparse.ArgumentParser()  # 创建一个 ArgumentParser 对象

# --- 2. 定义训练相关的参数 ---

# 定义设备参数 (cpu 或 cuda)，默认为自动检测
parser.add_argument('--device', type=str, default='Automatic detection')
# 定义训练的总轮数 (epochs)
# 原始参数设置
# parser.add_argument('--epochs', type=int, default=20)
# 修改训练epoch
parser.add_argument('--epochs', type=int, default=1)
# 定义每轮训练的迭代次数 (steps)
# 原始参数设置
# parser.add_argument('--iters_per_epoch', type=int, default=5000)
# 修改参数设置
parser.add_argument('--iters_per_epoch', type=int, default=100)
# 定义一个用于更精细评估的步数阈值
parser.add_argument('--finer_eval_step', type=int, default=100)
# 定义初始学习率
parser.add_argument('--start_lr', default=0.0001, type=float, help='start learning rate')
# 定义结束学习率（用于学习率调度）
parser.add_argument('--end_lr', default=0.000001, type=float, help='end learning rate')
# 定义一个动作参数，如果命令行中包含此参数，则不使用余弦学习率调度

# 损失权重
parser.add_argument('--no_lr_sche', action='store_true', help='no lr cos schedule')
parser.add_argument('--w_loss_L1', default=0.8, type=float, help='weight of loss L1')
parser.add_argument('--w_loss_SSIM', default=0.2, type=float, help='weight of loss SSIM')
parser.add_argument('--w_loss_Cr', default=0.05, type=float, help='weight of loss Cr')
parser.add_argument('--w_loss_Edge', default=0.05, type=float, help='weight of IR Edge consistency loss')
parser.add_argument('--w_loss_Content', default=0.05, type=float, help='weight of Perceptual Content loss (VGG)')
parser.add_argument('--w_loss_Style', default=0.05, type=float, help='weight of Perceptual Style loss (VGG Gram)')

# --- 3. 定义文件和目录相关的参数 ---

# 定义实验结果的根目录
parser.add_argument('--exp_dir', type=str, default='./experiment')
# 定义当前模型的名称
parser.add_argument('--model_name', type=str, default='THaze')
# 定义保存模型的子目录名称
parser.add_argument('--saved_model_dir', type=str, default='saved_model')
# 定义保存数据（如日志、损失）的子目录名称
parser.add_argument('--saved_data_dir', type=str, default='saved_data')
# 定义数据集名称（用于构建目录结构）
parser.add_argument('--dataset', type=str, default='Teacher')

# --- [新增] 训练期间的真实世界测试集路径 ---
# (请在运行时指定这些路径，或在此处设置你的默认值)
parser.add_argument('--real_test_hazy_path', type=str, default="E:/FLIR_zongti_quwu_ceshi/dataset/FLIR_zengqiang/Teacher_xunlian_guocheng_ceshi/hazy",
                    help='Path to real-world hazy images (e.g., ./real_test/hazy)')
parser.add_argument('--real_test_ir_path', type=str, default="E:/FLIR_zongti_quwu_ceshi/dataset/FLIR_zengqiang/Teacher_xunlian_guocheng_ceshi/ir",
                    help='Path to real-world ir images (e.g., ./real_test/ir)')
parser.add_argument('--real_test_mask_path', type=str, default="E:/FLIR_zongti_quwu_ceshi/dataset/FLIR_zengqiang/Teacher_xunlian_guocheng_ceshi/test_mask",
                    help='(Optional) Path to real-world mask images (e.g., ./real_test/mask)')
# --- 4. 解析参数并设置设备 ---

opt = parser.parse_args()  # 解析命令行传入的参数
# 自动检测设备：如果 CUDA 可用，则使用 'cuda'，否则使用 'cpu'
opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 5. 构建实验目录结构 ---

# 构建特定数据集的目录路径 (例如 ./experiment/Teacher)
dataset_dir = os.path.join(opt.exp_dir, opt.dataset)
# 构建特定模型的目录路径 (例如 ./experiment/Teacher/THaze)
model_dir = os.path.join(dataset_dir, opt.model_name)

# 自动创建不存在的目录
if not os.path.exists(opt.exp_dir):
    os.mkdir(opt.exp_dir)  # 创建 ./experiment
if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)  # 创建 ./experiment/Teacher
if not os.path.exists(model_dir):
    os.mkdir(model_dir)  # 创建 ./experiment/Teacher/THaze
    # 更新 opt 对象中的路径，使其指向新建的模型目录
    opt.saved_model_dir = os.path.join(model_dir, 'saved_model')
    opt.saved_data_dir = os.path.join(model_dir, 'saved_data')
    os.mkdir(opt.saved_model_dir)  # 创建 ./experiment/Teacher/THaze/saved_model
    os.mkdir(opt.saved_data_dir)  # 创建 ./experiment/Teacher/THaze/saved_data

# --- 6. 保存配置参数 ---

# 将所有配置参数 (opt 对象) 保存为 JSON 格式的文本文件
with open(os.path.join(model_dir, 'args.txt'), 'w') as f:
    json.dump(opt.__dict__, f, indent=2)  # indent=2 使 JSON 文件格式更易读