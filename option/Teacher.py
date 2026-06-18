﻿"""
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

# --- 2. 定义训练相关参数 ---

# 设备参数保留原接口；脚本解析后仍会根据 torch.cuda.is_available() 自动写成 cuda 或 cpu。
parser.add_argument('--device', type=str, default='Automatic detection',
                    help='运行设备占位参数；当前脚本会自动检测 CUDA，不建议手动修改。')

# ============================================================================
# 一、基础训练参数
# ============================================================================
# 总训练轮数。数据较多或尚未收敛时可增大；过大会延长训练并可能过拟合。初次训练保留 20。
parser.add_argument('--epochs', type=int, default=20,
                    help='总训练轮数；未收敛可增大，出现过拟合或训练时间过长可减小，初次训练建议保留 20。')
# 每个 epoch 实际执行的迭代步数。增大会让每轮学习更充分但耗时更长；减小适合快速试跑。
parser.add_argument('--iters_per_epoch', type=int, default=5000,
                    help='每个 epoch 的训练迭代数；正式训练保留 5000，快速排错可临时减小。')
# 达到该全局 step 后进入更频繁/精细的评估阶段。调小会更早评估但增加耗时，调大则减少中期评估。
parser.add_argument('--finer_eval_step', type=int, default=100000,
                    help='开始精细评估的全局步数；调小会更早评估但更耗时，初次训练建议保留 100000。')
# 初始学习率。过大可能震荡或发散，过小会收敛慢；只有确认训练稳定性问题时再改。
parser.add_argument('--start_lr', default=0.0001, type=float,
                    help='初始学习率；loss 抖动/发散时减小，收敛过慢且训练稳定时可小幅增大，默认 1e-4。')
# 学习率调度结束时的最低学习率。增大可保持后期更新力度，减小会让后期更稳定但更早停滞。
parser.add_argument('--end_lr', default=0.000001, type=float,
                    help='余弦调度的结束学习率；后期震荡可减小，后期几乎不更新可适当增大，默认 1e-6。')
# 命令行加入该开关后关闭余弦学习率调度。通常不要开启，除非需要固定学习率做对照实验。
parser.add_argument('--no_lr_sche', action='store_true',
                    help='关闭余弦学习率调度；默认不启用，只有固定学习率对照实验时使用。')

# ============================================================================
# 二、主监督损失
# ============================================================================
# 清晰 RGB 的 L1 重建主损失，决定整体去雾和颜色恢复。过大可能压制辅助任务，过小会削弱主目标。
parser.add_argument('--w_loss_rec', default=0.8, type=float,
                    help='清晰图 L1 重建权重；初次训练保留 0.8。调大更重视像素还原，调小会让辅助损失占比上升。')
# 雾密度图 L1 监督，约束 HDE 判断雾浓度。过大可能牺牲最终图像，过小会让密度引导不可靠。
parser.add_argument('--w_loss_density', default=1.0, type=float,
                    help='雾密度图监督权重；初次训练保留 1.0。密度预测偏差大可增大，主图像受干扰可减小。')
# 补全区掩码的 BCE+Dice 监督，决定 M=1/M=0 分流。过大可能让掩码任务压制重建，过小会导致区域分流不准。
parser.add_argument('--w_loss_mask', default=1.0, type=float,
                    help='补全掩码 BCE+Dice 权重；初次训练保留 1.0。掩码不准可增大，重建被掩码任务压制可减小。')
# SSIM 结构相似性损失，改善整体结构和局部对比。过大可能降低像素/颜色精度。
parser.add_argument('--w_loss_SSIM', default=0.2, type=float,
                    help='SSIM 结构损失权重；初次训练保留 0.2。结构模糊可增大，颜色或像素偏差增大时可减小。')
# 对比重建损失 Cr，帮助输出接近清晰图并远离有雾输入。过大可能出现过增强或颜色偏移。
parser.add_argument('--w_loss_Cr', default=0.05, type=float,
                    help='对比重建损失权重；初次训练保留 0.05。去雾不足可小幅增大，过增强/偏色时减小。')
# 旧边缘损失兼容项，当前训练默认关闭。除非复现实验，不建议开启或修改。
parser.add_argument('--w_loss_Edge', default=0.1, type=float,
                    help='旧版边缘损失兼容权重；当前默认 0.0 表示关闭，初次训练保持不变。')

# ============================================================================
# 三、语义对齐与补全区辅助损失
# ============================================================================
# 可靠区 IR/VIS 共享语义对齐。过大可能强迫不同模态过度一致，过小会让检索语义空间不稳定。
parser.add_argument('--w_loss_align', default=0.1, type=float,
                    help='可靠区跨模态语义对齐权重；初次训练保留 0.1。检索错配多可增大，模态细节被抹平可减小。')
# 补全区 transported RGB 与 clear GT 的像素重建，直接监督颜色搬运。过大可能过分依赖补全区颜色监督。
parser.add_argument('--w_loss_comp', default=1.0, type=float,
                    help='补全区颜色重建权重；初次训练保留 1.0。补全颜色偏差大可增大，局部颜色过拟合可减小。')
# 补全区多尺度梯度结构损失，不依赖 VGG/CLIP，主要提升边缘和纹理；过大可能颜色变硬或边缘过强。
parser.add_argument('--w_loss_comp_perc', default=0.5, type=float,
                    help='补全区多尺度梯度结构损失权重；初次训练保留 0.5。纹理模糊可增大，边缘过硬时减小。')
# 让颜色原型 attention 更稀疏、更聚焦。过大可能只依赖少数原型，导致颜色搬运单一。
parser.add_argument('--w_loss_sparse', default=0.01, type=float,
                    help='颜色原型 attention 稀疏权重；初次训练保留 0.01 并配合 warmup。颜色过于单一时减小。')
# IR 边缘感知的补全区颜色平滑项，减少颜色噪声并保护 IR 边缘；过大可能过度平滑。
parser.add_argument('--w_loss_ir_tv', default=0.05, type=float,
                    help='补全区 IR 边缘感知颜色平滑权重；初次训练保留 0.05。噪声多可增大，细节被抹平时减小。')
# 控制 IR 梯度对 TV 平滑权重的抑制强度。越大越保护强 IR 边缘但平滑区域变少，越小则平滑更均匀。
parser.add_argument('--ir_tv_edge_lambda', default=10.0, type=float,
                    help='IR-TV 的边缘敏感系数；初次训练保留 10.0。边缘被抹平可增大，噪声保留过多可减小。')

# ============================================================================
# 四、InfoNCE 语义对齐参数
# ============================================================================
# infonce 判别性更强但更依赖负样本；cosine 更简单稳定，适合消融或排查 InfoNCE 问题。
parser.add_argument('--align_mode', default='infonce', choices=['infonce', 'cosine'],
                    help='语义对齐模式；默认 infonce。训练不稳或做简单基线时可改 cosine。')
# InfoNCE softmax 温度。越小正负样本区分更尖锐但更易震荡，越大更平滑但判别力可能下降。
parser.add_argument('--align_temperature', default=0.07, type=float,
                    help='InfoNCE 温度；初次训练保留 0.07。对齐过软可减小，loss 抖动或梯度过激可增大。')
# 假负样本屏蔽阈值；越低，越多高相似非对角位置会被视为假负样本并屏蔽。
parser.add_argument('--infonce_fp_threshold', default=0.8, type=float,
                    help='InfoNCE 假负样本屏蔽阈值；初次训练保留 0.8。误把相似位置当负样本时降低，屏蔽过多时提高。')
# 每个 batch 样本最多参与 InfoNCE 的可靠位置数。增大可提供更多负样本但显存/计算上升，减小可省显存。
parser.add_argument('--infonce_max_samples', default=1024, type=int,
                    help='每张图参与 InfoNCE 的可靠位置上限；默认 1024。OOM 或训练慢时减小，对齐样本不足时增大。')
# 从“不屏蔽”退火到目标阈值的步数；初期语义空间不稳定，退火可避免误屏蔽真负样本。
parser.add_argument('--infonce_fp_warmup_steps', default=5000, type=int,
                    help='假负样本阈值退火步数；初次训练保留 5000。早期误屏蔽/不稳可增大，收敛很快可减小。')

# ============================================================================
# 五、Gumbel 掩码与颜色损失启动参数
# ============================================================================
# Gumbel-Sigmoid 初始温度。越大初期掩码越软、训练更稳；过大则二值分流形成较慢。
parser.add_argument('--gumbel_tau_start', default=1.0, type=float,
                    help='Gumbel 掩码初始温度；初次训练保留 1.0。早期掩码过硬/不稳可增大。')
# Gumbel-Sigmoid 最终温度。越小最终掩码越接近硬二值，但太小可能梯度不稳。
parser.add_argument('--gumbel_tau_end', default=0.1, type=float,
                    help='Gumbel 掩码最终温度；初次训练保留 0.1。最终掩码过软可减小，后期抖动可增大。')
# 控制补全颜色、语义对齐、稀疏、IR-TV 等辅助损失从第几步启用；太早会在语义空间未稳定时施压。
parser.add_argument('--color_loss_start_step', default=1000, type=int,
                    help='颜色相关辅助损失启动步数；初次训练建议 1000，前期不稳可改为 2000，启动太晚可减小。')
# 稀疏损失从 0 线性升到目标权重的步数。增大更平稳，减小会更早强化原型选择。
parser.add_argument('--sparse_warmup_steps', default=5000, type=int,
                    help='颜色原型稀疏损失 warmup 步数；初次训练保留 5000。早期原型塌缩可增大。')

# ============================================================================
# 六、颜色搬运与双向语义融合参数
# ============================================================================
# IR/VIS 共享语义向量维度。增大表达能力和显存/计算，减小更轻量但可能丢失语义细节。
parser.add_argument('--semantic_dim', default=128, type=int,
                    help='共享语义特征维度；初次训练保留 128。显存紧张可减小，语义表达不足可增大。')
# 可靠区颜色原型数量。增大可覆盖更多颜色类型但 attention 更分散，减小更稳定但颜色多样性下降。
parser.add_argument('--num_color_prototypes', default=32, type=int,
                    help='颜色搬运原型数；初次训练保留 32。颜色类型复杂可增大，原型冗余或不稳可减小。')
# 红外 query 到可见光颜色原型的 softmax 温度。越小选择更尖锐，越大颜色混合更平滑。
parser.add_argument('--transport_temperature', default=0.07, type=float,
                    help='颜色原型 attention 温度；默认 0.07。颜色混杂可减小，颜色跳变/不稳可增大。')
# VIS→IR 全图语义检索温度。越小更相信少数高相似 IR 位置但可能不稳定；越大更保守平滑但结构补充变弱。
parser.add_argument('--fusion_temperature', default=0.07, type=float,
                    help='VIS→IR 检索 softmax 温度；初次训练保留 0.07。检索过散可减小，注意力过尖或训练抖动可增大。')
# IR→VIS 校验门放行阈值。verify_gate 长期接近 0 表示过严，可降到 0.1；长期接近 1 表示过松，可升到 0.3。
parser.add_argument('--verify_threshold', default=0.2, type=float,
                    help='IR→VIS 校验门槛；初次训练保留 0.2。verify_gate 近 0 时降低，近 1 时提高。')
# verify gate 的 sigmoid 温度。越小越像硬开关，越大越平滑稳定但筛除热噪声能力可能变弱。
parser.add_argument('--verify_temperature', default=0.1, type=float,
                    help='IR→VIS 校验 sigmoid 温度；初次训练保留 0.1。训练抖动可试 0.2，门控过软可减小。')

# ============================================================================
# 七、训练/测试显存与加载参数
# H/4 双向融合构造 B×N×N attention。256×256 输入时 H/4=64×64，N=4096。
# 初次训练建议 batch_size=1 或 2；OOM 时先降 batch_size，再关可视化或降低输入尺寸。
# ============================================================================
parser.add_argument('--batch_size', default=8, type=int,
                    help='训练 batch；默认 8。H/4 全图 attention 占显存，OOM 时优先降为 1。')
parser.add_argument('--num_workers', default=16, type=int,
                    help='训练 DataLoader 进程数；默认 16。CPU/内存不足或 Windows 卡住时减小，GPU 等数据时可增大。')
parser.add_argument('--test_batch_size', default=8, type=int,
                    help='测试 batch；默认 8。测试显存不足时保持 1，显存充足且需加速时可增大。')
parser.add_argument('--test_num_workers', default=16, type=int,
                    help='测试 DataLoader 进程数；默认 16。CPU/内存不足或加载报错时减小。')

# ============================================================================
# 八、可视化和真实域测试开关
# ============================================================================
# 开启后训练过程中会跑真实域 hazy/IR 配对推理，会增加评估时间和显存峰值；排查 OOM 时可先关闭。
parser.add_argument('--run_real_infer_in_teacher', type=str2bool, nargs='?', const=True, default=True,
                    help='是否在 Teacher 训练中执行 real-domain overview；默认开启（default on）。训练太慢或评估阶段 OOM 时可设 false。')
# 保存 9 列训练区域可视化，便于检查 mask/融合/颜色搬运，但会增加 I/O 和少量显存占用。
parser.add_argument('--save_train_batch_region_vis', type=str2bool, nargs='?', const=True, default=True,
                    help='是否保存训练 batch 的 9 列监督图到 saved_data_dir；默认开启，I/O 慢或 OOM 排查时可关闭。')
# 每轮真实域 overview 最多保存的配对数量。增大可观察更多样本但更慢、占更多磁盘；<=0 表示全部。
parser.add_argument('--real_vis_max_images', default=0, type=int,
                    help='每个 epoch 最多展示的真实域 hazy/IR 配对数；默认 8，<=0 表示全部。')

# ============================================================================
# 九、路径参数
# ============================================================================
# 模型权重输出目录；脚本会自动创建，确保所在磁盘空间充足且当前用户可写。
parser.add_argument('--saved_model_dir', type=str, default='/root/autodl-tmp/quwu_daima_xiugai_shanchuan_github_TMMv2/Teacher_xunlian/saved_model',
                    help='模型 checkpoint（.pth）保存目录；会自动创建，需有写权限和足够磁盘空间。')
# 日志、损失、指标和训练可视化输出目录；脚本会自动创建。
parser.add_argument('--saved_data_dir', type=str, default='/root/autodl-tmp/quwu_daima_xiugai_shanchuan_github_TMMv2/Teacher_xunlian/saved_data',
                    help='训练日志、loss/指标数组及区域可视化保存目录；会自动创建并需要写权限。')
# 合成训练集根目录，五类子目录缺一不可，并应使用可配对的文件名。
parser.add_argument('--train_data_dir', type=str, default='/root/autodl-tmp/1_FLIR_M3FD/1_FLIR_autodl/1_train',
                    help='训练集根目录；必须包含 clear/、hazy/、ir/、Transmission_Map_GT/、IR_Completion_Mask_GT/。')
# 有 GT 的验证/测试集根目录，三个子目录中的文件应能按名称配对。
parser.add_argument('--test_data_dir', type=str, default='/root/autodl-tmp/1_FLIR_M3FD/1_FLIR_autodl/2_test',
                    help='测试集根目录；必须包含 hazy/、ir/、clear/ 三个子目录。')
# 真实域有雾可见光目录，与 real_test_ir_path 中的红外文件必须一一同名对应。
parser.add_argument('--real_test_hazy_path', type=str, default='/root/autodl-tmp/1_FLIR_M3FD/1_FLIR_autodl/3_train_test/hazy',
                    help='真实域有雾可见光目录；文件名必须与 real_test_ir_path 中红外图一一对应。')
# 真实域红外目录，与 real_test_hazy_path 中的可见光文件必须一一同名对应。
parser.add_argument('--real_test_ir_path', type=str, default='/root/autodl-tmp/1_FLIR_M3FD/1_FLIR_autodl/3_train_test/ir',
                    help='真实域红外目录；文件名必须与 real_test_hazy_path 中可见光图一一对应。')
# 真实域 overview 输出根目录，结果会继续写入 epoch_N/overview.png。
parser.add_argument('--real_test_output_dir', type=str,
                    default='/root/autodl-tmp/quwu_daima_xiugai_shanchuan_github_TMMv2/train_test/Teacher_guocheng_test',
                    help='真实域结果根目录；各轮 overview 写入该目录下的 epoch_N/overview.png。')
# 实验配置归档根目录；实际 args.txt 保存到 exp_dir/dataset/model_name/args.txt。
parser.add_argument('--exp_dir', type=str, default='./experiment',
                    help='实验配置根目录；需可写，内部按 dataset/model_name/ 保存 args.txt。')
# 实验子目录名称，不改变网络结构；建议填写能区分数据版本和实验设置的名称。
parser.add_argument('--model_name', type=str, default='1_FLIR_M3FD',
                    help='实验名称，用于组成 exp_dir/dataset/model_name/；修改它可避免覆盖不同实验的 args.txt。')
# 实验分组目录名称，不是训练数据路径；与 model_name 一起组织配置记录。
parser.add_argument('--dataset', type=str, default='Teacher',
                    help='实验分组名称，用于组成 exp_dir/dataset/model_name/，不控制实际数据读取路径。')

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
