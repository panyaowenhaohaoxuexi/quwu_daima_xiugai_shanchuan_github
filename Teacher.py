"""
这段PyTorch代码是一个完整的深度学习训练脚本，其核心作用是训练一个名为 Teacher 的神经网络模型来执行图像去雾（Image Dehazing）任务。
该脚本加载 THaze 数据集中的有雾图像和对应的清晰图像作为训练数据。在训练过程中，它将有雾图像输入 Teacher 模型，模型会输出一张预测的去雾图像。
接着，脚本会计算一个加权的复合损失函数来评估预测图像与真实清晰图像之间的差距，这个损失函数结合了L1损失（保证像素级准确性）、SSIM损失（保证结构相似性）和一种对比度损失（ContrastLoss，可能用于提升图像清晰度）。
脚本使用Adam优化器和余弦衰减学习率策略来反向传播损失并更新模型参数。在训练期间，脚本会周期性地在测试集上评估模型的性能，使用PSNR（峰值信噪比）和SSIM（结构相似性）这两个关键指标来衡量去雾效果。
最后，它会保存训练过程中的模型权重，并且会特别跟踪PSNR最高的模型，将其保存为 best.pth，以供后续使用。
"""

# import math  # 导入数学库
# import os  # 导入操作系统库，用于文件路径操作
# import time  # 导入时间库
# import numpy as np  # 导入Numpy库
# import torch  # 导入PyTorch
# import torch.nn.functional as F  # 导入PyTorch的函数式接口
# import torch.utils.data  # 导入PyTorch的数据工具
# from torch import optim, nn  # 导入PyTorch的优化器和神经网络模块
# from torch.backends import cudnn  # 导入cuDNN，用于加速GPU计算
# from torch.utils.data import DataLoader  # 导入数据加载器
# from loss import SSIM, ContrastLoss  # 导入自定义的SSIM损失和对比损失
# from data import RESIDE_Dataset, MultiModalHazeDataset,TestDataset  # 导入自定义的数据集类
# from metric import psnr, ssim  # 导入评估指标PSNR和SSIM
# from model import Teacher  # 导入教师模型
# from option.Teacher import opt  # 导入教师模型的配置选项
#
# start_time = time.time()  # 记录脚本开始时间
# steps = opt.iters_per_epoch * opt.epochs  # 计算总的训练迭代次数
# T = steps  # T（总步数）用于学习率调度
#
#
# def lr_schedule_cosdecay(t, T, init_lr=opt.start_lr, end_lr=opt.end_lr):
#     """
#     定义余弦衰减学习率调度函数
#     :param t: 当前步数
#     :param T: 总步数
#     :param init_lr: 初始学习率
#     :param end_lr: 最终学习率
#     :return: 当前步的学习率
#     """
#     lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / T))
#     return lr
#
# # 修改
# def collate_fn_skip_none(batch):
#     batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
#     if not batch:
#         # 如果整个批次都无效，返回空的 tensors 或 None
#         # 返回 None 可能需要在训练循环中处理
#         # 返回空 tensor 可能更安全
#         return torch.tensor([]), torch.tensor([]), torch.tensor([])
#     return torch.utils.data.dataloader.default_collate(batch)
#
#
#
# def train(teacher_net, loader_train_1, loader_test, optim, criterion):
#     """
#     定义训练函数
#     :param teacher_net: 教师模型
#     :param loader_train_1: 训练数据加载器
#     :param loader_test: 测试数据加载器
#     :param optim: 优化器
#     :param criterion: 损失函数列表
#     """
#     losses = []  # 存储每一步的总损失
#     loss_log = {'L1': [], 'SSIM': [], 'Cr': [], 'total': []}  # 存储每个epoch的平均损失
#     loss_log_tmp = {'L1': [], 'SSIM': [], 'Cr': [], 'total': []}  # 临时存储一个epoch内各步的损失
#     psnr_log = []  # 存储每个评估点的PSNR
#
#     start_step = 0  # 初始化起始步数
#     max_ssim = 0  # 初始化最佳SSIM
#     max_psnr = 0  # 初始化最佳PSNR
#     ssims = []  # 存储每个评估点的SSIM
#     psnrs = []  # 存储每个评估点的PSNR
#
#     loader_train_iter_1 = iter(loader_train_1)  # 创建训练数据加载器的迭代器
#
#     for step in range(start_step + 1, steps + 1):  # 开始训练循环，从1到总步数
#         teacher_net.train()  # 将教师模型设置为训练模式
#         lr = opt.start_lr  # 获取初始学习率
#         if not opt.no_lr_sche:  # 检查是否使用学习率调度
#             lr = lr_schedule_cosdecay(step, T)  # 计算当前步的学习率
#             for param_group in optim.param_groups:  # 遍历优化器中的参数组
#                 param_group["lr"] = lr  # 更新学习率
#
#         try:
#             x, y = next(loader_train_iter_1)  # 尝试获取下一个数据批次 (x: 有雾图像, y: 清晰图像)
#         except StopIteration:
#             loader_train_iter_1 = iter(loader_train_1)  # 如果迭代器耗尽（一个epoch结束），重置迭代器
#             x, y = next(loader_train_iter_1)  # 获取新epoch的第一个数据批次
#
#         x = x.to(opt.device, non_blocking=True)  # 将输入数据x移动到指定设备（如GPU）
#         y = y.to(opt.device, non_blocking=True)  # 将目标数据y移动到指定设备
#
#         teacher_out = teacher_net(x)  # 教师模型前向传播
#
#         # 根据配置计算各项损失
#         loss_L1 = criterion[0](teacher_out[0], y) if opt.w_loss_L1 > 0 else 0  # 计算L1损失
#         loss_SSIM = (1 - criterion[1](teacher_out[0], y)) if opt.w_loss_SSIM > 0 else 0  # 计算SSIM损失 (1 - SSIM)
#         loss_Cr = criterion[2](teacher_out[0], y, x) if opt.w_loss_Cr > 0 else 0  # 计算对比损失
#
#         # 计算加权总损失
#         loss = opt.w_loss_L1 * loss_L1 + opt.w_loss_SSIM * loss_SSIM + opt.w_loss_Cr * loss_Cr
#
#         loss.backward()  # 反向传播，计算梯度
#         optim.step()  # 更新模型参数
#         optim.zero_grad()  # 清空梯度
#
#         # 记录损失
#         losses.append(loss.item())
#         loss_log_tmp['L1'].append(loss_L1.item() if opt.w_loss_L1 > 0 else 0)
#         loss_log_tmp['SSIM'].append(loss_SSIM.item() if opt.w_loss_SSIM > 0 else 0)
#         loss_log_tmp['Cr'].append(loss_Cr.item() if opt.w_loss_Cr > 0 else 0)
#         loss_log_tmp['total'].append(loss.item())
#
#         # 打印当前训练状态
#         print(
#             f'\rloss:{loss.item():.5f} | L1:{opt.w_loss_L1 * (loss_L1.item() if opt.w_loss_L1 > 0 else 0):.5f} | SSIM:{opt.w_loss_SSIM * (loss_SSIM.item() if opt.w_loss_SSIM > 0 else 0):.5f} | Cr:{opt.w_loss_Cr * (loss_Cr.item() if opt.w_loss_Cr > 0 else 0):.5f}  | step :{step}/{steps} | lr :{lr :.9f} | time_used :{(time.time() - start_time) / 60 :.1f}',
#             end='', flush=True)
#
#         if step % len(loader_train_1) == 0:  # 检查是否到达一个epoch的末尾
#             loader_train_iter_1 = iter(loader_train_1)  # 重置迭代器 (虽然try-except里已有，但这里确保)
#             for key in loss_log.keys():  # 遍历所有损失类型
#                 loss_log[key].append(np.average(np.array(loss_log_tmp[key])))  # 计算并存储这个epoch的平均损失
#                 loss_log_tmp[key] = []  # 清空临时损失记录
#             np.save(os.path.join(opt.saved_data_dir, 'losses.npy'), losses)  # 保存所有步的损失到文件
#
#         # 检查是否到达评估点
#         if (step % opt.iters_per_epoch == 0 and step <= opt.finer_eval_step) or \
#                 (step > opt.finer_eval_step and (step - opt.finer_eval_step) % (5 * len(loader_train_1)) == 0):
#
#             # 计算当前是第几个epoch（用于日志和模型保存）
#             if step > opt.finer_eval_step:
#                 epoch = opt.finer_eval_step // opt.iters_per_epoch + (step - opt.finer_eval_step) // (
#                         5 * len(loader_train_1))
#             else:
#                 epoch = int(step / opt.iters_per_epoch)
#
#             with torch.no_grad():  # 禁用梯度计算
#                 ssim_eval, psnr_eval = test(teacher_net, loader_test)  # 在测试集上评估模型
#
#             # 准备并打印日志信息
#             log = f'\nstep :{step} | epoch: {epoch} | ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f} | lr:{lr:.12f}'
#             print(log)
#             # 将日志写入文件
#             with open(os.path.join(opt.saved_data_dir, 'log.txt'), 'a') as f:
#                 f.write(log + '\n')
#
#             # 记录评估结果
#             ssims.append(ssim_eval)
#             psnrs.append(psnr_eval)
#             psnr_log.append(psnr_eval)
#
#             state_dict = teacher_net.state_dict()  # 获取模型的状态字典
#
#             # 处理DataParallel（多GPU训练）保存模型时带有的'module.'前缀
#             if 'module' in list(state_dict.keys())[0]:
#                 from collections import OrderedDict
#                 new_state_dict = OrderedDict()
#                 for k, v in state_dict.items():
#                     name = k.replace('module.', '')  # 移除 'module.' 前缀
#                     new_state_dict[name] = v
#                 state_dict = new_state_dict
#
#             # 检查当前PSNR是否为历史最佳
#             if psnr_eval > max_psnr:
#                 max_ssim = max(max_ssim, ssim_eval)  # 更新最佳SSIM
#                 max_psnr = max(max_psnr, psnr_eval)  # 更新最佳PSNR
#                 print(f'model saved at step :{step}| epoch: {epoch} | max_psnr:{max_psnr:.4f}| max_ssim:{max_ssim:.4f}')
#                 saved_best_model_path = os.path.join(opt.saved_model_dir, 'best.pth')  # 设置最佳模型保存路径
#                 torch.save(state_dict, saved_best_model_path)  # 保存最佳模型
#
#             # 保存当前epoch的模型
#             saved_single_model_path = os.path.join(opt.saved_model_dir, str(epoch) + '.pth')
#             torch.save(state_dict, saved_single_model_path)
#
#             loader_train_iter_1 = iter(loader_train_1)  # 重置训练迭代器
#
#             # 保存评估指标的历史记录
#             np.save(os.path.join(opt.saved_data_dir, 'ssims.npy'), ssims)
#             np.save(os.path.join(opt.saved_data_dir, 'psnrs.npy'), psnrs)
#
#
# def pad_img(x, patch_size):
#     """
#     定义图像填充函数（用于测试时，确保输入尺寸能被模型网络结构整除）
#     :param x: 输入图像张量
#     :param patch_size: 要求的块大小（或因子）
#     :return: 填充后的图像
#     """
#     _, _, h, w = x.size()  # 获取图像的高和宽
#     mod_pad_h = (patch_size - h % patch_size) % patch_size  # 计算高度需要填充多少
#     mod_pad_w = (patch_size - w % patch_size) % patch_size  # 计算宽度需要填充多少
#     x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')  # 使用反射模式填充图像
#     return x
#
#
# def test(net, loader_test):
#     """
#     定义测试函数
#     :param net: 待评估的模型
#     :param loader_test: 测试数据加载器
#     :return: 平均SSIM和平均PSNR
#     """
#     net.eval()  # 将模型设置为评估模式
#     torch.cuda.empty_cache()  # 清空GPU缓存
#     ssims = []  # 存储SSIM值
#     psnrs = []  # 存储PSNR值
#
#     for i, (inputs, targets, hazy_name) in enumerate(loader_test):  # 遍历测试数据集
#         inputs = inputs.to(opt.device)  # 输入数据移动到设备
#         targets = targets.to(opt.device)  # 目标数据移动到设备
#         with torch.no_grad():  # 禁用梯度计算
#             H, W = inputs.shape[2:]  # 获取原始图像高宽
#             inputs = pad_img(inputs, 4)  # 填充输入图像（可能是因为模型结构要求4的倍数）
#             pred = net(inputs)[0].clamp(0, 1)  # 模型前向传播，并将输出限制在[0, 1]范围
#             pred = pred[:, :, :H, :W]  # 裁剪预测结果，恢复到原始尺寸
#
#         ssim_tmp = ssim(pred, targets).item()  # 计算SSIM
#         psnr_tmp = psnr(pred, targets)  # 计算PSNR
#         ssims.append(ssim_tmp)  # 记录SSIM
#         psnrs.append(psnr_tmp)  # 记录PSNR
#
#     return np.mean(ssims), np.mean(psnrs)  # 返回测试集上的平均SSIM和PSNR
#
#
# def set_seed_torch(seed=2024):
#     """
#     定义设置随机种子的函数，以确保实验可复现性
#     """
#     os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子
#     np.random.seed(seed)  # 设置Numpy随机种子
#     torch.manual_seed(seed)  # 设置PyTorch CPU随机种子
#     torch.cuda.manual_seed(seed)  # 设置PyTorch GPU随机种子
#     torch.backends.cudnn.deterministic = True  # 设置cuDNN为确定性模式
#
#
# if __name__ == "__main__":  # Python主程序入口
#
#     set_seed_torch(2024)  # 设置随机种子
#
#     # --- 数据准备 ---
#     train_dir_1 = './data/THaze/train'  # 训练数据目录
#     train_set_1 = RESIDE_Dataset(train_dir_1, True, 256, '.jpg')  # 实例化训练数据集
#
#     test_dir = './data/THaze/test'  # 测试数据目录
#     test_set = TestDataset(os.path.join(test_dir, 'hazy'), os.path.join(test_dir, 'clear'))  # 实例化测试数据集
#
#     # 创建数据加载器
#     loader_train_1 = DataLoader(dataset=train_set_1, batch_size=24, shuffle=True, num_workers=8)
#     loader_test = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=1)
#
#     # --- 模型初始化 ---
#     teacher_net = Teacher()  # 实例化教师模型
#     teacher_net = teacher_net.to(opt.device)  # 将模型移动到指定设备
#
#     epoch_size = len(loader_train_1)  # 计算一个epoch包含的批次数
#     print("epoch_size: ", epoch_size)
#
#     # --- GPU设置 ---
#     if opt.device == 'cuda':
#         teacher_net = torch.nn.DataParallel(teacher_net)  # 使用DataParallel实现多GPU训练
#         cudnn.benchmark = True  # 启用cuDNN的benchmark模式，加速计算
#
#     # --- 打印模型参数 ---
#     pytorch_total_params = sum(p.numel() for p in teacher_net.parameters() if p.requires_grad)  # 计算模型的可训练参数总量
#     print("Total_params: ==> {}".format(pytorch_total_params))
#     print("------------------------------------------------------------------")
#
#     # --- 损失函数定义 ---
#     criterion = []  # 初始化损失函数列表
#     criterion.append(nn.L1Loss().to(opt.device))  # 添加L1损失
#     criterion.append(SSIM().to(opt.device))  # 添加SSIM损失
#     criterion.append(ContrastLoss(ablation=False))  # 添加对比损失
#
#     # --- 优化器定义 ---
#     optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, teacher_net.parameters()), lr=opt.start_lr,
#                            betas=(0.9, 0.999),
#                            eps=1e-08)
#     optimizer.zero_grad()  # 优化器梯度清零
#
#     # --- 开始训练 ---
#     train(teacher_net, loader_train_1, loader_test, optimizer, criterion)


# Teacher.py

# 导入数学库，用于数学计算，例如余弦函数
import math
# 导入操作系统库，用于文件路径操作，例如创建目录
import os
# 导入时间库，用于记录时间
import time
# 导入 NumPy 库，用于数值计算，特别是数组操作
import numpy as np
# 导入 PyTorch 核心库
import torch
# 导入 PyTorch 神经网络函数库，例如 pad (填充)
import torch.nn.functional as F
# 导入 PyTorch 数据加载工具
import torch.utils.data
# 从 PyTorch 导入优化器 (optim) 和神经网络模块 (nn)
from torch import optim, nn
# 导入 PyTorch 的 cuDNN 库，用于加速 GPU 计算
from torch.backends import cudnn
# 从 PyTorch 数据加载工具中导入 DataLoader 类，用于批量加载数据
from torch.utils.data import DataLoader
# 从 loss 模块导入 SSIM 和 ContrastLoss 类，用于计算损失
from loss import SSIM, ContrastLoss
# 导入新的数据集类和模型类
from data import MultiModalHazeDataset, TestDataset # TestDataset 现在也支持三模态
from metric import psnr, ssim
from model import DualStreamTeacher # <--- 使用新的模型
from option.Teacher import opt


# 训练轮次
start_time = time.time()
# 计算总的训练步数 = 每个 epoch 的迭代次数 * 总 epoch 数
steps = opt.iters_per_epoch * opt.epochs
# 总步数 T，用于学习率调度
T = steps

# 定义函数 lr_schedule_cosdecay：实现学习率余弦衰减
# t: 当前步数
# T: 总步数
# init_lr: 初始学习率
# end_lr: 最终学习率
def lr_schedule_cosdecay(t, T, init_lr=opt.start_lr, end_lr=opt.end_lr):
    """
    计算余弦衰减后的学习率。

    参数:
        t (int): 当前训练步数。
        T (int): 总训练步数。
        init_lr (float): 初始学习率。
        end_lr (float): 最终学习率。

    返回:
        float: 当前步数对应的学习率。
    """
    # 根据余弦函数计算当前步的学习率
    lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / T))
    return lr

# 定义函数 collate_fn_skip_none：DataLoader 的整理函数，用于跳过数据加载中返回 None 的样本
# batch: DataLoader 从 Dataset 获取的一个批次的数据列表
def collate_fn_skip_none(batch):
    """
        DataLoader 的 collate_fn，用于过滤掉批次中值为 None 的样本。
        这通常用于处理 Dataset 的 __getitem__ 中可能发生的异常。
        修改: 使其能处理 TestDataset 返回的4个值。

    参数:
        batch (list): 从 Dataset 获取的样本列表。

    返回:
        tuple or None: 整理后的批次张量元组，如果整个批次无效则返回空元组。
    """
    # 过滤掉 batch 中第一个元素为 None 的项 (假设 None 表示加载失败)
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    # 如果过滤后批次为空
    if not batch:
        # 修改: 返回4个空值 (3个tensor, 1个list)
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), []
    # 使用 PyTorch 默认的 collate 函数将有效样本整理成批次张量/列表
    return torch.utils.data.dataloader.default_collate(batch)

# 定义函数 train：执行模型训练的主要逻辑
# teacher_net: 要训练的教师模型
# loader_train: 训练数据加载器
# loader_test: 测试数据加载器
# optim: 优化器
# criterion: 包含多个损失函数的列表
def train(teacher_net, loader_train, loader_test, optim, criterion):
    """
    执行教师模型的训练和评估过程。

    参数:
        teacher_net (nn.Module): 要训练的教师模型。
        loader_train (DataLoader): 训练数据加载器。
        loader_test (DataLoader): 测试数据加载器。
        optim (Optimizer): 优化器。
        criterion (list): 包含损失函数实例的列表。
    """
    losses = []
    loss_log = {'L1': [], 'SSIM': [], 'Cr': [], 'total': []}
    loss_log_tmp = {'L1': [], 'SSIM': [], 'Cr': [], 'total': []}
    psnr_log = []

    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    # 创建训练数据加载器的迭代器
    loader_train_iter = iter(loader_train)
    # 循环执行每一步训练
    for step in range(start_step + 1, steps + 1):
        teacher_net.train()
        lr = opt.start_lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr

        # --- 加载数据 ---
        try:
            batch_data = next(loader_train_iter)
            # 检查 collate_fn 是否返回了空 batch
            if not batch_data or not batch_data[0].numel():
                 print(f"\n警告: 在步骤 {step} 跳过空批次。")
                 loader_train_iter = iter(loader_train) # 重新迭代
                 continue # 跳过这个空的 batch
            hazy_vis, infrared, clear_vis = batch_data # 解包训练数据 (3项)
        except StopIteration:
            loader_train_iter = iter(loader_train)
            batch_data = next(loader_train_iter)
            if not batch_data or not batch_data[0].numel():
                 print(f"\n警告: 在步骤 {step} (StopIteration后) 跳过空批次。")
                 continue
            hazy_vis, infrared, clear_vis = batch_data
        except Exception as e:
            print(f"\n错误: 在步骤 {step} 加载数据时出错: {e}。跳过批次。")
            continue # 跳过有问题的 batch

        # 移动到设备
        hazy_vis = hazy_vis.to(opt.device, non_blocking=True)
        infrared = infrared.to(opt.device, non_blocking=True)
        clear_vis = clear_vis.to(opt.device, non_blocking=True)

        # --- 模型前向传播 ---
        teacher_out = teacher_net(hazy_vis, infrared)

        # --- 计算损失 ---
        loss_L1 = criterion[0](teacher_out[0], clear_vis) if opt.w_loss_L1 > 0 else torch.tensor(0.0).to(opt.device)
        loss_SSIM = (1 - criterion[1](teacher_out[0], clear_vis)) if opt.w_loss_SSIM > 0 else torch.tensor(0.0).to(opt.device)
        if opt.w_loss_Cr > 0 and criterion[2] is not None:
             loss_Cr = criterion[2](teacher_out[0], clear_vis, hazy_vis)
        else:
             loss_Cr = torch.tensor(0.0).to(opt.device)
        loss = opt.w_loss_L1 * loss_L1 + opt.w_loss_SSIM * loss_SSIM + opt.w_loss_Cr * loss_Cr

        # --- 反向传播和优化 ---
        loss.backward()
        optim.step()
        optim.zero_grad()

        # --- 记录和打印日志 ---
        losses.append(loss.item())
        loss_log_tmp['L1'].append(loss_L1.item() if isinstance(loss_L1, torch.Tensor) else loss_L1)
        loss_log_tmp['SSIM'].append(loss_SSIM.item() if isinstance(loss_SSIM, torch.Tensor) else loss_SSIM)
        loss_log_tmp['Cr'].append(loss_Cr.item() if isinstance(loss_Cr, torch.Tensor) else loss_Cr)
        loss_log_tmp['total'].append(loss.item())

        l1_val = (opt.w_loss_L1 * loss_L1.item()) if isinstance(loss_L1, torch.Tensor) and opt.w_loss_L1 > 0 else 0.0
        ssim_val = (opt.w_loss_SSIM * loss_SSIM.item()) if isinstance(loss_SSIM, torch.Tensor) and opt.w_loss_SSIM > 0 else 0.0
        cr_val = (opt.w_loss_Cr * loss_Cr.item()) if isinstance(loss_Cr, torch.Tensor) and opt.w_loss_Cr > 0 else 0.0
        print(f'\rloss:{loss.item():.5f} | L1:{l1_val:.5f} | SSIM:{ssim_val:.5f} | Cr:{cr_val:.5f}  | step :{step}/{steps} | lr :{lr :.9f} | time_used :{(time.time() - start_time) / 60 :.1f}', end='', flush=True)


        # --- 保存损失记录和执行评估的逻辑 ---
        steps_per_epoch = len(loader_train) if loader_train else 0
        if steps_per_epoch > 0 and step % steps_per_epoch == 0:
            for key in loss_log.keys():
                if loss_log_tmp[key]:
                    loss_log[key].append(np.mean(np.array(loss_log_tmp[key])))
                loss_log_tmp[key] = []
            os.makedirs(opt.saved_data_dir, exist_ok=True)
            try:
                np.save(os.path.join(opt.saved_data_dir, 'losses.npy'), losses)
            except Exception as e:
                 print(f"\n错误: 保存 losses.npy 失败: {e}")

        # 确定评估频率
        eval_freq_fine = 5 * steps_per_epoch if steps_per_epoch > 0 else opt.iters_per_epoch
        eval_freq_coarse = opt.iters_per_epoch

        # 判断当前步是否需要进行评估
        perform_eval = False
        current_epoch = 0
        if eval_freq_coarse > 0 and step <= opt.finer_eval_step:
            if step % eval_freq_coarse == 0:
                perform_eval = True
                current_epoch = step // eval_freq_coarse
        elif eval_freq_fine > 0 and step > opt.finer_eval_step:
             if (step - opt.finer_eval_step) % eval_freq_fine == 0:
                perform_eval = True
                base_epochs = opt.finer_eval_step // eval_freq_coarse if eval_freq_coarse > 0 else 0
                current_epoch = base_epochs + (step - opt.finer_eval_step) // eval_freq_fine

        # 如果当前步需要评估
        if perform_eval:
            try:
                loader_train_iter = iter(loader_train)
            except Exception as e:
                 print(f"\n错误: 评估前重新初始化训练迭代器失败: {e}")

            # 在评估时不计算梯度
            with torch.no_grad():
                # 调用 test 函数在测试集上评估模型性能
                if loader_test:
                    # 修改: 确保 test 函数被正确调用
                    ssim_eval, psnr_eval = test(teacher_net, loader_test)
                else:
                    print("\n警告: 测试加载器无效，跳过评估。")
                    ssim_eval, psnr_eval = 0.0, 0.0

            # 打印评估结果日志
            log = f'\nstep :{step} | epoch: {current_epoch} | ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f} | lr:{lr:.12f}'
            print(log)
            os.makedirs(opt.saved_data_dir, exist_ok=True)
            try:
                with open(os.path.join(opt.saved_data_dir, 'log.txt'), 'a') as f:
                    f.write(log + '\n')
            except Exception as e:
                print(f"\n错误: 写入 log.txt 失败: {e}")

            # 记录评估指标
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            psnr_log.append(psnr_eval)

            # 确保模型保存目录存在
            os.makedirs(opt.saved_model_dir, exist_ok=True)

            # 保存模型权重
            try:
                if psnr_eval > max_psnr:
                    max_ssim = max(max_ssim, ssim_eval)
                    max_psnr = max(max_psnr, psnr_eval)
                    print(f'模型在步骤 :{step}| epoch: {current_epoch} 保存 | 最高 psnr:{max_psnr:.4f}| 最高 ssim:{max_ssim:.4f}')
                    saved_best_model_path = os.path.join(opt.saved_model_dir, 'best.pth')
                    model_to_save = teacher_net.module if isinstance(teacher_net, nn.DataParallel) else teacher_net
                    torch.save(model_to_save.state_dict(), saved_best_model_path)

                saved_single_model_path = os.path.join(opt.saved_model_dir, str(current_epoch) + '.pth')
                model_to_save = teacher_net.module if isinstance(teacher_net, nn.DataParallel) else teacher_net
                torch.save(model_to_save.state_dict(), saved_single_model_path)
            except Exception as e:
                 print(f"\n错误: 保存模型权重失败 (epoch {current_epoch}): {e}")

            # 确保保存评估指标的目录存在
            os.makedirs(opt.saved_data_dir, exist_ok=True)
            try:
                np.save(os.path.join(opt.saved_data_dir, 'ssims.npy'), ssims)
                np.save(os.path.join(opt.saved_data_dir, 'psnrs.npy'), psnrs)
            except Exception as e:
                 print(f"\n错误: 保存 ssims.npy 或 psnrs.npy 失败: {e}")

            # 评估和保存后，也重新初始化训练迭代器
            try:
                loader_train_iter = iter(loader_train)
            except Exception as e:
                 print(f"\n错误: 评估后重新初始化训练迭代器失败: {e}")

# 定义函数 pad_img：对图像进行填充以满足特定尺寸要求
def pad_img(x, patch_size):
    """
    对图像进行反射填充，使其高度和宽度成为 patch_size 的整数倍。

    参数:
        x (Tensor): 输入图像张量 (B, C, H, W)。
        patch_size (int): 目标尺寸的因子。

    返回:
        Tensor: 填充后的图像张量。
    """
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

# 定义函数 test：在测试集上评估模型性能
def test(net, loader_test):
    """
    在测试集上评估模型，计算平均 SSIM 和 PSNR。
    修改: 使其能处理 TestDataset 返回的4个值并正确调用双流模型。

    参数:
        net (nn.Module): 要评估的模型。
        loader_test (DataLoader): 测试数据加载器 (应使用修改后的 TestDataset)。

    返回:
        tuple: 平均 SSIM 和 平均 PSNR。
    """
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []

    if loader_test is None:
        print("警告: test 函数接收到无效的 loader_test，返回 0 指标。")
        return 0.0, 0.0

    for i, batch_test in enumerate(loader_test):
        if not batch_test or not batch_test[0].numel():
             print(f"警告: 在测试加载器中跳过索引 {i} 的空批次。")
             continue

        # 修改: 解包4个返回值
        if len(batch_test) == 4:
            inputs_vis, inputs_ir, targets, hazy_name_list = batch_test
            hazy_name = hazy_name_list[0] if hazy_name_list else f"Unknown_Index_{i}"
        else:
            print(f"测试加载器返回了预期外的数据格式: {len(batch_test)} 项。跳过。")
            continue
        # 修改结束

        inputs_vis = inputs_vis.to(opt.device)
        inputs_ir = inputs_ir.to(opt.device) # 修改: 添加红外输入到设备
        targets = targets.to(opt.device)

        with torch.no_grad():
            H, W = inputs_vis.shape[2:] # 使用可见光尺寸作为基准
            try:
                # 修改: 填充两个输入
                inputs_vis_padded = pad_img(inputs_vis, 16)
                inputs_ir_padded = pad_img(inputs_ir, 16) # 填充红外输入
                # 修改结束
            except Exception as e:
                 print(f"\n错误: 测试时填充图像 {hazy_name} 失败: {e}。跳过。")
                 continue

            try:
                # 修改: 正确调用双流模型
                pred_output = net(inputs_vis_padded, inputs_ir_padded)
                # 修改结束

                if isinstance(pred_output, tuple):
                    pred = pred_output[0]
                else:
                    pred = pred_output
                pred = pred.clamp(0, 1)

            except Exception as e:
                 print(f"\n未知错误: 测试时模型前向传播失败 ({hazy_name}): {e}。跳过。")
                 continue

            # 裁剪回原始尺寸
            if pred.shape[2] > H or pred.shape[3] > W:
                 pred = pred[:, :, :H, :W]
            elif pred.shape[2] < H or pred.shape[3] < W:
                 print(f"警告: 预测尺寸 ({pred.shape}) 小于目标尺寸 ({H}, {W})，文件名 {hazy_name}。指标可能不准确。")

        # 计算指标
        try:
            ssim_tmp = ssim(pred, targets).item()
            psnr_tmp = psnr(pred, targets)
            if not np.isnan(ssim_tmp) and not np.isinf(ssim_tmp):
                 ssims.append(ssim_tmp)
            else:
                 print(f"警告: 无效 SSIM 值 ({ssim_tmp})，文件名 {hazy_name}。跳过。")
            if not np.isnan(psnr_tmp) and not np.isinf(psnr_tmp):
                 psnrs.append(psnr_tmp)
            else:
                 print(f"警告: 无效 PSNR 值 ({psnr_tmp})，文件名 {hazy_name}。跳过。")
        except Exception as e:
            print(f"\n错误: 计算指标失败 ({hazy_name}): {e}")

    mean_ssim = np.mean(ssims) if ssims else 0.0
    mean_psnr = np.mean(psnrs) if psnrs else 0.0
    return mean_ssim, mean_psnr

# 定义函数 set_seed_torch：设置随机种子以保证实验可复现性
def set_seed_torch(seed=2024):
    """
    设置 Python, NumPy 和 PyTorch 的随机种子以提高实验可复现性。

    参数:
        seed (int): 随机种子值。
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True


# Python 主程序入口点
if __name__ == "__main__":

    set_seed_torch(2024)

    # --- 训练数据集加载 ---
    hazy_vis_folder = 'E:/FLIR_zongti_quwu_ceshi/dataset/FLIR/train/hazy' # 训练集含雾可见光
    ir_folder = 'E:/FLIR_zongti_quwu_ceshi/dataset/FLIR/train/ir'      # 训练集红外
    clear_vis_folder = 'E:/FLIR_zongti_quwu_ceshi/dataset/FLIR/train/clear' # 训练集清晰可见光
    train_set = MultiModalHazeDataset(
        hazy_visible_path=hazy_vis_folder,
        infrared_path=ir_folder,
        clear_visible_path=clear_vis_folder,
        train=True,
        size=256,
        format='.jpg' # 确认训练集格式
    )

    # --- 修改: 测试数据集加载 ---
    test_dir = 'E:/FLIR_zongti_quwu_ceshi/dataset/FLIR/test' # 测试集根目录
    test_hazy_vis_folder = os.path.join(test_dir, 'hazy')    # 测试集含雾可见光
    test_ir_folder = os.path.join(test_dir, 'ir')        # 测试集红外
    test_clear_vis_folder = os.path.join(test_dir, 'clear')  # 测试集清晰可见光
    try:
        # 修改: 使用修改后的 TestDataset 加载三模态测试数据
        test_set = TestDataset(
            hazy_visible_path=test_hazy_vis_folder,
            infrared_path=test_ir_folder,
            clear_visible_path=test_clear_vis_folder,
            size=256, # 或者 'full', 与训练保持一致或根据需要调整
            format='.jpg' # 确认测试集格式
        )
    except Exception as e:
        print(f"错误: 初始化 TestDataset 失败: {e}。测试将跳过。")
        test_set = None
    # --- 修改结束 ---

    # --- DataLoader ---
    batch_size = opt.batch_size if hasattr(opt, 'batch_size') else 4
    num_workers = opt.num_workers if hasattr(opt, 'num_workers') else 4
    loader_train = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_skip_none,
        pin_memory=True
        )
    loader_test = None
    if test_set:
        # 修改: 测试加载器也使用 collate_fn_skip_none (因为 TestDataset 现在也可能返回 None)
        loader_test = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn_skip_none)
        # 修改结束

    # --- 模型初始化 ---
    teacher_net = DualStreamTeacher()
    teacher_net = teacher_net.to(opt.device)

    epoch_size = len(loader_train) if loader_train else 0
    print("epoch_size: ", epoch_size)
    if opt.device == 'cuda':
        teacher_net = torch.nn.DataParallel(teacher_net)
        cudnn.benchmark = True

    try:
        pytorch_total_params = sum(p.numel() for p in teacher_net.parameters() if p.requires_grad)
        print("Total_params: ==> {}".format(pytorch_total_params))
    except Exception as e:
        print(f"计算总参数量时出错: {e}")
    print("------------------------------------------------------------------")

    # --- 损失函数和优化器 ---
    criterion = []
    criterion.append(nn.L1Loss().to(opt.device))
    criterion.append(SSIM().to(opt.device))
    try:
        contrast_loss_instance = ContrastLoss(ablation=False).to(opt.device)
        criterion.append(contrast_loss_instance)
    except Exception as e:
         print(f"错误: 初始化 ContrastLoss 失败: {e}。将 Cr 损失权重设为 0。")
         criterion.append(None)
         opt.w_loss_Cr = 0
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, teacher_net.parameters()), lr=opt.start_lr,
                           betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer.zero_grad()

    # 开始训练
    train(teacher_net, loader_train, loader_test, optimizer, criterion)