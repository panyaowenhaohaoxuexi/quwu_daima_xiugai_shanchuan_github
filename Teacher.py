# -*- coding: utf-8 -*-
# Teacher.py (Training Script)

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
# --- [修改] 导入新的数据集类和模型类 ---
from data import MultiModalHazeDataset, TestDataset # TestDataset 现在也支持三模态
from metric import psnr, ssim
# from model import DualStreamTeacher # <--- 不再使用原始模型
from model import VIFNetInconsistencyTeacher, SobelEdgeDetector # <--- 使用新的融合模型
# --- [修改结束] ---
from option.Teacher import opt # 导入配置选项

# 训练轮次
start_time = time.time()
# 计算总的训练步数 = 每个 epoch 的迭代次数 * 总 epoch 数
steps = opt.iters_per_epoch * opt.epochs
# 总步数 T，用于学习率调度
T = steps

# 定义函数 lr_schedule_cosdecay：实现学习率余弦衰减
def lr_schedule_cosdecay(t, T, init_lr=opt.start_lr, end_lr=opt.end_lr):
    """
    计算余弦衰减后的学习率。
    """
    lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / T))
    return lr

# 定义函数 collate_fn_skip_none：DataLoader 的整理函数，用于跳过无效样本
def collate_fn_skip_none(batch):
    """
    DataLoader 的 collate_fn，用于过滤掉批次中值为 None 的样本。
    修改: 使其能处理 TestDataset 返回的4个值。
    """
    # 过滤掉 batch 中第一个元素为 None 的项
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch:
        # 如果整个批次都无效，根据训练/测试返回不同数量的空值
        # 假设通过 len(batch[0]) 判断是训练(3)还是测试(4)，但这不可靠
        # 更稳妥的方式是让调用者处理可能的空 batch
        # 这里返回适用于训练和测试的最小公倍数或根据需要调整
        # 返回空元组，让调用者检查
         return () # 返回空元组
    # 使用 PyTorch 默认的 collate 函数将有效样本整理成批次张量/列表
    return torch.utils.data.dataloader.default_collate(batch)

# 定义函数 train：执行模型训练的主要逻辑
def train(teacher_net, loader_train, loader_test, optim, criterion, edge_detector):
    """
    执行教师模型的训练和评估过程。
    """
    losses = []
    # --- [修改] 增加 Edge 日志 ---
    loss_log = {'L1': [], 'SSIM': [], 'Cr': [], 'Edge': [], 'total': []}
    loss_log_tmp = {'L1': [], 'SSIM': [], 'Cr': [], 'Edge': [], 'total': []}
    # --- [修改结束] ---
    psnr_log = []

    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    loader_train_iter = iter(loader_train)
    for step in range(start_step + 1, steps + 1):
        teacher_net.train()
        lr = opt.start_lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr

        # --- [修改] 加载数据 (三模态) ---
        try:
            batch_data = next(loader_train_iter)
            # 检查 collate_fn 是否返回了空元组
            if not batch_data:
                 print(f"\n警告: 在步骤 {step} 跳过空批次 (collate_fn 返回空)。")
                 loader_train_iter = iter(loader_train) # 重新迭代
                 continue # 跳过这个空的 batch
            # 检查返回的数据项数量是否正确（训练时应为3）
            if len(batch_data) != 3:
                 print(f"\n警告: 在步骤 {step} 加载数据项数量错误 ({len(batch_data)})。跳过批次。")
                 loader_train_iter = iter(loader_train) # 重新迭代
                 continue
            hazy_vis, infrared, clear_vis = batch_data # 解包训练数据 (3项)
        except StopIteration:
            loader_train_iter = iter(loader_train)
            try: # 尝试重新获取
                batch_data = next(loader_train_iter)
                if not batch_data:
                    print(f"\n警告: 在步骤 {step} (StopIteration后) 跳过空批次 (collate_fn 返回空)。")
                    continue
                if len(batch_data) != 3:
                    print(f"\n警告: 在步骤 {step} (StopIteration后) 加载数据项数量错误 ({len(batch_data)})。跳过批次。")
                    continue
                hazy_vis, infrared, clear_vis = batch_data
            except StopIteration: # 如果重新获取还是失败
                print("\n警告: 数据加载器在 epoch 开始时意外耗尽。")
                break # 提前结束训练可能更好
            except Exception as e:
                print(f"\n错误: 在步骤 {step} (StopIteration后) 加载数据时出错: {e}。跳过批次。")
                continue # 跳过有问题的 batch
        except Exception as e:
            print(f"\n错误: 在步骤 {step} 加载数据时出错: {e}。跳过批次。")
            continue # 跳过有问题的 batch

        # 移动到设备
        hazy_vis = hazy_vis.to(opt.device, non_blocking=True)
        infrared = infrared.to(opt.device, non_blocking=True)
        clear_vis = clear_vis.to(opt.device, non_blocking=True)
        # --- [修改结束] ---

        # --- [修改] 模型前向传播 (双输入) ---
        teacher_out = teacher_net(hazy_vis, infrared) # 输入可见光和红外
        # --- [修改结束] ---

        # --- [修改] 计算损失 (使用 clear_vis 作为 GT) ---
        pred_image = teacher_out[0] # 获取预测的去雾图像

        loss_L1 = criterion[0](pred_image, clear_vis) if opt.w_loss_L1 > 0 else torch.tensor(0.0).to(opt.device)
        loss_SSIM = (1 - criterion[1](pred_image, clear_vis)) if opt.w_loss_SSIM > 0 else torch.tensor(0.0).to(opt.device)
        if opt.w_loss_Cr > 0 and criterion[2] is not None:
             loss_Cr = criterion[2](pred_image, clear_vis, hazy_vis) # ContrastLoss 可能需要原始 hazy 输入
        else:
             loss_Cr = torch.tensor(0.0).to(opt.device)

        # --- [新增] 计算红外边缘损失 ---
        loss_Edge = torch.tensor(0.0, device=opt.device)
             # 确保 opt.w_loss_Edge > 0 并且 edge_detector 已经传入
        if opt.w_loss_Edge > 0 and edge_detector is not None:
            try:
                # 提取预测图像的边缘
                edge_pred = edge_detector(pred_image)
                # 提取红外图像的边缘 (作为目标，不计算梯度)
                with torch.no_grad():
                    edge_ir_target = edge_detector(infrared)
                # 使用 L1 损失 (criterion[0]) 计算边缘差距
                loss_Edge = criterion[0](edge_pred, edge_ir_target.detach())
            except Exception as e:
                print(f"\n错误: 计算 Edge 损失失败: {e}")
                loss_Edge = torch.tensor(0.0, device=opt.device)
        # --- [新增结束] ---

        # loss = opt.w_loss_L1 * loss_L1 + opt.w_loss_SSIM * loss_SSIM + opt.w_loss_Cr * loss_Cr
        loss = opt.w_loss_L1 * loss_L1 + opt.w_loss_SSIM * loss_SSIM + opt.w_loss_Cr * loss_Cr + opt.w_loss_Edge * loss_Edge
        # --- [修改结束] ---

        # --- 反向传播和优化 ---
        # 梯度清零（更推荐在 optimizer.step() 之后清零）
        optim.zero_grad()
        loss.backward()
        optim.step()
        # optim.zero_grad() # 移到这里更好

        # --- 记录和打印日志 ---
        losses.append(loss.item())
        # 确保在记录 .item() 前检查是否为 Tensor
        loss_log_tmp['L1'].append(loss_L1.item() if isinstance(loss_L1, torch.Tensor) else loss_L1)
        loss_log_tmp['SSIM'].append(loss_SSIM.item() if isinstance(loss_SSIM, torch.Tensor) else loss_SSIM)
        loss_log_tmp['Cr'].append(loss_Cr.item() if isinstance(loss_Cr, torch.Tensor) else loss_Cr)
        # AAA
        loss_log_tmp['Edge'].append(loss_Edge.item() if isinstance(loss_Edge, torch.Tensor) else loss_Edge)  # <-- [修改]
        # AAA
        loss_log_tmp['total'].append(loss.item())

        l1_val = (opt.w_loss_L1 * loss_L1.item()) if isinstance(loss_L1, torch.Tensor) and opt.w_loss_L1 > 0 else 0.0
        ssim_val = (opt.w_loss_SSIM * loss_SSIM.item()) if isinstance(loss_SSIM, torch.Tensor) and opt.w_loss_SSIM > 0 else 0.0
        cr_val = (opt.w_loss_Cr * loss_Cr.item()) if isinstance(loss_Cr, torch.Tensor) and opt.w_loss_Cr > 0 else 0.0
        # print(f'\rloss:{loss.item():.5f} | L1:{l1_val:.5f} | SSIM:{ssim_val:.5f} | Cr:{cr_val:.5f}  | step :{step}/{steps} | lr :{lr :.9f} | time_used :{(time.time() - start_time) / 60 :.1f}', end='', flush=True)
        # AAA
        edge_val = (opt.w_loss_Edge * loss_Edge.item()) if isinstance(loss_Edge,
                                                                      torch.Tensor) and opt.w_loss_Edge > 0 else 0.0  # <-- [修改]
        # AAA

        # AAA
        # --- [修改] 更新打印 ---
        print(
            f'\rloss:{loss.item():.5f} | L1:{l1_val:.5f} | SSIM:{ssim_val:.5f} | Cr:{cr_val:.5f} | Edge:{edge_val:.5f} | step :{step}/{steps} | lr :{lr :.9f} | time_used :{(time.time() - start_time) / 60 :.1f}',
            end='', flush=True)
        # --- [修改结束] ---
        # AAA

        # --- 保存损失记录和执行评估的逻辑 ---
        steps_per_epoch = len(loader_train) if loader_train else 0 # 获取每个 epoch 的步数
        # Epoch 结束统计
        if steps_per_epoch > 0 and step % steps_per_epoch == 0:
            # 重新初始化迭代器，确保下一轮能正确开始
            try:
                loader_train_iter = iter(loader_train)
            except Exception as e:
                print(f"\n错误: Epoch结束时重新初始化训练迭代器失败: {e}")

            for key in loss_log.keys():
                if loss_log_tmp[key]: # 确保列表不为空
                    loss_log[key].append(np.mean(np.array(loss_log_tmp[key])))
                loss_log_tmp[key] = [] # 清空临时记录
            os.makedirs(opt.saved_data_dir, exist_ok=True)
            try:
                np.save(os.path.join(opt.saved_data_dir, 'losses.npy'), losses)
            except Exception as e:
                 print(f"\n错误: 保存 losses.npy 失败: {e}")

        # 确定评估频率 (与之前逻辑保持一致)
        eval_freq_fine = 5 * steps_per_epoch if steps_per_epoch > 0 else opt.iters_per_epoch
        eval_freq_coarse = opt.iters_per_epoch if steps_per_epoch > 0 else steps # 如果 loader_train 为空，则只在最后评估一次

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
        elif step == steps: # 确保最后一步进行评估
            perform_eval = True
            # 计算最后一个epoch的编号
            if eval_freq_fine > 0 and step > opt.finer_eval_step:
                base_epochs = opt.finer_eval_step // eval_freq_coarse if eval_freq_coarse > 0 else 0
                current_epoch = base_epochs + math.ceil((step - opt.finer_eval_step) / eval_freq_fine)
            elif eval_freq_coarse > 0:
                 current_epoch = math.ceil(step / eval_freq_coarse)
            else:
                 current_epoch = opt.epochs # 或 1

        # 执行评估
        if perform_eval:
            # 在评估时不计算梯度
            with torch.no_grad():
                if loader_test:
                    ssim_eval, psnr_eval = test(teacher_net, loader_test)
                else:
                    print("\n警告: 测试加载器无效，跳过评估。")
                    ssim_eval, psnr_eval = 0.0, 0.0

            log = f'\nstep :{step} | epoch: {current_epoch} | ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f} | lr:{lr:.12f}'
            print(log)
            os.makedirs(opt.saved_data_dir, exist_ok=True)
            try:
                with open(os.path.join(opt.saved_data_dir, 'log.txt'), 'a') as f:
                    f.write(log + '\n')
            except Exception as e:
                print(f"\n错误: 写入 log.txt 失败: {e}")

            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            psnr_log.append(psnr_eval)

            os.makedirs(opt.saved_model_dir, exist_ok=True)
            try:
                model_to_save = teacher_net.module if isinstance(teacher_net, nn.DataParallel) else teacher_net
                state_dict = model_to_save.state_dict() # 直接获取 state_dict

                if psnr_eval > max_psnr:
                    max_ssim = max(max_ssim, ssim_eval)
                    max_psnr = max(max_psnr, psnr_eval)
                    print(f'模型在步骤 :{step}| epoch: {current_epoch} 保存 | 最高 psnr:{max_psnr:.4f}| 最高 ssim:{max_ssim:.4f}')
                    saved_best_model_path = os.path.join(opt.saved_model_dir, 'best.pth')
                    torch.save(state_dict, saved_best_model_path)

                saved_single_model_path = os.path.join(opt.saved_model_dir, str(current_epoch) + '.pth')
                torch.save(state_dict, saved_single_model_path)
            except Exception as e:
                 print(f"\n错误: 保存模型权重失败 (epoch {current_epoch}): {e}")

            os.makedirs(opt.saved_data_dir, exist_ok=True)
            try:
                np.save(os.path.join(opt.saved_data_dir, 'ssims.npy'), ssims)
                np.save(os.path.join(opt.saved_data_dir, 'psnrs.npy'), psnrs)
            except Exception as e:
                 print(f"\n错误: 保存 ssims.npy 或 psnrs.npy 失败: {e}")

            # 评估后也尝试重置迭代器，以防评估发生在epoch中间
            try:
                loader_train_iter = iter(loader_train)
            except Exception as e:
                 print(f"\n警告: 评估后重新初始化训练迭代器失败: {e}")


# 定义函数 pad_img：对图像进行填充以满足特定尺寸要求
def pad_img(x, patch_size):
    """
    对图像进行反射填充，使其高度和宽度成为 patch_size 的整数倍。
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
    """
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []

    if loader_test is None:
        print("警告: test 函数接收到无效的 loader_test，返回 0 指标。")
        return 0.0, 0.0

    for i, batch_test in enumerate(loader_test):
        # 检查 collate_fn 返回的是否为空
        if not batch_test:
             print(f"警告: 在测试加载器中跳过索引 {i} 的空批次 (collate_fn 返回空)。")
             continue

        # --- [修改] 解包4个返回值 ---
        # 确保返回的是4项
        if len(batch_test) == 4:
            inputs_vis, inputs_ir, targets, hazy_name_list = batch_test
            # 处理可能的空 batch 情况（如果 collate_fn 返回了带空 tensor 的元组）
            if not inputs_vis.numel():
                print(f"警告: 在测试加载器索引 {i} 遇到空数据。跳过。")
                continue
            # 获取文件名，处理列表情况
            hazy_name = hazy_name_list[0] if isinstance(hazy_name_list, (list, tuple)) and hazy_name_list else f"Unknown_Index_{i}"
        else:
            print(f"测试加载器返回了预期外的数据格式: {len(batch_test)} 项。跳过批次 {i}。")
            continue
        # --- [修改结束] ---

        inputs_vis = inputs_vis.to(opt.device, non_blocking=True)
        inputs_ir = inputs_ir.to(opt.device, non_blocking=True) # --- [修改] 添加红外输入到设备 ---
        targets = targets.to(opt.device, non_blocking=True)

        with torch.no_grad():
            H, W = inputs_vis.shape[2:] # 使用可见光尺寸作为基准
            try:
                # --- [修改] 填充两个输入 (假设需要16的倍数) ---
                inputs_vis_padded = pad_img(inputs_vis, 16)
                inputs_ir_padded = pad_img(inputs_ir, 16)
                # --- [修改结束] ---
            except Exception as e:
                 print(f"\n错误: 测试时填充图像 {hazy_name} 失败: {e}。跳过。")
                 continue

            try:
                # --- [修改] 正确调用双流模型 ---
                pred_output = net(inputs_vis_padded, inputs_ir_padded)
                # --- [修改结束] ---

                # 获取图像输出 (假设在元组的第一个位置)
                if isinstance(pred_output, tuple):
                    pred = pred_output[0]
                else:
                    pred = pred_output # 如果模型只返回图像
                pred = pred.clamp(0, 1) # 限制范围

            except Exception as e:
                 # 打印更详细的错误信息
                 import traceback
                 print(f"\n未知错误: 测试时模型前向传播失败 ({hazy_name}): {e}")
                 # traceback.print_exc() # 取消注释以打印详细堆栈
                 continue

            # 裁剪回原始尺寸
            if pred.shape[2] > H or pred.shape[3] > W:
                 pred = pred[:, :, :H, :W]
            elif pred.shape[2] < H or pred.shape[3] < W:
                 # 尺寸不匹配可能是 padding 或模型内部下采样/上采样的问题
                 print(f"警告: 预测尺寸 ({pred.shape}) 小于目标尺寸 ({H}, {W})，文件名 {hazy_name}。指标可能不准确。")

        # 计算指标
        try:
            # 确保 pred 和 targets 维度匹配
            if pred.shape != targets.shape:
                print(f"警告: 预测 ({pred.shape}) 和目标 ({targets.shape}) 尺寸不匹配，文件名 {hazy_name}。跳过指标计算。")
                continue
            ssim_tmp = ssim(pred, targets).item()
            psnr_tmp = psnr(pred, targets)
            # 增加对 NaN 和 Inf 值的检查
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

    # 计算平均值前检查列表是否为空
    mean_ssim = np.mean(ssims) if ssims else 0.0
    mean_psnr = np.mean(psnrs) if psnrs else 0.0
    return mean_ssim, mean_psnr

# 定义函数 set_seed_torch：设置随机种子以保证实验可复现性
def set_seed_torch(seed=2024):
    """
    设置 Python, NumPy 和 PyTorch 的随机种子以提高实验可复现性。
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 对于需要确定性的场景，取消下面两行的注释，但这可能会降低性能
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # 通常 benchmark = True 可以加速训练
    torch.backends.cudnn.benchmark = True


# Python 主程序入口点
if __name__ == "__main__":

    set_seed_torch(2024)

    # --- [修改] 数据集路径和实例化 ---
    # !! 请将下面的路径修改为你实际的数据集路径 !!
    train_base_dir = 'E:/FLIR_zongti_quwu_ceshi/dataset/FLIR/train' # 训练集根目录
    test_base_dir = 'E:/FLIR_zongti_quwu_ceshi/dataset/FLIR/test'   # 测试集根目录

    # 训练数据集路径
    hazy_vis_folder = os.path.join(train_base_dir, 'hazy')
    ir_folder = os.path.join(train_base_dir, 'ir')
    clear_vis_folder = os.path.join(train_base_dir, 'clear')
    try:
        train_set = MultiModalHazeDataset(
            hazy_visible_path=hazy_vis_folder,
            infrared_path=ir_folder,
            clear_visible_path=clear_vis_folder,
            train=True,
            size=256, # 训练时使用随机裁剪
            format='.jpg' # 确认训练集格式
        )
        print(f"成功加载训练数据集，共 {len(train_set)} 个样本。")
    except Exception as e:
        print(f"错误: 初始化训练数据集 MultiModalHazeDataset 失败: {e}")
        train_set = None # 设置为 None 以便后续检查
        exit() # 训练集加载失败则退出

    # 测试数据集路径
    test_hazy_vis_folder = os.path.join(test_base_dir, 'hazy')
    test_ir_folder = os.path.join(test_base_dir, 'ir')
    test_clear_vis_folder = os.path.join(test_base_dir, 'clear')
    try:
        test_set = TestDataset(
            hazy_visible_path=test_hazy_vis_folder,
            infrared_path=test_ir_folder,
            clear_visible_path=test_clear_vis_folder,
            size=256, # 测试时使用中心裁剪或缩放
            format='.jpg' # 确认测试集格式
        )
        print(f"成功加载测试数据集，共 {len(test_set)} 个样本。")
    except Exception as e:
        print(f"错误: 初始化测试数据集 TestDataset 失败: {e}。测试将跳过。")
        test_set = None
    # --- [修改结束] ---

    # --- DataLoader ---
    # 从配置中读取 batch_size 和 num_workers，提供默认值
    batch_size = getattr(opt, 'batch_size', 4) # 使用 opt 中的 batch_size，默认为 4
    num_workers = getattr(opt, 'num_workers', 4) # 使用 opt 中的 num_workers，默认为 4

    loader_train = None
    if train_set: # 仅在 train_set 成功加载时创建 DataLoader
        loader_train = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn_skip_none,
            pin_memory=True, # 如果内存充足，可以加速数据传输
            drop_last=True # 丢弃最后一个不完整的 batch，避免 BN 层问题
            )
    else:
        print("错误：训练数据集加载失败，无法创建训练 DataLoader。")
        exit()

    loader_test = None
    if test_set:
        loader_test = DataLoader(
            dataset=test_set,
            batch_size=1, # 测试时通常 batch_size=1
            shuffle=False,
            num_workers=1, # 测试时 worker 少一些通常没问题
            collate_fn=collate_fn_skip_none
            )

    # --- [修改] 模型初始化 ---
    teacher_net = VIFNetInconsistencyTeacher() # 实例化新的模型
    teacher_net = teacher_net.to(opt.device)
    # --- [修改结束] ---

    # AAA
    # --- [新增] 初始化边缘检测器 ---
    edge_detector = SobelEdgeDetector().to(opt.device)
    # 确保它不参与训练（Sobel 没有可训练参数，但这是个好习惯）
    for param in edge_detector.parameters():
        param.requires_grad = False
    edge_detector.eval()
    # --- [新增结束] ---
    # AAA

    epoch_size = len(loader_train) if loader_train else 0
    if epoch_size == 0:
         print("错误：训练 DataLoader 为空或长度为 0。请检查数据集和批处理大小。")
         exit()
    print("每个 Epoch 的步数 (epoch_size): ", epoch_size)

    if opt.device == 'cuda':
        # 如果有多张 GPU，DataParallel 会自动使用
        print(f"检测到 CUDA 设备，使用 DataParallel (可用 GPU 数量: {torch.cuda.device_count()})。")
        teacher_net = torch.nn.DataParallel(teacher_net)
        cudnn.benchmark = True # 启用 benchmark 加速

    try:
        pytorch_total_params = sum(p.numel() for p in teacher_net.parameters() if p.requires_grad)
        print("模型可训练参数总量: ==> {}".format(pytorch_total_params))
    except Exception as e:
        print(f"计算总参数量时出错: {e}")
    print("------------------------------------------------------------------")

    # --- 损失函数和优化器 ---
    criterion = []
    # L1 Loss
    criterion.append(nn.L1Loss().to(opt.device))
    # SSIM Loss
    criterion.append(SSIM().to(opt.device))
    # Contrast Loss
    try:
        contrast_loss_instance = ContrastLoss(ablation=False).to(opt.device)
        criterion.append(contrast_loss_instance)
    except Exception as e:
         # 如果 ContrastLoss 初始化失败（例如缺少 VGG 权重），则禁用它
         print(f"错误: 初始化 ContrastLoss 失败: {e}。将 Cr 损失权重设为 0。")
         criterion.append(None)
         opt.w_loss_Cr = 0 # 禁用对比度损失

    # 确保 criterion 列表长度至少为 3，即使 Cr Loss 失败
    while len(criterion) < 3:
        criterion.append(None)

    # Adam 优化器
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, teacher_net.parameters()), lr=opt.start_lr,
                           betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer.zero_grad() # 初始化梯度

    # 开始训练
    print("开始训练...")
    # AAA
    # --- [修改] 传入 edge_detector ---
    train(teacher_net, loader_train, loader_test, optimizer, criterion, edge_detector)
    # --- [修改结束] ---
    # AAA
    print("训练完成。")