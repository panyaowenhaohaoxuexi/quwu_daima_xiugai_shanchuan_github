import math
import os
import time
import numpy as np
import torch
import clip
import torch.nn.functional as F
from torch import optim, nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from metric import psnr, ssim
from loss import SSIM  # 仅用于测试阶段时可用；训练不直接用
# 使用你 data/ 下“已修改”的 TestDataset（三模态：hazy/ir/clear）
from data import MultiModalCLIPLoader, TestDataset
from model import DualStreamTeacher
from CLIP import L_clip_from_feature
from collections import OrderedDict
from option.EMA import opt

start_time = time.time()
steps = opt.iters_per_epoch * opt.epochs
T = steps


# 定义函数 lr_schedule_cosdecay：实现学习率余弦衰减
def lr_schedule_cosdecay(t, T, init_lr=opt.start_lr, end_lr=opt.end_lr):
    lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / T))
    return lr


# ---------- collate_fn（分开训练/测试两套，避免空批次时解包长度不匹配） ----------
from torch.utils.data.dataloader import default_collate

# 定义函数 collate_train：用于整理训练批次数据
def collate_train(batch):
    """
    训练集（MultiModalCLIPLoader）：期望返回二元组 (vis, ir)
    过滤掉无效样本并使用默认collate函数整理批次。
    """
    batch = [b for b in batch if b is not None and b[0] is not None]
    return default_collate(batch) if batch else (torch.tensor([]), torch.tensor([]))

# 定义函数 collate_test：用于整理测试批次数据
def collate_test(batch):
    """
    测试集（TestDataset 三模态）：期望返回四元组 (vis, ir, clear, name)
    过滤掉无效样本并使用默认collate函数整理批次。
    """
    batch = [b for b in batch if b is not None and b[0] is not None]
    return default_collate(batch) if batch else (torch.tensor([]), torch.tensor([]), torch.tensor([]), [''])
# -----------------------------------------------------------------------------

# 定义类 TextEncoder：封装CLIP文本编码器的部分功能
class TextEncoder(nn.Module):
    """
    轻量封装的 CLIP 文本编码器（仅复用 CLIP 的文本分支模块）。
    作用：将 embedding_prompt 与 tokenized_prompts 输入到 CLIP 文本分支，
         输出与视觉空间对齐的文本特征 text_features。
    """
    # 定义方法 __init__：初始化 TextEncoder 实例
    def __init__(self, clip_model):
        """ 初始化方法，接收预加载的CLIP模型实例 """
        super().__init__()
        self.transformer = clip_model.transformer # transformer 模块
        self.positional_embedding = clip_model.positional_embedding # 位置编码
        self.ln_final = clip_model.ln_final # 最终的 LayerNorm
        self.text_projection = clip_model.text_projection # 文本投影层
        self.dtype = clip_model.dtype # 数据类型

    # 定义方法 forward：执行前向传播
    def forward(self, prompts, tokenized_prompts):
        """
        前向传播逻辑。

        参数:
            prompts (Tensor): 形状 [B, L, C]，预计算的嵌入。
            tokenized_prompts (Tensor): 形状 [B, L]，对应的 token 序列。

        返回:
            Tensor: 文本特征，形状 [B, C]。
        """
        x = prompts + self.positional_embedding.type(self.dtype) # 添加位置编码
        x = x.permute(1, 0, 2) # [L, B, C]
        x = self.transformer(x) # 通过 Transformer
        x = x.permute(1, 0, 2) # [B, L, C]
        x = self.ln_final(x).type(self.dtype) # LayerNorm

        # 检查批次大小是否匹配，如果不匹配则打印警告并使用最后一个 token 特征
        if x.shape[0] == tokenized_prompts.shape[0]:
            # 提取 [EOT] token 对应的特征并投影
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        else:
            print(f"警告: TextEncoder 中形状不匹配 ({x.shape} vs {tokenized_prompts.shape})。使用最后一个 token 的特征。")
            x = x[:, -1, :] @ self.text_projection # 使用最后一个 token 的特征
        return x

# 定义函数 update_ema_variables：执行指数移动平均更新教师模型权重
def update_ema_variables(student_model, teacher_model, alpha):
    """
    EMA 更新：teacher = alpha * teacher + (1 - alpha) * student
    支持 DataParallel 包装和跨设备参数。

    参数:
        student_model (nn.Module): 学生模型。
        teacher_model (nn.Module): 教师模型 (将被原地更新)。
        alpha (float): EMA 平滑系数。
    """
    # 获取学生模型的状态字典，处理 DataParallel 情况
    student_state_dict = student_model.module.state_dict() if isinstance(student_model, nn.DataParallel) else student_model.state_dict()
    # 获取教师模型实例，处理 DataParallel 情况
    teacher_model_instance = teacher_model.module if isinstance(teacher_model, nn.DataParallel) else teacher_model

    # 在不计算梯度的情况下更新教师模型参数
    with torch.no_grad():
        # 遍历教师模型的命名参数
        for name, teacher_param in teacher_model_instance.named_parameters():
            # 确保学生模型也有同名参数
            if name in student_state_dict:
                student_param = student_state_dict[name]
                # 如果参数在同一设备，直接更新
                if teacher_param.device == student_param.device:
                    teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)
                else:
                    # 如果不在同一设备，先将学生参数移动到教师参数设备再更新
                    teacher_param.data.mul_(alpha).add_(student_param.data.to(teacher_param.device), alpha=1 - alpha)

# 定义函数 train：执行主要的训练逻辑
def train(teacher_net, student_net, loader_train, loader_test, optim, criterion, text_features):
    """
    训练主循环：Student 学习，Teacher 以 EMA 跟随。
    - 前向：双输入 (hazy_vis, infrared)
    - 损失：L1(student, teacher_ema) + λ * CLIP(student, text_feat)
    - 评估：周期性在 test(loader_test) 上评估（真双路）

    参数:
        teacher_net (nn.Module): EMA 教师网络 (eval 模式)。
        student_net (nn.Module): 学生网络 (train 模式)。
        loader_train (DataLoader): 训练数据加载器。
        loader_test (DataLoader or None): 测试数据加载器。
        optim (Optimizer): 优化器 (仅作用于 student_net)。
        criterion (list): 损失函数列表 [L1Loss, ClipLoss or None]。
        text_features (Tensor or None): 预计算的文本特征。
    """
    losses = [] # 记录每步总损失
    loss_log = {'L1_r': [], 'Clip': [], 'total': []} # 记录每 epoch 平均损失
    loss_log_tmp = {'L1_r': [], 'Clip': [], 'total': []} # 临时记录当前 epoch 损失
    psnr_log = [] # 记录每次评估的 PSNR

    start_step = 0 # 起始步数
    max_ssim = 0 # 最高 SSIM
    max_psnr = 0 # 最高 PSNR
    ssims = [] # 记录每次评估的 SSIM
    psnrs = [] # 记录每次评估的 PSNR

    loader_train_iter = iter(loader_train) # 获取训练数据迭代器
    alpha = 0.95 # EMA 平滑系数
    save_count = 0 # 保存计数器

    # 确保保存目录存在
    os.makedirs(opt.saved_model_dir, exist_ok=True)
    os.makedirs(opt.saved_data_dir, exist_ok=True)

    # 主训练循环
    for step in range(start_step + 1, steps + 1):
        teacher_net.eval() # 设置教师模型为评估模式
        student_net.train() # 设置学生模型为训练模式

        # 获取当前学习率
        lr = opt.start_lr
        if not opt.no_lr_sche: # 如果使用学习率调度
            lr = lr_schedule_cosdecay(step, T)
            # 应用学习率到优化器
            for param_group in optim.param_groups:
                param_group["lr"] = lr

        # ======== 加载双模态训练数据 ========
        try:
            batch_data = next(loader_train_iter) # 获取下一批数据
            # 检查是否为空批次
            if not batch_data or not batch_data[0].numel():
                print(f"\n警告: 在步骤 {step} 跳过空批次。")
                loader_train_iter = iter(loader_train) # 重新获取迭代器
                continue # 跳过当前步骤
            hazy_vis, infrared = batch_data  # 解包可见光和红外图像
        except StopIteration: # 如果迭代器耗尽 (epoch结束)
            loader_train_iter = iter(loader_train) # 重新获取迭代器
            try:
                batch_data = next(loader_train_iter) # 获取新 epoch 的第一批
                if not batch_data or not batch_data[0].numel():
                    print(f"\n警告: 在步骤 {step} (StopIteration后) 跳过空批次。")
                    continue
                hazy_vis, infrared = batch_data
            except StopIteration: # 如果重置后仍然没有数据
                 print("\n警告: 数据加载器在重置后意外耗尽。提前结束 epoch。")
                 break # 提前结束训练循环
            except Exception as e:
                 print(f"\n错误: 在步骤 {step} (StopIteration后) 加载数据失败: {e}。跳过批次。")
                 continue
        except Exception as e: # 捕获其他加载错误
            print(f"\n错误: 在步骤 {step} 加载数据失败: {e}。跳过批次。")
            continue

        # 将数据移动到设备
        hazy_vis = hazy_vis.to(opt.device, non_blocking=True)
        infrared = infrared.to(opt.device, non_blocking=True)

        # ======== 前向传播 ========
        # 教师模型前向 (不计算梯度)
        with torch.no_grad():
            teacher_output = teacher_net(hazy_vis, infrared)
            # 获取教师模型的图像输出
            teacher_image = teacher_output[0] if isinstance(teacher_output, tuple) else teacher_output

        # 学生模型前向 (计算梯度)
        student_out = student_net(hazy_vis, infrared)
        # 获取学生模型的图像输出
        student_image = student_out[0] if isinstance(student_out, tuple) else student_out

        # ======== 计算损失 ========
        loss_L1_r = torch.tensor(0.0, device=opt.device) # 初始化 L1_r 损失
        loss_Clip = torch.tensor(0.0, device=opt.device) # 初始化 Clip 损失

        # 计算 L1 一致性损失 (学生 vs 教师)
        if opt.w_loss_L1_r > 0:
            loss_L1_r = criterion[0](student_image, teacher_image.detach()) # 使用 detach 阻止梯度流向教师

        # 计算 CLIP 损失 (学生 vs 文本特征)
        if opt.w_loss_Clip > 0 and criterion[1] is not None and text_features is not None:
            try:
                loss_Clip = criterion[1](student_image, text_features)  # 调用 L_clip_from_feature
            except Exception as e:
                print(f"\n错误: 在步骤 {step} 计算 CLIP 损失失败: {e}。将 loss_Clip 设为 0。")
                loss_Clip = torch.tensor(0.0, device=opt.device)
        else: # 如果禁用 CLIP 损失或相关组件无效
            loss_Clip = torch.tensor(0.0, device=opt.device)

        # 计算加权总损失
        loss = opt.w_loss_L1_r * loss_L1_r + opt.w_loss_Clip * loss_Clip

        # 检查损失是否有效 (非 NaN 或 Inf)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n警告: 在步骤 {step} 检测到无效损失 (L1_r: {loss_L1_r.item()}, Clip: {loss_Clip.item()})。跳过步骤。")
            optim.zero_grad() # 清除可能存在的无效梯度
            continue # 跳过当前步骤

        # ======== 反向传播和优化 ========
        optim.zero_grad() # 清除之前的梯度
        loss.backward() # 计算梯度
        # 可选: 添加梯度裁剪
        # torch.nn.utils.clip_grad_norm_(student_net.parameters(), max_norm=1.0)
        optim.step() # 更新学生模型参数

        # ======== EMA 更新教师模型 ========
        update_ema_variables(student_net, teacher_net, alpha)

        # ======== 日志记录 ========
        losses.append(loss.item()) # 记录总损失
        loss_log_tmp['L1_r'].append(loss_L1_r.item()) # 记录 L1_r
        loss_log_tmp['Clip'].append(loss_Clip.item()) # 记录 Clip
        loss_log_tmp['total'].append(loss.item()) # 记录总损失

        # 计算用于打印的加权损失值
        l1r_val = (opt.w_loss_L1_r * loss_L1_r.item()) if opt.w_loss_L1_r > 0 else 0.0
        clip_val = (opt.w_loss_Clip * loss_Clip.item()) if opt.w_loss_Clip > 0 else 0.0
        # 打印当前步骤的训练信息
        print(
            f'\rloss:{loss.item():.5f} | L1_r:{l1r_val:.5f}  |Clip:{clip_val:.5f} | step :{step}/{steps} | lr :{lr :.9f} | time_used :{(time.time() - start_time) / 60 :.1f}',
            end='', flush=True)

        # ======== 保存损失曲线 & Epoch 结束统计 ========
        # 计算每个 epoch 的步数
        steps_per_epoch = len(loader_train) if len(loader_train) > 0 else opt.iters_per_epoch
        # 如果当前 step 是 epoch 的最后一步
        if steps_per_epoch > 0 and step % steps_per_epoch == 0:
            # 计算并记录这个 epoch 的平均损失
            for key in loss_log.keys():
                if loss_log_tmp[key]: # 确保临时列表不为空
                    loss_log[key].append(np.mean(np.array(loss_log_tmp[key])))
                loss_log_tmp[key] = [] # 清空临时列表
            # 确保保存目录存在
            os.makedirs(opt.saved_data_dir, exist_ok=True)
            try:
                # 保存所有步数的总损失列表
                np.save(os.path.join(opt.saved_data_dir, 'losses.npy'), losses)
            except Exception as e:
                print(f"\n错误: 保存 losses.npy 失败: {e}")

        # ======== 确定评估频率 ========
        eval_freq_fine = 5 * steps_per_epoch if steps_per_epoch > 0 else opt.iters_per_epoch # 后期频率
        eval_freq_coarse = opt.iters_per_epoch # 前期频率
        perform_eval = False # 是否执行评估的标志
        current_epoch = 0 # 当前评估周期编号

        # 判断是否达到评估点
        if eval_freq_coarse > 0 and step <= opt.finer_eval_step: # 前期
            if step % eval_freq_coarse == 0:
                perform_eval = True
                current_epoch = step // eval_freq_coarse
        elif eval_freq_fine > 0 and step > opt.finer_eval_step: # 后期
            if (step - opt.finer_eval_step) % eval_freq_fine == 0:
                perform_eval = True
                base_epochs = opt.finer_eval_step // eval_freq_coarse if eval_freq_coarse > 0 else 0
                current_epoch = base_epochs + (step - opt.finer_eval_step) // eval_freq_fine

        # ======== 执行评估 ========
        if perform_eval:
            # 重新初始化训练迭代器 (以防评估耗时较长)
            try:
                loader_train_iter = iter(loader_train)
            except Exception as e:
                print(f"\n错误: 评估前重新初始化训练迭代器失败: {e}")

            # 在评估时不计算梯度
            with torch.no_grad():
                # 调用 test 函数评估学生模型
                ssim_eval, psnr_eval = test(student_net, loader_test)

            # 打印评估日志
            log = f'\nstep :{step} | epoch: {current_epoch} | ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f} | lr:{lr:.12f}'
            print(log)
            # 确保日志目录存在并写入日志文件
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

            # 保存学生模型的状态
            os.makedirs(opt.saved_model_dir, exist_ok=True)
            try:
                # 获取学生模型的状态字典 (处理 DataParallel)
                model_to_save = student_net.module if isinstance(student_net, nn.DataParallel) else student_net
                state_dict = model_to_save.state_dict()
                # 如果当前 PSNR 是历史最佳
                if psnr_eval > max_psnr:
                    max_ssim = max(max_ssim, ssim_eval)
                    max_psnr = max(max_psnr, psnr_eval)
                    print(f'模型在步骤 :{step}| epoch: {current_epoch} 保存 | 最高 psnr:{max_psnr:.4f}| 最高 ssim:{max_ssim:.4f}')
                    # 保存为 best_student.pth
                    torch.save(state_dict, os.path.join(opt.saved_model_dir, 'best_student.pth'))
                # 保存当前周期的学生模型
                torch.save(state_dict, os.path.join(opt.saved_model_dir, f'student_{current_epoch}.pth'))
                save_count += 1
            except Exception as e:
                print(f"\n错误: 保存学生模型失败 (epoch {current_epoch}): {e}")

            # 保存 EMA 教师模型的快照
            try:
                # 获取教师模型的状态字典 (处理 DataParallel)
                teacher_model_to_save = teacher_net.module if isinstance(teacher_net, nn.DataParallel) else teacher_net
                teacher_state_dict = teacher_model_to_save.state_dict()
                # 保存为 teacher_ema_EPOCH.pth
                torch.save(teacher_state_dict, os.path.join(opt.saved_model_dir, f'teacher_ema_{current_epoch}.pth'))
                print(f'EMA 教师模型在 epoch 保存: {current_epoch}')
            except Exception as e:
                print(f"\n错误: 保存 EMA 教师模型失败 (epoch {current_epoch}): {e}")

            # 保存评估指标列表
            os.makedirs(opt.saved_data_dir, exist_ok=True)
            try:
                np.save(os.path.join(opt.saved_data_dir, 'ssims.npy'), ssims)
                np.save(os.path.join(opt.saved_data_dir, 'psnrs.npy'), psnrs)
            except Exception as e:
                print(f"\n错误: 保存 ssims.npy 或 psnrs.npy 失败: {e}")

            # 评估后重新初始化训练迭代器
            try:
                loader_train_iter = iter(loader_train)
            except Exception as e:
                print(f"\n错误: 评估后重新初始化训练迭代器失败: {e}")

# 定义函数 pad_img：对图像进行填充
def pad_img(x, patch_size):
    """ 对图像进行反射填充以满足尺寸要求 """
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

# 定义函数 test：执行测试评估
def test(net, loader_test):
    """
    评估函数：在测试集上计算平均 SSIM 与 PSNR。
    适配返回4项数据的 TestDataset 并正确调用双流模型。
    """
    net.eval() # 设置为评估模式
    torch.cuda.empty_cache() # 清空缓存
    ssims = [] # 存储 SSIM
    psnrs = [] # 存储 PSNR

    # 检查加载器是否有效
    if loader_test is None:
        print("警告: test() 收到 None 的 loader_test，返回 0 指标。")
        return 0.0, 0.0

    # 遍历测试数据
    for i, batch_test in enumerate(loader_test):
        # 处理空批次
        if not batch_test or not batch_test[0].numel():
            print(f"警告: 在测试加载器中跳过索引 {i} 的空批次。")
            continue

        # 解包测试数据 (预期4项)
        if len(batch_test) == 4:
            inputs_vis, inputs_ir, targets, hazy_name_list = batch_test
            # 获取文件名 (batch size 为 1)
            hazy_name = hazy_name_list[0] if isinstance(hazy_name_list, (list, tuple)) and len(hazy_name_list) > 0 else f"Unknown_Index_{i}"
        else: # 数据格式错误
            print(f"测试加载器返回了预期外的数据格式: {len(batch_test)} 项。跳过。")
            continue

        # 将数据移动到设备
        inputs_vis = inputs_vis.to(opt.device, non_blocking=True)
        inputs_ir = inputs_ir.to(opt.device, non_blocking=True)
        targets = targets.to(opt.device, non_blocking=True)

        # 在不计算梯度的情况下执行
        with torch.no_grad():
            H, W = inputs_vis.shape[2:] # 获取原始尺寸
            try:
                # 填充输入图像
                inputs_vis_padded = pad_img(inputs_vis, 16)
                inputs_ir_padded = pad_img(inputs_ir, 16)
            except Exception as e:
                print(f"\n错误: 测试时 pad 失败 {hazy_name}: {e}。跳过。")
                continue

            try:
                # 模型前向传播 (双输入)
                pred_output = net(inputs_vis_padded, inputs_ir_padded)
                # 获取图像输出
                pred = pred_output[0] if isinstance(pred_output, tuple) else pred_output
                # 限制输出范围
                pred = pred.clamp(0, 1)
            except Exception as e: # 捕获前向传播错误
                print(f"\n未知错误: 测试前向失败 ({hazy_name}): {e}。跳过。")
                continue

            # 裁剪回原始尺寸
            if pred.shape[2] > H or pred.shape[3] > W:
                pred = pred[:, :, :H, :W]
            elif pred.shape[2] < H or pred.shape[3] < W: # 检查尺寸是否意外变小
                print(f"警告: 预测尺寸 {pred.shape} 小于目标尺寸 ({H}, {W})，文件 {hazy_name}。指标可能不准。")

        # 计算评估指标
        try:
            ssim_tmp = ssim(pred, targets).item()
            psnr_tmp = psnr(pred, targets)
            # 检查并记录有效指标
            if not np.isnan(ssim_tmp) and not np.isinf(ssim_tmp):
                ssims.append(ssim_tmp)
            else:
                print(f"警告: 无效 SSIM ({ssim_tmp})，文件 {hazy_name}。跳过。")
            if not np.isnan(psnr_tmp) and not np.isinf(psnr_tmp):
                psnrs.append(psnr_tmp)
            else:
                print(f"警告: 无效 PSNR ({psnr_tmp})，文件 {hazy_name}。跳过。")
        except Exception as e: # 捕获指标计算错误
            print(f"\n错误: 计算指标失败 ({hazy_name}): {e}")

    # 计算平均指标
    mean_ssim = np.mean(ssims) if ssims else 0.0
    mean_psnr = np.mean(psnrs) if psnrs else 0.0
    return mean_ssim, mean_psnr

# 定义函数 set_seed_torch：设置随机种子
def set_seed_torch(seed=2018):
    """ 设置随机种子以提高可复现性 """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 设置所有 GPU 种子
    torch.backends.cudnn.benchmark = True # 启用 benchmark 加速


# ======== 全局加载 CLIP 模型与文本特征 ========
try:
    # 加载 CLIP ViT 模型
    clip_model, _ = clip.load("ViT-B/32", device=torch.device("cpu"),
                              download_root="D:/liu_lan_qi_xia_zai/CoA-main_daima_xiugai/clip_model/") # CLIP 模型文件根目录
    clip_model.to(opt.device) # 移动到设备
    for param in clip_model.parameters(): # 冻结参数
        param.requires_grad = False
    clip_model.eval() # 设置为评估模式
except Exception as e:
    print(f"错误: 加载 CLIP 模型失败: {e}。CLIP 损失将被禁用。")
    clip_model = None # 标记为失败

text_features = None # 初始化文本特征为 None
if clip_model is not None: # 仅当 CLIP 模型加载成功时
    try:
        # 加载预计算的 prompt 嵌入
        data = torch.load('D:/liu_lan_qi_xia_zai/CoA-main_daima_xiugai/clip_model/haze_prompt.pth',
                          map_location=opt.device) # 加载 prompt 文件
        new_state_dict = OrderedDict() # 创建有序字典
        # 处理可能的 'module.' 前缀
        for k, v in data.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        # 获取 prompt 嵌入并设为不可训练
        embedding_prompt = nn.Parameter(new_state_dict['embedding_prompt']).to(opt.device) # shape is [2, 77, 512]
        embedding_prompt.requires_grad = False
        # 实例化文本编码器
        text_encoder = TextEncoder(clip_model)
        # 创建占位符 tokenized prompts (batch size 1)
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in [" ".join(["X"] * 16)]]).to(opt.device) # shape is [1, 77]

        # --- 修改这里: 只选择第一个 prompt ---
        # 确保 embedding_prompt 和 tokenized_prompts 的 batch size (第0维) 一致
        if embedding_prompt.shape[0] != tokenized_prompts.shape[0]:
             print(f"信息: 从 embedding_prompt (形状 {embedding_prompt.shape}) "
                   f"选择第一个 prompt 以匹配 tokenized_prompts (形状 {tokenized_prompts.shape})。")
             selected_embedding_prompt = embedding_prompt[0:1] # 选择第一个 prompt, 保持 batch dim = 1
        else:
             selected_embedding_prompt = embedding_prompt

        # 使用批次大小匹配的 prompt 进行编码得到最终文本特征
        text_features = text_encoder(selected_embedding_prompt, tokenized_prompts)
        # --- 修改结束 ---

    except FileNotFoundError: # 如果 prompt 文件未找到
        print("错误: haze_prompt.pth 未找到。CLIP 损失将被禁用。")
        text_features = None
    except Exception as e: # 捕获其他加载错误
        print(f"错误: 加载文本特征失败: {e}。CLIP 损失将被禁用。")
        text_features = None
# =================================================


# Python 主程序入口点
if __name__ == "__main__":

    set_seed_torch(2024) # 设置随机种子

    # ======== 训练数据：真实雾（无GT），二元组 (vis, ir) ========
    # 指定训练数据路径 (硬编码)
    vis_hazy_folder = "E:/FLIR_zongti_quwu_ceshi/dataset/REAL_FOGGY/hazy"  # 可见光 .png
    ir_hazy_folder = "E:/FLIR_zongti_quwu_ceshi/dataset/REAL_FOGGY/ir"     # 红外 .png
    # 创建训练数据集实例
    train_set = MultiModalCLIPLoader(
        hazy_visible_path=vis_hazy_folder,
        infrared_path=ir_hazy_folder,
        train=True, # 训练模式
        size=256,   # 裁剪尺寸
        format='.png' # 文件格式
    )

    # ======== 测试数据：三模态 (vis, ir, clear)，均为 .jpg ========
    # 指定测试数据路径 (硬编码)
    test_dir = 'E:/FLIR_zongti_quwu_ceshi/dataset/FLIR/test'
    test_hazy_vis_folder = os.path.join(test_dir, 'hazy') # 测试集含雾可见光
    test_ir_folder = os.path.join(test_dir, 'ir')       # 测试集红外
    test_clear_vis_folder = os.path.join(test_dir, 'clear') # 测试集清晰可见光
    # 创建测试数据集实例
    try:
        test_set = TestDataset(
            hazy_visible_path=test_hazy_vis_folder,
            infrared_path=test_ir_folder,
            clear_visible_path=test_clear_vis_folder,
            size=256,     # 测试时尺寸处理方式 (与训练一致或 'full')
            format='.jpg' # 测试集文件格式
        )
    except Exception as e: # 捕获初始化错误
        print(f"错误: 初始化 TestDataset 失败: {e}。测试将跳过。")
        test_set = None # 标记为无效

    # ======== DataLoader ========
    # 获取 batch size 和 num workers (优先从 opt 获取，否则用默认值)
    batch_size = opt.batch_size if hasattr(opt, 'batch_size') else 4 # 默认 batch size 4
    num_workers = opt.num_workers if hasattr(opt, 'num_workers') else 4 # 默认 num workers 4

    # 创建训练数据加载器
    loader_train = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True, # 打乱训练数据
        num_workers=num_workers,
        collate_fn=collate_train, # 使用训练集的 collate 函数
        pin_memory=True # 加速数据传输
    )

    # 创建测试数据加载器 (如果 test_set 有效)
    loader_test = None
    if test_set:
        loader_test = DataLoader(
            dataset=test_set,
            batch_size=1, # 测试时 batch size 通常为 1
            shuffle=False, # 测试时不打乱
            num_workers=1, # 测试时 worker 通常为 1
            collate_fn=collate_test # 使用测试集的 collate 函数
        )

    # ======== 模型初始化 ========
    teacher_net = DualStreamTeacher().to(opt.device) # EMA 教师模型
    student_net = DualStreamTeacher().to(opt.device) # 学生模型

    # 指定预训练权重路径 (硬编码)
    pretrained_teacher_path = "D:/liu_lan_qi_xia_zai/CoA-main_daima_xiugai/saved_model/best.pth"
    print(f"加载预训练教师模型: {pretrained_teacher_path}")

    # 加载预训练权重到 teacher 和 student
    try:
        checkpoint = torch.load(pretrained_teacher_path, map_location=opt.device)
        teacher_state_dict = OrderedDict()
        student_state_dict = OrderedDict()
        has_module_prefix = any(k.startswith('module.') for k in checkpoint.keys())

        # 复制权重并处理 'module.' 前缀
        for k, v in checkpoint.items():
            name = k[7:] if has_module_prefix else k
            teacher_state_dict[name] = v.clone() # 必须 clone 以免共享内存
            student_state_dict[name] = v.clone()

        # 加载权重到模型 (strict=False 允许部分不匹配)
        teacher_net_load_result = teacher_net.load_state_dict(teacher_state_dict, strict=False)
        print("加载 EMA 教师网络状态:", teacher_net_load_result)
        teacher_net.eval() # 设置为评估模式

        student_net_load_result = student_net.load_state_dict(student_state_dict, strict=False)
        print("加载学生网络状态:", student_net_load_result)
        # 打印不匹配的键，帮助调试
        if teacher_net_load_result.missing_keys or teacher_net_load_result.unexpected_keys:
            print("EMA 教师不匹配键:", teacher_net_load_result)
        if student_net_load_result.missing_keys or student_net_load_result.unexpected_keys:
            print("学生不匹配键:", student_net_load_result)
    except FileNotFoundError: # 文件未找到
        print(f"错误: 预训练教师模型未找到 {pretrained_teacher_path}。将从头开始训练。")
    except Exception as e: # 其他加载错误
        print(f"错误: 加载预训练教师模型失败: {e}。将从头开始训练。")

    # 如果使用 CUDA，应用 DataParallel
    if opt.device == 'cuda':
        teacher_net = torch.nn.DataParallel(teacher_net)
        student_net = torch.nn.DataParallel(student_net)
        cudnn.benchmark = True # 启用 benchmark

    # 打印可训练参数量
    try:
        pytorch_total_params = sum(p.numel() for p in student_net.parameters() if p.requires_grad)
        print("学生网络可训练参数量: ==> {}".format(pytorch_total_params))
    except Exception as e:
        print(f"无法计算总参数量: {e}")
    print("------------------------------------------------------------------")

    # ======== 损失函数 ========
    criterion = []
    criterion.append(nn.L1Loss().to(opt.device))  # L1_r
    clip_loss_instance = None # 初始化 CLIP 损失实例为 None
    # 仅当 CLIP 模型和文本特征都成功加载时，才创建 CLIP 损失实例
    if clip_model is not None and text_features is not None:
        try:
            clip_loss_instance = L_clip_from_feature().to(opt.device)
        except Exception as e:
            print(f"错误: 创建 L_clip_from_feature 实例失败: {e}。CLIP 损失已禁用。")
            clip_loss_instance = None
    criterion.append(clip_loss_instance) # 添加实例或 None 到列表

    # 如果 CLIP 损失实例无效，强制权重为 0
    if criterion[1] is None:
        opt.w_loss_Clip = 0
        print("警告: CLIP 损失已禁用。")

    # ======== 优化器（仅学生） ========
    optimizer = optim.Adam(
        params=filter(lambda x: x.requires_grad, student_net.parameters()), # 只优化学生网络的可训练参数
        lr=opt.start_lr, betas=(0.9, 0.999), eps=1e-08
    )
    optimizer.zero_grad() # 初始化梯度

    # ======== 开始训练 ========
    # 调用 train 函数，传入所有必要的对象CLIP 损失已禁用
    train(teacher_net, student_net, loader_train, loader_test if loader_test else None, optimizer, criterion, text_features)