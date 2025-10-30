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
from model import VIFNetInconsistencyTeacher, SobelEdgeDetector
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
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer  # transformer 模块
        self.positional_embedding = clip_model.positional_embedding  # 位置编码
        self.ln_final = clip_model.ln_final  # 最终的 LayerNorm
        self.text_projection = clip_model.text_projection  # 文本投影层
        self.dtype = clip_model.dtype  # 数据类型

    def forward(self, prompts, tokenized_prompts):
        """
        参数:
            prompts (Tensor): [B, L, C]，预计算的嵌入。
            tokenized_prompts (Tensor): [B, L]，对应的 token 序列。
        返回:
            Tensor: 文本特征 [B, C]。
        """
        x = prompts + self.positional_embedding.type(self.dtype)  # 添加位置编码
        x = x.permute(1, 0, 2)  # [L, B, C]
        x = self.transformer(x)  # 通过 Transformer
        x = x.permute(1, 0, 2)  # [B, L, C]
        x = self.ln_final(x).type(self.dtype)  # LayerNorm

        if x.shape[0] == tokenized_prompts.shape[0]:
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        else:
            print(f"警告: TextEncoder 中形状不匹配 ({x.shape} vs {tokenized_prompts.shape})。使用最后一个 token 的特征。")
            x = x[:, -1, :] @ self.text_projection
        return x


# 仅切换 BatchNorm 的 train/eval，不影响其他层
def _set_batchnorm_mode(module, train):
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        if train:
            module.train()
        else:
            module.eval()


# EMA 更新（兼容 DP）
def update_ema_variables(student_model, teacher_model, alpha):
    student_state_dict = student_model.module.state_dict() if isinstance(student_model, nn.DataParallel) else student_model.state_dict()
    teacher_model_instance = teacher_model.module if isinstance(teacher_model, nn.DataParallel) else teacher_model
    with torch.no_grad():
        for name, teacher_param in teacher_model_instance.named_parameters():
            if name in student_state_dict:
                student_param = student_state_dict[name].to(teacher_param.device)
                teacher_param.data.mul_(alpha).add_(student_param, alpha=1 - alpha)
        for name, teacher_buffer in teacher_model_instance.named_buffers():
            if name in student_state_dict:
                student_buffer = student_state_dict[name]
                if isinstance(student_buffer, torch.Tensor):
                    teacher_buffer.data.copy_(student_buffer.to(teacher_buffer.device))


# [MOD] 将文本特征按当前 batch 对齐（重复/裁剪）
def _align_text_features_to_batch(text_features: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    将全局 text_features 对齐到当前训练 batch 的大小。
    规则：
      - 若 B_text == batch_size: 直接返回
      - 若 B_text == 1: 重复到 batch_size
      - 其他：重复并裁剪到 batch_size
    """
    if text_features is None:
        return None
    tf = text_features
    if tf.dim() == 1:
        tf = tf.unsqueeze(0)  # [1, D]
    b = tf.shape[0]
    if b == batch_size:
        return tf
    reps = (batch_size + b - 1) // b
    tf = tf.repeat(reps, 1)[:batch_size]
    return tf


# 定义函数 train：执行主要的训练逻辑
def train(teacher_net, student_net, loader_train, loader_test, optim, criterion, text_features, edge_detector):
    """
    训练主循环：Student 学习，Teacher 以 EMA 跟随。
    - 前向：双输入 (hazy_vis, infrared)
    - 损失：L1(student, teacher_ema) + λ * CLIP(student, text_feat)  ←（参考代码B）
    - 评估：周期性在 test(loader_test) 上评估（真双路）
    """
    losses = []
    loss_log = {'L1_r': [], 'Clip': [], 'Edge': [], 'total': []}
    loss_log_tmp = {'L1_r': [], 'Clip': [], 'Edge': [], 'total': []}
    psnr_log = []

    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []

    loader_train_iter = iter(loader_train)
    alpha = 0.95
    save_count = 0

    os.makedirs(opt.saved_model_dir, exist_ok=True)
    os.makedirs(opt.saved_data_dir, exist_ok=True)

    for step in range(start_step + 1, steps + 1):
        teacher_net.eval()
        student_net.train()

        lr = opt.start_lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr

        # ======== 加载双模态训练数据 ========
        try:
            batch_data = next(loader_train_iter)
            if not batch_data or not batch_data[0].numel():
                print(f"\n警告: 在步骤 {step} 跳过空批次。")
                loader_train_iter = iter(loader_train)
                continue
            hazy_vis, infrared = batch_data
        except StopIteration:
            loader_train_iter = iter(loader_train)
            try:
                batch_data = next(loader_train_iter)
                if not batch_data or not batch_data[0].numel():
                    print(f"\n警告: 在步骤 {step} (StopIteration后) 跳过空批次。")
                    continue
                hazy_vis, infrared = batch_data
            except StopIteration:
                print("\n警告: 数据加载器在重置后意外耗尽。提前结束 epoch。")
                break
            except Exception as e:
                print(f"\n错误: 在步骤 {step} (StopIteration后) 加载数据失败: {e}。跳过批次。")
                continue
        except Exception as e:
            print(f"\n错误: 在步骤 {step} 加载数据失败: {e}。跳过批次。")
            continue

        # 移动到设备
        hazy_vis = hazy_vis.to(opt.device, non_blocking=True)
        infrared = infrared.to(opt.device, non_blocking=True)

        # ======== 前向传播 ========
        # 教师（不计算梯度），仅切换 BN 为 train（使用 batch 统计）
        teacher_module = teacher_net.module if isinstance(teacher_net, nn.DataParallel) else teacher_net
        teacher_module.apply(lambda m: _set_batchnorm_mode(m, True))
        with torch.no_grad():
            teacher_output = teacher_net(hazy_vis, infrared)
        teacher_module.apply(lambda m: _set_batchnorm_mode(m, False))
        teacher_image = teacher_output[0] if isinstance(teacher_output, tuple) else teacher_output

        # 学生（计算梯度）
        student_out = student_net(hazy_vis, infrared)
        student_image = student_out[0] if isinstance(student_out, tuple) else student_out

        # ======== 计算损失（参考代码B的 CLIP 调用方式） ========
        loss_L1_r = torch.tensor(0.0, device=opt.device)
        loss_Clip = torch.tensor(0.0, device=opt.device)
        loss_Edge = torch.tensor(0.0, device=opt.device)  # 新增

        # L1(student, teacher.detach())
        if opt.w_loss_L1_r > 0:
            loss_L1_r = criterion[0](student_image, teacher_image.detach())

        # [MOD] CLIP(student, text_features) —— 先按当前 batch 对齐 text_features
        if opt.w_loss_Clip > 0 and criterion[1] is not None and text_features is not None:
            try:
                tf_batch = _align_text_features_to_batch(text_features, student_image.shape[0]).to(opt.device)
                loss_Clip = criterion[1](student_image, tf_batch)  # 与代码B保持一致：L_clip_from_feature(img_pred, text_features)
            except Exception as e:
                print(f"\n错误: 在步骤 {step} 计算 CLIP 损失失败: {e}。将 loss_Clip 设为 0。")
                loss_Clip = torch.tensor(0.0, device=opt.device)
        else:
            loss_Clip = torch.tensor(0.0, device=opt.device)

            # [新增] 边缘一致性损失 L1(edge(student), edge(ir).detach())
        if opt.w_loss_Edge > 0 and criterion[2] is not None:
            try:
                # 确保 edge_detector 在正确的设备上
                edge_detector.to(opt.device)
                # 计算学生输出的边缘
                edge_student = edge_detector(student_image)
                # 计算真实红外图像的边缘 (作为目标，不计算梯度)
                with torch.no_grad():
                     edge_ir_target = edge_detector(infrared)
                # 计算 L1 损失
                loss_Edge = criterion[2](edge_student, edge_ir_target)
            except Exception as e:
                print(f"\n错误: 在步骤 {step} 计算 Edge 损失失败: {e}。将 loss_Edge 设为 0。")
                loss_Edge = torch.tensor(0.0, device=opt.device)
        # [新增结束]

        # 总损失
        loss = opt.w_loss_L1_r * loss_L1_r + opt.w_loss_Clip * loss_Clip + opt.w_loss_Edge * loss_Edge

        # 数值健壮性检查
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n警告: 在步骤 {step} 检测到无效损失 (L1_r: {loss_L1_r.item()}, Clip: {loss_Clip.item()})。跳过步骤。")
            optim.zero_grad()
            continue

        # 反向与更新
        optim.zero_grad()
        loss.backward()
        # 可选梯度裁剪：
        # torch.nn.utils.clip_grad_norm_(student_net.parameters(), max_norm=1.0)
        optim.step()

        # EMA
        update_ema_variables(student_net, teacher_net, alpha)

        # 日志
        losses.append(loss.item())
        loss_log_tmp['L1_r'].append(loss_L1_r.item())
        loss_log_tmp['Clip'].append(loss_Clip.item())
        loss_log_tmp['Edge'].append(loss_Edge.item())
        loss_log_tmp['total'].append(loss.item())

        l1r_val = (opt.w_loss_L1_r * loss_L1_r.item()) if opt.w_loss_L1_r > 0 else 0.0
        clip_val = (opt.w_loss_Clip * loss_Clip.item()) if opt.w_loss_Clip > 0 else 0.0
        edge_val = (opt.w_loss_Edge * loss_Edge.item()) if opt.w_loss_Edge > 0 else 0.0  # 新增
        print(
            f'\rloss:{loss.item():.5f} | L1_r:{l1r_val:.5f} | Clip:{clip_val:.5f} | Edge:{edge_val:.5f} '
    f'| step:{step}/{steps} | lr:{lr:.9f} | time_used:{(time.time() - start_time) / 60:.1f}',
    end='', flush=True)

        # ======== 保存损失曲线 & Epoch 结束统计 ========
        steps_per_epoch = len(loader_train) if len(loader_train) > 0 else opt.iters_per_epoch
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

        # ======== 确定评估频率 ========
        eval_freq_fine = 5 * steps_per_epoch if steps_per_epoch > 0 else opt.iters_per_epoch
        eval_freq_coarse = opt.iters_per_epoch
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

        # ======== 执行评估 ========
        if perform_eval:
            try:
                loader_train_iter = iter(loader_train)
            except Exception as e:
                print(f"\n错误: 评估前重新初始化训练迭代器失败: {e}")

            with torch.no_grad():
                ssim_eval, psnr_eval = test(student_net, loader_test)

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
                model_to_save = student_net.module if isinstance(student_net, nn.DataParallel) else student_net
                state_dict = model_to_save.state_dict()
                if psnr_eval > max_psnr:
                    max_ssim = max(max_ssim, ssim_eval)
                    max_psnr = max(max_psnr, psnr_eval)
                    print(f'模型在步骤 :{step}| epoch: {current_epoch} 保存 | 最高 psnr:{max_psnr:.4f}| 最高 ssim:{max_ssim:.4f}')
                    torch.save(state_dict, os.path.join(opt.saved_model_dir, 'best_student.pth'))
                torch.save(state_dict, os.path.join(opt.saved_model_dir, f'student_{current_epoch}.pth'))
                save_count += 1
            except Exception as e:
                print(f"\n错误: 保存学生模型失败 (epoch {current_epoch}): {e}")

            try:
                teacher_model_to_save = teacher_net.module if isinstance(teacher_net, nn.DataParallel) else teacher_net
                teacher_state_dict = teacher_model_to_save.state_dict()
                torch.save(teacher_state_dict, os.path.join(opt.saved_model_dir, f'teacher_ema_{current_epoch}.pth'))
                print(f'EMA 教师模型在 epoch 保存: {current_epoch}')
            except Exception as e:
                print(f"\n错误: 保存 EMA 教师模型失败 (epoch {current_epoch}): {e}")

            os.makedirs(opt.saved_data_dir, exist_ok=True)
            try:
                np.save(os.path.join(opt.saved_data_dir, 'ssims.npy'), ssims)
                np.save(os.path.join(opt.saved_data_dir, 'psnrs.npy'), psnrs)
            except Exception as e:
                print(f"\n错误: 保存 ssims.npy 或 psnrs.npy 失败: {e}")

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
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []

    if loader_test is None:
        print("警告: test() 收到 None 的 loader_test，返回 0 指标。")
        return 0.0, 0.0

    for i, batch_test in enumerate(loader_test):
        if not batch_test or not batch_test[0].numel():
            print(f"警告: 在测试加载器中跳过索引 {i} 的空批次。")
            continue

        if len(batch_test) == 4:
            inputs_vis, inputs_ir, targets, hazy_name_list = batch_test
            hazy_name = hazy_name_list[0] if isinstance(hazy_name_list, (list, tuple)) and len(hazy_name_list) > 0 else f"Unknown_Index_{i}"
        else:
            print(f"测试加载器返回了预期外的数据格式: {len(batch_test)} 项。跳过。")
            continue

        inputs_vis = inputs_vis.to(opt.device, non_blocking=True)
        inputs_ir = inputs_ir.to(opt.device, non_blocking=True)
        targets = targets.to(opt.device, non_blocking=True)

        with torch.no_grad():
            H, W = inputs_vis.shape[2:]
            try:
                inputs_vis_padded = pad_img(inputs_vis, 16)
                inputs_ir_padded = pad_img(inputs_ir, 16)
            except Exception as e:
                print(f"\n错误: 测试时 pad 失败 {hazy_name}: {e}。跳过。")
                continue

            try:
                pred_output = net(inputs_vis_padded, inputs_ir_padded)
                pred = pred_output[0] if isinstance(pred_output, tuple) else pred_output
                pred = pred.clamp(0, 1)
            except Exception as e:
                print(f"\n未知错误: 测试前向失败 ({hazy_name}): {e}。跳过。")
                continue

            if pred.shape[2] > H or pred.shape[3] > W:
                pred = pred[:, :, :H, :W]
            elif pred.shape[2] < H or pred.shape[3] < W:
                print(f"警告: 预测尺寸 {pred.shape} 小于目标尺寸 ({H}, {W})，文件 {hazy_name}。指标可能不准。")

        try:
            ssim_tmp = ssim(pred, targets).item()
            psnr_tmp = psnr(pred, targets)
            if not np.isnan(ssim_tmp) and not np.isinf(ssim_tmp):
                ssims.append(ssim_tmp)
            else:
                print(f"警告: 无效 SSIM ({ssim_tmp})，文件 {hazy_name}。跳过。")
            if not np.isnan(psnr_tmp) and not np.isinf(psnr_tmp):
                psnrs.append(psnr_tmp)
            else:
                print(f"警告: 无效 PSNR ({psnr_tmp})，文件 {hazy_name}。跳过。")
        except Exception as e:
            print(f"\n错误: 计算指标失败 ({hazy_name}): {e}")

    mean_ssim = np.mean(ssims) if ssims else 0.0
    mean_psnr = np.mean(psnrs) if psnrs else 0.0
    return mean_ssim, mean_psnr


# 定义函数 set_seed_torch：设置随机种子
def set_seed_torch(seed=2018):
    """ 设置随机种子以提高可复现性 """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ======== 全局加载 CLIP 模型与文本特征 ========
try:
    # 加载 CLIP ViT 模型
    clip_model, _ = clip.load("ViT-B/32", device=torch.device("cpu"),
                              download_root="/root/CoA-main_daima_xiugai/clip_model/")
    clip_model.to(opt.device)
    for param in clip_model.parameters():
        param.requires_grad = False
    clip_model.eval()
except Exception as e:
    print(f"错误: 加载 CLIP 模型失败: {e}。CLIP 损失将被禁用。")
    clip_model = None

text_features = None
if clip_model is not None:
    try:
        # 加载预计算的 prompt 嵌入
        data = torch.load('/root/CoA-main_daima_xiugai/clip_model/haze_prompt.pth',
                          map_location=opt.device)
        new_state_dict = OrderedDict()
        for k, v in data.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        # [MOD] 不再只取第一个 prompt；使用全部 embedding_prompt，并让 tokenized_prompts 的 batch 与之匹配
        embedding_prompt = nn.Parameter(new_state_dict['embedding_prompt']).to(opt.device)  # [B_prompt, 77, 512]
        embedding_prompt.requires_grad = False

        text_encoder = TextEncoder(clip_model)

        # [MOD] 让 tokenized_prompts 的 batch 维与 embedding_prompt 对齐（参考代码B）
        B_prompt = embedding_prompt.shape[0]
        token_str = " ".join(["X"] * 16)
        tokenized_prompts = torch.cat([clip.tokenize(token_str) for _ in range(B_prompt)], dim=0).to(opt.device)  # [B_prompt, 77]

        # 计算整批文本特征（与代码B一致的做法）
        text_features = text_encoder(embedding_prompt, tokenized_prompts)  # [B_prompt, D]
    except FileNotFoundError:
        print("错误: haze_prompt.pth 未找到。CLIP 损失将被禁用。")
        text_features = None
    except Exception as e:
        print(f"错误: 加载文本特征失败: {e}。CLIP 损失将被禁用。")
        text_features = None
# =================================================


# Python 主程序入口点
if __name__ == "__main__":

    set_seed_torch(2024)

    # ======== 训练数据：真实雾（无GT），二元组 (vis, ir) ========
    vis_hazy_folder = "/root/autodl-tmp/REAL_FOGGY/hazy"
    ir_hazy_folder = "/root/autodl-tmp/REAL_FOGGY/ir"
    train_set = MultiModalCLIPLoader(
        hazy_visible_path=vis_hazy_folder,
        infrared_path=ir_hazy_folder,
        train=True,
        size=256,
        format='.png'
    )

    # ======== 测试数据：三模态 (vis, ir, clear)，均为 .jpg ========
    test_dir = '/root/autodl-tmp/FLIR/test'
    test_hazy_vis_folder = os.path.join(test_dir, 'hazy')
    test_ir_folder = os.path.join(test_dir, 'ir')
    test_clear_vis_folder = os.path.join(test_dir, 'clear')
    try:
        test_set = TestDataset(
            hazy_visible_path=test_hazy_vis_folder,
            infrared_path=test_ir_folder,
            clear_visible_path=test_clear_vis_folder,
            size=256,
            format='.jpg'
        )
    except Exception as e:
        print(f"错误: 初始化 TestDataset 失败: {e}。测试将跳过。")
        test_set = None

    # ======== DataLoader ========
    batch_size = opt.batch_size if hasattr(opt, 'batch_size') else 24
    num_workers = opt.num_workers if hasattr(opt, 'num_workers') else 16

    loader_train = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_train,
        pin_memory=True
    )

    loader_test = None
    if test_set:
        loader_test = DataLoader(
            dataset=test_set,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_test
        )

    # ======== 模型初始化 ========
    teacher_net = VIFNetInconsistencyTeacher().to(opt.device)
    student_net = VIFNetInconsistencyTeacher().to(opt.device)

    # --- [新增] 初始化边缘检测器 ---
    edge_detector = SobelEdgeDetector().to(opt.device)
    # 确保它不参与训练（如果它有参数的话，Sobel 没有可训练参数，但以防万一）
    for param in edge_detector.parameters():
        param.requires_grad = False
    edge_detector.eval()
    # --- [新增结束] ---

    pretrained_teacher_path = "/root/autodl-tmp/CoA_daima_xiugai/Teacher_train/saved_model/best.pth"
    print(f"加载预训练教师模型: {pretrained_teacher_path}")

    try:
        checkpoint = torch.load(pretrained_teacher_path, map_location=opt.device)
        teacher_state_dict = OrderedDict()
        student_state_dict = OrderedDict()
        has_module_prefix = any(k.startswith('module.') for k in checkpoint.keys())

        for k, v in checkpoint.items():
            name = k[7:] if has_module_prefix else k
            teacher_state_dict[name] = v.clone()
            student_state_dict[name] = v.clone()

        teacher_net_load_result = teacher_net.load_state_dict(teacher_state_dict, strict=False)
        print("加载 EMA 教师网络状态:", teacher_net_load_result)
        teacher_net.eval()

        student_net_load_result = student_net.load_state_dict(student_state_dict, strict=False)
        print("加载学生网络状态:", student_net_load_result)
        if teacher_net_load_result.missing_keys or teacher_net_load_result.unexpected_keys:
            print("EMA 教师不匹配键:", teacher_net_load_result)
        if student_net_load_result.missing_keys or student_net_load_result.unexpected_keys:
            print("学生不匹配键:", student_net_load_result)
    except FileNotFoundError:
        print(f"错误: 预训练教师模型未找到 {pretrained_teacher_path}。将从头开始训练。")
    except Exception as e:
        print(f"错误: 加载预训练教师模型失败: {e}。将从头开始训练。")

    if opt.device == 'cuda':
        teacher_net = torch.nn.DataParallel(teacher_net)
        student_net = torch.nn.DataParallel(student_net)
        cudnn.benchmark = True

    try:
        pytorch_total_params = sum(p.numel() for p in student_net.parameters() if p.requires_grad)
        print("学生网络可训练参数量: ==> {}".format(pytorch_total_params))
    except Exception as e:
        print(f"无法计算总参数量: {e}")
    print("------------------------------------------------------------------")

    # ======== 损失函数 ========
    criterion = []
    criterion.append(nn.L1Loss().to(opt.device))  # L1_r
    clip_loss_instance = None
    if clip_model is not None and text_features is not None:
        try:
            clip_loss_instance = L_clip_from_feature().to(opt.device)
        except Exception as e:
            print(f"错误: 创建 L_clip_from_feature 实例失败: {e}。CLIP 损失已禁用。")
            clip_loss_instance = None
    criterion.append(clip_loss_instance)

    if criterion[1] is None:
        opt.w_loss_Clip = 0
        print("警告: CLIP 损失已禁用。")


    criterion.append(nn.L1Loss().to(opt.device))  # Edge Loss (criterion[2])

    # ======== 优化器（仅学生） ========
    optimizer = optim.Adam(
        params=filter(lambda x: x.requires_grad, student_net.parameters()),
        lr=opt.start_lr, betas=(0.9, 0.999), eps=1e-08
    )
    optimizer.zero_grad()

    # ======== 开始训练 ========
    train(
        teacher_net,
        student_net,
        loader_train,
        loader_test if loader_test else None,
        optimizer,
        criterion,
        text_features,
        edge_detector  # <--- 传入边缘检测器
    )
