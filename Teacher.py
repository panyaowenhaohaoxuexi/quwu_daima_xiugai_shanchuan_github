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

# --- [修改] 导入 SSIM, ContrastLoss 和 PerceptualLoss ---
from loss import SSIM
from loss.teacher_region_loss import compute_teacher_region_loss
# --- [修改结束] ---

# --- [修改] 导入新的数据集类和模型类 ---
from data import MultiModalHazeDataset, TestDataset, SynthMultiModalDataset, collate_synth  # TestDataset 现在也支持三模态
from metric import psnr, ssim
# from model import DualStreamTeacher # <--- 不再使用原始模型
from model import VIFNetInconsistencyTeacher, CannyEdgeDetector  # <--- 使用新的融合模型
# --- [修改结束] ---
from option.Teacher import opt  # 导入配置选项

# --- [新增] 导入 Eval.py 所需的模块 ---
import glob
import torchvision
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode

# --- [新增结束] ---
from utils.visualize_mask import visualize_epoch_mask
from utils.visualize_teacher_region import save_teacher_region_visualization
# --- [新增] 将 device 和 transform 移至全局 ---
# (以便 dehaze 函数和 train 函数都能访问)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = Compose([
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])
# --- [新增结束] ---

# --- [新增] 定义掩码的预处理流程 (仅 ToTensor) ---
transform_mask = Compose([
    ToTensor()
])
# --- [新增结束] ---


def find_sky_mask_path(sky_mask_dir, image_name, suffix='', ext=''):
    if not sky_mask_dir:
        return None

    stem, original_ext = os.path.splitext(image_name)
    ext_or_original = ext if ext else original_ext
    candidates = [
        os.path.join(sky_mask_dir, stem + suffix + ext_or_original),
        os.path.join(sky_mask_dir, image_name),
        os.path.join(sky_mask_dir, stem + ".png"),
        os.path.join(sky_mask_dir, stem + ".jpg"),
        os.path.join(sky_mask_dir, stem + ".jpeg"),
        os.path.join(sky_mask_dir, stem + "_sky.png"),
        os.path.join(sky_mask_dir, stem + "_sky.jpg"),
        os.path.join(sky_mask_dir, stem + "_sky.jpeg"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None

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
    支持训练/测试返回的 3-item / 4-item / 5-item batch。
    """
    # 过滤掉 batch 中第一个元素为 None 的项
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch:
        # 如果整个批次都无效，根据训练/测试返回不同数量的空值
        # 假设通过 len(batch[0]) 判断是训练(3)还是测试(4)，但这不可靠
        # 更稳妥的方式是让调用者处理可能的空 batch
        # 这里返回适用于训练和测试的最小公倍数或根据需要调整
        # 返回空元组，让调用者检查
        return ()  # 返回空元组
    # 使用 PyTorch 默认的 collate 函数将有效样本整理成批次张量/列表
    return torch.utils.data.dataloader.default_collate(batch)


# --- [新增] 从 Eval.py 复制的 dehaze 函数 ---
# (它依赖于全局定义的 device 和 transform)
# --- [修改] dehaze 函数现在接受 mask_image_path=None ---
def dehaze(model, vis_image_path, ir_image_path, mask_image_path, sky_mask_image_path, folder):
    """
    使用加载的双流模型对指定路径的可见光、红外和可选的掩码进行去雾处理，
    并将结果保存到指定文件夹。
    (此版本已更新，支持掩码加载)
    """
    try:
        # 1. 加载并预处理可见光图像
        haze_vis = transform(Image.open(vis_image_path).convert("RGB")).unsqueeze(0).to(device)
        # 2. 加载并预处理红外图像
        haze_ir = transform(Image.open(ir_image_path).convert("RGB")).unsqueeze(0).to(device)  # 假设红外也用相同 transform

        haze_mask_tensor = None  # 默认掩码为 None
        sky_mask_tensor = None

        # --- [新增] 掩码加载逻辑 (参考 Eval_EMA.py) ---
        if mask_image_path:  # 检查路径是否非空
            if os.path.exists(mask_image_path):
                # 掩码存在，加载它 (使用 "L" 模式加载单通道灰度图)
                haze_mask_tensor = transform_mask(Image.open(mask_image_path).convert("L")).unsqueeze(0).to(device)
                # 确保掩码是 0-1 范围 (ToTensor() 已经做到了)
            else:
                # 提供了掩码路径但文件丢失 (对应"无掩码"情况)
                print(f"\n警告: 提供了掩码路径但文件未找到: {mask_image_path}。将回退到基础注入模式。")
                # haze_mask_tensor 保持为 None
        # --- [新增结束] ---

        if sky_mask_image_path:
            if os.path.exists(sky_mask_image_path):
                sky_mask_tensor = transform_mask(Image.open(sky_mask_image_path).convert("L")).unsqueeze(0).to(device)
                sky_mask_tensor = (sky_mask_tensor >= 0.5).float()
            else:
                print(f"\n警告: 提供了 sky mask 路径但文件未找到: {sky_mask_image_path}。将回退到 sky_mask=None。")

        # 3. 获取原始图像尺寸 (以可见光为准)
        h, w = haze_vis.shape[2], haze_vis.shape[3]

        # 4. 调整尺寸
        target_h = (h // 16) * 16
        target_w = (w // 16) * 16
        if target_h == 0: target_h = 16
        if target_w == 0: target_w = 16

        if h != target_h or w != target_w:
            haze_vis_resized = Resize((target_h, target_w), interpolation=InterpolationMode.BICUBIC, antialias=True)(
                haze_vis)
            haze_ir_resized = Resize((target_h, target_w), interpolation=InterpolationMode.BICUBIC, antialias=True)(
                haze_ir)
        else:
            haze_vis_resized = haze_vis
            haze_ir_resized = haze_ir

        # --- [新增] 仅当掩码张量存在时才调整其尺寸 ---
        haze_mask_resized = None  # 默认 resized 掩码为 None
        if haze_mask_tensor is not None:
            # 掩码使用 BILINEAR (最近邻也行，但 BILINEAR 更平滑)
            resize_mask_fn = Resize((target_h, target_w), interpolation=InterpolationMode.BILINEAR, antialias=False)
            haze_mask_resized = resize_mask_fn(haze_mask_tensor) if (
                    h != target_h or w != target_w) else haze_mask_tensor
        # --- [新增结束] ---

        sky_mask_resized = None
        if sky_mask_tensor is not None:
            resize_mask_fn = Resize((target_h, target_w), interpolation=InterpolationMode.BILINEAR, antialias=False)
            sky_mask_resized = resize_mask_fn(sky_mask_tensor) if (
                    h != target_h or w != target_w) else sky_mask_tensor
            sky_mask_resized = (sky_mask_resized >= 0.5).float()

        # 5. 模型推理 (传入三个输入)
        #    - [核心] 传入 haze_mask_resized (它要么是掩码张量，要么是 None)
        if sky_mask_resized is not None and haze_mask_resized is not None:
            print("提示: sky_mask 只在 haze_mask=None 时生效；当前传入了 haze_mask，因此会跳过 CMDN，sky_mask 不参与本次推理。")
        pred_output = model(
            haze_vis_resized,
            haze_ir_resized,
            haze_mask=haze_mask_resized,
            sky_mask=sky_mask_resized
        )

        if isinstance(pred_output, tuple):
            out_tensor = pred_output[0]
        else:
            out_tensor = pred_output

        out = out_tensor.squeeze(0)  # 移除批次维度
        out = out.clamp(0, 1)

        # 6. 将输出图像尺寸恢复到原始尺寸
        if h != target_h or w != target_w:
            out = Resize((h, w), interpolation=InterpolationMode.BICUBIC, antialias=True)(out)

        # 7. 保存
        output_filename = os.path.basename(vis_image_path)
        torchvision.utils.save_image(out, os.path.join(folder, output_filename))

    except FileNotFoundError as e:
        print(f"\n错误: 找不到图像文件 {e}。跳过。")
    except Exception as e:
        base_name = os.path.basename(vis_image_path)
        print(f"\n处理图像 {base_name} 时发生错误: {e}。跳过。")


# --- [新增结束] ---
# --- [修改] run_real_world_test 函数，使其查找并传递掩码 ---
def run_real_world_test(model, epoch, hazy_dir, ir_dir):
    """
    在指定的真实（无标签）数据集上运行推理。
    (此版本已更新，支持掩码加载)
    """
    if not hazy_dir or not ir_dir:
        print(f"\n跳过真实世界测试：未指定 'real_test_hazy_path' 或 'real_test_ir_path'。")
        return

    if not os.path.isdir(hazy_dir):
        print(f"\n警告: 真实测试 hazy 目录不存在: {hazy_dir}。跳过。")
        return

    if not os.path.isdir(ir_dir):
        print(f"\n警告: 真实测试 ir 目录不存在: {ir_dir}。跳过。")
        return

    # --- [新增] 检查掩码文件夹是否有效 (参考 Eval_EMA.py) ---
    use_mask_if_available = False
    mask_folder = opt.real_test_mask_path  # 从 opt 读取新路径

    if mask_folder and os.path.isdir(mask_folder):
        print(f"掩码模式: ON。将从以下路径加载掩码 (如果存在): {mask_folder}")
        use_mask_if_available = True
    else:
        print(f"掩码模式: OFF。未提供或未找到掩码文件夹: '{mask_folder}'。")
        print("所有图像将使用模型的基础注入模式 (base_weight) 运行。")
    # --- [新增结束] ---

    using_specific = bool(opt.real_test_specific_hazy_dir) and (
        os.path.abspath(hazy_dir) == os.path.abspath(opt.real_test_specific_hazy_dir)
    )
    sky_mask_folder = opt.real_test_sky_mask_dir
    if using_specific and opt.real_test_specific_sky_mask_dir:
        sky_mask_folder = opt.real_test_specific_sky_mask_dir

    use_sky_mask_if_available = False
    if sky_mask_folder and os.path.isdir(sky_mask_folder):
        print(f"天空掩码模式: ON。将从以下路径读取 sky mask: {sky_mask_folder}")
        use_sky_mask_if_available = True
    else:
        print("天空掩码模式: OFF。未提供 sky mask 目录，或目录不存在。")

    # 1. 设置输出目录
    output_folder = os.path.join(opt.real_test_output_dir, f'epoch_{epoch}')
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n正在对真实世界图像运行推理 (Epoch {epoch}) -> 保存至 {output_folder}")

    # 2. 查找图像
    vis_images = sorted(glob.glob(os.path.join(hazy_dir, '*.jpg')) + \
                        glob.glob(os.path.join(hazy_dir, '*.png')) + \
                        glob.glob(os.path.join(hazy_dir, '*.jpeg')))

    if not vis_images:
        print(f"警告: 在 {hazy_dir} 中未找到图像文件。")
        return

    # 3. 设置模型为评估模式
    model.eval()

    # 4. 禁用梯度并开始推理
    with torch.no_grad():
        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} | {rate_fmt}"
        for vis_path in tqdm(vis_images, bar_format=bar_format, desc=f"Epoch {epoch} 真实测试"):
            base_filename = os.path.basename(vis_path)
            ir_path = os.path.join(ir_dir, base_filename)

            # --- [修改] 动态构造掩码路径 ---
            mask_path = None  # 默认为 None (无掩码模式)
            if use_mask_if_available:
                # 仅当掩码文件夹有效时，才构造路径
                mask_path = os.path.join(mask_folder, base_filename)
                # 注意: 我们不在这里检查 os.path.exists(mask_path)
                # 我们把 mask_path (可能存在也可能不存在) 传递给 dehaze 函数
                # dehaze 函数内部会处理 "文件不存在" 的情况 (即视为"无掩码")
            # --- [修改结束] ---

            sky_mask_path = None
            if use_sky_mask_if_available:
                sky_mask_path = find_sky_mask_path(
                    sky_mask_folder,
                    base_filename,
                    suffix=opt.sky_mask_suffix,
                    ext=opt.sky_mask_ext
                )
                if sky_mask_path is None:
                    print(f"\n警告: 未找到 {base_filename} 对应的 sky mask；回退为 sky_mask=None。")

            if os.path.exists(ir_path):
                # [修改] 调用 dehaze，传入 mask_path (可能是路径字符串，也可能是 None)
                dehaze(model, vis_path, ir_path, mask_path, sky_mask_path, output_folder)
            else:
                print(f"\n警告: 找不到 {base_filename} 对应的红外图像: {ir_path}。跳过。")

    # 5. （可选）恢复训练模式
    model.train()


# --- [新增结束] ---


# 定义函数 train：执行模型训练的主要逻辑
def train(teacher_net, loader_train, loader_test, optim, criterion, edge_detector):
    """
    执行合成域 Teacher 区域补全监督训练。
    """
    losses = []
    loss_log = {'rec': [], 'density': [], 'mask': [], 'ssim': [], 'edge': [], 'total': []}
    loss_log_tmp = {'rec': [], 'density': [], 'mask': [], 'ssim': [], 'edge': [], 'total': []}
    psnr_log = []

    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    loader_train_iter = iter(loader_train)
    ssim_loss_module = criterion[1] if criterion and len(criterion) > 1 else None

    for step in range(start_step + 1, steps + 1):
        teacher_net.train()
        lr = opt.start_lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr

        try:
            batch_data = next(loader_train_iter)
        except StopIteration:
            loader_train_iter = iter(loader_train)
            try:
                batch_data = next(loader_train_iter)
            except StopIteration:
                print("\n警告: 数据加载器在 epoch 开始时意外耗尽。")
                break
            except Exception as e:
                print(f"\n错误: 在步骤 {step} (StopIteration后) 加载数据时出错: {e}。跳过批次。")
                continue
        except Exception as e:
            print(f"\n错误: 在步骤 {step} 加载数据时出错: {e}。跳过批次。")
            continue

        if not batch_data or len(batch_data) != 5:
            raise RuntimeError(
                "New synthetic Teacher training requires hazy_vis, clear_vis, infrared, density_gt, mask_gt."
            )

        hazy_vis, clear_vis, infrared, density_gt, mask_gt = batch_data
        if not hazy_vis.numel():
            print(f"\n警告: 在步骤 {step} 跳过空批次。")
            continue

        hazy_vis = hazy_vis.to(opt.device, non_blocking=True)
        clear_vis = clear_vis.to(opt.device, non_blocking=True)
        infrared = infrared.to(opt.device, non_blocking=True)
        density_gt = density_gt.to(opt.device, non_blocking=True)
        mask_gt = mask_gt.to(opt.device, non_blocking=True)

        tau_start = getattr(opt, "gumbel_tau_start", 1.0)
        tau_end = getattr(opt, "gumbel_tau_end", 0.1)
        progress = min(1.0, (step - 1) / max(1, steps - 1))
        tau = tau_start + (tau_end - tau_start) * progress
        if hasattr(teacher_net, "module"):
            teacher_net.module.set_gumbel_tau(tau)
        else:
            teacher_net.set_gumbel_tau(tau)

        out = teacher_net(hazy_vis, infrared, return_dict=True)
        pred_image = out["pred_clear"]

        lambda_rec = getattr(opt, "w_loss_rec", getattr(opt, "w_loss_L1", 1.0))
        lambda_density = getattr(opt, "w_loss_density", 1.0)
        lambda_mask = getattr(opt, "w_loss_mask", 1.0)
        lambda_ssim = getattr(opt, "w_loss_SSIM", 0.0)
        lambda_edge = getattr(opt, "w_loss_Edge", 0.0)

        loss_dict = compute_teacher_region_loss(
            pred_clear=pred_image,
            clear_gt=clear_vis,
            density_map=out["density_map"],
            density_gt=density_gt,
            mask_logits=out["mask_logits"],
            mask_prob=out["mask_prob"],
            mask_gt=mask_gt,
            lambda_rec=lambda_rec,
            lambda_density=lambda_density,
            lambda_mask=lambda_mask,
            lambda_ssim=lambda_ssim,
            lambda_edge=lambda_edge,
            ssim_module=ssim_loss_module,
        )
        loss = loss_dict["total"]

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.append(loss.item())
        for key in ("rec", "density", "mask", "ssim", "edge"):
            loss_log_tmp[key].append(loss_dict[key].item())
        loss_log_tmp['total'].append(loss.item())

        with torch.no_grad():
            train_psnr = psnr(pred_image.detach().clamp(0, 1), clear_vis)
            train_ssim = ssim(pred_image.detach().clamp(0, 1), clear_vis).item()
            mask_ratio = out["binary_mask"].mean().item()
            density_mean = out["density_map"].mean().item()

        print(
            f'\rloss:{loss.item():.5f} | rec:{loss_dict["rec"].item():.5f} '
            f'| density:{loss_dict["density"].item():.5f} | mask:{loss_dict["mask"].item():.5f} '
            f'| ssim_loss:{loss_dict["ssim"].item():.5f} | edge:{loss_dict["edge"].item():.5f} '
            f'| mask_ratio:{mask_ratio:.4f} | density_mean:{density_mean:.4f} | tau:{tau:.4f} '
            f'| PSNR:{train_psnr:.4f} | SSIM:{train_ssim:.4f} '
            f'| step :{step}/{steps} | lr :{lr :.9f} | time_used :{(time.time() - start_time) / 60 :.1f}',
            end='', flush=True)

        steps_per_epoch = len(loader_train) if loader_train else 0
        # Epoch 结束统计
        if steps_per_epoch > 0 and step % steps_per_epoch == 0:
            try:
                loader_train_iter = iter(loader_train)
            except Exception as e:
                print(f"\n错误: Epoch结束时重新初始化训练迭代器失败: {e}")

            for key in loss_log.keys():
                if loss_log_tmp[key]:  # 确保列表不为空
                    loss_log[key].append(np.mean(np.array(loss_log_tmp[key])))
                loss_log_tmp[key] = []  # 清空临时记录
            os.makedirs(opt.saved_data_dir, exist_ok=True)
            try:
                np.save(os.path.join(opt.saved_data_dir, 'losses.npy'), losses)
            except Exception as e:
                print(f"\n错误: 保存 losses.npy 失败: {e}")

            try:
                epoch_idx = step // steps_per_epoch
                save_teacher_region_visualization(
                    opt.saved_data_dir,
                    f"epoch_{epoch_idx}",
                    hazy_vis.detach().cpu(),
                    infrared.detach().cpu(),
                    pred_image.detach().cpu(),
                    clear_vis.detach().cpu(),
                    out["density_map"].detach().cpu(),
                    density_gt.detach().cpu(),
                    out["mask_prob"].detach().cpu(),
                    out["binary_mask"].detach().cpu(),
                    mask_gt.detach().cpu(),
                )
            except Exception as e:
                print(f"\n[teacher_region_vis] 可视化失败，跳过: {e}")

        # 确定评估频率 (与之前逻辑保持一致)
        eval_freq_fine = 5 * steps_per_epoch if steps_per_epoch > 0 else opt.iters_per_epoch
        eval_freq_coarse = opt.iters_per_epoch if steps_per_epoch > 0 else steps  # 如果 loader_train 为空，则只在最后评估一次

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
        elif step == steps:  # 确保最后一步进行评估
            perform_eval = True
            # 计算最后一个epoch的编号
            if eval_freq_fine > 0 and step > opt.finer_eval_step:
                base_epochs = opt.finer_eval_step // eval_freq_coarse if eval_freq_coarse > 0 else 0
                current_epoch = base_epochs + math.ceil((step - opt.finer_eval_step) / eval_freq_fine)
            elif eval_freq_coarse > 0:
                current_epoch = math.ceil(step / eval_freq_coarse)
            else:
                current_epoch = opt.epochs  # 或 1

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
                state_dict = model_to_save.state_dict()  # 直接获取 state_dict

                if psnr_eval > max_psnr:
                    max_ssim = max(max_ssim, ssim_eval)
                    max_psnr = max(max_psnr, psnr_eval)
                    print(
                        f'模型在步骤 :{step}| epoch: {current_epoch} 保存 | 最高 psnr:{max_psnr:.4f}| 最高 ssim:{max_ssim:.4f}')
                    saved_best_model_path = os.path.join(opt.saved_model_dir, 'best.pth')
                    torch.save(state_dict, saved_best_model_path)

                saved_single_model_path = os.path.join(opt.saved_model_dir, str(current_epoch) + '.pth')
                torch.save(state_dict, saved_single_model_path)
            except Exception as e:
                print(f"\n错误: 保存模型权重失败 (epoch {current_epoch}): {e}")

            if getattr(opt, "run_real_infer_in_teacher", False):
                hazy_source = opt.real_test_specific_hazy_dir if opt.real_test_specific_hazy_dir else opt.real_test_hazy_path
                ir_source = opt.real_test_specific_ir_dir if opt.real_test_specific_ir_dir else opt.real_test_ir_path
                run_real_world_test(
                    teacher_net,
                    current_epoch,
                    hazy_source,
                    ir_source
                )

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


def pad_mask(x, patch_size):
    """
    对二值 mask 做常数 0 padding，使高度和宽度成为 patch_size 的整数倍。
    mask 不使用 reflect padding，避免把天空区域反射到边界外。
    """
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='constant', value=0)
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

        sky_mask = None
        if len(batch_test) == 5:
            inputs_vis, inputs_ir, targets, sky_mask, hazy_name_list = batch_test
            if not inputs_vis.numel():
                print(f"警告: 在测试加载器索引 {i} 遇到空数据。跳过。")
                continue
            hazy_name = hazy_name_list[0] if isinstance(hazy_name_list,
                                                        (list, tuple)) and hazy_name_list else f"Unknown_Index_{i}"
        elif len(batch_test) == 4:
            inputs_vis, inputs_ir, targets, hazy_name_list = batch_test
            # 处理可能的空 batch 情况（如果 collate_fn 返回了带空 tensor 的元组）
            if not inputs_vis.numel():
                print(f"警告: 在测试加载器索引 {i} 遇到空数据。跳过。")
                continue
            # 获取文件名，处理列表情况
            hazy_name = hazy_name_list[0] if isinstance(hazy_name_list,
                                                        (list, tuple)) and hazy_name_list else f"Unknown_Index_{i}"
        else:
            print(f"测试加载器返回了预期外的数据格式: {len(batch_test)} 项。跳过批次 {i}。")
            continue
        # --- [修改结束] ---

        inputs_vis = inputs_vis.to(opt.device, non_blocking=True)
        inputs_ir = inputs_ir.to(opt.device, non_blocking=True)  # --- [修改] 添加红外输入到设备 ---
        targets = targets.to(opt.device, non_blocking=True)
        if sky_mask is not None:
            sky_mask = sky_mask.to(opt.device, non_blocking=True)

        with torch.no_grad():
            H, W = inputs_vis.shape[2:]  # 使用可见光尺寸作为基准
            try:
                # --- [修改] 填充两个输入 (假设需要16的倍数) ---
                inputs_vis_padded = pad_img(inputs_vis, 16)
                inputs_ir_padded = pad_img(inputs_ir, 16)
                if sky_mask is not None:
                    sky_mask_padded = pad_mask(sky_mask, 16)
                    sky_mask_padded = (sky_mask_padded >= 0.5).float()
                else:
                    sky_mask_padded = None
                # --- [修改结束] ---
            except Exception as e:
                print(f"\n错误: 测试时填充图像 {hazy_name} 失败: {e}。跳过。")
                continue

            try:
                # --- [修改] 正确调用双流模型 (测试时不用掩码, haze_mask=None) ---
                pred_output = net(
                    inputs_vis_padded,
                    inputs_ir_padded,
                    haze_mask=None,
                    sky_mask=sky_mask_padded
                )[0]
                # --- [修改结束] ---

                pred = pred_output  # pred_output 已经是图像
                pred = pred.clamp(0, 1)  # 限制范围

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


if __name__ == "__main__":

    set_seed_torch(2024)

    # --- [修改] 数据集路径和实例化 ---
    # !! 请将下面的路径修改为你实际的数据集路径 !!
    train_base_dir = opt.train_data_dir  # 训练集根目录
    test_base_dir = opt.test_data_dir  # 测试集根目录

    # 训练数据集路径：合成域五元组 hazy_vis, clear_vis, infrared, density_gt, mask_gt
    try:
        train_set = SynthMultiModalDataset(
            root=train_base_dir,
            train=True,
            size=256,
        )
        print(f"成功加载训练数据集，共 {len(train_set)} 个样本。")
    except Exception as e:
        print(f"错误: 初始化训练数据集 SynthMultiModalDataset 失败: {e}")
        train_set = None  # 设置为 None 以便后续检查
        exit()  # 训练集加载失败则退出

    # 测试数据集路径
    test_hazy_vis_folder = os.path.join(test_base_dir, 'hazy')
    test_ir_folder = os.path.join(test_base_dir, 'ir')
    test_clear_vis_folder = os.path.join(test_base_dir, 'clear')
    print("[SkyMask][Test] use_test_sky_mask=", opt.use_test_sky_mask)
    print("[SkyMask][Test] test_sky_mask_dir=", opt.test_sky_mask_dir)
    print("[SkyMask][Test] sky_mask_suffix=", opt.sky_mask_suffix)
    print("[SkyMask][Test] sky_mask_ext=", opt.sky_mask_ext)
    print("[SkyMask][Test] require_test_sky_mask=", opt.require_test_sky_mask)
    try:
        test_set = TestDataset(
            hazy_visible_path=test_hazy_vis_folder,
            infrared_path=test_ir_folder,
            clear_visible_path=test_clear_vis_folder,
            size=256,  # 测试时使用中心裁剪或缩放
            format='auto',  # 自动兼容 jpg/png/multi-level 数据集
            sky_mask_path=opt.test_sky_mask_dir,
            use_sky_mask=opt.use_test_sky_mask,
            sky_mask_suffix=opt.sky_mask_suffix,
            sky_mask_ext=opt.sky_mask_ext,
            require_sky_mask=opt.require_test_sky_mask
        )
        print(f"成功加载测试数据集，共 {len(test_set)} 个样本。")
    except Exception as e:
        print(f"错误: 初始化测试数据集 TestDataset 失败: {e}。测试将跳过。")
        test_set = None
    # --- [修改结束] ---

    # --- DataLoader ---
    # 从配置中读取 batch_size 和 num_workers，提供默认值
    batch_size = getattr(opt, 'batch_size', 8)  # 使用 opt 中的 batch_size，默认为 4
    num_workers = getattr(opt, 'num_workers', 16)  # 使用 opt 中的 num_workers，默认为 4

    loader_train = None
    if train_set:  # 仅在 train_set 成功加载时创建 DataLoader
        loader_train = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_synth,
            pin_memory=True,  # 如果内存充足，可以加速数据传输
            drop_last=True  # 丢弃最后一个不完整的 batch，避免 BN 层问题
        )
    else:
        print("错误：训练数据集加载失败，无法创建训练 DataLoader。")
        exit()

    loader_test = None
    if test_set:
        loader_test = DataLoader(
            dataset=test_set,
            batch_size=8,  # 测试时通常 batch_size=1
            shuffle=False,
            num_workers=16,  # 测试时 worker 少一些通常没问题
            collate_fn=collate_fn_skip_none
        )

    # --- [修改] 模型初始化 ---
    teacher_net = VIFNetInconsistencyTeacher(
        tau_min=opt.tau_min,
        tau_max=opt.tau_max,
        gate_temperature=opt.gate_temperature,
        support_gamma=opt.support_gamma,
        support_floor=opt.support_floor,
        support_threshold=opt.support_threshold,
        support_temperature=opt.support_temperature,
        hard_gate_threshold=opt.hard_gate_threshold,
    ).to(opt.device)  # 实例化新的模型
    teacher_net = teacher_net.to(opt.device)
    # --- [修改结束] ---

    # 新区域 loss 内部使用 Sobel edge；不再初始化旧 Canny/Boundary/CrossModal 链路。
    edge_detector = None

    epoch_size = len(loader_train) if loader_train else 0
    if epoch_size == 0:
        print("错误：训练 DataLoader 为空或长度为 0。请检查数据集和批处理大小。")
        exit()
    print("每个 Epoch 的步数 (epoch_size): ", epoch_size)

    if opt.device == 'cuda':
        # 如果有多张 GPU，DataParallel 会自动使用
        print(f"检测到 CUDA 设备，使用 DataParallel (可用 GPU 数量: {torch.cuda.device_count()})。")
        teacher_net = torch.nn.DataParallel(teacher_net)
        cudnn.benchmark = True  # 启用 benchmark 加速

    try:
        pytorch_total_params = sum(p.numel() for p in teacher_net.parameters() if p.requires_grad)
        print("模型可训练参数总量: ==> {}".format(pytorch_total_params))
    except Exception as e:
        print(f"计算总参数量时出错: {e}")
    print("------------------------------------------------------------------")

    # 新合成域 Teacher 只需要 SSIM module 供 compute_teacher_region_loss 复用。
    criterion = [nn.L1Loss().to(opt.device), SSIM().to(opt.device)]

    # Adam 优化器
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, teacher_net.parameters()), lr=opt.start_lr,
                           betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer.zero_grad()  # 初始化梯度

    # 开始训练
    print("开始训练...")
    # AAA
    # --- [修改] 传入 edge_detector ---
    train(teacher_net, loader_train, loader_test, optimizer, criterion, edge_detector)
    # --- [修改结束] ---
    # AAA
    print("训练完成。")
