"""
这段代码的核心作用是使用知识蒸馏（Knowledge Distillation）技术来训练一个用于图像去雾（Image Dehazing）的 Student (学生) 模型。
它首先加载一个已经预训练好的、性能强大的 Teacher (教师) 模型，并将其设置为评估模式（即冻结其参数）。
然后，脚本初始化一个新的 Student 模型（通常更轻量级）。
在训练过程中，Student 模型不仅像传统训练一样，通过 L1、SSIM 和对比度损失来学习将有雾图像恢复为清晰的真实图像，它还必须同时学习模仿 Teacher 模型的内部行为。
这是通过 FA (特征亲和力) 损失实现的，该损失迫使 Student 模型的中间特征图谱接近 Teacher 模型的特征图谱。最终目标是训练出一个高效、轻量的 Student 模型，使其去雾性能接近那个更复杂的 Teacher 模型。
脚本会周期性地评估 Student 模型的 PSNR 和 SSIM 指标，并保存性能最佳的 Student 模型权重。
用的是model/Student.py
使用的是RESIDE_Dataset
这里的知识蒸馏并不仅仅是让学生模型模仿教师模型的最终输出图像，更重要的是强制学生模型在网络中间层学习和生成与教师模型相似的特征。
通过最小化这些中间特征之间的差异（loss_FA），学生模型被引导去学习教师模型处理和理解图像的方式，从而继承教师模型的知识。
当然，学生模型的最终输出也会受到传统监督损失 (L1, SSIM, Cr) 的约束，以确保其生成的图像接近真实的清晰图像。
这个脚本
"""

import math  # 导入数学库
import os  # 导入操作系统库
import time  # 导入时间库
import numpy as np  # 导入Numpy库
import itertools  # 导入迭代器工具（此脚本中未显式使用）
import torch  # 导入PyTorch
import clip  # 导入CLIP模型库（此脚本中未显式使用，可能在data.CLIP_loader中用到）
import torch.nn.functional as F  # 导入PyTorch函数式接口
from torch import optim, nn  # 导入PyTorch优化器和神经网络模块
from torch.backends import cudnn  # 导入cuDNN，用于GPU加速
from torch.utils.data import DataLoader  # 导入数据加载器
import torch.utils.data  # 导入PyTorch数据工具
from metric import psnr, ssim  # 导入自定义的PSNR和SSIM评估指标
from loss import SSIM, FA, ContrastLoss  # 导入自定义的SSIM损失、FA（Feature Affinity）损失、对比损失
from data import RESIDE_Dataset, TestDataset, CLIP_loader  # 导入自定义的数据集
from model import Teacher, Student  # 导入教师模型和学生模型
from collections import OrderedDict  # 导入有序字典，用于处理模型权重
from option.KD import opt  # 导入知识蒸馏（KD）的配置选项

start_time = time.time()  # 记录脚本开始时间
steps = opt.iters_per_epoch * opt.epochs  # 计算总训练迭代次数
T = steps  # T（总步数）用于学习率调度


def lr_schedule_cosdecay(t, T, init_lr=opt.start_lr, end_lr=opt.end_lr):
    """
    定义余弦衰减学习率调度函数
    :param t: 当前步数
    :param T: 总步数
    :param init_lr: 初始学习率
    :param end_lr: 最终学习率
    :return: 当前步的学习率
    """
    lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / T))
    return lr


def train(teacher_net, student_net, loader_train_1, loader_test, optim, criterion):
    """
    定义训练函数
    :param teacher_net: 教师模型（预训练且固定）
    :param student_net: 学生模型（待训练）
    :param loader_train_1: 训练数据加载器
    :param loader_test: 测试数据加载器
    :param optim: 优化器（用于学生模型）
    :param criterion: 损失函数列表
    """
    losses = []  # 存储每一步的总损失
    # 存储每个epoch的平均损失
    loss_log = {'FA': [], 'L1': [], 'SSIM': [], 'Cr': [], 'total': []}
    # 临时存储一个epoch内各步的损失
    loss_log_tmp = {'FA': [], 'L1': [], 'SSIM': [], 'Cr': [], 'total': []}
    psnr_log = []  # 存储每个评估点的PSNR

    start_step = 0  # 初始化起始步数
    max_ssim = 0  # 初始化最佳SSIM
    max_psnr = 0  # 初始化最佳PSNR
    ssims = []  # 存储每个评估点的SSIM
    psnrs = []  # 存储每个评估点的PSNR

    loader_train_iter_1 = iter(loader_train_1)  # 创建训练数据加载器的迭代器

    for step in range(start_step + 1, steps + 1):  # 开始训练循环
        teacher_net.eval()  # !! 关键：将教师模型设置为评估模式（不更新参数，BN层等固定）
        student_net.train()  # 将学生模型设置为训练模式

        lr = opt.start_lr  # 获取初始学习率
        if not opt.no_lr_sche:  # 检查是否使用学习率调度
            lr = lr_schedule_cosdecay(step, T)  # 计算当前步的学习率
            for param_group in optim.param_groups:  # 遍历优化器中的参数组
                param_group["lr"] = lr  # 更新学习率

        # 获取一批训练数据 (x: 有雾图像, y: 清晰图像)
        x, y = next(loader_train_iter_1)
        x = x.to(opt.device)  # 数据移动到GPU
        y = y.to(opt.device)

        # --- 知识蒸馏 ---
        with torch.no_grad():  # 教师模型不计算梯度
            teacher_output = teacher_net(x)  # 获取教师模型的输出（[0]为图像, [1]为特征）
        student_out = student_net(x)  # 获取学生模型的输出（[0]为图像, [1]为特征）
        # -----------------

        # 初始化各项损失
        loss_FA = 0
        loss_L1 = 0
        loss_SSIM = 0
        loss_Cr = 0

        # --- 计算损失 ---
        # 1. 特征亲和力损失 (FA loss) - 知识蒸馏损失
        #    让学生模型的中间特征(student_out[1])模仿教师模型的中间特征(teacher_output[1])
        if opt.w_loss_FA > 0:
            loss_FA = criterion[0](student_out[1], teacher_output[1])

        # 2. L1 损失 - 监督损失
        #    计算学生模型的输出图像(student_out[0])与真实清晰图像(y)的L1距离
        if opt.w_loss_L1 > 0:
            loss_L1 = criterion[1](student_out[0], y)

        # 3. SSIM 损失 - 监督损失
        #    计算学生输出与真实图像的SSIM
        if opt.w_loss_SSIM > 0:
            loss_SSIM = (1 - criterion[2](student_out[0], y))

        # 4. 对比度损失 (Cr loss) - 监督损失
        if opt.w_loss_Cr > 0:
            loss_Cr = criterion[3](student_out[0], y, x)

        # 计算加权总损失
        loss = opt.w_loss_FA * loss_FA + opt.w_loss_L1 * loss_L1 + opt.w_loss_SSIM * loss_SSIM + opt.w_loss_Cr * loss_Cr

        # --- 反向传播（只更新学生模型） ---
        loss.backward()  # 反向传播，计算梯度
        optim.step()  # 更新学生模型的参数
        optim.zero_grad()  # 清空梯度

        # --- 日志记录 ---
        losses.append(loss.item())
        loss_log_tmp['FA'].append(loss_FA.item())
        loss_log_tmp['L1'].append(loss_L1.item())
        loss_log_tmp['SSIM'].append(loss_SSIM.item())
        loss_log_tmp['Cr'].append(loss_Cr.item())
        loss_log_tmp['total'].append(loss.item())

        # 打印当前训练状态
        print(
            f'\rloss:{loss.item():.5f} | FA:{opt.w_loss_FA * (loss_FA.item() if opt.w_loss_FA > 0 else 0):.5f} | L1:{opt.w_loss_L1 * (loss_L1.item() if opt.w_loss_L1 > 0 else 0):.5f} | SSIM:{opt.w_loss_SSIM * (loss_SSIM.item() if opt.w_loss_SSIM > 0 else 0):.5f} | Cr:{opt.w_loss_Cr * (loss_Cr.item() if opt.w_loss_Cr > 0 else 0):.5f} | step :{step}/{steps} | lr :{lr :.9f} | time_used :{(time.time() - start_time) / 60 :.1f}',
            end='', flush=True)

        # --- Epoch 结束时的日志 ---
        if step % len(loader_train_1) == 0:
            loader_train_iter_1 = iter(loader_train_1)  # 重置迭代器
            for key in loss_log.keys():
                loss_log[key].append(np.average(np.array(loss_log_tmp[key])))  # 记录epoch平均损失
                loss_log_tmp[key] = []
            np.save(os.path.join(opt.saved_data_dir, 'losses.npy'), losses)  # 保存损失历史

        # --- 周期性评估和保存模型 ---
        if (step % opt.iters_per_epoch == 0 and step <= opt.finer_eval_step) or (
                step > opt.finer_eval_step and (step - opt.finer_eval_step) % (5 * len(loader_train_1)) == 0):

            # 计算当前epoch数（用于日志和保存）
            if step > opt.finer_eval_step:
                epoch = opt.finer_eval_step // opt.iters_per_epoch + (step - opt.finer_eval_step) // (
                        5 * len(loader_train_1))
            else:
                epoch = int(step / opt.iters_per_epoch)

            with torch.no_grad():  # 评估时禁用梯度
                ssim_eval, psnr_eval = test(student_net, loader_test)  # 在测试集上评估学生模型

            # 打印并保存日志
            log = f'\nstep :{step} | epoch: {epoch} | ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f} | lr:{lr:.12f}'
            print(log)
            with open(os.path.join(opt.saved_data_dir, 'log.txt'), 'a') as f:
                f.write(log + '\n')

            # 记录评估指标
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            psnr_log.append(psnr_eval)

            state_dict = student_net.state_dict()  # 获取学生模型的状态字典

            # 处理DataParallel（多GPU）保存模型时带有的'module.'前缀
            if 'module' in list(state_dict.keys())[0]:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '')
                    new_state_dict[name] = v
                state_dict = new_state_dict

            # 检查是否为最佳PSNR
            if psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                print(
                    f'model saved at step :{step}| epoch: {epoch} | max_psnr:{max_psnr:.4f}| max_ssim:{max_ssim:.4f}')
                saved_best_model_path = os.path.join(opt.saved_model_dir, 'best.pth')  # 最佳模型保存路径
                torch.save(state_dict, saved_best_model_path)  # 保存最佳学生模型

            # 保存当前epoch的模型
            saved_single_model_path = os.path.join(opt.saved_model_dir, str(epoch) + '.pth')
            torch.save(state_dict, saved_single_model_path)

            loader_train_iter_1 = iter(loader_train_1)  # 重置训练迭代器

            # 保存评估指标的历史记录
            np.save(os.path.join(opt.saved_data_dir, 'ssims.npy'), ssims)
            np.save(os.path.join(opt.saved_data_dir, 'psnrs.npy'), psnrs)


def pad_img(x, patch_size):
    """
    定义图像填充函数（用于测试，确保输入尺寸能被模型整除）
    """
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size  # 计算高度需填充量
    mod_pad_w = (patch_size - w % patch_size) % patch_size  # 计算宽度需填充量
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')  # 反射填充
    return x


def test(net, loader_test):
    """
    定义测试函数（用于评估学生模型）
    :param net: 待评估的网络（即学生网络）
    :param loader_test: 测试数据加载器
    :return: 平均SSIM和平均PSNR
    """
    net.eval()  # 设置为评估模式
    torch.cuda.empty_cache()  # 清空GPU缓存
    ssims = []
    psnrs = []

    for i, (inputs, targets, hazy_name) in enumerate(loader_test):
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        with torch.no_grad():
            H, W = inputs.shape[2:]  # 记录原始高宽
            inputs = pad_img(inputs, 4)  # 填充输入
            pred = net(inputs)[0].clamp(0, 1)  # 模型推理，取[0]（图像输出），并限制范围
            pred = pred[:, :, :H, :W]  # 裁剪回原始尺寸

        ssim_tmp = ssim(pred, targets).item()  # 计算SSIM
        psnr_tmp = psnr(pred, targets)  # 计算PSNR
        ssims.append(ssim_tmp)
        psnrs.append(psnr_tmp)

    return np.mean(ssims), np.mean(psnrs)  # 返回平均值


def set_seed_torch(seed=2018):
    """
    设置随机种子以保证可复现性
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":  # 主程序入口

    set_seed_torch(2024)  # 设置随机种子

    # --- 数据准备 ---
    train_dir_1 = './data/THaze/train'
    train_set_1 = RESIDE_Dataset(train_dir_1, True, 256, '.jpg')

    test_dir = './data/THaze/test'
    test_set = TestDataset(os.path.join(test_dir, 'hazy'), os.path.join(test_dir, 'clear'))

    loader_train_1 = DataLoader(dataset=train_set_1, batch_size=24, shuffle=True, num_workers=8)
    loader_test = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=1)

    # --- 加载预训练的教师模型 ---
    teacher_net = Teacher()  # 实例化教师模型
    teacher_net = teacher_net.to(opt.device)  # 移动到GPU
    teacher_net.load_state_dict(
        torch.load('./model/Teacher_model/Teacher.pth', map_location=torch.device("cpu")))  # 加载预训练权重
    teacher_net.eval()  # !! 设置教师模型为评估模式

    # --- 初始化学生模型 ---
    student_net = Student()  # 实例化学生模型
    student_net = student_net.to(opt.device)  # 移动到GPU

    epoch_size = len(loader_train_1)
    print("epoch_size: ", epoch_size)

    # --- GPU设置（针对学生模型） ---
    if opt.device == 'cuda':
        student_net = torch.nn.DataParallel(student_net)  # 为学生模型设置多GPU
        cudnn.benchmark = True

    # --- 打印学生模型参数量 ---
    pytorch_total_params = sum(p.numel() for p in student_net.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(pytorch_total_params))
    print("------------------------------------------------------------------")

    # --- 损失函数定义 ---
    criterion = []
    criterion.append(FA().to(opt.device))  # 0: 特征亲和力损失 (蒸馏)
    criterion.append(nn.L1Loss().to(opt.device))  # 1: L1 损失 (监督)
    criterion.append(SSIM().to(opt.device))  # 2: SSIM 损失 (监督)
    criterion.append(ContrastLoss(ablation=False))  # 3: 对比度损失 (监督)

    # --- 优化器定义（只优化学生模型） ---
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, student_net.parameters()), lr=opt.start_lr,
                           betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer.zero_grad()

    # --- 开始训练 ---
    train(teacher_net, student_net, loader_train_1, loader_test, optimizer, criterion)