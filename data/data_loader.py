"""
这段Python代码定义了一套用于加载图像去雾（Image Dehazing）数据的PyTorch Dataset 类，它们是数据加载器（DataLoader）的蓝图。
这段代码的核心是一个名为 preprocess_feature 的辅助函数，它负责将输入的有雾图像转换为Tensor，并使用CLIP模型特定的均值和标准差进行归一化，这表明有雾图像将被用于与CLIP相关的计算。
代码文件主要包含两类数据集：第一类是用于有监督训练和评估的。
RESIDE_Dataset 和 RESIDE_Dataset_2 都是为加载成对的（有雾图像, 清晰图像）数据而设计的，它们在训练时会进行随机裁剪、翻转和旋转等数据增强。
它们之间的唯一区别在于匹配有雾图像和清晰图像的文件名逻辑。TestDataset 也加载成对数据，但主要用于评估，因此只进行裁剪而不进行随机增强，并会额外返回图像文件名。
第二类是 CLIP_loader，这是一个无监督的数据集。它只加载有雾图像（例如来自 real_foggy 目录），而不加载任何对应的清晰图像。
这个类对有雾图像进行数据增强和CLIP预处理后，只返回处理后的有雾图像。
这种数据加载器是为那些不需要成对清晰图像的训练方法而设计的，例如使用CLIP损失进行语义引导或使用EMA（指数移动平均）教师模型进行自我训练的场景。
"""

import os
import random
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, Resize
from torchvision.transforms import functional as FF  # 导入 torchvision 的 functional 接口，用于更灵活的变换


def preprocess_feature(img):
    """
    对图像进行预处理，使其适用于CLIP模型的输入。
    1. 将PIL图像转换为Tensor (范围 [0, 1])。
    2. 使用CLIP特定的均值和标准差进行归一化。
    """
    img = ToTensor()(img)
    # CLIP模型专用的归一化参数
    clip_normalizer = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    img = clip_normalizer(img)
    return img


class RESIDE_Dataset(data.Dataset):
    """
    用于加载 RESIDE (或类似格式) 数据集的数据加载器。
    它假定有雾图像和清晰图像是通过文件名中的ID（例如 '0001_1.png' 和 '0001.png'）来配对的。
    """

    def __init__(self, path, train, size=256, format='.png'):
        super(RESIDE_Dataset, self).__init__()
        self.size = size  # 裁剪的目标尺寸
        self.train = train  # 是否为训练模式 (决定是否进行数据增强)
        self.format = format  # 清晰图像的文件格式
        # 获取所有有雾图像的文件名列表
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))
        # 获取所有有雾图像的完整路径列表
        self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]
        # 清晰图像的目录路径
        self.clear_dir = os.path.join(path, 'clear')

    def __getitem__(self, index):
        # 1. 加载有雾图像
        haze = Image.open(self.haze_imgs[index])

        # 如果图像尺寸小于目标尺寸，则随机换一张图像 (防止RandomCrop失败)
        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, 100)
                haze = Image.open(self.haze_imgs[index])

        # 2. 根据有雾图像的文件名找到对应的清晰图像
        img = self.haze_imgs[index]
        split_name = os.path.split(img)[-1].split('_')  # 例如 '0001_1.png' -> ['0001', '1.png']
        id = split_name[0]  # 获取ID '0001'
        clear_name = id + self.format  # 构造清晰图像名 '0001.png'
        clear = Image.open(os.path.join(self.clear_dir, clear_name))  # 加载清晰图像

        # 3. 随机裁剪
        if not isinstance(self.size, str):
            # 获取随机裁剪参数 (保证haze和clear使用相同的裁剪位置)
            i, j, h, w = RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)

        # 4. 数据增强和预处理
        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
        return haze, clear

    def augData(self, data, target):
        """ 对 (data, target) 图像对进行数据增强和预处理 """
        if self.train:
            # 随机水平翻转
            rand_hor = random.randint(0, 1)
            # 随机旋转 (0, 90, 180, 270 度)
            rand_rot = random.randint(0, 3)
            data = RandomHorizontalFlip(rand_hor)(data)
            target = RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)

        # 将目标图像(clear)转换为Tensor (范围 [0, 1])
        target = ToTensor()(target)
        # 将输入图像(haze)进行CLIP预处理 (ToTensor + Normalize)
        return preprocess_feature(data), target

    def __len__(self):
        # 返回数据集中有雾图像的总数
        return len(self.haze_imgs)


class RESIDE_Dataset_2(data.Dataset):
    """
    与 RESIDE_Dataset 类似的数据加载器，但文件名配对逻辑不同。
    它假设有雾图像和清晰图像的文件名主体完全相同，只是扩展名可能不同。
    例如 'image1.jpg' (hazy) 配对 'image1.jpg' (clear) 或 'image1.png' (clear)。
    """

    def __init__(self, path, train, size=256, format='.jpg'):
        super(RESIDE_Dataset_2, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))
        self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'clear')

    def __getitem__(self, index):
        haze = Image.open(self.haze_imgs[index])
        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, 100)
                haze = Image.open(self.haze_imgs[index])

        img = self.haze_imgs[index]
        split_name = os.path.split(img)[-1]  # 'image1.jpg'

        # --- 关键区别 ---
        id = os.path.splitext(split_name)[0]  # 'image1'
        # ----------------

        clear_name = f"{id}{self.format}"  # 'image1.jpg' (或 .png, 取决于 self.format)
        clear = Image.open(os.path.join(self.clear_dir, clear_name))

        if not isinstance(self.size, str):
            i, j, h, w = RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)

        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
        return haze, clear

    def augData(self, data, target):
        """ 数据增强和预处理 (与 RESIDE_Dataset 相同) """
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = RandomHorizontalFlip(rand_hor)(data)
            target = RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        target = ToTensor()(target)
        return preprocess_feature(data), target

    def __len__(self):
        return len(self.haze_imgs)


# --- 修改后的 TestDataset 类 ---
class TestDataset(data.Dataset):
    """
    用于测试阶段的数据加载器，加载含雾可见光、红外和清晰可见光图像。
    假设三个文件夹下的文件名一一对应。 (基于用户提供的版本修改)
    """
    # 定义方法 __init__：初始化 TestDataset 实例
    def __init__(self, hazy_visible_path, infrared_path, clear_visible_path, size=256, format='.jpg'):
        """
        初始化测试数据集实例。

        参数:
            hazy_visible_path (str): 测试集含雾可见光图像文件夹路径。
            infrared_path (str): 测试集红外图像文件夹路径。
            clear_visible_path (str): 测试集清晰可见光图像文件夹路径 (Ground Truth)。
            size (int or str): 图像裁剪或缩放尺寸。测试时通常不随机裁剪，
                               但可能需要缩放或固定裁剪以适应模型输入。
                               设为 'full' 表示不改变尺寸 (需要模型支持任意尺寸)。
                               设为整数则进行中心裁剪或缩放（这里实现为中心裁剪）。
            format (str): 图像文件扩展名。
        """
        super(TestDataset, self).__init__()
        self.hazy_visible_path = hazy_visible_path
        self.infrared_path = infrared_path # <--- 新增红外路径属性
        self.clear_visible_path = clear_visible_path # <--- 使用清晰可见光路径属性
        self.size = size
        self.format = format # <--- 新增格式属性

        # 以含雾可见光文件夹为基准列出文件
        try:
            # 只列出指定格式的文件
            self.image_list = [f for f in os.listdir(hazy_visible_path) if f.endswith(format)]
            self.image_list.sort() # 确保顺序一致
            if not self.image_list:
                raise FileNotFoundError(f"在 {hazy_visible_path} 中未找到格式为 {format} 的图像")
            # 简单检查其他路径是否存在
            if not os.path.isdir(infrared_path):
                 raise FileNotFoundError(f"红外图像目录未找到: {infrared_path}")
            if not os.path.isdir(clear_visible_path):
                 raise FileNotFoundError(f"清晰图像目录未找到: {clear_visible_path}")
        except FileNotFoundError as e:
            print(f"错误: 初始化 TestDataset 失败 - {e}")
            self.image_list = []
        except Exception as e:
            print(f"列出文件时发生未知错误: {e}")
            self.image_list = []

    # 定义方法 __getitem__：根据索引获取一个测试数据样本
    def __getitem__(self, index):
        """
        根据索引加载含雾可见光、红外和清晰可见光图像用于测试。

        参数:
            index (int): 数据样本的索引。

        返回:
            tuple: (处理后的含雾可见光 Tensor, 处理后的红外 Tensor,
                    处理后的清晰可见光 Tensor, 图像文件名 str)。
                   如果加载失败则返回 (None, None, None, None)。
        """
        image_name = self.image_list[index] # 使用 image_list

        # 构建三个图像的完整路径
        hazy_vis_img_path = os.path.join(self.hazy_visible_path, image_name)
        infrared_img_path = os.path.join(self.infrared_path, image_name) # <--- 新增红外路径
        clear_vis_img_path = os.path.join(self.clear_visible_path, image_name) # <--- 使用清晰可见光路径

        try:
            # 加载三个图像
            hazy_vis = Image.open(hazy_vis_img_path).convert('RGB')
            infrared = Image.open(infrared_img_path).convert('RGB') # <--- 加载红外图像
            clear_vis = Image.open(clear_vis_img_path).convert('RGB')

            # --- 测试时的尺寸处理 ---
            if isinstance(self.size, int):
                # 选择: 中心裁剪 或 缩放
                # 方案 A: 中心裁剪 (如果图像尺寸大于等于 self.size)
                w, h = hazy_vis.size # 假设所有图像尺寸相同
                if w >= self.size and h >= self.size:
                    hazy_vis = FF.center_crop(hazy_vis, (self.size, self.size))
                    infrared = FF.center_crop(infrared, (self.size, self.size)) # <--- 对红外应用
                    clear_vis = FF.center_crop(clear_vis, (self.size, self.size))
                else:
                    # 方案 B: 如果图像尺寸小于要求，强制缩放到 self.size x self.size
                    hazy_vis = FF.resize(hazy_vis, [self.size, self.size], interpolation=FF.InterpolationMode.BILINEAR)
                    infrared = FF.resize(infrared, [self.size, self.size], interpolation=FF.InterpolationMode.BILINEAR) # <--- 对红外应用
                    clear_vis = FF.resize(clear_vis, [self.size, self.size], interpolation=FF.InterpolationMode.BILINEAR)
                    print(f"警告: 图像 {image_name} 尺寸小于 {self.size}x{self.size}，已强制缩放。")

            # --- 调用新的预处理方法 ---
            hazy_vis_tensor, infrared_tensor, clear_vis_tensor = self.preprocess_test(hazy_vis, infrared, clear_vis)

            # --- 返回四个值 ---
            return hazy_vis_tensor, infrared_tensor, clear_vis_tensor, image_name

        except FileNotFoundError as e:
            print(f"错误: 加载图像失败: {e}. 跳过索引 {index}.")
            return None, None, None, None # 返回 None 让 collate_fn 处理
        except Exception as e:
            print(f"处理索引 {index} ({image_name}) 时发生未知错误: {e}")
            return None, None, None, None

    # 定义方法 preprocess_test：对测试数据进行预处理
    def preprocess_test(self, hazy_vis, infrared, clear_vis):
        """
        对测试图像进行预处理 (ToTensor 和 Normalize)。
        测试时不进行随机数据增强。

        参数:
            hazy_vis (PIL.Image): 含雾可见光图像。
            infrared (PIL.Image): 红外图像。
            clear_vis (PIL.Image): 清晰可见光图像。

        返回:
            tuple: (处理后的含雾可见光 Tensor, 处理后的红外 Tensor,
                    处理后的清晰可见光 Tensor)。
        """
        # 对两个输入应用 preprocess_feature (包含 ToTensor 和 Normalize)
        hazy_vis_processed = preprocess_feature(hazy_vis)
        infrared_processed = preprocess_feature(infrared) # 暂时使用相同预处理
        # 对清晰图像(GT)只应用 ToTensor
        clear_vis_processed = ToTensor()(clear_vis)
        return hazy_vis_processed, infrared_processed, clear_vis_processed

    # 定义方法 __len__：返回数据集的大小
    def __len__(self):
        """ 返回数据集中样本的总数。 """
        return len(self.image_list) # 使用 image_list

class CLIP_loader(data.Dataset):
    """
    用于无监督训练的数据加载器 (例如用于EMA和CLIP损失的训练)。
    它 **只** 加载有雾图像，不加载清晰图像。
    """

    def __init__(self, hazy_path, train, size=256):
        self.hazy_path = hazy_path
        self.train = train
        self.hazy_image_list = os.listdir(hazy_path)
        self.hazy_image_list.sort()
        self.size = size

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        hazy = Image.open(hazy_image_path).convert('RGB')
        width, height = hazy.size
        # 裁剪尺寸为 (size, height, width) 中的最小值
        crop_size = min(self.size, height, width)

        # 随机裁剪
        if not isinstance(self.size, str):
            i, j, h, w = RandomCrop.get_params(hazy, output_size=(crop_size, crop_size))
            hazy = FF.crop(hazy, i, j, h, w)

        # 缩放到统一尺寸
        hazy = Resize((self.size, self.size))(hazy)

        # 数据增强和预处理
        hazy = self.augData(hazy.convert("RGB"))

        # 只返回有雾图像
        return hazy

    def augData(self, data):
        """ 只对单张图像进行数据增强和预处理 """
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = RandomHorizontalFlip(rand_hor)(data)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
        # 进行CLIP预处理
        return preprocess_feature(data)

    def __len__(self):
        return len(self.hazy_image_list)

# --- 添加新的 MultiModalHazeDataset 类 ---
class MultiModalHazeDataset(data.Dataset):
    def __init__(self, hazy_visible_path, infrared_path, clear_visible_path, train, size=256, format='.jpg'):
        """
        初始化多模态雾霾数据集。

        Args:
            hazy_visible_path (str): 含雾可见光图像文件夹路径。
            infrared_path (str): 红外图像文件夹路径。
            clear_visible_path (str): 清晰可见光图像文件夹路径 (Ground Truth)。
            train (bool): 是否为训练模式 (决定是否进行数据增强)。
            size (int or str): 图像裁剪尺寸，如果是 'full' 则不裁剪。
            format (str): 图像文件扩展名。
        """
        super(MultiModalHazeDataset, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.hazy_visible_path = hazy_visible_path
        self.infrared_path = infrared_path
        self.clear_visible_path = clear_visible_path

        # 假设三个文件夹下的文件名一一对应，以含雾可见光文件夹为基准列出文件
        try:
            self.image_list = [f for f in os.listdir(hazy_visible_path) if f.endswith(format)]
            self.image_list.sort() # 确保顺序一致
            if not self.image_list:
                 raise FileNotFoundError(f"No images found in {hazy_visible_path} with format {format}")
        except FileNotFoundError:
            print(f"Error: Directory not found - {hazy_visible_path}")
            self.image_list = []
        except Exception as e:
            print(f"An error occurred while listing files: {e}")
            self.image_list = []

    def __getitem__(self, index):
        """
        获取数据集中的一个样本。
        """
        image_name = self.image_list[index]

        hazy_vis_img_path = os.path.join(self.hazy_visible_path, image_name)
        infrared_img_path = os.path.join(self.infrared_path, image_name)
        clear_vis_img_path = os.path.join(self.clear_visible_path, image_name)

        try:
            hazy_vis = Image.open(hazy_vis_img_path).convert('RGB')
            # 假设红外图像也是三通道或可以转为三通道，如果不是需要调整 .convert()
            infrared = Image.open(infrared_img_path).convert('RGB')
            clear_vis = Image.open(clear_vis_img_path).convert('RGB')

            # --- 图像尺寸检查与随机裁剪 (如果需要) ---
            if isinstance(self.size, int):
                # 确保所有图像尺寸都足够大
                min_h = min(hazy_vis.size[1], infrared.size[1], clear_vis.size[1])
                min_w = min(hazy_vis.size[0], infrared.size[0], clear_vis.size[0])
                if min_w < self.size or min_h < self.size:
                    # 如果尺寸不足，可以选择resize或者跳过，这里选择resize到最小满足要求
                    target_h = max(self.size, min_h)
                    target_w = max(self.size, min_w)
                    hazy_vis = hazy_vis.resize((target_w, target_h), Image.BILINEAR)
                    infrared = infrared.resize((target_w, target_h), Image.BILINEAR)
                    clear_vis = clear_vis.resize((target_w, target_h), Image.BILINEAR)

                # 应用相同的随机裁剪参数
                i, j, h, w = RandomCrop.get_params(hazy_vis, output_size=(self.size, self.size))
                hazy_vis = FF.crop(hazy_vis, i, j, h, w)
                infrared = FF.crop(infrared, i, j, h, w)
                clear_vis = FF.crop(clear_vis, i, j, h, w)

            # --- 数据增强与预处理 ---
            hazy_vis_tensor, infrared_tensor, clear_vis_tensor = self.aug_and_preprocess(hazy_vis, infrared, clear_vis)

            # 返回两个输入模态和一个GT
            # 注意：返回格式可能需要根据你的模型输入调整，例如是否需要合并通道
            return hazy_vis_tensor, infrared_tensor, clear_vis_tensor

        except FileNotFoundError as e:
            print(f"Error loading image: {e}. Skipping index {index}.")
            # 返回 None 或者上一个有效样本，或者抛出异常，取决于你的策略
            # 这里简单返回 None，DataLoader 需要配置忽略 None 的 batch
            return None, None, None
        except Exception as e:
            print(f"An unexpected error occurred at index {index} ({image_name}): {e}")
            return None, None, None


    def aug_and_preprocess(self, hazy_vis, infrared, clear_vis):
        """
        对输入的三个图像应用数据增强和预处理。
        """
        if self.train:
            # 应用相同的随机水平翻转
            rand_hor = random.randint(0, 1)
            if rand_hor == 1:
                hazy_vis = FF.hflip(hazy_vis)
                infrared = FF.hflip(infrared)
                clear_vis = FF.hflip(clear_vis)

            # 应用相同的随机旋转 (0, 90, 180, 270 度)
            rand_rot = random.randint(0, 3)
            if rand_rot > 0:
                hazy_vis = FF.rotate(hazy_vis, 90 * rand_rot)
                infrared = FF.rotate(infrared, 90 * rand_rot)
                clear_vis = FF.rotate(clear_vis, 90 * rand_rot)

        # 预处理：ToTensor 和 Normalize
        # 注意：preprocess_feature 包含 Normalize，可能需要为红外图像调整
        hazy_vis_processed = preprocess_feature(hazy_vis)
        infrared_processed = preprocess_feature(infrared) # 暂时使用相同预处理
        clear_vis_processed = ToTensor()(clear_vis) # GT 通常只需要 ToTensor

        return hazy_vis_processed, infrared_processed, clear_vis_processed

    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        return len(self.image_list)

# 修改
# --- 添加新的 MultiModalCLIPLoader 类 ---
class MultiModalCLIPLoader(data.Dataset):
    def __init__(self, hazy_visible_path, infrared_path, train, size=256, format='.jpg'):
        """
        初始化用于真实场景适配的多模态雾霾数据集 (无 GT)。

        Args:
            hazy_visible_path (str): 真实含雾可见光图像文件夹路径。
            infrared_path (str): 真实红外图像文件夹路径。
            train (bool): 是否为训练模式 (决定是否进行数据增强)。
            size (int or str): 图像裁剪尺寸，如果是 'full' 则不裁剪。
            format (str): 图像文件扩展名。
        """
        super(MultiModalCLIPLoader, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.hazy_visible_path = hazy_visible_path
        self.infrared_path = infrared_path

        # 假设两个文件夹下的文件名一一对应，以含雾可见光文件夹为基准
        try:
            self.image_list = [f for f in os.listdir(hazy_visible_path) if f.endswith(format)]
            self.image_list.sort()
            if not self.image_list:
                raise FileNotFoundError(f"No images found in {hazy_visible_path} with format {format}")
            # 简单检查红外路径是否存在
            if not os.path.isdir(infrared_path):
                 raise FileNotFoundError(f"Infrared image directory not found: {infrared_path}")
        except FileNotFoundError as e:
            print(f"Error initializing MultiModalCLIPLoader: {e}")
            self.image_list = []
        except Exception as e:
            print(f"An unexpected error occurred while listing files: {e}")
            self.image_list = []


    def __getitem__(self, index):
        image_name = self.image_list[index]
        hazy_vis_img_path = os.path.join(self.hazy_visible_path, image_name)
        infrared_img_path = os.path.join(self.infrared_path, image_name)

        try:
            hazy_vis = Image.open(hazy_vis_img_path).convert('RGB')
            # 同样假设红外可以转为 RGB
            infrared = Image.open(infrared_img_path).convert('RGB')

            # --- 尺寸检查与随机裁剪 (同 MultiModalHazeDataset) ---
            if isinstance(self.size, int):
                min_h = min(hazy_vis.size[1], infrared.size[1])
                min_w = min(hazy_vis.size[0], infrared.size[0])
                if min_w < self.size or min_h < self.size:
                    target_h = max(self.size, min_h)
                    target_w = max(self.size, min_w)
                    hazy_vis = hazy_vis.resize((target_w, target_h), Image.BILINEAR)
                    infrared = infrared.resize((target_w, target_h), Image.BILINEAR)

                i, j, h, w = RandomCrop.get_params(hazy_vis, output_size=(self.size, self.size))
                hazy_vis = FF.crop(hazy_vis, i, j, h, w)
                infrared = FF.crop(infrared, i, j, h, w)

            # --- 数据增强与预处理 ---
            hazy_vis_tensor, infrared_tensor = self.aug_and_preprocess(hazy_vis, infrared)

            return hazy_vis_tensor, infrared_tensor # 只返回两个输入

        except FileNotFoundError as e:
            print(f"Error loading image: {e}. Skipping index {index}.")
            return None, None
        except Exception as e:
            print(f"An unexpected error occurred at index {index} ({image_name}): {e}")
            return None, None

    def aug_and_preprocess(self, hazy_vis, infrared):
        """对输入的两个图像应用数据增强和预处理。"""
        if self.train:
            rand_hor = random.randint(0, 1)
            if rand_hor == 1:
                hazy_vis = FF.hflip(hazy_vis)
                infrared = FF.hflip(infrared)

            rand_rot = random.randint(0, 3)
            if rand_rot > 0:
                hazy_vis = FF.rotate(hazy_vis, 90 * rand_rot)
                infrared = FF.rotate(infrared, 90 * rand_rot)

        # 应用相同的预处理 (包括 Normalize)
        hazy_vis_processed = preprocess_feature(hazy_vis)
        infrared_processed = preprocess_feature(infrared) # 暂时使用相同预处理

        return hazy_vis_processed, infrared_processed

    def __len__(self):
        return len(self.image_list)

# --- 确保文件末尾有换行 ---