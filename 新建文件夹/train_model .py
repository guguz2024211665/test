import os
import warnings
warnings.filterwarnings('ignore')
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset  # 修复基类缺失问题
from PIL import Image  # 后续代码需要 PIL 库
import csv 
import glob  # 用于文件路径匹配
import albumentations as A  # 用于图像增强
from albumentations.pytorch import ToTensorV2  # 用于转换到张量
import torchvision.transforms as T  # 用于图像变换
# ====== 优化后的Patch级别增强类 ======
class AdvancedAugmentation:
    def __init__(self, is_training=True):
        self.is_training = is_training
        
        # 训练集增强 - 专门为patch设计
        self.train_transform = A.Compose([
            # ====== 局部几何变换 ======
            A.OneOf([
                # 轻微弹性变形 - 模拟地形微变形
                A.ElasticTransform(alpha=30, sigma=30 * 0.05, p=0.3),
                
                # 网格变形 - 模拟图像采集时的微小畸变
                A.GridDistortion(num_steps=3, distort_limit=0.1, p=0.3),
                
                # 光学畸变 - 模拟镜头畸变
                A.OpticalDistortion(distort_limit=0.1, p=0.3)
            ], p=0.5),
            
            # 旋转和翻转 - 在patch级别合理
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            
            # 轻微缩放 - 模拟不同距离/分辨率
            A.Affine(scale=(0.9, 1.1), keep_ratio=True, p=0.4),
            
            # ====== 局部颜色和光照变换 ======
            A.OneOf([
                # 对比度和亮度变化 - 模拟不同光照条件
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                
                # Gamma校正 - 模拟曝光变化
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                
                # 自适应直方图均衡化 - 增强局部对比度
                A.CLAHE(p=0.5),
                
                # 轻微颜色抖动
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
            ], p=0.7),
            
            # 饱和度变化 - 模拟不同季节植被
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
            
            # ====== 噪声和模糊 - 模拟传感器噪声和传输失真 ======
            A.OneOf([
                # 运动模糊 - 模拟卫星移动
                A.MotionBlur(blur_limit=(3, 7), p=0.2),
                
                # 高斯模糊 - 模拟大气扰动
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                
                # 中值模糊 - 减少椒盐噪声
                A.MedianBlur(blur_limit=3, p=0.2),
                
                # 玻璃模糊 - 模拟大气折射
                A.GlassBlur(sigma=0.5, max_delta=2, iterations=1, p=0.1),
            ], p=0.6),
            
            A.OneOf([
                # 高斯噪声 - 模拟传感器噪声
                A.GaussNoise(p=0.4),
                
                # ISO噪声 - 模拟高ISO噪声
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=0.3),
                
                # 乘性噪声 - 模拟信号干扰
                A.MultiplicativeNoise(multiplier=(0.95, 1.05), per_channel=True, p=0.3),
            ], p=0.5),
            
            # ====== 局部遮挡 - 模拟云层、阴影等遮挡 ======
            A.OneOf([
                # 随机遮挡 - 模拟云层
                                A.CoarseDropout(p=0.4),
                
                # 网格遮挡 - 模拟规则遮挡
                A.GridDropout(ratio=0.1, p=0.3),
                
                # 随机阴影 - 模拟地形阴影
                                A.RandomShadow(p=0.3),
            ], p=0.5),
            
            # ====== 压缩和传输失真 ======
            A.OneOf([
                # JPEG压缩失真
                A.ImageCompression(p=0.3),
                
                # 色调分离 - 模拟低比特传输
                A.Posterize(num_bits=5, p=0.2),
                
                # 锐化 - 增强边缘
                A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.0), p=0.2),
            ], p=0.4),
            
            # ====== 格式转换 ======
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
        
        # 验证集转换 - 仅基础预处理
        self.val_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

    def __call__(self, image, mask):
        if self.is_training:
            augmented = self.train_transform(image=image, mask=mask)
        else:
            augmented = self.val_transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']

class DamageAwareDataset(Dataset):
    def __init__(self, base_dataset, damage_boost=5):
        self.base_dataset = base_dataset
        self.damage_indices = []
        self.normal_indices = []
        
        # 预扫描数据集统计损坏样本
        for i in tqdm(range(len(base_dataset)), desc="Scanning damage samples"):
            _, mask, *_ = base_dataset[i]
            if mask.sum() > 0:
                self.damage_indices.append(i)
            else:
                self.normal_indices.append(i)
        
        self.damage_boost = damage_boost
        print(f"Found {len(self.damage_indices)} damage samples and {len(self.normal_indices)} normal samples")

    def __len__(self):
        return len(self.normal_indices) + len(self.damage_indices) * self.damage_boost

    def __getitem__(self, idx):
        if idx < len(self.damage_indices) * self.damage_boost:
            damage_idx = idx % len(self.damage_indices)
            return self.base_dataset[self.damage_indices[damage_idx]]
        else:
            normal_idx = (idx - len(self.damage_indices) * self.damage_boost) % len(self.normal_indices)
            return self.base_dataset[self.normal_indices[normal_idx]]

class YOLOLandslideDataset(Dataset):
    """YOLO格式的山体滑坡数据集"""
    def __init__(self, images_dir, labels_dir, transform=None, disaster_class_ids=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        
        # 灾害相关类别ID (COCO 80类中的相关ID)
        self.disaster_class_ids = disaster_class_ids or [
            0, 1, 2, 3, 5, 6, 7, 8, 10, 24, 25, 27, 28, 29, 33, 
            44, 56, 57, 58, 59, 60, 62, 63, 67, 73
        ]
        
        # 获取所有图像文件
        self.image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
        if not self.image_files:
            # 尝试其他图像格式
            self.image_files = glob.glob(os.path.join(images_dir, "*.jpeg")) + \
                             glob.glob(os.path.join(images_dir, "*.png"))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # 加载图像
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            # 如果图像损坏，创建替代图像
            img = Image.new('RGB', (224, 224), 
                           color=(random.randint(0, 255), 
                                  random.randint(0, 255), 
                                  random.randint(0, 255)))
        
        # 获取对应的标注文件路径
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.labels_dir, f"{base_name}.txt")
        
        # 二分类标签: 1=包含灾害相关物体，0=不包含
        label = 0
        
        # 如果标注文件存在，检查是否包含灾害类别
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        try:
                            class_id = int(parts[0])
                            if class_id in self.disaster_class_ids:
                                label = 1
                                break  # 只要有一个灾害物体就标记为1
                        except ValueError:
                            continue
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

class CombinedLandslideDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = glob.glob(os.path.join(images_dir, '*.*'))
        self.image_files = [f for f in self.image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        base_name = os.path.basename(img_path)
        label = 0
        if "xview2" in base_name.lower() and "post" in base_name.lower():
            label = 1
        elif "xview2" in base_name.lower() and "pre" in base_name.lower():
            label = 0
        else:
            label_path = os.path.join(self.labels_dir, os.path.splitext(base_name)[0] + ".txt")
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            label = 1
                            break
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224), color=(random.randint(0,255),random.randint(0,255),random.randint(0,255)))
        if self.transform:
            img = self.transform(img)
        return img, label

def get_segmentation_dataloaders(data_root="data/combined_dataset", batch_size=4, num_workers=2):
    """
    获取分割任务的数据加载器
    返回: (train_loader, val_loader, test_loader)
    """
    # 数据增强和转换
    train_transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = XView2SegmentationDataset(
        os.path.join(data_root, "images", "train2017"),
        os.path.join(data_root, "masks", "train2017"),
        transform=train_transform
    )
    
    val_dataset = XView2SegmentationDataset(
        os.path.join(data_root, "images", "val2017"),
        os.path.join(data_root, "masks", "val2017"), 
        transform=val_transform
    )
    
    test_dataset = XView2SegmentationDataset(
        os.path.join(data_root, "images", "test2017"),
        os.path.join(data_root, "masks", "test2017"),
        transform=val_transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

def get_landslide_dataloaders(data_root="data/combined_dataset", batch_size=4):
    """
    获取山体滑坡分类数据加载器
    适用于二分类任务（滑坡/非滑坡）
    """
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def make_loader(split):
        images_dir = os.path.join(data_root, "images", split)
        labels_dir = os.path.join(data_root, "labels", split)
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"警告: {split} 目录不存在: {images_dir} 或 {labels_dir}")
            return None
        
        dataset = CombinedLandslideDataset(images_dir, labels_dir, transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train2017"), num_workers=2, pin_memory=True)
    
    return make_loader("train2017"), make_loader("val2017"), make_loader("test2017")

def get_calibration_loader(data_root="data/combined_dataset", batch_size=32, num_samples=100):
    train_images_dir = os.path.join(data_root, "images", "train2017")
    train_labels_dir = os.path.join(data_root, "labels", "train2017")
    dataset = CombinedLandslideDataset(train_images_dir, train_labels_dir, transform=T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    if len(dataset) < num_samples:
        num_samples = len(dataset)
    indices = random.sample(range(len(dataset)), num_samples)
    calib_subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(calib_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

def load_sim_features(sim_feature_csv='data/sim_features.csv'):
    sim_dict = {}
    with open(sim_feature_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 数值型特征
            float_feats = [
                float(row['comm_snr']),
                float(row['radar_feat']),
                float(row['radar_max']),
                float(row['radar_std']),
                float(row['radar_peak_idx']),
                float(row['path_loss']),
                float(row['shadow_fading']),
                float(row['rain_attenuation']),
                float(row['target_rcs']),
                float(row['bandwidth']),
                float(row['ber'])
            ]
            # 字符串型特征
            str_feats = [row['channel_type'], row['modulation']]
            sim_dict[row['img_path']] = (float_feats, str_feats)
    return sim_dict

def process_xview2_mask(mask_tensor):
    """
    将mask中2、3、4视为损坏像素，返回二值化mask。
    """
    return (mask_tensor >= 2).float()

class XView2SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, sim_feature_dict=None, transform=None, mask_transform=None, 
                 damage_sample_txts=None, damage_prob=0.7, is_training=True):
        """
        xView2数据集加载器
        数据集特性：
        - 包含灾前(pre-disaster)和灾后(post-disaster)图像
        - 掩码用于定位和损伤评估任务
        - 掩码是单通道PNG图像，值含义：
            0: 背景
            1: 未损坏
            2: 损坏
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.sim_feature_dict = sim_feature_dict
        self.transform = transform
        self.mask_transform = mask_transform
        self.is_training = is_training

        # 默认行为：全部样本
        self.image_files = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            and os.path.exists(os.path.join(masks_dir, os.path.splitext(f)[0] + "_target.png"))
        ]

        self.use_weighted_sampling = False
        if damage_sample_txts is not None:
            # damage_sample_txts: (has_damage_txt, no_damage_txt)
            has_damage_txt, no_damage_txt = damage_sample_txts
            with open(has_damage_txt) as f:
                self.has_damage = [line.strip() for line in f if line.strip()]
            with open(no_damage_txt) as f:
                self.no_damage = [line.strip() for line in f if line.strip()]
            self.use_weighted_sampling = True
            self.damage_prob = damage_prob
            self.length = len(self.has_damage) + len(self.no_damage)
        else:
            self.length = len(self.image_files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        try:
            if self.use_weighted_sampling:
                # 增强采样：保证85%概率采样有损坏区域
                found_damage = False
                for _ in range(10):
                    if random.random() < 0.85 and len(self.has_damage) > 0:
                        mask_name = random.choice(self.has_damage)
                    else:
                        mask_name = random.choice(self.no_damage)
                    img_name = mask_name.replace('_target', '')
                    img_path = os.path.join(self.images_dir, img_name)
                    mask_path = os.path.join(self.masks_dir, mask_name)
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        mask = Image.open(mask_path)
                        if mask.mode != 'L':
                            mask = mask.convert('L')
                        mask = mask.resize((256, 256), resample=Image.Resampling.NEAREST)
                        mask_np = np.array(mask)
                        if (mask_np >= 2).sum() > 0:
                            found_damage = True
                            break
                if not found_damage:
                    # fallback到无损坏样本
                    mask_name = random.choice(self.no_damage)
                    img_name = mask_name.replace('_target', '')
                    img_path = os.path.join(self.images_dir, img_name)
                    mask_path = os.path.join(self.masks_dir, mask_name)
                    mask = Image.open(mask_path)
                    if mask.mode != 'L':
                        mask = mask.convert('L')
                    mask = mask.resize((256, 256), resample=Image.Resampling.NEAREST)
                    mask_np = np.array(mask)
            else:
                img_name = self.image_files[idx]
                mask_name = os.path.splitext(img_name)[0] + "_target.png"
                img_path = os.path.join(self.images_dir, img_name)
                mask_path = os.path.join(self.masks_dir, mask_name)
                mask = Image.open(mask_path)
                if mask.mode != 'L':
                    mask = mask.convert('L')
                mask = mask.resize((256, 256), resample=Image.Resampling.NEAREST)
                mask_np = np.array(mask)  # uint8, 0/1/2/3/4
                mask_bin = (mask_np >= 2).astype(np.float32)  # 2/3/4为损坏
                mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0)  # [1, H, W]
            image = Image.open(img_path).convert('RGB')
            if mask.mode != 'L':
                mask = mask.convert('L')
            mask_tensor = T.ToTensor()(mask)
            mask_tensor = process_xview2_mask(mask_tensor)
            if self.transform:
                if 'albumentations' in str(type(self.transform)):
                    augmented = self.transform(image=image, mask=mask)
                    image = augmented['image']
                    mask = augmented['mask']
                else:
                    image = self.transform(image)
            if self.use_weighted_sampling and idx == 0:
                print(f"采样掩码: {mask_name}, 损坏像素数: {(mask_np == 2).sum()}")
            if idx == 0 and not self.use_weighted_sampling:
                print(f"\n第一个样本调试信息:")
                print(f"图像路径: {img_path}")
                print(f"掩码路径: {mask_path}")
                print(f"原始掩码值范围: min={mask_np.min()}, max={mask_np.max()}")
                print(f"原始掩码唯一值: {np.unique(mask_np)}")
                print(f"处理后掩码形状: {mask_tensor.shape}")
                print(f"处理后掩码唯一值: {torch.unique(mask_tensor)}")
            # 加载sim特征
            sim_feat_tensor = torch.zeros(11)
            str_feats = ["", ""]
            if hasattr(self, 'sim_feature_dict') and self.sim_feature_dict is not None:
                key = os.path.basename(img_path)
                if key in self.sim_feature_dict:
                    sim_feats = self.sim_feature_dict[key]
                    sim_feat_tensor = torch.tensor(sim_feats[:11], dtype=torch.float32)
                    str_feats = sim_feats[11:]
            # 自动跳过全为0的掩码（无损坏像素）
            return image, mask_tensor, sim_feat_tensor, str_feats
        except Exception as e:
            print(f"[警告] 加载样本 {idx} 时出错: {e}, 自动跳过，尝试下一个样本。")
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)

class PatchSegmentationDataset(Dataset):
    """
    用于加载npy格式的patch图像和掩码。
    """
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.npy')])
        self.mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.npy')])
        assert len(self.image_files) == len(self.mask_files), "图像和掩码数量不一致"
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        image = np.load(img_path)  # shape: (C, H, W) or (H, W, C)
        mask = np.load(mask_path)  # shape: (H, W)
        # 保证image为float32, mask为float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        # 如果image是(H, W, C)，转为(C, H, W)
        if image.ndim == 3 and image.shape[0] != 1 and image.shape[0] != 3:
            image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor, mask_tensor


def get_patch_dataloaders(data_root="data/patch_dataset", batch_size=4, num_workers=2):
    """
    获取patch分割任务的数据加载器
    返回: (train_loader, val_loader)
    """
    train_images_dir = os.path.join(data_root, "train/images")
    train_masks_dir = os.path.join(data_root, "train/masks")
    val_images_dir = os.path.join(data_root, "val/images")
    val_masks_dir = os.path.join(data_root, "val/masks")

    train_dataset = PatchSegmentationDataset(train_images_dir, train_masks_dir)
    val_dataset = PatchSegmentationDataset(val_images_dir, val_masks_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader

class MultiModalPatchSegmentationDataset(Dataset):
    """
    支持多模态patch分割：每个patch加载patch、掩码、原图仿真特征。
    """
    def __init__(self, images_dir, masks_dir, patch_index_csv, sim_feature_dict, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.sim_feature_dict = sim_feature_dict
        # 加载patch->原图索引
        self.patch2img = {}
        with open(patch_index_csv, 'r') as f:
            next(f)
            for line in f:
                patch, img = line.strip().split(',')
                self.patch2img[patch] = img
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.npy')])
        self.mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.npy')])
        assert len(self.image_files) == len(self.mask_files), "图像和掩码数量不一致"
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_patch_name = self.image_files[idx]
        mask_patch_name = self.mask_files[idx]
        img_patch_path = os.path.join(self.images_dir, img_patch_name)
        mask_patch_path = os.path.join(self.masks_dir, mask_patch_name)
        image = np.load(img_patch_path)
        mask = np.load(mask_patch_path)
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        if image.ndim == 3 and image.shape[0] != 1 and image.shape[0] != 3:
            image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        # 查找原图名并加载仿真特征
        origin_img = self.patch2img.get(img_patch_name, None)
        if origin_img is not None and self.sim_feature_dict is not None:
            # 尝试多种路径格式来匹配sim_feature_dict中的键
            possible_keys = [
                origin_img,  # 原始格式
                f"combined_dataset/images/train2017/{origin_img}",  # 训练集路径
                f"combined_dataset/images/val2017/{origin_img}",    # 验证集路径
                f"combined_dataset/images/test2017/{origin_img}",   # 测试集路径
                origin_img.replace('/', '\\'),  # Windows路径格式
                f"combined_dataset\\images\\train2017\\{origin_img}",
                f"combined_dataset\\images\\val2017\\{origin_img}",
                f"combined_dataset\\images\\test2017\\{origin_img}"
            ]
            
            sim_feats = None
            for key in possible_keys:
                if key in self.sim_feature_dict:
                    sim_feats_tuple = self.sim_feature_dict[key]
                    # 只取数值型特征，忽略字符串特征
                    sim_feats = sim_feats_tuple[0] if isinstance(sim_feats_tuple, tuple) else sim_feats_tuple
                    break
            
            if sim_feats is None:
                # 如果找不到匹配的键，使用零向量
                sim_feats = np.zeros(11, dtype=np.float32)
        
            # 对sim_feats进行归一化处理，防止数值过大
            sim_feats = np.array(sim_feats, dtype=np.float32)
            if np.std(sim_feats) > 0:
                sim_feats = (sim_feats - np.mean(sim_feats)) / np.std(sim_feats)
        
            sim_feats_tensor = torch.tensor(sim_feats, dtype=torch.float32)
        else:
            sim_feats = np.zeros(11, dtype=np.float32)
            # 对sim_feats进行归一化处理，防止数值过大
            sim_feats = np.array(sim_feats, dtype=np.float32)
            if np.std(sim_feats) > 0:
                sim_feats = (sim_feats - np.mean(sim_feats)) / np.std(sim_feats)
        
            sim_feats_tensor = torch.tensor(sim_feats, dtype=torch.float32)
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor, mask_tensor, sim_feats_tensor


# 数据增强包装类
class AugmentedDataset(Dataset):
    def __init__(self, dataset, augment_fn):
        self.dataset = dataset
        self.augment_fn = augment_fn
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        img, mask, sim_feat = self.dataset[idx]
        
        # 检查图像数据类型和范围
        if img.dtype == torch.float32:
            # 如果已经是float32，需要反归一化到0-255范围
            if img.min() >= -3 and img.max() <= 3:
                # 反归一化：从ImageNet标准化范围转换回0-255
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = img * 255.0
                img = torch.clamp(img, 0, 255)
        
        # 转换为NumPy数组 (H, W, C)
        img = img.permute(1, 2, 0).numpy().astype(np.uint8)
        mask = mask.squeeze().numpy()
        
        # 应用增强
        img, mask = self.augment_fn(img, mask)
        
        return img, mask, sim_feat

def get_multimodal_patch_dataloaders(data_root="data/patch_dataset", 
                                    sim_feature_csv="data/sim_features.csv", 
                                    batch_size=4, 
                                    num_workers=2,
                                    damage_boost=5):
    
    sim_feature_dict = load_sim_features(sim_feature_csv)
    
    # 创建基础数据集
    train_dataset = MultiModalPatchSegmentationDataset(
        os.path.join(data_root, "train/images"),
        os.path.join(data_root, "train/masks"),
        os.path.join(data_root, "patch_index_train.csv"),
        sim_feature_dict,
        transform=None  # 稍后应用增强
    )
    
    val_dataset = MultiModalPatchSegmentationDataset(
        os.path.join(data_root, "val/images"),
        os.path.join(data_root, "val/masks"),
        os.path.join(data_root, "patch_index_val.csv"),
        sim_feature_dict,
        transform=None
    )
    
    # 应用过采样 - 启用数据增强
    aug = AdvancedAugmentation(is_training=True)
    train_dataset = DamageAwareDataset(train_dataset, damage_boost=damage_boost)
    
    # 启用增强
    train_dataset = AugmentedDataset(train_dataset, aug)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 避免最后不完整的batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

# 多模态分割模型
class DeepLabWithSimFeature(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, sim_feat_dim=11):
        super().__init__()
        self.deeplab = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b1",
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
        dummy = torch.zeros(1, in_channels, 64, 64)
        with torch.no_grad():
            features = self.deeplab.encoder(dummy)
            self.encoder_out_dim = features[-1].shape[1]
        self.sim_fc = nn.Sequential(
            nn.Linear(sim_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # 添加dropout防止过拟合
            nn.Linear(64, self.encoder_out_dim)
        )
        self.dropout2d = nn.Dropout2d(0.2)  # 取消注释，启用dropout
        # 权重初始化
        for m in self.sim_fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, img, sim_feat):
        features = self.deeplab.encoder(img)
        x = features[-1]
        B, C, H, W = x.shape
        sim_proj = self.sim_fc(sim_feat)
        sim_proj = sim_proj.view(B, C, 1, 1).expand(-1, C, H, W)
        fused = x + sim_proj
        features = list(features)
        features[-1] = fused
        out = self.deeplab.decoder(features)
        out = self.dropout2d(out)
        out = self.deeplab.segmentation_head(out)
        return out

# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    def forward(self, inputs, targets):
        # 数据合法性检查
        if torch.isnan(inputs).any():
            print("[警告] inputs中有nan值")
            inputs = torch.nan_to_num(inputs, nan=0.0)
        if torch.isnan(targets).any():
            print("[警告] targets中有nan值")
            targets = torch.nan_to_num(targets, nan=0.0)
        
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 确保输入在合理范围内
        inputs = torch.clamp(inputs, 0, 1)
        targets = torch.clamp(targets, 0, 1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        # 检查计算结果
        if torch.isnan(dice) or torch.isinf(dice):
            print(f"[警告] Dice计算异常: dice={dice}, intersection={intersection}, inputs_sum={inputs.sum()}, targets_sum={targets.sum()}")
            return torch.tensor(1.0, device=inputs.device, requires_grad=True)
        
        return 1 - dice

# 评估指标
def iou_score(outputs, masks, smooth=1e-5):
    # 数据合法性检查
    if torch.isnan(outputs).any():
        outputs = torch.nan_to_num(outputs, nan=0.0)
    if torch.isnan(masks).any():
        masks = torch.nan_to_num(masks, nan=0.0)
    
    preds = (torch.sigmoid(outputs) > 0.5).float()
    masks = masks.float()
    intersection = (preds * masks).sum()
    union = (preds + masks).sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    # 检查计算结果
    if torch.isnan(iou) or torch.isinf(iou):
        return torch.tensor(0.0, device=outputs.device)
    
    return iou

def dice_score(outputs, masks, smooth=1e-5):
    # 数据合法性检查
    if torch.isnan(outputs).any():
        outputs = torch.nan_to_num(outputs, nan=0.0)
    if torch.isnan(masks).any():
        masks = torch.nan_to_num(masks, nan=0.0)
    
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > 0.5).float()
    masks = masks.float()
    intersection = (outputs * masks).sum()
    dice = (2. * intersection + smooth) / (outputs.sum() + masks.sum() + smooth)
    
    # 检查计算结果
    if torch.isnan(dice) or torch.isinf(dice):
        return torch.tensor(0.0, device=outputs.device)
    
    return dice

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_multimodal_patch_dataloaders(
        data_root="data/patch_dataset",
        sim_feature_csv="data/sim_features.csv",
        batch_size=64,  # 降低batch size来减少过拟合
        num_workers=8   # 相应调整worker数量
    )
    model = DeepLabWithSimFeature(in_channels=3, num_classes=1, sim_feat_dim=11).to(device)
    criterion = DiceLoss()
    # 调整优化器参数来防止过拟合
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # 替换为余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)
    scaler = GradScaler()

    # === 模型恢复相关 ===
    start_epoch = 1
    best_val_iou = 0.0
    iou_log = []
    checkpoint_path = "models/checkpoint.pth"
    best_model_path = "models/best_multimodal_patch_model.pth"

    if os.path.exists(checkpoint_path):
        print(f"发现检查点文件: {checkpoint_path}, 尝试恢复训练...")
        checkpoint = torch.load(checkpoint_path)
        
        # 处理模型权重加载的兼容性问题
        model_state_dict = checkpoint['model_state_dict']
        current_state_dict = model.state_dict()
        
        # 检查是否需要处理Dropout层导致的键名变化
        if 'sim_fc.3.weight' in current_state_dict and 'sim_fc.3.weight' not in model_state_dict:
            print("检测到模型结构变化（添加了Dropout层），进行权重兼容性处理...")
            
            # 创建新的状态字典
            new_state_dict = {}
            for key, value in model_state_dict.items():
                if key in current_state_dict:
                    new_state_dict[key] = value
                else:
                    # 处理sim_fc层的键名变化
                    if key == 'sim_fc.2.weight' and 'sim_fc.3.weight' in current_state_dict:
                        new_state_dict['sim_fc.3.weight'] = value
                    elif key == 'sim_fc.2.bias' and 'sim_fc.3.bias' in current_state_dict:
                        new_state_dict['sim_fc.3.bias'] = value
                    else:
                        print(f"跳过不兼容的键: {key}")
            
            # 加载兼容的权重
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            if missing_keys:
                print(f"缺失的键: {missing_keys}")
            if unexpected_keys:
                print(f"意外的键: {unexpected_keys}")
        else:
            # 正常加载
            model.load_state_dict(model_state_dict)
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        # 恢复调度器状态（如果存在）
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_iou = checkpoint['best_val_iou']
        iou_log = checkpoint['iou_log']
        print(f"恢复训练成功! 将从 epoch {start_epoch} 开始训练")
        print(f"历史最佳 IoU: {best_val_iou:.4f}")

    # 如果有保存的最佳模型，也加载
            if os.path.exists(best_model_path):
        print(f"发现最佳模型文件: {best_model_path}, 加载模型权重...")
        best_model_state_dict = torch.load(best_model_path)
        # 同样处理最佳模型的兼容性
        current_state_dict = model.state_dict()
        if 'sim_fc.3.weight' in current_state_dict and 'sim_fc.3.weight' not in best_model_state_dict:
            print("处理最佳模型的权重兼容性...")
            new_best_state_dict = {}
            for key, value in best_model_state_dict.items():
                if key in current_state_dict:
                    new_best_state_dict[key] = value
            else:
                    if key == 'sim_fc.2.weight' and 'sim_fc.3.weight' in current_state_dict:
                        new_best_state_dict['sim_fc.3.weight'] = value
                    elif key == 'sim_fc.2.bias' and 'sim_fc.3.bias' in current_state_dict:
                        new_best_state_dict['sim_fc.3.bias'] = value
            missing_keys, unexpected_keys = model.load_state_dict(new_best_state_dict, strict=False)
            if missing_keys:
                print(f"最佳模型缺失的键: {missing_keys}")
            if unexpected_keys:
                print(f"最佳模型意外的键: {unexpected_keys}")
        else:
            model.load_state_dict(best_model_state_dict)
        print("最佳模型权重加载完成!")

    # === 分阶段训练逻辑 ===
    # 计算当前阶段
    current_stage = (start_epoch - 1) // 20 + 1
    stage_start_epoch = (current_stage - 1) * 20 + 1
    stage_end_epoch = current_stage * 20
    print(f"\n当前训练阶段: {current_stage}")
    print(f"阶段范围: epoch {stage_start_epoch} - {stage_end_epoch}")
    # 检查是否需要训练
    if start_epoch > stage_end_epoch:
        print(f"\n当前阶段已完成（从epoch {start_epoch}恢复），直接进行量化...")
    else:
        # 正常训练循环（当前阶段）
        for epoch in range(start_epoch, stage_end_epoch + 1):
        model.train()
        total_loss = 0
        total_iou = 0
        total_dice = 0
            for batch_idx, (images, masks, sim_feats) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{stage_end_epoch} - Training")):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                sim_feats = sim_feats.to(device, non_blocking=True)
                optimizer.zero_grad()
                with autocast('cuda'):
                    outputs = model(images, sim_feats)
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"[调试] 训练 outputs存在NaN/Inf! batch_idx={batch_idx}")
                    print(f"outputs min={outputs.min().item()}, max={outputs.max().item()}, mean={outputs.mean().item()}")
                    print(f"images min={images.min().item()}, max={images.max().item()}, mean={images.mean().item()}")
                    print(f"sim_feats min={sim_feats.min().item()}, max={sim_feats.max().item()}, mean={sim_feats.mean().item()}")
                    if outputs.shape[-2:] != masks.shape[-2:]:
                        outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                    outputs = outputs.squeeze(1) if outputs.shape[1] == 1 else outputs
                    masks = masks.squeeze(1) if masks.shape[1] == 1 else masks
                if torch.isnan(masks).any() or torch.isinf(masks).any():
                    print(f"[调试] 训练 masks存在NaN/Inf! batch_idx={batch_idx}")
                    print(f"masks min={masks.min().item()}, max={masks.max().item()}, mean={masks.mean().item()}")
                    loss = criterion(outputs, masks)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[调试] 训练 loss为NaN/Inf! batch_idx={batch_idx}")
                    continue
                scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                total_loss += loss.item()
                total_iou += iou_score(outputs, masks).item()
                total_dice += dice_score(outputs, masks).item()
            avg_loss = total_loss / len(train_loader)
            avg_iou = total_iou / len(train_loader)
            avg_dice = total_dice / len(train_loader)
            print(f"Epoch {epoch} Train Loss: {avg_loss:.4f} IoU: {avg_iou:.4f} Dice: {avg_dice:.4f}")
            # 验证阶段
            model.eval()
            val_loss = 0
            val_iou = 0
            val_dice = 0
            with torch.no_grad():
                for batch_idx, (images, masks, sim_feats) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch}/{stage_end_epoch} - Validation")):
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    sim_feats = sim_feats.to(device, non_blocking=True)
                    with autocast('cuda'):
                        outputs = model(images, sim_feats)
                    # 调试：每5个epoch打印前2个batch的掩码唯一值和输出范围
                    if batch_idx < 2 and epoch % 5 == 0:
                        print(f"[调试] 验证掩码唯一值: {torch.unique(masks)}")
                        print(f"[调试] 验证输出范围: min={outputs.min().item()}, max={outputs.max().item()}, mean={outputs.mean().item()}")
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        print(f"[调试] 验证 outputs存在NaN/Inf! batch_idx={batch_idx}")
                        print(f"outputs min={outputs.min().item()}, max={outputs.max().item()}, mean={outputs.mean().item()}")
                        print(f"images min={images.min().item()}, max={images.max().item()}, mean={images.mean().item()}")
                        print(f"sim_feats min={sim_feats.min().item()}, max={sim_feats.max().item()}, mean={sim_feats.mean().item()}")
                        if outputs.shape[-2:] != masks.shape[-2:]:
                            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                        outputs = outputs.squeeze(1) if outputs.shape[1] == 1 else outputs
                        masks = masks.squeeze(1) if masks.shape[1] == 1 else masks
                    if torch.isnan(masks).any() or torch.isinf(masks).any():
                        print(f"[调试] 验证 masks存在NaN/Inf! batch_idx={batch_idx}")
                        print(f"masks min={masks.min().item()}, max={masks.max().item()}, mean={masks.mean().item()}")
                        loss = criterion(outputs, masks)
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"[调试] 验证 loss为NaN/Inf! batch_idx={batch_idx}")
                        continue
                    val_loss += loss.item()
                    val_iou += iou_score(outputs, masks).item()
                    val_dice += dice_score(outputs, masks).item()
            avg_val_loss = val_loss / len(val_loader)
            avg_val_iou = val_iou / len(val_loader)
            avg_val_dice = val_dice / len(val_loader)
            print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f} IoU: {avg_val_iou:.4f} Dice: {avg_val_dice:.4f}")
            iou_log.append(avg_val_iou)
            # 保存最佳模型逻辑
            if avg_val_iou > best_val_iou:
                best_val_iou = avg_val_iou
                torch.save(model.state_dict(), best_model_path)
                print(f"[保存] 新最佳模型，Val IoU: {best_val_iou:.4f} (历史最佳)")
            else:
                print(f"[未保存] 当前IoU {avg_val_iou:.4f} < 历史最佳 {best_val_iou:.4f}")
            # 更新学习率调度器
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr:.2e}")
            # 保存检查点（包括模型、优化器、scaler等状态）
            checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_iou': best_val_iou,
                'iou_log': iou_log
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"检查点已保存: {checkpoint_path}")
        # 保存IoU历史
        with open("iou_history.csv", "w") as f:
            f.write("epoch,iou\n")
            for i, iou in enumerate(iou_log):
                f.write(f"{i+1},{iou:.6f}\n")
    # === 分阶段量化逻辑 ===
    # 检查当前阶段是否完成
    if start_epoch > stage_end_epoch or (start_epoch <= stage_end_epoch and epoch == stage_end_epoch):
        print(f"\n" + "="*50)
        print(f"阶段 {current_stage} 训练完成！开始自动量化模型...")
        print("="*50)
        try:
            import subprocess
            import sys
            # 确保最佳模型存在
            if not os.path.exists(best_model_path):
                print(f"警告：最佳模型文件 {best_model_path} 不存在，跳过量化")
                return
            # 调用量化脚本
        quantize_script = os.path.join("inference", "quantize_model.py")
        if os.path.exists(quantize_script):
                print("开始量化模型...")
                cmd = [
                    sys.executable, quantize_script,
                    "--model_path", best_model_path,
                    "--quant_path", f"models/quantized_seg_model_stage{current_stage}.pt",
                    "--data_root", "data/combined_dataset"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ 模型量化成功！")
                    print("量化输出:")
                    print(result.stdout)
                    # 检查量化模型文件
                    quantized_model_path = f"models/quantized_seg_model_stage{current_stage}.pt"
                    if os.path.exists(quantized_model_path):
                        file_size = os.path.getsize(quantized_model_path) / (1024 * 1024)  # MB
                        print(f"量化模型文件大小: {file_size:.2f} MB")
                        # 对比原始模型大小
                        original_size = os.path.getsize(best_model_path) / (1024 * 1024)  # MB
                        compression_ratio = original_size / file_size
                        print(f"原始模型大小: {original_size:.2f} MB")
                        print(f"压缩比: {compression_ratio:.2f}x")
        else:
                        print("❌ 量化模型文件未生成")
    else:
                    print("❌ 模型量化失败！")
                    print("错误输出:")
                    print(result.stderr)
            else:
                print(f"❌ 量化脚本 {quantize_script} 不存在")
        except Exception as e:
            print(f"❌ 量化过程中发生错误: {e}")
        print(f"\n" + "="*50)
        print(f"阶段 {current_stage} 训练和量化流程完成！")
        print("="*50)
        # 提示下一阶段
        next_stage = current_stage + 1
        next_stage_start = next_stage * 20 - 19
        next_stage_end = next_stage * 20
        print(f"\n下一阶段: 阶段 {next_stage} (epoch {next_stage_start}-{next_stage_end})")
        print("重新运行脚本继续下一阶段训练...")
    else:
        print(f"\n当前阶段 {current_stage} 未完成，请继续训练...")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    main()
