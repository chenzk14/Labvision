# backend/core/dataset.py
"""
试剂图像数据集
支持：
1. 常规训练数据集（按文件夹分类）
2. Triplet数据集（用于Triplet Loss）
3. 推理时的单图处理
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2


def get_train_transforms(img_size: int = 224):
    """
    训练时数据增强（优化版）

    细粒度识别的增强策略：
    - 保留细节特征（不要过度裁剪）
    - 模拟实验室光照变化
    - 模拟摄像头角度轻微变化
    - 增加更多样化的增强
    """
    return A.Compose([
        # 先放大再随机裁剪
        A.Resize(img_size + 48, img_size + 48),
        A.RandomCrop(img_size, img_size),
        # 水平翻转（试剂瓶对称性）
        A.HorizontalFlip(p=0.5),
        # 旋转（试剂瓶放置可能有角度）
        A.Rotate(limit=20, p=0.6, border_mode=cv2.BORDER_CONSTANT, value=0),
        # 光照变化（实验室灯光不稳定）
        A.RandomBrightnessContrast(
            brightness_limit=0.4,
            contrast_limit=0.4,
            p=0.7
        ),
        # 模拟摄像头噪声
        A.GaussNoise(var_limit=(5, 30), p=0.4),
        # 轻微模糊（摄像头对焦问题）
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        # 颜色抖动（模拟不同摄像头色彩偏差）
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=30,
            val_shift_limit=30,
            p=0.6
        ),
        # 随机缩放（模拟不同距离）
        A.RandomScale(scale_limit=0.1, p=0.4),
        # 确保最终尺寸一致（重要！）
        A.Resize(img_size, img_size),
        # 归一化（ImageNet均值方差）
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 224):
    """验证/推理时的变换（不增强）"""
    return A.Compose([
        A.Resize(img_size, img_size),
        # 中心裁剪（保留主体）
        A.CenterCrop(img_size, img_size, p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size: int = 224):
    """
    测试时增强（TTA）
    通过多次增强推理取平均提高准确率
    """
    transforms = [
        # 原图
        A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ]),
        # 水平翻转
        A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ]),
        # 轻微旋转
        A.Compose([
            A.Resize(img_size, img_size),
            A.Rotate(limit=10, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ]),
        # 亮度调整
        A.Compose([
            A.Resize(img_size, img_size),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ]),
        # 裁剪
        A.Compose([
            A.Resize(img_size + 32, img_size + 32),
            A.RandomCrop(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ]),
    ]
    return transforms


class ReagentDataset(Dataset):
    """
    标准试剂数据集

    目录结构:
    data/images/
    ├── 乙醇001/
    │   ├── img_001.jpg
    │   ├── img_002.jpg
    │   └── ...
    ├── 乙醇002/
    │   ├── img_001.jpg
    │   └── ...
    └── 盐酸001/
        └── ...
    """

    def __init__(
            self,
            root_dir: str,
            transform=None,
            min_samples: int = 1,
            verbose: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.verbose = verbose

        # 扫描数据集
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.samples: List[Tuple[str, int]] = []

        self._scan_dataset(min_samples)

    def _scan_dataset(self, min_samples: int):
        """扫描目录，构建类别→索引映射"""
        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        idx = 0

        for class_dir in class_dirs:
            images = []
            for f in class_dir.iterdir():
                if f.is_file() and f.suffix.lower() in valid_exts:
                    try:
                        with open(f, 'rb') as test_file:
                            test_file.read(1)
                        images.append(f)
                    except (IOError, OSError, UnicodeDecodeError):
                        continue

            if len(images) < min_samples:
                if self.verbose:
                    print(f"[Dataset] 跳过 {class_dir.name}：样本不足({len(images)})")
                continue

            self.class_to_idx[class_dir.name] = idx
            self.idx_to_class[idx] = class_dir.name
            for img_path in images:
                self.samples.append((str(img_path.resolve()), idx))
            idx += 1

            if self.verbose:
                print(f"[Dataset] 加载类别 {class_dir.name}: {len(images)} 张图片")

        if self.verbose:
            print(f"[Dataset] 共 {len(self.class_to_idx)} 类试剂，{len(self.samples)} 张图片")

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        try:
            # 读取图像，处理中文路径
            image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法解码图像: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[Dataset] 加载图像失败 {img_path}: {str(e)}")
            # 返回一个黑色图像作为占位符
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

    def get_class_samples(self, class_name: str) -> List[str]:
        """获取某类所有图片路径"""
        if class_name not in self.class_to_idx:
            return []
        target_idx = self.class_to_idx[class_name]
        return [path for path, label in self.samples if label == target_idx]


class TripletDataset(Dataset):
    """
    Triplet数据集
    每次返回 (anchor, positive, negative) 三元组

    用于辅助训练，使同类试剂特征更近，不同试剂特征更远
    """

    def __init__(self, base_dataset: ReagentDataset, transform=None):
        self.base = base_dataset
        self.transform = transform

        # 按类别索引样本
        self.class_to_samples: Dict[int, List[str]] = {}
        for img_path, label in base_dataset.samples:
            if label not in self.class_to_samples:
                self.class_to_samples[label] = []
            self.class_to_samples[label].append(img_path)

        # 过滤只有1个样本的类（无法构成正对）
        self.valid_classes = [
            cls for cls, samples in self.class_to_samples.items()
            if len(samples) >= 2
        ]

        if len(self.valid_classes) == 0:
            print("[TripletDataset] 警告: 没有类别有>=2个样本，Triplet Loss将被跳过")

    def _load_image(self, path: str) -> np.ndarray:
        try:
            # 使用imread读取图像，处理中文路径
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法解码图像: {path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[TripletDataset] 加载图像失败 {path}: {str(e)}")
            # 返回一个黑色图像作为占位符
            return np.zeros((224, 224, 3), dtype=np.uint8)

    def _apply_transform(self, image: np.ndarray) -> torch.Tensor:
        if self.transform:
            augmented = self.transform(image=image)
            return augmented['image']
        return torch.from_numpy(image.transpose(2, 0, 1)).float()

    def __len__(self) -> int:
        return len(self.base.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Anchor类别
        _, anchor_class = self.base.samples[idx]

        if anchor_class not in self.valid_classes:
            # 降级：返回重复anchor
            anchor_path = self.base.samples[idx][0]
            img = self._load_image(anchor_path)
            t = self._apply_transform(img)
            return t, t, t

        # Anchor
        anchor_path = random.choice(self.class_to_samples[anchor_class])
        anchor_img = self._load_image(anchor_path)

        # Positive（同类不同图）
        pos_candidates = [
            p for p in self.class_to_samples[anchor_class]
            if p != anchor_path
        ]
        positive_path = random.choice(pos_candidates)
        positive_img = self._load_image(positive_path)

        # Negative（不同类）
        neg_class = random.choice([
            c for c in self.valid_classes if c != anchor_class
        ])
        negative_path = random.choice(self.class_to_samples[neg_class])
        negative_img = self._load_image(negative_path)

        return (
            self._apply_transform(anchor_img),
            self._apply_transform(positive_img),
            self._apply_transform(negative_img),
        )


def create_dataloaders(
        data_dir: str,
        img_size: int = 224,
        batch_size: int = 16,
        val_split: float = 0.2,
        num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, ReagentDataset]:
    """
    创建训练/验证数据加载器

    Returns:
        train_loader, val_loader, full_dataset
    """
    from sklearn.model_selection import train_test_split

    full_dataset = ReagentDataset(
        root_dir=data_dir,
        transform=get_train_transforms(img_size),
        min_samples=1,
    )

    # 分割训练/验证集
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.samples[i][1] for i in indices]

    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_split,
            stratify=labels,
            random_state=42,
        )
    except ValueError:
        # 样本太少时不分层
        train_idx, val_idx = train_test_split(
            indices, test_size=val_split, random_state=42
        )

    from torch.utils.data import Subset

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    # 验证集用不同transform（不打印信息）
    val_dataset = ReagentDataset(
        root_dir=data_dir,
        transform=get_val_transforms(img_size),
        min_samples=1,
        verbose=False,
    )
    val_subset = Subset(val_dataset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[DataLoader] 训练集: {len(train_idx)} | 验证集: {len(val_idx)}")
    return train_loader, val_loader, full_dataset