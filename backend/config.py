# backend/config.py
"""
全局配置文件 - 所有超参数集中管理
针对 1050Ti 4GB + Fine-Grained 图像识别优化
"""

import os
from pathlib import Path
import torch

# ===================== 路径配置 =====================
BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
DB_DIR = DATA_DIR / "db"

MODELS_DIR = BASE_DIR / "saved_models"
LOGS_DIR = BASE_DIR / "logs"

# 自动创建目录
for d in [IMAGES_DIR, EMBEDDINGS_DIR, DB_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ===================== 模型配置 =====================
MODEL_CONFIG = {

    # 模型类型: 单流EfficientNet + Metric Learning
    "model_type": "single_efficientnet",

    # 主干网络
    # efficientnet_b2 在 fine-grained 任务表现明显好于 b0
    # 显存占用仍可被 1050Ti 接受
    "backbone": "efficientnet_b2",

    # 特征提取器类型
    "feature_extractor": "efficientnet",

    # 特征向量维度
    "embedding_dim": 256,

    # 输入图像尺寸
    # efficientnet_b2 推荐 260
    "img_size": 260,

    # ArcFace 参数
    "arcface_margin": 0.35,
    "arcface_scale": 30,

    # 是否使用 Dropout
    "dropout": 0.3,

    # 是否使用特征归一化
    "feature_norm": True,

    # Triplet Loss 参数
    "triplet_margin": 0.3,
    "triplet_weight": 0.5,

    # ArcFace Loss 权重
    "arcface_weight": 1.0,
}

# ===================== 训练配置 =====================
TRAIN_CONFIG = {

    # batch
    "batch_size": 8,

    # 训练轮数
    "epochs": 120,

    # 初始学习率
    "lr": 5e-5,

    # 权重衰减
    "weight_decay": 5e-5,

    # dataloader
    "num_workers": 4,

    # 学习率策略
    "scheduler": "cosine",

    # warmup
    "warmup_epochs": 5,

    # 数据增强概率
    "aug_prob": 0.8,

    # mixup
    "use_mixup": False,
    "mixup_alpha": 0.2,

    # label smoothing
    "label_smoothing": 0.0,

    # 验证集比例
    "val_split": 0.3,

    # Triplet Loss
    "triplet_margin": 0.3,

    # Loss权重
    "arcface_weight": 0.5,
    "triplet_weight": 1.0,

    # temperature
    "temperature": 0.05,

    # early stop
    "early_stop_patience": 30,

    # checkpoint
    "save_every": 5,
}

# ===================== 数据增强 =====================
AUG_CONFIG = {

    "horizontal_flip": 0.5,

    "color_jitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.05,
    },

    "random_rotate": 10,

    "random_crop_scale": (0.85, 1.0),

    "gaussian_blur": 0.1,

}

# ===================== FAISS配置 =====================
FAISS_CONFIG = {

    # 索引类型
    "index_type": "HNSW",

    # 向量维度
    "embedding_dim": MODEL_CONFIG["embedding_dim"],

    # HNSW参数
    "M": 32,
    "efConstruction": 200,
    "efSearch": 64,

}

# ===================== 识别配置 =====================
INFERENCE_CONFIG = {

    # 相似度阈值
    "similarity_threshold": 0.68,

    # TopK
    "topk": 5,

    # TTA 关闭 TTA（当前阶段）
    "use_tta": False,
    # "tta_augments": 8,

    # FAISS index
    "faiss_index_path": str(EMBEDDINGS_DIR / "reagent.index"),

    # metadata
    "metadata_path": str(EMBEDDINGS_DIR / "metadata.json"),

    # 模型路径
    "model_path": str(MODELS_DIR / "best_model.pth"),
}

# ===================== 设备 =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 数据库 =====================
DATABASE_URL = f"sqlite+aiosqlite:///{DB_DIR}/reagent.db"