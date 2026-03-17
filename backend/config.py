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

    # 主干网络
    # efficientnet_b2 在 fine-grained 任务表现明显好于 b0
    # 显存占用仍可被 1050Ti 接受
    # "backbone": "efficientnet_b2",
    "backbone": "None",

    # 特征提取器类型：
    # - "efficientnet": 使用当前项目内置 EfficientNetEmbedder（需要/可选训练权重）
    # - "dinov2": 使用 DINOv2 自监督基础模型提特征（transformers），细粒度识别强，自监督
    # - "clip": 使用 CLIP 视觉编码器提特征（transformers），零样本能力强，通用性好
    # 小样本、角度不固定时，推荐 "dinov2" 或 "clip"

    # "feature_extractor": "dinov2",
    "feature_extractor": "clip  ",

    # 基础模型名称（transformers hub id）
    # "dinov2_model_name": "facebook/dinov2-base",
    "clip_model_name": "openai/clip-vit-base-patch32",

    # 特征向量维度
    # DINOv2 输出特征维度：dinov2-base = 768
    "embedding_dim": 512,

    # 输入图像尺寸
    # efficientnet 推荐 260
    "img_size": 224,

    # ArcFace 参数
    "arcface_margin": 0.35,
    "arcface_scale": 32,

    # 是否使用 Dropout
    "dropout": 0.0,

    # 是否使用特征归一化
    "feature_norm": True,
}

# ===================== 训练配置 =====================
TRAIN_CONFIG = {

    # batch
    "batch_size": 8,

    # 训练轮数
    "epochs": 120,

    # 初始学习率
    "lr": 2e-4,

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
    "use_mixup": True,
    "mixup_alpha": 0.2,

    # label smoothing
    "label_smoothing": 0.1,

    # 验证集比例
    "val_split": 0.2,

    # Triplet Loss
    "triplet_margin": 0.3,

    # Loss权重
    # "arcface_weight": 1.0,
    # "triplet_weight": 0.5,

    "arcface_weight": 0,
    "triplet_weight": 0,

    # temperature
    "temperature": 0.05,

    # early stop
    "early_stop_patience": 20,

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
    # 注意：当 feature_extractor 为 dinov2/clip 时，实际维度将由模型决定，
    # 这里仅作为默认值保留（避免影响旧逻辑）。
    "embedding_dim": MODEL_CONFIG["embedding_dim"],

    # HNSW参数
    "M": 32,
    "efConstruction": 200,
    "efSearch": 64,

}

# ===================== 识别配置 =====================
INFERENCE_CONFIG = {

    # 相似度阈值
    "similarity_threshold": 0.70,

    # TopK
    "topk": 5,

    # TTA
    "use_tta": True,
    "tta_augments": 8,

    # FAISS index
    "faiss_index_path": str(EMBEDDINGS_DIR / "reagent.index"),

    # metadata
    "metadata_path": str(EMBEDDINGS_DIR / "metadata.json"),

    # 模型路径
    "model_path": str(MODELS_DIR / "best_model.pth"),
}

# ===================== 设备 =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(f"[Config] 使用设备: {DEVICE}")

# ===================== 数据库 =====================
DATABASE_URL = f"sqlite+aiosqlite:///{DB_DIR}/reagent.db"