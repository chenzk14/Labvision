# backend/config.py
"""
全局配置文件 - 所有超参数集中管理
"""
import os
from pathlib import Path

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
    # 主干网络: efficientnet_b0 适配1050Ti 4GB显存
    "backbone": "efficientnet_b0",
    # 嵌入向量维度
    "embedding_dim": 512,
    # 输入图像尺寸
    "img_size": 224,
    # ArcFace 参数
    "arcface_margin": 0.5,   # 角度边距
    "arcface_scale": 64.0,   # 特征缩放
}

# ===================== 训练配置 =====================
TRAIN_CONFIG = {
    "batch_size": 16,          # 1050Ti 4GB显存 安全值
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "num_workers": 2,
    # 数据增强强度 (fine-grained需要适度增强)
    "aug_prob": 0.5,
    # 验证集比例
    "val_split": 0.2,
    # 早停耐心
    "early_stop_patience": 10,
    # 每N个epoch保存一次
    "save_every": 5,
    # Triplet Loss margin
    "triplet_margin": 0.3,
}

# ===================== 识别配置 =====================
INFERENCE_CONFIG = {
    # 相似度阈值(低于此值认为未知试剂)
    "similarity_threshold": 0.75,
    # Top-K候选
    "topk": 5,
    # FAISS索引文件
    "faiss_index_path": str(EMBEDDINGS_DIR / "reagent.index"),
    # 试剂元数据
    "metadata_path": str(EMBEDDINGS_DIR / "metadata.json"),
    # 模型权重
    "model_path": str(MODELS_DIR / "best_model.pth"),
}

# ===================== 设备配置 =====================
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Config] 使用设备: {DEVICE}")

# ===================== 数据库 =====================
DATABASE_URL = f"sqlite+aiosqlite:///{DB_DIR}/reagent.db"