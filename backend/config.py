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
    # "backbone": "swin_tiny_patch4_window7_224",
    # 嵌入向量维度
    "embedding_dim": 512,
    # 输入图像尺寸
    "img_size": 224,
    # ArcFace 参数（优化后）
    "arcface_margin": 0.3,   # 角度边距（降低以提高泛化能力）
    "arcface_scale": 30.0,   # 特征缩放（降低以避免过拟合）
}

# ===================== 训练配置 =====================
TRAIN_CONFIG = {
    "batch_size": 8,          # 1050Ti 4GB显存 安全值
    "epochs": 150,            # 增加训练轮数
    "lr": 3e-4,               # 提高学习率
    "weight_decay": 1e-4,
    "num_workers": 2,
    # 数据增强强度 (fine-grained需要适度增强)
    "aug_prob": 0.7,          # 增强数据增强
    # 验证集比例
    "val_split": 0.15,        # 减少验证集比例，增加训练数据
    # 早停耐心
    "early_stop_patience": 15, # 增加耐心
    # 每N个epoch保存一次
    "save_every": 5,
    # Triplet Loss margin
    "triplet_margin": 0.2,    # 降低margin
    # 温度参数（用于softmax）
    "temperature": 0.07,      # 降低温度以提高区分度
}

# ===================== 识别配置 =====================
INFERENCE_CONFIG = {
    # 相似度阈值(降低阈值以提高召回率)
    "similarity_threshold": 0.65,
    # Top-K候选
    "topk": 5,
    # FAISS索引文件
    "faiss_index_path": str(EMBEDDINGS_DIR / "reagent.index"),
    # 试剂元数据
    "metadata_path": str(EMBEDDINGS_DIR / "metadata.json"),
    # 模型权重
    "model_path": str(MODELS_DIR / "best_model.pth"),
    # 推理时是否使用TTA（测试时增强）
    "use_tta": True,
    # TTA增强次数
    "tta_augments": 5,
}

# ===================== 设备配置 =====================
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Config] 使用设备: {DEVICE}")

# ===================== 数据库 =====================
DATABASE_URL = f"sqlite+aiosqlite:///{DB_DIR}/reagent.db"