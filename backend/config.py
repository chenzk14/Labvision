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

    # 模型类型: 单流EfficientNetV2 + Metric Learning
    "model_type": "single_efficientnetv2",

    # 主干网络
    # efficientnetv2_s 在 fine-grained 任务表现优秀
    # V2版本训练更快、参数效率更高、鲁棒性更强
    # 显存占用仍可被 1050Ti 接受
    "backbone": "efficientnetv2_s",

    # 特征提取器类型
    "feature_extractor": "efficientnetv2",

    # 特征向量维度
    "embedding_dim": 256,

    # 输入图像尺寸
    # efficientnetv2_s 推荐 384（V2版本支持更大的输入尺寸）
    # "img_size": 384,
    "img_size": 288,

    # ArcFace 参数
    "arcface_margin": 0.35,
    "arcface_scale": 60,

    # 是否使用 Dropout
    "dropout": 0.3,

    # 是否使用特征归一化
    "feature_norm": True,

    # Triplet Loss 参数
    "triplet_margin": 0.3,
}

# ===================== 训练配置 =====================
TRAIN_CONFIG = {

    # batch
    # "batch_size": 8,
    "batch_size": 4,

    # 训练轮数
    "epochs": 120,

    # 初始学习率
    # "lr": 5e-5,   #5e-5 在小数据上收敛极慢甚至不收敛
    "lr": 3e-4,

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

    # Loss权重  也是ArcFace  暂时关闭
    #ArcFace 是主loss（分类结构）
    #Triplet 是辅助（需要大量样本才稳定）
    # "arcface_weight": 0.5,
    # "triplet_weight": 1.0,
    "arcface_weight": 1.0,
    "triplet_weight": 0.0,   # 或直接0.0先关闭

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

    "random_rotate": 5,
    "random_crop_scale": (0.85, 1.0),
    "gaussian_blur": 0.0,

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
    "similarity_threshold": 0.7,

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

# ===================== 目标检测配置 (YOLOv11) =====================
DETECTION_CONFIG = {

    # YOLO模型名称 (yolo11n/yolo11s/yolo11m/yolo11l/yolo11x)
    # n=nan(最小), s=small, m=medium, l=large, x=xlarge
    "model_name": "yolo11m.pt",

    # 运行设备 ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
    "device": "auto",

    # 检测置信度阈值
    "confidence_threshold": 0.7,

    # NMS的IOU阈值
    "iou_threshold": 0.45,

    # 最大检测数量
    "max_det": 100,

    # 裁剪时的边界框扩展像素数
    "crop_padding": 10,

    # 是否绘制检测结果
    "draw_detections": True,

    # 绘制边界框颜色 (B, G, R)
    "bbox_color": (0, 255, 0),

    # 边界框线条粗细
    "bbox_thickness": 2,
}

# ===================== 设备 =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 数据库 =====================
DATABASE_URL = f"sqlite+aiosqlite:///{DB_DIR}/reagent.db"