# backend/core/recognition_engine.py
"""
识别引擎
核心功能：
1. 注册新试剂（提取嵌入 → 存入FAISS）
2. 识别未知试剂（提取嵌入 → FAISS检索 → 返回最近邻）
3. 增量学习（新试剂不需要重新训练，直接注册）

架构：
摄像头图像 → EfficientNet → 512维向量 → FAISS检索 → 试剂ID
"""

import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import faiss
import cv2
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A

from backend.config import INFERENCE_CONFIG, MODEL_CONFIG, DEVICE
from backend.models.metric_model import EfficientNetEmbedder
from backend.core.dataset import get_val_transforms


class FAISSIndex:
    """
    FAISS向量索引管理

    使用 IndexFlatIP（内积=余弦相似度，因为向量已L2归一化）
    支持增量添加向量
    """

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        # 内积索引（对于L2归一化向量等价于余弦相似度）
        self.index = faiss.IndexFlatIP(embedding_dim)
        # 向量ID → 元数据映射
        self.id_map: List[Dict[str, Any]] = []

    def add(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> int:
        """
        添加一个嵌入向量

        Args:
            embedding: 形状 [embedding_dim]，需已L2归一化
            metadata: {'reagent_id': '乙醇001', 'image_path': '...', ...}

        Returns:
            vector_id: 在索引中的位置
        """
        # 确保正确形状 [1, embedding_dim]
        vec = embedding.reshape(1, -1).astype(np.float32)
        # L2归一化（保险起见再次归一化）
        faiss.normalize_L2(vec)

        vector_id = self.index.ntotal
        self.index.add(vec)
        self.id_map.append(metadata)

        return vector_id

    def search(
            self,
            query: np.ndarray,
            k: int = 5
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        检索最相似的K个向量

        Returns:
            similarities: 相似度分数 [k]
            metadatas: 对应的元数据列表
        """
        if self.index.ntotal == 0:
            return np.array([]), []

        q = query.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)

        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(q, k)

        similarities = scores[0]
        metadatas = [self.id_map[i] for i in indices[0] if i >= 0]

        return similarities, metadatas

    def save(self, index_path: str, metadata_path: str):
        """持久化索引"""
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, ensure_ascii=False, indent=2)
        print(f"[FAISS] 已保存 {self.index.ntotal} 个向量")

    def load(self, index_path: str, metadata_path: str) -> bool:
        """加载索引"""
        if not Path(index_path).exists():
            return False
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)
        print(f"[FAISS] 加载完成，共 {self.index.ntotal} 个向量")
        return True

    @property
    def total(self) -> int:
        return self.index.ntotal


class ReagentRecognitionEngine:
    """
    试剂识别引擎

    主要对外接口：
    - register_reagent(): 注册新试剂（录入流程调用）
    - recognize(): 识别试剂
    - get_all_reagents(): 获取所有已注册试剂
    """

    def __init__(self):
        self.embedding_dim = MODEL_CONFIG["embedding_dim"]
        self.img_size = MODEL_CONFIG["img_size"]
        self.threshold = INFERENCE_CONFIG["similarity_threshold"]

        # 初始化嵌入提取器
        self.embedder = None
        self.transform = get_val_transforms(self.img_size)

        # FAISS索引
        self.faiss_index = FAISSIndex(self.embedding_dim)

        # 加载模型和索引
        self._load_model()
        self._load_index()

    def _load_model(self):
        """加载训练好的模型"""
        model_path = INFERENCE_CONFIG["model_path"]

        self.embedder = EfficientNetEmbedder(
            embedding_dim=self.embedding_dim,
            pretrained=False  # 推理时不需要重新加载预训练权重
        ).to(DEVICE)

        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=DEVICE)
            # 只加载embedder部分的权重
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            # 过滤出embedder的权重
            embedder_dict = {
                k.replace("embedder.", ""): v
                for k, v in state_dict.items()
                if k.startswith("embedder.")
            }
            if embedder_dict:
                self.embedder.load_state_dict(embedder_dict)
                print(f"[Engine] 加载模型权重: {model_path}")
            else:
                print("[Engine] 未找到embedder权重，使用预训练权重")
                self.embedder = EfficientNetEmbedder(
                    embedding_dim=self.embedding_dim,
                    pretrained=True
                ).to(DEVICE)
        else:
            print("[Engine] ⚠️  未找到训练模型，使用ImageNet预训练权重（仅用于演示）")
            self.embedder = EfficientNetEmbedder(
                embedding_dim=self.embedding_dim,
                pretrained=True
            ).to(DEVICE)

        self.embedder.eval()

    def _load_index(self):
        """加载FAISS索引"""
        index_path = INFERENCE_CONFIG["faiss_index_path"]
        meta_path = INFERENCE_CONFIG["metadata_path"]

        if Path(index_path).exists() and Path(meta_path).exists():
            self.faiss_index.load(index_path, meta_path)
        else:
            print("[Engine] 索引文件不存在，将在首次注册后创建")

    def _preprocess_image(self, image_input) -> torch.Tensor:
        """
        预处理图像

        Args:
            image_input: 可以是：
                - numpy array (BGR from OpenCV)
                - PIL Image
                - 文件路径 str
        """
        if isinstance(image_input, str):
            # 处理中文路径
            image_array = np.fromfile(image_input, dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if img is None:
                # 尝试使用PIL作为备选方案
                try:
                    pil_image = Image.open(image_input).convert('RGB')
                    img = np.array(pil_image)
                except Exception as e:
                    raise ValueError(f"无法读取图像: {image_input}, 错误: {str(e)}")
        elif isinstance(image_input, np.ndarray):
            if image_input.shape[2] == 3:
                # 假设BGR（OpenCV默认）
                img = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            else:
                img = image_input
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input.convert("RGB"))
        else:
            raise ValueError(f"不支持的图像类型: {type(image_input)}")

        # albumentations变换
        augmented = self.transform(image=img)
        tensor = augmented['image'].unsqueeze(0).to(DEVICE)  # [1, 3, H, W]
        return tensor

    @torch.no_grad()
    def extract_embedding(self, image_input) -> np.ndarray:
        """提取图像嵌入向量"""
        tensor = self._preprocess_image(image_input)
        embedding = self.embedder(tensor)  # [1, 512]
        return embedding.cpu().numpy()[0]  # [512]

    def register_reagent(
            self,
            image_input,
            reagent_id: str,
            reagent_name: str,
            extra_info: Optional[Dict] = None,
            image_save_path: Optional[str] = None,
    ) -> Dict:
        """
        注册新试剂到识别系统

        在录入流程中，用户在一体机录入后放入柜中，
        摄像头拍摄图像，调用此函数注册。

        Args:
            image_input: 摄像头捕获的图像
            reagent_id: 试剂唯一ID（如'乙醇001'）
            reagent_name: 试剂名称（如'乙醇'）
            extra_info: 额外信息（批次、保质期等）
            image_save_path: 保存图像的路径（可选）

        Returns:
            注册结果
        """
        # 提取嵌入
        embedding = self.extract_embedding(image_input)

        # 构建元数据
        metadata = {
            "reagent_id": reagent_id,
            "reagent_name": reagent_name,
            "vector_id": self.faiss_index.total,
            "timestamp": time.time(),
            "image_path": image_save_path or "",
            **(extra_info or {}),
        }

        # 添加到FAISS索引
        vid = self.faiss_index.add(embedding, metadata)

        # 持久化
        self._save_index()

        return {
            "success": True,
            "reagent_id": reagent_id,
            "vector_id": vid,
            "message": f"试剂 {reagent_id} 注册成功",
        }

    def recognize(
            self,
            image_input,
            topk: int = 5,
    ) -> Dict:
        """
        识别试剂

        Args:
            image_input: 摄像头图像
            topk: 返回前K个候选

        Returns:
            {
                "recognized": True/False,
                "reagent_id": "乙醇001",
                "confidence": 0.95,
                "candidates": [...],
            }
        """
        if self.faiss_index.total == 0:
            return {
                "recognized": False,
                "message": "系统中尚无注册试剂",
                "candidates": []
            }

        # 提取嵌入
        embedding = self.extract_embedding(image_input)

        # FAISS检索
        similarities, metadatas = self.faiss_index.search(embedding, k=topk)

        if len(similarities) == 0:
            return {"recognized": False, "message": "检索失败", "candidates": []}

        best_score = float(similarities[0])
        best_match = metadatas[0]

        # 构建候选列表
        candidates = [
            {
                "reagent_id": m["reagent_id"],
                "reagent_name": m["reagent_name"],
                "similarity": float(s),
                "confidence_pct": f"{float(s) * 100:.1f}%",
            }
            for s, m in zip(similarities, metadatas)
        ]

        # 判断是否识别成功
        recognized = best_score >= self.threshold

        return {
            "recognized": recognized,
            "reagent_id": best_match["reagent_id"] if recognized else None,
            "reagent_name": best_match["reagent_name"] if recognized else None,
            "confidence": best_score,
            "confidence_pct": f"{best_score * 100:.1f}%",
            "candidates": candidates,
            "threshold": self.threshold,
            "message": (
                f"识别成功: {best_match['reagent_id']} ({best_score * 100:.1f}%)"
                if recognized
                else f"置信度不足({best_score * 100:.1f}% < {self.threshold * 100:.0f}%)，可能是新试剂"
            ),
        }

    def _save_index(self):
        """保存索引"""
        self.faiss_index.save(
            INFERENCE_CONFIG["faiss_index_path"],
            INFERENCE_CONFIG["metadata_path"],
        )

    def get_all_reagents(self) -> List[Dict]:
        """获取所有已注册试剂"""
        return self.faiss_index.id_map

    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        reagents = self.faiss_index.id_map
        unique_ids = set(r["reagent_id"] for r in reagents)
        unique_names = set(r["reagent_name"] for r in reagents)
        return {
            "total_registrations": len(reagents),
            "unique_reagent_ids": len(unique_ids),
            "unique_reagent_names": len(unique_names),
            "faiss_vectors": self.faiss_index.total,
            "device": DEVICE,
            "model": MODEL_CONFIG["backbone"],
        }

    def rebuild_index_from_images(self, data_dir: str):
        """
        从图片目录重建索引
        （目录结构：data/images/乙醇001/*.jpg）
        """
        from pathlib import Path
        import cv2

        data_path = Path(data_dir).resolve()
        self.faiss_index = FAISSIndex(self.embedding_dim)

        total = 0
        for class_dir in sorted(data_path.iterdir()):
            if not class_dir.is_dir():
                continue

            reagent_id = class_dir.name
            # 从ID提取名称（去掉末尾数字）
            reagent_name = reagent_id.rstrip('0123456789')

            print(f"[Engine] 处理类别: {reagent_id}")
            
            for img_file in class_dir.glob("*.[jJpP][pPnN][gG]"):
                try:
                    self.register_reagent(
                        image_input=str(img_file.resolve()),
                        reagent_id=reagent_id,
                        reagent_name=reagent_name,
                        image_save_path=str(img_file.resolve()),
                    )
                    total += 1
                except Exception as e:
                    print(f"  处理 {img_file.name} 失败: {str(e)}")

        print(f"[Engine] 索引重建完成，共注册 {total} 张图片")
        self._save_index()


# 单例模式 - 全局唯一引擎实例
_engine_instance: Optional[ReagentRecognitionEngine] = None


def get_engine() -> ReagentRecognitionEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ReagentRecognitionEngine()
    return _engine_instance