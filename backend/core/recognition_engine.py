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
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import torch
import faiss
import cv2
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A

from backend.config import INFERENCE_CONFIG, MODEL_CONFIG, DEVICE, IMAGES_DIR
from backend.models.metric_model import EfficientNetEmbedder
from backend.models.foundation_embedder import create_foundation_embedder
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
        self.feature_extractor = (MODEL_CONFIG.get("feature_extractor") or "efficientnet").lower().strip()
        self.img_size = MODEL_CONFIG["img_size"]
        self.threshold = INFERENCE_CONFIG["similarity_threshold"]

        # 初始化嵌入提取器
        self.embedder = None
        self.model_id = None

        # 预处理：默认沿用 albumentations（efficientnet）
        self.transform = get_val_transforms(self.img_size)
        self.preprocess_mode = "albumentations"

        # FAISS索引
        # embedding_dim 可能由基础模型决定（dinov2/clip）
        self.embedding_dim = MODEL_CONFIG["embedding_dim"]
        self.faiss_index = FAISSIndex(self.embedding_dim)

        # 加载模型和索引
        self._load_model()
        self._load_index()

    def _load_model(self):
        """加载训练好的模型"""
        if self.feature_extractor in ("dinov2", "clip"):
            bundle = create_foundation_embedder(
                self.feature_extractor,
                dinov2_model_name=MODEL_CONFIG.get("dinov2_model_name", "facebook/dinov2-base"),
                clip_model_name=MODEL_CONFIG.get("clip_model_name", "openai/clip-vit-base-patch32"),
                device=DEVICE,
            )
            self.embedder = bundle.model
            self.embedding_dim = bundle.embedding_dim
            self.model_id = bundle.model_id
            self.preprocess_mode = "foundation"
            self.foundation_preprocess = bundle.preprocess

            # 重置 FAISS 维度（避免沿用旧配置）
            self.faiss_index = FAISSIndex(self.embedding_dim)
            print(f"[Engine] 使用基础模型特征：{self.model_id} (dim={self.embedding_dim})")
            return

        # 默认：efficientnet（保持旧逻辑）
        model_path = INFERENCE_CONFIG["model_path"]
        self.model_id = f"efficientnet_embedder:{MODEL_CONFIG.get('embedding_dim')}"

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
            # 维度不一致时不要硬加载（否则 search/add 会出错）
            try:
                loaded = self.faiss_index.load(index_path, meta_path)
                if loaded and getattr(self.faiss_index.index, "d", None) != self.embedding_dim:
                    d = getattr(self.faiss_index.index, "d", None)
                    print(
                        f"[Engine] ⚠️  现有索引维度({d})与当前特征维度({self.embedding_dim})不一致，"
                        f"将忽略旧索引。请运行 scripts/build_index.py 重建索引。"
                    )
                    self.faiss_index = FAISSIndex(self.embedding_dim)
            except Exception as e:
                print(f"[Engine] ⚠️  索引加载失败，将在首次注册后创建。原因: {e}")
                self.faiss_index = FAISSIndex(self.embedding_dim)
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

        if self.preprocess_mode == "foundation":
            # foundation preprocess expects RGB float tensor in [0,1]
            pil = Image.fromarray(img)
            t = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float() / 255.0  # [3,H,W]
            t = t.unsqueeze(0).to(DEVICE)
            return t

        # albumentations变换（efficientnet）
        augmented = self.transform(image=img)
        tensor = augmented["image"].unsqueeze(0).to(DEVICE)  # [1, 3, H, W]
        return tensor

    @torch.no_grad()
    def extract_embedding(self, image_input) -> np.ndarray:
        """提取图像嵌入向量"""
        tensor = self._preprocess_image(image_input)
        if self.preprocess_mode == "foundation":
            pixel_values = self.foundation_preprocess(tensor)  # -> pixel_values
            embedding = self.embedder(pixel_values)
        else:
            embedding = self.embedder(tensor)  # [1, D]
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
        识别试剂（单物体）

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

    def recognize_multiple(
            self,
            image_input,
            topk: int = 5,
            min_confidence: float = 0.5,
    ) -> Dict:
        """
        识别多个试剂（多物体识别）

        Args:
            image_input: 摄像头图像（可能包含多个试剂）
            topk: 返回前K个候选
            min_confidence: 检测置信度阈值

        Returns:
            {
                "total_objects": 3,
                "recognized_objects": [
                    {
                        "bbox": [x1, y1, x2, y2],
                        "reagent_id": "乙醇001",
                        "reagent_name": "乙醇",
                        "confidence": 0.92,
                    },
                    ...
                ],
                "unrecognized_objects": [
                    {
                        "bbox": [x1, y1, x2, y2],
                        "best_candidate": "乙醇002",
                        "confidence": 0.68,
                    }
                ]
            }
        """
        if self.faiss_index.total == 0:
            return {
                "total_objects": 0,
                "recognized_objects": [],
                "unrecognized_objects": [],
                "message": "系统中尚无注册试剂",
            }

        try:
            from backend.core.object_detector import get_detector
        except ImportError:
            return {
                "total_objects": 0,
                "recognized_objects": [],
                "unrecognized_objects": [],
                "message": "目标检测模块未安装，请先安装 ultralytics: pip install ultralytics",
            }

        # 使用全局检测器单例（避免每次请求都重复加载 YOLO）
        detector = get_detector()

        # 预处理图像
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                image_array = np.fromfile(image_input, dtype=np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
        else:
            img = np.array(image_input)

        h, w = img.shape[:2]

        # 检测所有物体
        detections = detector.detect(img, confidence_threshold=min_confidence)
        # 按置信度排序，避免低质量框影响结果
        detections = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)

        recognized_objects = []
        unrecognized_objects = []

        # 对每个检测到的物体进行识别
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']

            # 过滤过小的框（通常是误检/噪声）
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            if bw < max(20, int(0.03 * w)) or bh < max(20, int(0.03 * h)):
                continue

            # 裁剪检测区域（带 padding，提升识别稳定性）
            pad = max(10, int(0.08 * max(bw, bh)))
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(w, x2 + pad)
            cy2 = min(h, y2 + pad)
            crop = img[cy1:cy2, cx1:cx2]

            if crop.size == 0:
                continue

            # 识别裁剪的图像
            result = self.recognize(crop, topk=topk)

            obj_info = {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "crop_bbox": [int(cx1), int(cy1), int(cx2), int(cy2)],
                "detection_confidence": float(confidence),
                "detector_class": det.get("class_name", "unknown"),
            }

            if result["recognized"]:
                obj_info.update({
                    "reagent_id": result["reagent_id"],
                    "reagent_name": result["reagent_name"],
                    "confidence": result["confidence"],
                    "confidence_pct": result["confidence_pct"],
                })
                recognized_objects.append(obj_info)
            else:
                if result["candidates"]:
                    obj_info.update({
                        "best_candidate": result["candidates"][0]["reagent_id"],
                        "best_candidate_name": result["candidates"][0]["reagent_name"],
                        "confidence": result["confidence"],
                        "confidence_pct": result["confidence_pct"],
                    })
                unrecognized_objects.append(obj_info)

        return {
            "total_objects": len(recognized_objects) + len(unrecognized_objects),
            "recognized_count": len(recognized_objects),
            "unrecognized_count": len(unrecognized_objects),
            "recognized_objects": recognized_objects,
            "unrecognized_objects": unrecognized_objects,
            "message": f"检测到 {len(recognized_objects) + len(unrecognized_objects)} 个物体，识别成功 {len(recognized_objects)} 个",
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
            "model": self.model_id or MODEL_CONFIG.get("backbone", "N/A"),
        }

    async def rebuild_index_from_images(self, data_dir: str, db=None):
        """
        从图片目录重建索引
        （目录结构：data/images/乙醇001/*.jpg）
        
        Args:
            data_dir: 图片目录路径
            db: 数据库会话（可选，用于获取试剂名称和更新图片数）
        """
        from pathlib import Path
        import cv2
        from backend.core.database import Reagent
        from sqlalchemy import select, update

        data_path = Path(data_dir).resolve()
        self.faiss_index = FAISSIndex(self.embedding_dim)

        # 从数据库获取试剂名称映射
        reagent_name_map = {}
        db_reagent_ids = None
        if db is not None:
            try:
                result = await db.execute(select(Reagent))
                reagents = result.scalars().all()
                reagent_name_map = {r.reagent_id: r.reagent_name for r in reagents}
                db_reagent_ids = set(reagent_name_map.keys())
                print(f"[Engine] 从数据库加载了 {len(reagent_name_map)} 个试剂名称")
            except Exception as e:
                print(f"[Engine] 从数据库加载试剂名称失败: {e}")

        # 统计每个试剂的图片数量
        reagent_image_counts = {}
        total = 0
        
        for class_dir in sorted(data_path.iterdir()):
            if not class_dir.is_dir():
                continue

            reagent_id = class_dir.name
            # 跳过明显非试剂目录
            if reagent_id in {"corrections", "__pycache__", ".git"}:
                continue
            # 如果提供了数据库，则以数据库为准过滤，避免把无关目录当试剂
            if db_reagent_ids is not None and reagent_id not in db_reagent_ids:
                print(f"[Engine] 跳过目录 '{reagent_id}'（数据库中不存在该 reagent_id）")
                continue

            # 优先使用数据库中的名称，否则从ID推断
            reagent_name = reagent_name_map.get(reagent_id)
            if not reagent_name:
                reagent_name = reagent_id.rstrip('0123456789')
                print(f"[Engine] 警告: 未找到 '{reagent_id}' 的数据库记录，使用推断名称: {reagent_name}")

            print(f"[Engine] 处理类别: {reagent_id} - {reagent_name}")
            
            image_count = 0
            for img_file in class_dir.glob("*.[jJpP][pPnN][gG]"):
                try:
                    self.register_reagent(
                        image_input=str(img_file.resolve()),
                        reagent_id=reagent_id,
                        reagent_name=reagent_name,
                        image_save_path=str(img_file.resolve()),
                    )
                    total += 1
                    image_count += 1
                except Exception as e:
                    print(f"  处理 {img_file.name} 失败: {str(e)}")
            
            reagent_image_counts[reagent_id] = image_count

        print(f"[Engine] 索引重建完成，共注册 {total} 张图片")
        self._save_index()
        
        # 更新数据库中的图片数量
        if db is not None and reagent_image_counts:
            try:
                for reagent_id, count in reagent_image_counts.items():
                    await db.execute(
                        update(Reagent)
                        .where(Reagent.reagent_id == reagent_id)
                        .values(image_count=count)
                    )
                await db.commit()
                print(f"[Engine] 已更新 {len(reagent_image_counts)} 个试剂的图片数量")
            except Exception as e:
                print(f"[Engine] 更新数据库图片数量失败: {e}")

    def delete_reagent(self, reagent_id: str) -> Dict:
        """
        删除试剂的所有特征向量

        Args:
            reagent_id: 试剂唯一ID

        Returns:
            删除结果
        """
        if self.faiss_index.total == 0:
            return {
                "success": True,
                "deleted_count": 0,
                "message": f"试剂 {reagent_id} 的索引为空",
            }

        # 找到所有属于该试剂的向量
        to_delete = set()
        for i, metadata in enumerate(self.faiss_index.id_map):
            if metadata.get("reagent_id") == reagent_id:
                to_delete.add(i)

        if not to_delete:
            return {
                "success": True,
                "deleted_count": 0,
                "message": f"未找到试剂 {reagent_id} 的特征向量",
            }

        # 重建索引（排除要删除的向量）
        new_id_map = []
        new_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # 重新添加未被删除的向量
        for i, metadata in enumerate(self.faiss_index.id_map):
            if i not in to_delete:
                # 从旧索引中重建向量
                vec = self.faiss_index.index.reconstruct(i)
                new_index.add(vec.reshape(1, -1))
                new_id_map.append(metadata)
        
        self.faiss_index.index = new_index
        self.faiss_index.id_map = new_id_map
        
        # 持久化
        self._save_index()

        print(f"[Engine] 已删除试剂 {reagent_id} 的 {len(to_delete)} 个特征向量")
        print(f"[Engine] 剩余特征向量: {self.faiss_index.total}")

        return {
            "success": True,
            "deleted_count": len(to_delete),
            "message": f"已删除试剂 {reagent_id} 的 {len(to_delete)} 个特征向量",
        }

    def apply_correction(
            self,
            image_input,
            corrected_reagent_id: str,
            corrected_reagent_name: str,
            original_recognition_id: Optional[str] = None,
            original_image_path: Optional[str] = None,
            save_image: bool = True,
            correction_source: str = "manual",
            notes: Optional[str] = None,
    ) -> Dict:
        """
        应用纠错 - 将纠正后的样本注册到FAISS索引
        
        Args:
            image_input: 纠正后的图像
            corrected_reagent_id: 纠正后的试剂ID
            corrected_reagent_name: 纠正后的试剂名称
            original_recognition_id: 原识别结果ID（用于记录）
            original_image_path: 原识别图片路径
            save_image: 是否保存纠错图片
            correction_source: 纠正来源（manual/auto）
            notes: 纠正备注
        
        Returns:
            应用结果
        """
        import os
        from pathlib import Path
        
        # 保存纠错图片
        corrected_image_path = None
        if save_image:
            corrections_dir = Path(IMAGES_DIR) / "corrections"
            corrections_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time() * 1000)
            filename = f"{corrected_reagent_id}_{timestamp}.jpg"
            corrected_image_path = str(corrections_dir / filename)
            
            if isinstance(image_input, str):
                shutil.copy(image_input, corrected_image_path)
            elif isinstance(image_input, np.ndarray):
                cv2.imwrite(corrected_image_path, image_input)
            elif isinstance(image_input, Image.Image):
                image_input.save(corrected_image_path)
        
        # 提取嵌入
        embedding = self.extract_embedding(image_input)
        
        # 构建元数据
        metadata = {
            "reagent_id": corrected_reagent_id,
            "reagent_name": corrected_reagent_name,
            "vector_id": self.faiss_index.total,
            "timestamp": time.time(),
            "image_path": corrected_image_path or "",
            "is_correction": True,
            "original_recognition_id": original_recognition_id,
            "correction_source": correction_source,
            "notes": notes,
        }
        
        # 添加到FAISS索引
        vid = self.faiss_index.add(embedding, metadata)
        
        # 持久化
        self._save_index()
        
        return {
            "success": True,
            "reagent_id": corrected_reagent_id,
            "reagent_name": corrected_reagent_name,
            "vector_id": vid,
            "corrected_image_path": corrected_image_path,
            "message": f"纠错已应用，试剂 {corrected_reagent_id} 的特征向量已添加到索引",
        }

    def get_correction_statistics(self) -> Dict:
        """
        获取纠错统计信息
        
        Returns:
            纠错统计数据
        """
        total_vectors = self.faiss_index.total
        correction_vectors = [
            m for m in self.faiss_index.id_map
            if m.get("is_correction", False)
        ]
        
        unique_corrected_ids = set(m["reagent_id"] for m in correction_vectors)
        correction_sources = {}
        for m in correction_vectors:
            source = m.get("correction_source", "unknown")
            correction_sources[source] = correction_sources.get(source, 0) + 1
        
        return {
            "total_vectors": total_vectors,
            "correction_count": len(correction_vectors),
            "correction_ratio": f"{len(correction_vectors) / total_vectors * 100:.2f}%" if total_vectors > 0 else "0%",
            "unique_corrected_reagents": len(unique_corrected_ids),
            "correction_sources": correction_sources,
        }

    def export_corrections_for_training(
            self,
            output_dir: str = None,
            include_original: bool = False,
    ) -> Dict:
        """
        导出纠错样本用于模型重训
        
        Args:
            output_dir: 输出目录（默认为 data/corrections）
            include_original: 是否包含原始识别错误的图片
        
        Returns:
            导出结果
        """
        from pathlib import Path
        import shutil
        
        if output_dir is None:
            output_dir = Path("data/corrections")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有纠错向量
        correction_vectors = [
            m for m in self.faiss_index.id_map
            if m.get("is_correction", False)
        ]
        
        if not correction_vectors:
            return {
                "success": False,
                "exported_count": 0,
                "message": "没有纠错样本可导出",
            }
        
        exported_count = 0
        exported_reagents = set()
        
        for metadata in correction_vectors:
            reagent_id = metadata["reagent_id"]
            image_path = metadata.get("image_path", "")
            
            if not image_path or not Path(image_path).exists():
                print(f"[Engine] 警告: 图片不存在，跳过: {image_path}")
                continue
            
            # 创建试剂目录
            reagent_dir = output_dir / reagent_id
            reagent_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制图片
            filename = Path(image_path).name
            dest_path = reagent_dir / filename
            shutil.copy(image_path, dest_path)
            
            exported_count += 1
            exported_reagents.add(reagent_id)
            
            # 如果需要，也复制原始错误图片
            if include_original:
                original_image_path = metadata.get("original_image_path", "")
                if original_image_path and Path(original_image_path).exists():
                    original_filename = f"original_{filename}"
                    original_dest_path = reagent_dir / original_filename
                    shutil.copy(original_image_path, original_dest_path)
        
        # 生成导出报告
        report_path = output_dir / "export_report.json"
        import json
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "export_time": datetime.now().isoformat(),
                "exported_count": exported_count,
                "exported_reagents": list(exported_reagents),
                "output_directory": str(output_dir),
                "include_original": include_original,
            }, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "exported_count": exported_count,
            "exported_reagents": len(exported_reagents),
            "output_directory": str(output_dir),
            "report_path": str(report_path),
            "message": f"已导出 {exported_count} 个纠错样本到 {output_dir}",
        }

    def verify_correction_quality(
            self,
            reagent_id: str,
            min_samples: int = 3,
    ) -> Dict:
        """
        验证纠错质量 - 检查某个试剂的纠错样本是否足够用于训练
        
        Args:
            reagent_id: 试剂ID
            min_samples: 最小样本数
        
        Returns:
            验证结果
        """
        correction_vectors = [
            m for m in self.faiss_index.id_map
            if m.get("is_correction", False) and m["reagent_id"] == reagent_id
        ]
        
        all_vectors = [
            m for m in self.faiss_index.id_map
            if m["reagent_id"] == reagent_id
        ]
        
        return {
            "reagent_id": reagent_id,
            "total_samples": len(all_vectors),
            "correction_samples": len(correction_vectors),
            "meets_minimum": len(correction_vectors) >= min_samples,
            "correction_ratio": f"{len(correction_vectors) / len(all_vectors) * 100:.2f}%" if all_vectors else "0%",
            "ready_for_retraining": len(correction_vectors) >= min_samples,
        }


# 单例模式 - 全局唯一引擎实例
_engine_instance: Optional[ReagentRecognitionEngine] = None


def get_engine() -> ReagentRecognitionEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ReagentRecognitionEngine()
    return _engine_instance