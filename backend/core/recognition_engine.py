"""
backend/core/recognition_engine.py

主要修复：
[Fix-1] delete_vector / delete_reagent: HNSW 不支持 reconstruct(), 改为统一用 IndexFlatIP
[Fix-2] 纠错后立即识别仍错误: apply_correction / register_reagent 改为 force=True 强制写盘
         并且在内存索引写入后立即刷新 efSearch，保证新向量可被检索
[Fix-3] 索引类型统一: 全部改用 IndexFlatIP（小数据量精度最高，支持 reconstruct）
         如果向量数 > 10000 才考虑换 HNSW
[Fix-4] FAISSIndex.save 原子写: 已有 tmp->rename，保留
[Fix-5] register_reagent 默认 force=True，确保每次注册后索引立即持久化
"""

import json
import time
import shutil
import threading
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
from backend.models.metric_model import EfficientNetV2Embedder
from backend.core.dataset import get_val_transforms
from backend.models.metric_model import ReagentRecognitionModel


class FAISSIndex:
    """
    FAISS 向量索引管理

    [Fix-3] 统一使用 IndexFlatIP:
    - 支持 reconstruct() → delete_vector/delete_reagent 不再崩溃
    - 对于小数据集（<10000条）精度与 HNSW 相当
    - 不需要 efSearch/efConstruction 调参
    """

    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        # [Fix-3] 统一用 IndexFlatIP，小数据集精度最高且支持 reconstruct
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.id_map: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._pending_saves = 0
        self._last_save_time = 0

    def add(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> int:
        with self._lock:
            vec = embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(vec)
            vector_id = self.index.ntotal
            self.index.add(vec)
            self.id_map.append(metadata)
            self._pending_saves += 1
            return vector_id

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[Dict]]:
        with self._lock:
            if self.index.ntotal == 0:
                return np.array([]), []
            q = query.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(q)
            k = min(k, self.index.ntotal)
            scores, indices = self.index.search(q, k)
            similarities = scores[0]
            metadatas = [self.id_map[i] for i in indices[0] if i >= 0]
            return similarities, metadatas

    def save(self, index_path: str, metadata_path: str, force: bool = False) -> bool:
        with self._lock:
            current_time = time.time()
            # [Fix-2] force=True 时跳过批量保存策略
            if not force and self._pending_saves < 10 and (current_time - self._last_save_time) < 5:
                return False

            index_path = Path(index_path)
            metadata_path = Path(metadata_path)
            temp_index_path = index_path.with_suffix('.tmp')
            temp_metadata_path = metadata_path.with_suffix('.tmp')

            try:
                faiss.write_index(self.index, str(temp_index_path))
                with open(temp_metadata_path, "w", encoding="utf-8") as f:
                    json.dump(self.id_map, f, ensure_ascii=False, indent=2)
                temp_index_path.replace(index_path)
                temp_metadata_path.replace(metadata_path)
                self._pending_saves = 0
                self._last_save_time = current_time
                print(f"[FAISS] 已保存 {self.index.ntotal} 个向量 (force={force})")
                return True
            except Exception as e:
                print(f"[FAISS] 保存失败: {e}")
                if temp_index_path.exists():
                    temp_index_path.unlink()
                if temp_metadata_path.exists():
                    temp_metadata_path.unlink()
                return False

    def load(self, index_path: str, metadata_path: str) -> bool:
        with self._lock:
            if not Path(index_path).exists():
                return False
            try:
                loaded_index = faiss.read_index(index_path)
                with open(metadata_path, "r", encoding="utf-8") as f:
                    loaded_id_map = json.load(f)

                if len(loaded_id_map) != loaded_index.ntotal:
                    print(
                        f"[FAISS] 警告: 元数据({len(loaded_id_map)})与向量数"
                        f"({loaded_index.ntotal})不一致，重建空索引"
                    )
                    return False

                # [Fix-3] 如果加载的是 HNSW 索引，迁移到 IndexFlatIP
                if not isinstance(loaded_index, faiss.IndexFlatIP):
                    print("[FAISS] 检测到旧版 HNSW 索引，正在迁移到 IndexFlatIP...")
                    new_index = faiss.IndexFlatIP(loaded_index.d)
                    # HNSW 支持有限的 reconstruct，但我们从旧 id_map 重新提取嵌入
                    # 如果 reconstruct 失败，提示用户重建索引
                    try:
                        for i in range(loaded_index.ntotal):
                            vec = loaded_index.reconstruct(i)
                            new_index.add(vec.reshape(1, -1))
                        self.index = new_index
                        self.id_map = loaded_id_map
                        print(f"[FAISS] HNSW → FlatIP 迁移完成，共 {new_index.ntotal} 个向量")
                        # 立即保存为新格式
                        faiss.write_index(new_index, index_path)
                        return True
                    except Exception as e:
                        print(f"[FAISS] HNSW 迁移失败: {e}")
                        print("[FAISS] 请运行 scripts/build_index.py 重建索引")
                        return False

                self.index = loaded_index
                self.id_map = loaded_id_map
                print(f"[FAISS] 加载完成，共 {self.index.ntotal} 个向量")
                return True
            except Exception as e:
                print(f"[FAISS] 加载失败: {e}")
                return False

    @property
    def total(self) -> int:
        return self.index.ntotal


class ReagentRecognitionEngine:

    def __init__(self):
        self.img_size = MODEL_CONFIG["img_size"]
        self.threshold = INFERENCE_CONFIG["similarity_threshold"]
        self.embedder = None
        self.transform = get_val_transforms(self.img_size)
        self.embedding_dim = None
        self.faiss_index = None
        self._load_model()
        self._load_index()

    def _load_model(self):
        model_path = INFERENCE_CONFIG["model_path"]

        if not Path(model_path).exists():
            print("[Engine] ⚠️  未找到训练模型，使用 ImageNet 预训练权重（仅用于演示）")
            self.embedder = EfficientNetV2Embedder(
                embedding_dim=MODEL_CONFIG["embedding_dim"],
                pretrained=True
            ).to(DEVICE)
            self.embedding_dim = self.embedder.embedding_dim
            self.faiss_index = FAISSIndex(self.embedding_dim)
            self.embedder.eval()
            return

        checkpoint = torch.load(model_path, map_location=DEVICE)

        class_mapping_path = Path(model_path).parent / "class_mapping.json"
        if class_mapping_path.exists():
            with open(str(class_mapping_path), "r", encoding="utf-8") as f:
                class_mapping = json.load(f)
            num_classes = len(class_mapping["class_to_idx"])
        else:
            num_classes = 100

        self.model = ReagentRecognitionModel(
            num_classes=num_classes,
            embedding_dim=MODEL_CONFIG["embedding_dim"],
            pretrained=False,
            arcface_margin=MODEL_CONFIG["arcface_margin"],
            arcface_scale=MODEL_CONFIG["arcface_scale"],
        ).to(DEVICE)
        self.model.eval()

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        new_state_dict = {
            k[7:] if k.startswith('module.') else k: v
            for k, v in state_dict.items()
        }
        missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"[Engine] 警告: 缺少的键: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"[Engine] 警告: 意外的键: {unexpected_keys[:5]}...")

        print(f"[Engine] 加载模型权重: {model_path}")
        self.embedder = self.model.embedder
        self.embedding_dim = self.model.embedding_dim
        self.faiss_index = FAISSIndex(self.embedding_dim)

        if hasattr(self.embedder, 'eval'):
            self.embedder.eval()

    def _load_index(self):
        index_path = INFERENCE_CONFIG["faiss_index_path"]
        meta_path = INFERENCE_CONFIG["metadata_path"]

        if Path(index_path).exists() and Path(meta_path).exists():
            try:
                loaded = self.faiss_index.load(index_path, meta_path)
                if loaded:
                    # 检查维度一致性
                    if self.faiss_index.index.d != self.embedding_dim:
                        print(
                            f"[Engine] ⚠️  索引维度({self.faiss_index.index.d})"
                            f"与模型维度({self.embedding_dim})不一致，重建空索引"
                        )
                        self.faiss_index = FAISSIndex(self.embedding_dim)
            except Exception as e:
                print(f"[Engine] ⚠️  索引加载失败: {e}")
                self.faiss_index = FAISSIndex(self.embedding_dim)
        else:
            print("[Engine] 索引文件不存在，将在首次注册后创建")

    def _preprocess_image(self, image_input) -> torch.Tensor:
        if isinstance(image_input, str):
            image_array = np.fromfile(image_input, dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if img is None:
                try:
                    pil_image = Image.open(image_input).convert('RGB')
                    img = np.array(pil_image)
                except Exception as e:
                    raise ValueError(f"无法读取图像: {image_input}, 错误: {str(e)}")
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                img = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            else:
                img = image_input
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input.convert("RGB"))
        else:
            raise ValueError(f"不支持的图像类型: {type(image_input)}")

        augmented = self.transform(image=img)
        tensor = augmented["image"].unsqueeze(0).to(DEVICE)
        return tensor

    @torch.no_grad()
    def extract_embedding(self, image_input) -> np.ndarray:
        tensor = self._preprocess_image(image_input)
        embedding = self.embedder(tensor)
        embedding = embedding.cpu().numpy()[0]
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding

    def register_reagent(
        self,
        image_input,
        reagent_id: str,
        reagent_name: str,
        extra_info: Optional[Dict] = None,
        image_save_path: Optional[str] = None,
        force_save: bool = True,  # [Fix-2] 默认强制写盘，确保注册后立即可检索
    ) -> Dict:
        """
        注册新试剂

        [Fix-2] force_save 默认 True：
        每次注册后立即写盘，保证前端纠错后下次识别能命中新向量。
        之前批量策略（10次才写一次）会导致刚注册的样本在内存中但下次
        请求仍从旧磁盘索引加载时丢失。
        """
        embedding = self.extract_embedding(image_input)
        metadata = {
            "reagent_id":   reagent_id,
            "reagent_name": reagent_name,
            "vector_id":    self.faiss_index.total,
            "timestamp":    time.time(),
            "image_path":   image_save_path or "",
            **(extra_info or {}),
        }
        vid = self.faiss_index.add(embedding, metadata)

        # [Fix-2] 强制写盘
        saved = self.faiss_index.save(
            INFERENCE_CONFIG["faiss_index_path"],
            INFERENCE_CONFIG["metadata_path"],
            force=force_save,
        )

        return {
            "success":    True,
            "reagent_id": reagent_id,
            "vector_id":  vid,
            "saved":      saved,
            "message":    f"试剂 {reagent_id} 注册成功",
        }

    def recognize(self, image_input, topk: int = 5) -> Dict:
        if self.faiss_index.total == 0:
            return {
                "recognized": False,
                "message":    "系统中尚无注册试剂",
                "candidates": [],
            }

        embedding = self.extract_embedding(image_input)
        similarities, metadatas = self.faiss_index.search(embedding, k=topk)

        if len(similarities) == 0:
            return {"recognized": False, "message": "检索失败", "candidates": []}

        best_score = float(similarities[0])
        best_match = metadatas[0]

        candidates = [
            {
                "reagent_id":      m["reagent_id"],
                "reagent_name":    m["reagent_name"],
                "similarity":      float(s),
                "confidence_pct":  f"{float(s) * 100:.1f}%",
            }
            for s, m in zip(similarities, metadatas)
        ]

        recognized = best_score >= self.threshold
        return {
            "recognized":     recognized,
            "reagent_id":     best_match["reagent_id"] if recognized else None,
            "reagent_name":   best_match["reagent_name"] if recognized else None,
            "confidence":     best_score,
            "confidence_pct": f"{best_score * 100:.1f}%",
            "candidates":     candidates,
            "threshold":      self.threshold,
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
        if self.faiss_index.total == 0:
            return {
                "total_objects":       0,
                "recognized_objects":  [],
                "unrecognized_objects": [],
                "message":             "系统中尚无注册试剂",
            }

        try:
            from backend.core.object_detector import get_detector
        except ImportError:
            return {
                "total_objects":       0,
                "recognized_objects":  [],
                "unrecognized_objects": [],
                "message":             "目标检测模块未安装，请先安装 ultralytics",
            }

        detector = get_detector()

        if isinstance(image_input, str):
            image_array = np.fromfile(image_input, dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
        else:
            img = np.array(image_input)

        h, w = img.shape[:2]
        detections = detector.detect(img, confidence_threshold=min_confidence)
        detections = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)

        recognized_objects = []
        unrecognized_objects = []

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']

            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            if bw < max(20, int(0.03 * w)) or bh < max(20, int(0.03 * h)):
                continue

            pad = max(10, int(0.08 * max(bw, bh)))
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(w, x2 + pad)
            cy2 = min(h, y2 + pad)
            crop = img[cy1:cy2, cx1:cx2]

            if crop.size == 0:
                continue

            result = self.recognize(crop, topk=topk)
            obj_info = {
                "bbox":                 [int(x1), int(y1), int(x2), int(y2)],
                "crop_bbox":            [int(cx1), int(cy1), int(cx2), int(cy2)],
                "detection_confidence": float(confidence),
                "detector_class":       det.get("class_name", "unknown"),
            }

            if result["recognized"]:
                obj_info.update({
                    "reagent_id":     result["reagent_id"],
                    "reagent_name":   result["reagent_name"],
                    "confidence":     result["confidence"],
                    "confidence_pct": result["confidence_pct"],
                })
                recognized_objects.append(obj_info)
            else:
                if result["candidates"]:
                    obj_info.update({
                        "best_candidate":      result["candidates"][0]["reagent_id"],
                        "best_candidate_name": result["candidates"][0]["reagent_name"],
                        "confidence":          result["confidence"],
                        "confidence_pct":      result["confidence_pct"],
                    })
                unrecognized_objects.append(obj_info)

        return {
            "total_objects":        len(recognized_objects) + len(unrecognized_objects),
            "recognized_count":     len(recognized_objects),
            "unrecognized_count":   len(unrecognized_objects),
            "recognized_objects":   recognized_objects,
            "unrecognized_objects": unrecognized_objects,
            "message": (
                f"检测到 {len(recognized_objects) + len(unrecognized_objects)} 个物体，"
                f"识别成功 {len(recognized_objects)} 个"
            ),
        }

    def _save_index(self, force: bool = False):
        self.faiss_index.save(
            INFERENCE_CONFIG["faiss_index_path"],
            INFERENCE_CONFIG["metadata_path"],
            force=force,
        )

    def get_all_reagents(self) -> List[Dict]:
        return self.faiss_index.id_map

    def get_stats(self) -> Dict:
        reagents = self.faiss_index.id_map
        unique_ids   = set(r["reagent_id"]   for r in reagents)
        unique_names = set(r["reagent_name"] for r in reagents)
        return {
            "total_registrations":   len(reagents),
            "unique_reagent_ids":    len(unique_ids),
            "unique_reagent_names":  len(unique_names),
            "faiss_vectors":         self.faiss_index.total,
            "device":                DEVICE,
            "model":                 MODEL_CONFIG.get("backbone", "N/A"),
        }

    async def rebuild_index_from_images(self, data_dir: str, db=None):
        """从图片目录重建索引"""
        from pathlib import Path
        from backend.core.database import Reagent
        from sqlalchemy import select, update

        data_path = Path(data_dir).resolve()
        self.faiss_index = FAISSIndex(self.embedding_dim)

        reagent_name_map = {}
        db_reagent_ids = None
        if db is not None:
            try:
                result = await db.execute(select(Reagent))
                reagents = result.scalars().all()
                reagent_name_map = {r.reagent_id: r.reagent_name for r in reagents}
                db_reagent_ids = set(reagent_name_map.keys())
            except Exception as e:
                print(f"[Engine] 从数据库加载试剂名称失败: {e}")

        total = 0
        for class_dir in sorted(data_path.iterdir()):
            if not class_dir.is_dir():
                continue
            reagent_id = class_dir.name
            if reagent_id in {"corrections", "__pycache__", ".git"}:
                continue
            if db_reagent_ids is not None and reagent_id not in db_reagent_ids:
                continue

            reagent_name = reagent_name_map.get(reagent_id) or reagent_id.rstrip('0123456789')

            for img_file in class_dir.glob("*.[jJpP][pPnN][gG]"):
                try:
                    self.register_reagent(
                        image_input=str(img_file.resolve()),
                        reagent_id=reagent_id,
                        reagent_name=reagent_name,
                        image_save_path=str(img_file.resolve()),
                        force_save=False,   # 批量重建时暂不写盘，最后统一写
                    )
                    total += 1
                except Exception as e:
                    print(f"  处理 {img_file.name} 失败: {str(e)}")

        # 重建完成后强制写盘一次
        self._save_index(force=True)
        print(f"[Engine] 索引重建完成，共注册 {total} 张图片")

        if db is not None:
            try:
                from backend.core.database import ReagentImage
                img_result = await db.execute(
                    select(ReagentImage.reagent_id, ReagentImage.id)
                )
                img_records = img_result.all()
                image_counts = {}
                for rid, _ in img_records:
                    image_counts[rid] = image_counts.get(rid, 0) + 1
                for rid, count in image_counts.items():
                    await db.execute(
                        update(Reagent)
                        .where(Reagent.reagent_id == rid)
                        .values(image_count=count)
                    )
                await db.commit()
            except Exception as e:
                print(f"[Engine] 更新数据库图片数量失败: {e}")

    def delete_reagent(self, reagent_id: str) -> Dict:
        """
        删除试剂的所有特征向量

        [Fix-1] 改用重建 IndexFlatIP（不再依赖 reconstruct 的索引类型）
        """
        if self.faiss_index.total == 0:
            return {"success": True, "deleted_count": 0, "message": f"索引为空"}

        to_delete = {
            i for i, metadata in enumerate(self.faiss_index.id_map)
            if metadata.get("reagent_id") == reagent_id
        }

        if not to_delete:
            return {
                "success": True,
                "deleted_count": 0,
                "message": f"未找到试剂 {reagent_id} 的特征向量",
            }

        self._rebuild_index_excluding(to_delete)
        self._save_index(force=True)

        print(f"[Engine] 已删除试剂 {reagent_id} 的 {len(to_delete)} 个特征向量")
        return {
            "success":       True,
            "deleted_count": len(to_delete),
            "message":       f"已删除试剂 {reagent_id} 的 {len(to_delete)} 个特征向量",
        }

    def delete_vector(self, vector_id: int) -> Dict:
        """
        删除单个特征向量

        [Fix-1] 原来用 HNSW reconstruct() 会崩溃，改用安全的重建方式
        """
        if self.faiss_index.total == 0:
            return {"success": True, "deleted_count": 0, "message": "索引为空"}

        target_index = next(
            (i for i, m in enumerate(self.faiss_index.id_map) if m.get("vector_id") == vector_id),
            None,
        )

        if target_index is None:
            return {
                "success": True,
                "deleted_count": 0,
                "message": f"未找到 vector_id={vector_id} 的特征向量",
            }

        self._rebuild_index_excluding({target_index})
        self._save_index(force=True)

        print(f"[Engine] 已删除 vector_id={vector_id} 的特征向量")
        return {
            "success":       True,
            "deleted_count": 1,
            "message":       f"已删除 vector_id={vector_id} 的特征向量",
        }

    def _rebuild_index_excluding(self, exclude_positions: set):
        """
        [Fix-1] 安全重建索引，排除指定位置的向量

        使用 IndexFlatIP.reconstruct() 提取原始向量（FlatIP 支持此操作）
        """
        new_index  = faiss.IndexFlatIP(self.embedding_dim)
        new_id_map = []

        for i, metadata in enumerate(self.faiss_index.id_map):
            if i in exclude_positions:
                continue
            try:
                # [Fix-1] IndexFlatIP 支持 reconstruct
                vec = self.faiss_index.index.reconstruct(i)
                new_index.add(vec.reshape(1, -1))
                new_id_map.append(metadata)
            except Exception as e:
                print(f"[Engine] 重建索引时跳过位置 {i}: {e}")

        self.faiss_index.index  = new_index
        self.faiss_index.id_map = new_id_map
        print(f"[Engine] 索引重建完成，剩余 {new_index.ntotal} 个向量")

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
        应用纠错 — [Fix-2] 强制写盘，确保纠错后下次识别立即生效
        """
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

        embedding = self.extract_embedding(image_input)
        metadata = {
            "reagent_id":              corrected_reagent_id,
            "reagent_name":            corrected_reagent_name,
            "vector_id":               self.faiss_index.total,
            "timestamp":               time.time(),
            "image_path":              corrected_image_path or "",
            "is_correction":           True,
            "original_recognition_id": original_recognition_id,
            "correction_source":       correction_source,
            "notes":                   notes,
        }
        vid = self.faiss_index.add(embedding, metadata)

        # [Fix-2] 强制写盘
        self._save_index(force=True)

        return {
            "success":               True,
            "reagent_id":            corrected_reagent_id,
            "reagent_name":          corrected_reagent_name,
            "vector_id":             vid,
            "corrected_image_path":  corrected_image_path,
            "message": f"纠错已应用，试剂 {corrected_reagent_id} 的特征向量已添加到索引",
        }

    def get_correction_statistics(self) -> Dict:
        total_vectors = self.faiss_index.total
        correction_vectors = [
            m for m in self.faiss_index.id_map if m.get("is_correction", False)
        ]
        unique_corrected_ids = set(m["reagent_id"] for m in correction_vectors)
        correction_sources: Dict[str, int] = {}
        for m in correction_vectors:
            source = m.get("correction_source", "unknown")
            correction_sources[source] = correction_sources.get(source, 0) + 1

        return {
            "total_vectors":             total_vectors,
            "correction_count":          len(correction_vectors),
            "correction_ratio":          (
                f"{len(correction_vectors) / total_vectors * 100:.2f}%"
                if total_vectors > 0 else "0%"
            ),
            "unique_corrected_reagents": len(unique_corrected_ids),
            "correction_sources":        correction_sources,
        }

    def export_corrections_for_training(
        self,
        output_dir: str = None,
        include_original: bool = False,
    ) -> Dict:
        if output_dir is None:
            output_dir = Path("data/corrections")
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        correction_vectors = [
            m for m in self.faiss_index.id_map if m.get("is_correction", False)
        ]
        if not correction_vectors:
            return {"success": False, "exported_count": 0, "message": "没有纠错样本可导出"}

        exported_count = 0
        exported_reagents = set()

        for metadata in correction_vectors:
            reagent_id  = metadata["reagent_id"]
            image_path  = metadata.get("image_path", "")
            if not image_path or not Path(image_path).exists():
                continue
            reagent_dir = output_dir / reagent_id
            reagent_dir.mkdir(parents=True, exist_ok=True)
            dest_path = reagent_dir / Path(image_path).name
            shutil.copy(image_path, dest_path)
            exported_count += 1
            exported_reagents.add(reagent_id)

        report_path = output_dir / "export_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "export_time":       datetime.now().isoformat(),
                "exported_count":    exported_count,
                "exported_reagents": list(exported_reagents),
                "output_directory":  str(output_dir),
            }, f, ensure_ascii=False, indent=2)

        return {
            "success":           True,
            "exported_count":    exported_count,
            "exported_reagents": len(exported_reagents),
            "output_directory":  str(output_dir),
            "message":           f"已导出 {exported_count} 个纠错样本到 {output_dir}",
        }

    def verify_correction_quality(self, reagent_id: str, min_samples: int = 3) -> Dict:
        all_vectors = [m for m in self.faiss_index.id_map if m["reagent_id"] == reagent_id]
        correction_vectors = [m for m in all_vectors if m.get("is_correction", False)]
        return {
            "reagent_id":         reagent_id,
            "total_samples":      len(all_vectors),
            "correction_samples": len(correction_vectors),
            "meets_minimum":      len(correction_vectors) >= min_samples,
            "correction_ratio": (
                f"{len(correction_vectors) / len(all_vectors) * 100:.2f}%"
                if all_vectors else "0%"
            ),
            "ready_for_retraining": len(correction_vectors) >= min_samples,
        }


# ── 单例 ──────────────────────────────────────────────────────────────────────
_engine_instance: Optional[ReagentRecognitionEngine] = None


def get_engine() -> ReagentRecognitionEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ReagentRecognitionEngine()
    return _engine_instance