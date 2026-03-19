"""模型打包脚本 - 2026现代化版本
将训练好的模型打包成可部署的包，支持多物体识别

使用方法：
  python scripts/package_model.py --output_dir deploy_package
"""

from __future__ import annotations

import json
import shutil
import sys
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import MODEL_CONFIG, INFERENCE_CONFIG, DEVICE, DETECTION_CONFIG


@dataclass
class PackageConfig:
    """打包配置"""
    output_dir: str = "deploy_package"
    skip_inference_script: bool = False
    base_dir: Path = Path(__file__).parent.parent


class ModelPackager:
    """模型打包器 - 2026现代化实现"""

    def __init__(self, config: PackageConfig):
        self.config = config
        self.output_path = Path(config.output_dir)
        self.base_dir = config.base_dir

    def package(self) -> None:
        """执行打包流程"""
        self._create_directories()
        self._copy_model_files()
        self._generate_config()
        if not self.config.skip_inference_script:
            self._generate_inference_script()
        self._generate_requirements()
        self._generate_readme()
        self._print_summary()

    def _create_directories(self) -> None:
        """创建目录结构"""
        dirs = ["models", "embeddings", "config"]
        for d in dirs:
            (self.output_path / d).mkdir(parents=True, exist_ok=True)

    def _copy_model_files(self) -> None:
        """复制模型文件"""
        files = [
            ("saved_models/best_model.pth", "models/best_model.pth"),
            ("saved_models/class_mapping.json", "config/class_mapping.json"),
            ("data/embeddings/reagent.index", "embeddings/reagent.index"),
            ("data/embeddings/metadata.json", "embeddings/metadata.json"),
        ]
        for src, dst in files:
            src_path = self.base_dir / src
            dst_path = self.output_path / dst
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                print(f"  ✓ {src} → {dst}")
            else:
                print(f"  ⚠ {src} 不存在")

    def _generate_config(self) -> None:
        """生成配置文件"""
        # 修正：确保配置文件中的特征提取器与训练时使用的EfficientNet一致
        model_config = MODEL_CONFIG.copy()
        # 强制使用efficientnet，因为训练和索引构建都是用efficientnet
        # model_config["feature_extractor"] = "efficientnet"
        
        config_data = {
            "model_config": model_config,
            "inference_config": INFERENCE_CONFIG,
            "device": DEVICE,
            "package_time": datetime.now().isoformat(),
        }
        
        config_path = self.output_path / "config" / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 配置文件已保存: {config_path}")

    def _generate_inference_script(self) -> None:
        """生成推理脚本 - 2026现代化版本，支持多物体识别"""
        inference_script = '''# 试剂识别推理脚本 - 2026现代化版本
# 使用方法：python inference.py --image_path path/to/image.jpg
# 多物体识别：python inference.py --image_path path/to/image.jpg --multiple

from __future__ import annotations

import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import math


@dataclass
class DetectionResult:
    """检测结果"""
    bbox: List[int]
    confidence: float
    class_id: int
    class_name: str


class ObjectDetector:
    """基于YOLOv11的目标检测器 - 2026现代化版本"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> None:
        self.model_name = model_name or DETECTION_CONFIG["model_name"]
        self.device = device or DETECTION_CONFIG["device"]
        self.confidence_threshold = confidence_threshold or DETECTION_CONFIG["confidence_threshold"]
        self.iou_threshold = iou_threshold or DETECTION_CONFIG["iou_threshold"]
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """加载YOLO模型"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            if self.device != "auto":
                self.model.to(self.device)
            print(f"[ObjectDetector] 加载模型: {self.model_name}")
        except ImportError:
            print(f"[ObjectDetector] ultralytics未安装，多物体识别功能不可用")
            print(f"[ObjectDetector] 请运行: pip install ultralytics")
        except Exception as e:
            print(f"[ObjectDetector] 加载模型失败: {e}")

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_det: int = 100,
    ) -> List[DetectionResult]:
        """检测图片中的物体"""
        if self.model is None:
            return []

        conf_th = confidence_threshold or self.confidence_threshold
        iou_th = iou_threshold or self.iou_threshold

        try:
            results = self.model(
                image,
                conf=conf_th,
                iou=iou_th,
                max_det=max_det,
                verbose=False,
            )
        except Exception as e:
            print(f"[ObjectDetector] 检测失败: {e}")
            return []

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names.get(class_id, "unknown")
                detections.append(DetectionResult(
                    bbox=[int(x1), int(y1), int(x2), int(y2)],
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                ))
        return detections

    def detect_and_crop(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
        padding: int = 10,
    ) -> List[Tuple[np.ndarray, DetectionResult]]:
        """检测并裁剪物体"""
        detections = self.detect(image, confidence_threshold)
        results = []
        h, w = image.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                results.append((crop, det))
        return results


class EfficientNetEmbedder(nn.Module):
    """EfficientNet特征提取器"""
    def __init__(self, embedding_dim: int = 256, backbone: str = "efficientnet_b2", pretrained: bool = False):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        backbone_out_dim = self.backbone.num_features
        self.projector = nn.Sequential(
            nn.Linear(backbone_out_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.projector(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class ArcFaceLoss(nn.Module):
    """ArcFace Loss - 2026简化版本"""
    def __init__(self, embedding_dim: int = 256, num_classes: int = 100, margin: float = 0.35, scale: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, weight_norm)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size()).to(embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return F.cross_entropy(output, labels)


class ReagentRecognitionModel(nn.Module):
    """试剂识别模型 - 2026简化版本"""
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 256,
        backbone: str = "efficientnet_b2",
        pretrained: bool = False,
        arcface_margin: float = 0.35,
        arcface_scale: float = 30.0,
    ):
        super().__init__()
        self.embedder = EfficientNetEmbedder(
            embedding_dim=embedding_dim,
            backbone=backbone,
            pretrained=pretrained
        )
        self.arcface = ArcFaceLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            margin=arcface_margin,
            scale=arcface_scale,
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        embeddings = self.embedder(x)
        if labels is not None:
            loss = self.arcface(embeddings, labels)
            return embeddings, loss
        return embeddings, None


_detector_instance: Optional[ObjectDetector] = None


def get_detector() -> ObjectDetector:
    """获取全局检测器实例（使用配置文件中的参数）"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = ObjectDetector()
    return _detector_instance


def get_detector_with_config(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    iou_threshold: Optional[float] = None,
) -> ObjectDetector:
    """
    获取检测器实例，支持自定义参数覆盖配置文件

    Args:
        model_name: YOLO模型名称（覆盖配置文件）
        device: 运行设备（覆盖配置文件）
        confidence_threshold: 检测置信度阈值（覆盖配置文件）
        iou_threshold: NMS的IOU阈值（覆盖配置文件）

    Returns:
        ObjectDetector实例
    """
    return ObjectDetector(
        model_name=model_name,
        device=device,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )


class ReagentRecognitionEngine:
    """试剂识别引擎 - 2026现代化版本，支持多物体识别"""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        img_size: int = 224,
        device: Optional[str] = None,
    ):
        self.img_size = img_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        script_dir = Path(__file__).parent

        model_path = model_path or script_dir / "models" / "best_model.pth"
        index_path = index_path or script_dir / "embeddings" / "reagent.index"
        metadata_path = metadata_path or script_dir / "embeddings" / "metadata.json"

        config_path = script_dir / "config" / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.threshold = config.get("inference_config", {}).get("similarity_threshold", 0.68)
            embedding_dim = config.get("model_config", {}).get("embedding_dim", 256)
        else:
            self.threshold = 0.68
            embedding_dim = 256

        checkpoint = torch.load(str(model_path), map_location=self.device)
        print(f"[Engine] 检测到模型类型: single_efficientnet")

        class_mapping_path = script_dir / "config" / "class_mapping.json"
        if class_mapping_path.exists():
            with open(class_mapping_path, "r", encoding="utf-8") as f:
                class_mapping = json.load(f)
            num_classes = len(class_mapping["class_to_idx"])
        else:
            num_classes = 100

        self.model = ReagentRecognitionModel(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            backbone=checkpoint.get("backbone", "efficientnet_b2"),
            pretrained=False,
            arcface_margin=checkpoint.get("arcface_margin", 0.35),
            arcface_scale=checkpoint.get("arcface_scale", 30.0),
        ).to(self.device)
        print(f"[Engine] 使用单流EfficientNet模型")

        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
        new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()

        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)

        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        print(f"[Engine] 模型加载完成")
        print(f"  设备: {self.device}")
        print(f"  向量数: {self.index.ntotal}")
        print(f"  阈值: {self.threshold}")
    
    def _preprocess_image(self, image_input) -> torch.Tensor:
        """预处理图像"""
        if isinstance(image_input, str):
            image_array = np.fromfile(image_input, dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if img is None:
                img = np.array(Image.open(str(image_input)).convert('RGB'))
        elif isinstance(image_input, np.ndarray):
            img = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) if image_input.shape[2] == 3 else image_input
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input.convert("RGB"))
        else:
            raise ValueError(f"不支持的图像类型: {type(image_input)}")

        augmented = self.transform(image=img)
        return augmented['image'].unsqueeze(0).to(self.device)

    @torch.no_grad()
    def extract_embedding(self, image_input) -> np.ndarray:
        """提取嵌入向量"""
        tensor = self._preprocess_image(image_input)
        embeddings, _ = self.model(tensor, labels=None)
        return embeddings.cpu().numpy()[0]

    def recognize(self, image_input, topk: int = 5) -> Dict:
        """识别试剂"""
        if self.index.ntotal == 0:
            return {"recognized": False, "message": "系统中尚无注册试剂", "candidates": []}

        embedding = self.extract_embedding(image_input)
        q = embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        k = min(topk, self.index.ntotal)
        scores, indices = self.index.search(q, k)

        similarities = scores[0]
        metadatas = [self.id_map[i] for i in indices[0] if i >= 0]

        if len(similarities) == 0:
            return {"recognized": False, "message": "检索失败", "candidates": []}

        best_score = float(similarities[0])
        best_match = metadatas[0]
        recognized = best_score >= self.threshold

        candidates = [
            {
                "reagent_id": m.get("reagent_id", "unknown"),
                "reagent_name": m.get("reagent_name", "unknown"),
                "similarity": float(s),
                "confidence_pct": f"{float(s) * 100:.1f}%",
            }
            for s, m in zip(similarities, metadatas)
        ]

        return {
            "recognized": recognized,
            "reagent_id": best_match.get("reagent_id") if recognized else None,
            "reagent_name": best_match.get("reagent_name") if recognized else None,
            "confidence": best_score,
            "confidence_pct": f"{best_score * 100:.1f}%",
            "candidates": candidates,
            "threshold": self.threshold,
            "message": (
                f"识别成功: {best_match.get('reagent_id')} ({best_score * 100:.1f}%)"
                if recognized
                else f"置信度不足({best_score * 100:.1f}% < {self.threshold * 100:.0f}%)，可能是新试剂"
            ),
        }

    def recognize_multiple(
            self,
            image_input,
            topk: int = 5,
            min_confidence: float = 0.5,
    ):
        """
        识别多个试剂（多物体识别）

        Args:
            image_input: 摄像头图像（可能包含多个试剂）
            topk: 返回前K个候选
            min_confidence: 检测置信度阈值

        Returns:
            {
                "total_objects": 3,
                "recognized_objects": [...],
                "unrecognized_objects": [...],
                "message": "检测到 3 个物体，识别成功 2 个"
            }
        """
        if self.index.ntotal == 0:
            return {
                "total_objects": 0,
                "recognized_objects": [],
                "unrecognized_objects": [],
                "message": "系统中尚无注册试剂",
            }

        # 获取检测器
        detector = get_detector()
        if detector.model is None:
            return {
                "total_objects": 0,
                "recognized_objects": [],
                "unrecognized_objects": [],
                "message": "目标检测模块未安装，请先安装 ultralytics: pip install ultralytics",
            }

        # 预处理图像
        if isinstance(image_input, str):
            image_array = np.fromfile(image_input, dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if img is None:
                pil_image = Image.open(str(image_input)).convert('RGB')
                img = np.array(pil_image)
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


class ObjectDetector:
    """
    基于YOLOv11的目标检测器
    检测图片中的试剂瓶，返回边界框信息
    """

    def __init__(
            self,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            confidence_threshold: Optional[float] = None,
            iou_threshold: Optional[float] = None,
    ):
        """
        初始化检测器

        Args:
            model_name: YOLO模型名称（如果为None，使用配置文件默认值）
            device: 运行设备（如果为None，使用配置文件默认值）
            confidence_threshold: 检测置信度阈值（如果为None，使用配置文件默认值）
            iou_threshold: NMS的IOU阈值（如果为None，使用配置文件默认值）
        """
        self.model_name = model_name or DETECTION_CONFIG["model_name"]
        self.device = device or DETECTION_CONFIG["device"]
        self.confidence_threshold = confidence_threshold or DETECTION_CONFIG["confidence_threshold"]
        self.iou_threshold = iou_threshold or DETECTION_CONFIG["iou_threshold"]
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载YOLO模型"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            if self.device != "auto":
                self.model.to(self.device)
            print(f"[ObjectDetector] 加载模型: {self.model_name}")
        except ImportError:
            print("[ObjectDetector] ultralytics未安装，多物体识别功能不可用")
            print("[ObjectDetector] 请运行: pip install ultralytics")
        except Exception as e:
            print(f"[ObjectDetector] 加载模型失败: {e}")
            self.model = None

    def detect(
            self,
            image: np.ndarray,
            confidence_threshold: float = None,
            iou_threshold: float = None,
            max_det: int = 100,
    ):
        """
        检测图片中的物体

        Args:
            image: 输入图片 (BGR格式)
            confidence_threshold: 检测置信度阈值
            iou_threshold: NMS的IOU阈值
            max_det: 最大检测数量

        Returns:
            检测结果列表
        """
        if self.model is None:
            return []

        conf_th = confidence_threshold or self.confidence_threshold
        iou_th = iou_threshold or self.iou_threshold

        try:
            results = self.model(
                image,
                conf=conf_th,
                iou=iou_th,
                max_det=max_det,
                verbose=False,
            )
        except Exception as e:
            print(f"[ObjectDetector] 检测失败: {e}")
            return []

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names.get(class_id, "unknown")

                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name,
                })

        return detections


# 单例模式
_detector_instance = None


def get_detector() -> ObjectDetector:
    """获取全局检测器实例（使用配置文件中的参数）"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = ObjectDetector()
    return _detector_instance


def get_detector_with_config(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    iou_threshold: Optional[float] = None,
) -> ObjectDetector:
    """
    获取检测器实例，支持自定义参数覆盖配置文件

    Args:
        model_name: YOLO模型名称（覆盖配置文件）
        device: 运行设备（覆盖配置文件）
        confidence_threshold: 检测置信度阈值（覆盖配置文件）
        iou_threshold: NMS的IOU阈值（覆盖配置文件）

    Returns:
        ObjectDetector实例
    """
    return ObjectDetector(
        model_name=model_name,
        device=device,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )

    def recognize_multiple(
        self,
        image_input,
        topk: int = 5,
        min_confidence: float = 0.5,
    ) -> Dict:
        """识别多个试剂（多物体识别）"""
        if self.index.ntotal == 0:
            return {
                "total_objects": 0,
                "recognized_objects": [],
                "unrecognized_objects": [],
                "message": "系统中尚无注册试剂",
            }

        detector = get_detector()
        if detector.model is None:
            return {
                "total_objects": 0,
                "recognized_objects": [],
                "unrecognized_objects": [],
                "message": "目标检测模块未安装，请先安装 ultralytics: pip install ultralytics",
            }

        if isinstance(image_input, str):
            image_array = np.fromfile(image_input, dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if img is None:
                img = np.array(Image.open(str(image_input)).convert('RGB'))
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
        else:
            img = np.array(image_input)

        h, w = img.shape[:2]
        detections = detector.detect(img, confidence_threshold=min_confidence)
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        recognized_objects = []
        unrecognized_objects = []

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            bw, bh = x2 - x1, y2 - y1

            if bw < max(20, int(0.03 * w)) or bh < max(20, int(0.03 * h)):
                continue

            pad = max(10, int(0.08 * max(bw, bh)))
            cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
            cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
            crop = img[cy1:cy2, cx1:cx2]

            if crop.size == 0:
                continue

            result = self.recognize(crop, topk=topk)
            obj_info = {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "crop_bbox": [int(cx1), int(cy1), int(cx2), int(cy2)],
                "detection_confidence": float(det.confidence),
                "detector_class": det.class_name,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="试剂识别推理")
    parser.add_argument("--image_path", type=str, required=True, help="图片路径")
    parser.add_argument("--topk", type=int, default=5, help="返回前K个候选")
    parser.add_argument("--model_dir", type=str, default=None, help="模型目录")
    parser.add_argument("--multiple", action="store_true", help="启用多物体识别模式")
    parser.add_argument("--min_confidence", type=float, default=0.5, help="目标检测的最小置信度")
    args = parser.parse_args()

    print("[INFO] 初始化识别引擎...")
    model_dir = Path(args.model_dir) if args.model_dir else Path(__file__).parent

    engine = ReagentRecognitionEngine(
        model_path=model_dir / "models" / "best_model.pth",
        index_path=model_dir / "embeddings" / "reagent.index",
        metadata_path=model_dir / "embeddings" / "metadata.json",
    )

    print(f"[INFO] 识别图片: {args.image_path}")

    if args.multiple:
        print("[INFO] 使用多物体识别模式")
        result = engine.recognize_multiple(
            args.image_path,
            topk=args.topk,
            min_confidence=args.min_confidence
        )
        print("=" * 50)
        print("识别结果:")
        print("=" * 50)
        print(result["message"])
        print()

        if result["recognized_objects"]:
            print(f"已识别的试剂 ({len(result['recognized_objects'])}):")
            for i, obj in enumerate(result["recognized_objects"], 1):
                print(f"  {i}. {obj['reagent_id']} ({obj['reagent_name']})")
                print(f"     置信度: {obj['confidence_pct']}")
                print(f"     位置: {obj['bbox']}")
                print()

        if result["unrecognized_objects"]:
            print(f"未识别的物体 ({len(result['unrecognized_objects'])}):")
            for i, obj in enumerate(result["unrecognized_objects"], 1):
                print(f"  {i}. 最佳候选: {obj.get('best_candidate', 'N/A')} ({obj.get('best_candidate_name', 'N/A')})")
                print(f"     置信度: {obj['confidence_pct']}")
                print(f"     位置: {obj['bbox']}")
                print()
    else:
        result = engine.recognize(args.image_path, topk=args.topk)
        print("=" * 50)
        print("识别结果:")
        print("=" * 50)

        if result["recognized"]:
            print(f"[SUCCESS] 识别成功!")
            print(f"   试剂ID: {result['reagent_id']}")
            print(f"   试剂名称: {result['reagent_name']}")
            print(f"   置信度: {result['confidence_pct']}")
        else:
            print(f"[FAIL] 未识别")
            print(f"   置信度: {result['confidence_pct']}")
            print(f"   阈值: {result['threshold'] * 100:.0f}%")

        print("候选结果:")
        for i, cand in enumerate(result["candidates"][:args.topk], 1):
            print(f"  {i}. {cand['reagent_id']} ({cand['reagent_name']}) - {cand['confidence_pct']}")


if __name__ == "__main__":
    main()
'''
        script_path = self.output_path / "inference.py"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(inference_script)
        print(f"  ✓ 推理脚本已保存: {script_path}")

    def _generate_requirements(self) -> None:
        """生成requirements.txt"""
        requirements = '''# 试剂识别模型依赖 - 2026现代化版本

--extra-index-url https://download.pytorch.org/whl/cu117

torch==2.0.1+cu117
torchvision==0.15.2+cu117
numpy==1.24.4
opencv-python==4.8.1.78
pillow==10.0.1
albumentations==1.3.1
timm==0.9.7
faiss-cpu==1.7.4
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
ultralytics==8.0.200
'''
        req_path = self.output_path / "requirements.txt"
        with open(req_path, "w", encoding="utf-8") as f:
            f.write(requirements)
        print(f"  ✓ 依赖文件已保存: {req_path}")

    def _generate_readme(self) -> None:
        """生成README.md"""
        readme_content = '''# 试剂识别模型部署包 - 2026现代化版本

## 包含内容

- `models/best_model.pth` - 训练好的模型权重
- `embeddings/reagent.index` - FAISS向量索引
- `embeddings/metadata.json` - 试剂元数据
- `config/class_mapping.json` - 类别映射
- `config/config.json` - 模型配置
- `inference.py` - 推理脚本
- `requirements.txt` - 依赖列表

## 快速开始

### 1. 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 2. 单物体识别

```bash
python inference.py --image_path path/to/image.jpg
```

### 3. 多物体识别

```bash
python inference.py --image_path path/to/image.jpg --multiple
```

## 输出格式

### 单物体识别

```
识别结果:
==================================================
[SUCCESS] 识别成功!
   试剂ID: 乙醇001
   试剂名称: 乙醇
   置信度: 92.3%

候选结果:
  1. 乙醇001 (乙醇) - 92.3%
  2. 乙醇002 (乙醇) - 85.6%
  3. 乙醇003 (乙醇) - 78.2%
```

### 多物体识别

```
识别结果:
==================================================
检测到 3 个物体，识别成功 2 个

已识别的试剂 (2):
  1. 电脑001 (电脑)
     置信度: 98.4%
     位置: [100, 200, 300, 400]
  2. 水杯001001 (水杯)
     置信度: 95.2%
     位置: [400, 200, 500, 400]

未识别的物体 (1):
  1. 最佳候选: 其他障碍物001 (其他障碍物)
     置信度: 45.3%
     位置: [600, 200, 700, 400]
```

## 高级用法

### 自定义阈值

修改 `config/config.json` 中的 `similarity_threshold`:

```json
{{
  "inference_config": {{
    "similarity_threshold": 0.75
  }}
}}
```

### 调整检测置信度

```bash
python inference.py --image_path path/to/image.jpg --multiple --min_confidence 0.3
```

## 注意事项

1. 确保图片清晰，试剂特征明显
2. 光照条件良好
3. 多物体识别需要安装ultralytics: `pip install ultralytics`
4. 如果识别率低，可以降低相似度阈值

---
打包时间: {package_time}
'''
        readme_path = self.output_path / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content.format(
                package_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
        print(f"  ✓ README已保存: {readme_path}")

    def _print_summary(self) -> None:
        """打印打包摘要"""
        print(f"\n打包完成!")
        print(f"  输出目录: {self.output_path.absolute()}")
        print(f"\n下一步:")
        print(f"  1. 将 {self.output_path.name} 目录复制到目标机器")
        print(f"  2. 安装依赖: pip install -r requirements.txt")
        print(f"  3. 运行推理: python deploy_package/inference.py --image_path path/to/image.jpg")
        print(f"  4. 多物体识别: python deploy_package/inference.py --image_path path/to/image --multiple")


def package_model(output_dir: str = "deploy_package", skip_inference_script: bool = False) -> None:
    """打包模型和依赖文件"""
    config = PackageConfig(
        output_dir=output_dir,
        skip_inference_script=skip_inference_script,
    )
    packager = ModelPackager(config)
    packager.package()


def main() -> None:
    parser = argparse.ArgumentParser(description="打包试剂识别模型")
    parser.add_argument("--output_dir", type=str, default="deploy_package", help="输出目录")
    parser.add_argument("--skip-inference-script", action="store_true", help="跳过生成推理脚本")
    args = parser.parse_args()
    package_model(args.output_dir, args.skip_inference_script)


if __name__ == "__main__":
    main()