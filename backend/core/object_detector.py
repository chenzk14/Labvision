"""
backend/core/object_detector.py

"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.config import DETECTION_CONFIG

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    warnings.warn(
        "ultralytics 未安装，多物体识别功能不可用。"
        "请运行: pip install ultralytics"
    )


class ObjectDetector:

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ):
        self.model_name           = model_name           if model_name           is not None else DETECTION_CONFIG["model_name"]
        self.device               = device               if device               is not None else DETECTION_CONFIG["device"]
        # [Fix-1] 用 is not None 而不是 or，避免 0.0 被当作 falsy
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else DETECTION_CONFIG["confidence_threshold"]
        self.iou_threshold        = iou_threshold        if iou_threshold        is not None else DETECTION_CONFIG["iou_threshold"]
        self.model = None

        if YOLO_AVAILABLE:
            self._load_model()
        else:
            warnings.warn("YOLO 模型未加载，请先安装 ultralytics")

    def _load_model(self):
        try:
            self.model = YOLO(self.model_name)
            if self.device != "auto":
                self.model.to(self.device)
            print(f"[ObjectDetector] 加载模型: {self.model_name}")
        except FileNotFoundError:
            print(f"[ObjectDetector] 模型文件不存在: {self.model_name}")
            print(f"[ObjectDetector] 请手动下载 {self.model_name} 并放到项目根目录")
            self.model = None
        except Exception as e:
            print(f"[ObjectDetector] 加载模型失败: {e}")
            self.model = None

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_det: int = 100,
    ) -> List[Dict]:
        if self.model is None:
            return []

        # [Fix-1] is not None 判断
        conf_th = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        iou_th  = iou_threshold        if iou_threshold        is not None else self.iou_threshold

        try:
            results = self.model(image, conf=conf_th, iou=iou_th, max_det=max_det, verbose=False)
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
                class_id   = int(box.cls[0].cpu().numpy())
                class_name = result.names.get(class_id, "unknown")
                detections.append({
                    "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": confidence,
                    "class_id":   class_id,
                    "class_name": class_name,
                })
        return detections

    def detect_and_crop(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
        padding: int = 10,
    ) -> List[Tuple[np.ndarray, Dict]]:
        detections = self.detect(image, confidence_threshold)
        results = []
        h, w = image.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                results.append((crop, det))
        return results

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        show_confidence: bool = True,
        show_class: bool = True,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        img = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            confidence = det["confidence"]
            class_name = det["class_name"]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            label_parts = []
            if show_class:
                label_parts.append(class_name)
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            if label_parts:
                label = " ".join(label_parts)
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img

    def get_stats(self) -> Dict:
        return {
            "model_name":           self.model_name,
            "device":               self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold":        self.iou_threshold,
            "available":            YOLO_AVAILABLE,
            "model_loaded":         self.model is not None,
        }


_detector_instance: Optional[ObjectDetector] = None


def get_detector() -> ObjectDetector:
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
    return ObjectDetector(
        model_name=model_name,
        device=device,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )