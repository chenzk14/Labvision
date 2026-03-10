# backend/core/object_detector.py
"""
目标检测模块 - 用于多物体识别
使用YOLOv8检测图片中的多个试剂瓶

功能：
1. 检测图片中的所有试剂瓶
2. 返回每个试剂瓶的边界框
3. 支持自定义检测模型
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    warnings.warn(
        "ultralytics未安装，多物体识别功能不可用。"
        "请运行: pip install ultralytics"
    )


class ObjectDetector:
    """
    基于YOLOv8的目标检测器

    检测图片中的试剂瓶，返回边界框信息
    """

    def __init__(
            self,
            model_name: str = "yolov8n.pt",
            device: str = "auto",
            confidence_threshold: float = 0.5,
            iou_threshold: float = 0.45,
    ):
        """
        初始化检测器

        Args:
            model_name: YOLO模型名称 (yolov8n/yolov8s/yolov8m/yolov8l/yolov8x)
                       n=nan(最小), s=small, m=medium, l=large, x=xlarge
            device: 运行设备 ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
            confidence_threshold: 检测置信度阈值
            iou_threshold: NMS的IOU阈值
        """
        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self.model = None

        if YOLO_AVAILABLE:
            self._load_model()
        else:
            warnings.warn("YOLO模型未加载，请先安装ultralytics")

    def _load_model(self):
        """加载YOLO模型"""
        try:
            self.model = YOLO(self.model_name)
            if self.device != "auto":
                self.model.to(self.device)
            print(f"[ObjectDetector] 加载模型: {self.model_name}")
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
        """
        检测图片中的物体

        Args:
            image: 输入图片 (BGR格式，来自OpenCV)
            confidence_threshold: 检测置信度阈值（覆盖初始化值）
            iou_threshold: NMS的IOU阈值（覆盖初始化值）
            max_det: 最大检测数量

        Returns:
            检测结果列表，每个元素包含：
            {
                "bbox": [x1, y1, x2, y2],  # 边界框坐标
                "confidence": 0.95,         # 检测置信度
                "class_id": 0,              # 类别ID
                "class_name": "bottle",     # 类别名称
            }
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

    def detect_and_crop(
            self,
            image: np.ndarray,
            confidence_threshold: Optional[float] = None,
            padding: int = 10,
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        检测并裁剪物体

        Args:
            image: 输入图片
            confidence_threshold: 检测置信度阈值
            padding: 边界框扩展像素数

        Returns:
            [(cropped_image, detection_info), ...]
        """
        detections = self.detect(image, confidence_threshold)

        results = []
        h, w = image.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            # 添加padding并确保在图片范围内
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            # 裁剪
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
        """
        在图片上绘制检测结果

        Args:
            image: 输入图片
            detections: 检测结果列表
            show_confidence: 是否显示置信度
            show_class: 是否显示类别
            color: 边界框颜色 (B, G, R)
            thickness: 边界框线条粗细

        Returns:
            绘制了检测结果的图片
        """
        img = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            confidence = det["confidence"]
            class_name = det["class_name"]

            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # 绘制标签
            label_parts = []
            if show_class:
                label_parts.append(class_name)
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")

            if label_parts:
                label = " ".join(label_parts)

                # 计算标签背景大小
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )

                # 绘制标签背景
                cv2.rectangle(
                    img,
                    (x1, y1 - text_h - 10),
                    (x1 + text_w, y1),
                    color,
                    -1,
                )

                # 绘制标签文字
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        return img

    def get_stats(self) -> Dict:
        """获取检测器统计信息"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "available": YOLO_AVAILABLE,
            "model_loaded": self.model is not None,
        }


# 单例模式
_detector_instance: Optional[ObjectDetector] = None


def get_detector() -> ObjectDetector:
    """获取全局检测器实例"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = ObjectDetector()
    return _detector_instance


if __name__ == "__main__":
    # 测试代码
    detector = ObjectDetector()
    print("检测器信息:", detector.get_stats())