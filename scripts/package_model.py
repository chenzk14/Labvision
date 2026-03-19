# C:\InsightFaceFAISS\FSFGIC\scripts\package_model.py
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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import MODEL_CONFIG, INFERENCE_CONFIG, DEVICE, DETECTION_CONFIG, TRAIN_CONFIG


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
        dirs = ["models", "embeddings", "config", "backend"]
        for d in dirs:
            (self.output_path / d).mkdir(parents=True, exist_ok=True)

    def _copy_model_files(self) -> None:
        """复制模型文件和backend模块"""
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
        
        self._copy_backend_module()
    
    def _copy_backend_module(self) -> None:
        """复制backend模块到部署包"""
        backend_src = self.base_dir / "backend"
        backend_dst = self.output_path / "backend"
        
        if not backend_src.exists():
            print(f"  ⚠ backend 目录不存在")
            return
        
        try:
            if backend_dst.exists():
                shutil.rmtree(backend_dst)
            shutil.copytree(backend_src, backend_dst, 
                           ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
            print(f"  ✓ backend 模块已复制")
        except Exception as e:
            print(f"  ⚠ 复制 backend 模块失败: {e}")

    def _generate_config(self) -> None:
        """生成配置文件"""
        model_config = MODEL_CONFIG.copy()
        
        config_data = {
            "model_config": model_config,
            "inference_config": INFERENCE_CONFIG,
            "train_config": TRAIN_CONFIG,
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

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
import faiss

sys.path.insert(0, str(Path(__file__).parent))

try:
    from backend.core.recognition_engine import ReagentRecognitionEngine as BackendEngine
    from backend.core.object_detector import get_detector
    from backend.config import INFERENCE_CONFIG, MODEL_CONFIG, DEVICE
except ImportError:
    print("[ERROR] 无法导入 backend 模块，请确保已安装所有依赖")
    print("[ERROR] 运行: pip install -r requirements.txt")
    sys.exit(1)


class ReagentRecognitionEngine:
    """试剂识别引擎 - 2026现代化版本，支持多物体识别"""

    def __init__(
        self,
        model_path: Path = None,
        index_path: Path = None,
        metadata_path: Path = None,
        img_size: int = None,
        device: str = None,
    ):
        self.img_size = img_size or MODEL_CONFIG["img_size"]
        self.device = device or DEVICE
        script_dir = Path(__file__).parent

        model_path = model_path or script_dir / "models" / "best_model.pth"
        index_path = index_path or script_dir / "embeddings" / "reagent.index"
        metadata_path = metadata_path or script_dir / "embeddings" / "metadata.json"

        config_path = script_dir / "config" / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.threshold = config.get("inference_config", {}).get("similarity_threshold", 0.7)
            embedding_dim = config.get("model_config", {}).get("embedding_dim", 256)
        else:
            self.threshold = INFERENCE_CONFIG.get("similarity_threshold", 0.7)
            embedding_dim = MODEL_CONFIG.get("embedding_dim", 256)

        checkpoint = torch.load(str(model_path), map_location=self.device)
        print(f"[Engine] 加载模型: {model_path}")

        class_mapping_path = script_dir / "config" / "class_mapping.json"
        if class_mapping_path.exists():
            with open(class_mapping_path, "r", encoding="utf-8") as f:
                class_mapping = json.load(f)
            num_classes = len(class_mapping["class_to_idx"])
        else:
            num_classes = 100

        from backend.models.metric_model import ReagentRecognitionModel
        self.model = ReagentRecognitionModel(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            pretrained=False,
            arcface_margin=checkpoint.get("arcface_margin", 0.35),
            arcface_scale=checkpoint.get("arcface_scale", 60.0),
        ).to(self.device)

        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
        new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()

        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)

        from backend.core.dataset import get_val_transforms
        self.transform = get_val_transforms(self.img_size)

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
                from PIL import Image
                img = np.array(Image.open(str(image_input)).convert('RGB'))
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            img = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) if image_input.shape[2] == 3 else image_input
        elif hasattr(image_input, 'convert'):
            img = np.array(image_input.convert("RGB"))
        else:
            raise ValueError(f"不支持的图像类型: {type(image_input)}")

        augmented = self.transform(image=img)
        return augmented['image'].unsqueeze(0).to(self.device)

    @torch.no_grad()
    def extract_embedding(self, image_input) -> np.ndarray:
        """提取嵌入向量"""
        tensor = self._preprocess_image(image_input)
        embeddings = self.model.embedder(tensor)
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
# 根据项目实际依赖生成

# PyTorch (根据CUDA版本选择)
# CUDA 11.8
--extra-index-url https://download.pytorch.org/whl/cu118
torch>=2.0.0
torchvision>=0.15.0

# 或者使用 CPU 版本
# torch>=2.0.0
# torchvision>=0.15.0

# 核心依赖
numpy>=1.24.0
opencv-python>=4.8.0
pillow>=10.0.0
albumentations>=1.3.0
timm>=0.9.0
faiss-cpu>=1.7.4

# 机器学习
scikit-learn>=1.3.0

# 目标检测
ultralytics>=8.0.0

# 数据库 (可选)
# sqlalchemy>=2.0.0
# aiosqlite>=0.19.0

# 可视化 (可选)
matplotlib>=3.7.0
seaborn>=0.12.0

# Web API (可选)
# fastapi>=0.100.0
# uvicorn>=0.23.0
# python-multipart>=0.0.6
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
- `backend/` - 后端模块（包含核心识别逻辑）

## 系统要求

- Python 3.8+
- CUDA 11.8+ (如果使用GPU)
- 4GB+ 显存 (推荐)

## 快速开始

### 1. 安装依赖

```bash
# 使用 CUDA 11.8
python -m pip install -r requirements.txt

# 如果使用 CPU，先注释掉 requirements.txt 中的 CUDA 相关行
# python -m pip install -r requirements.txt
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

### 指定模型目录

```bash
python inference.py --image_path path/to/image.jpg --model_dir /path/to/model/dir
```

## 技术架构

- **特征提取**: EfficientNetV2-S
- **损失函数**: ArcFace Loss
- **向量检索**: FAISS (IndexFlatIP)
- **目标检测**: YOLOv11 (多物体识别)
- **图像尺寸**: 288x288
- **特征维度**: 256

## 注意事项

1. **图像质量**: 确保图片清晰，试剂特征明显
2. **光照条件**: 光照条件良好，避免过暗或过亮
3. **多物体识别**: 需要安装ultralytics: `pip install ultralytics`
4. **识别率调整**: 如果识别率低，可以降低相似度阈值
5. **GPU支持**: 推荐使用GPU加速推理

## 故障排除

### 导入错误

如果遇到 `ModuleNotFoundError`，请确保已安装所有依赖：

```bash
pip install -r requirements.txt
```

### CUDA 错误

如果遇到 CUDA 相关错误，请检查：
1. CUDA 版本是否正确
2. PyTorch 是否支持 CUDA
3. 显存是否足够

### 模型加载失败

如果模型加载失败，请检查：
1. 模型文件是否存在
2. 模型文件是否损坏
3. 索引文件是否完整

## 性能优化

- 使用 GPU 加速推理
- 批量处理图像
- 调整图像尺寸
- 使用量化模型

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
        print(f"  模型类型: {MODEL_CONFIG.get('model_type', 'unknown')}")
        print(f"  主干网络: {MODEL_CONFIG.get('backbone', 'unknown')}")
        print(f"  特征维度: {MODEL_CONFIG.get('embedding_dim', 256)}")
        print(f"\n下一步:")
        print(f"  1. 将 {self.output_path.name} 目录复制到目标机器")
        print(f"  2. 安装依赖: pip install -r requirements.txt")
        print(f"  3. 运行推理: python {self.output_path.name}/inference.py --image_path path/to/image.jpg")
        print(f"  4. 多物体识别: python {self.output_path.name}/inference.py --image_path path/to/image --multiple")
        print(f"\n注意事项:")
        print(f"  - 推理脚本依赖 backend 模块，请确保完整复制项目结构")
        print(f"  - 如果只需要推理功能，可以只复制必要的文件")


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