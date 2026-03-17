# scripts/package_model.py
"""
模型打包脚本
将训练好的模型打包成可部署的包

使用方法：
  python scripts/package_model.py --output_dir deploy_package
"""

import os
import sys
import shutil
import argparse
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import MODEL_CONFIG, INFERENCE_CONFIG, DEVICE


def package_model(output_dir: str = "deploy_package", skip_inference_script: bool = False):
    """
    打包模型和依赖文件

    Args:
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"开始打包模型...")
    print(f"  输出目录: {output_path.absolute()}")

    # 1. 创建目录结构
    dirs_to_create = [
        output_path / "models",
        output_path / "embeddings",
        output_path / "config",
        # output_path / "scripts",
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

    # 2. 复制模型文件
    model_files = [
        ("saved_models/best_model.pth", "models/best_model.pth"),
        ("saved_models/class_mapping.json", "config/class_mapping.json"),
        ("data/embeddings/reagent.index", "embeddings/reagent.index"),
        ("data/embeddings/metadata.json", "embeddings/metadata.json"),
    ]

    for src, dst in model_files:
        src_path = Path(src)
        dst_path = output_path / dst
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"  {src} → {dst}")
        else:
            print(f"  警告: {src} 不存在")

    # 3. 生成配置文件
    config_data = {
        "model_config": MODEL_CONFIG,
        "inference_config": INFERENCE_CONFIG,
        "device": DEVICE,
        "package_time": datetime.now().isoformat(),
    }

    config_path = output_path / "config" / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    print(f"  配置文件已保存: {config_path}")

    # 4. 创建推理脚本（可选）
    if not skip_inference_script:
        inference_script = '''# 试剂识别推理脚本
# 使用方法：python inference.py --image_path path/to/image.jpg

import sys
import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import faiss
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm


class EfficientNetEmbedder(nn.Module):
    """EfficientNet特征提取器"""
    def __init__(self, embedding_dim: int = 512, pretrained: bool = False):
        super().__init__()
        import timm
        
        # 加载EfficientNet-B0主干
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,  # 去掉分类头，只要特征
            global_pool="avg",  # 全局平均池化
        )
        
        # backbone输出维度
        backbone_out_dim = self.backbone.num_features  # 1280
        
        # 投影头: 1280 → 512
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
        # 提取backbone特征
        features = self.backbone(x)  # [B, 1280]
        
        # 投影到512维
        embeddings = self.projector(features)  # [B, 512]
        
        # L2归一化 - ArcFace必须在单位超球面上
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class ReagentRecognitionEngine:
    """试剂识别引擎（独立版）"""
    def __init__(
        self,
        model_path=None,
        index_path=None,
        metadata_path=None,
        img_size=224,
        device=None,
    ):
        self.img_size = img_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 获取脚本所在目录
        script_dir = Path(__file__).parent
        
        # 如果没有提供路径，使用默认相对路径
        if model_path is None:
            model_path = script_dir / "models" / "best_model.pth"
        if index_path is None:
            index_path = script_dir / "embeddings" / "reagent.index"
        if metadata_path is None:
            metadata_path = script_dir / "embeddings" / "metadata.json"
        
        # 加载配置
        config_path = script_dir / "config" / "config.json"
        if config_path.exists():
            with open(str(config_path), "r", encoding="utf-8") as f:
                config = json.load(f)
            self.threshold = config.get("inference_config", {}).get("similarity_threshold", 0.75)
            embedding_dim = config.get("model_config", {}).get("embedding_dim", 512)
        else:
            self.threshold = 0.75
            embedding_dim = 512
        
        # 加载模型
        checkpoint = torch.load(str(model_path), map_location=self.device)
        self.embedder = EfficientNetEmbedder(embedding_dim=embedding_dim, pretrained=False).to(self.device)
        
        # 尝试不同的键名
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            # 直接使用整个检查点
            state_dict = checkpoint
        
        # 移除键名中的 "module." 前缀（如果使用DataParallel训练）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # 加载状态字典
        missing_keys, unexpected_keys = self.embedder.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"[WARNING] 缺少的键: {missing_keys}")
        if unexpected_keys:
            print(f"[WARNING] 意外的键: {unexpected_keys}")
        
        self.embedder.eval()
        
        # 加载FAISS索引
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)
        
        # 数据变换
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
        
        print(f"[Engine] 模型加载完成")
        print(f"  设备: {self.device}")
        print(f"  向量数: {self.index.ntotal}")
        print(f"  阈值: {self.threshold}")
    
    def _preprocess_image(self, image_input):
        """预处理图像"""
        if isinstance(image_input, str):
            image_array = np.fromfile(image_input, dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if img is None:
                pil_image = Image.open(str(image_input)).convert('RGB')
                img = np.array(pil_image)
        elif isinstance(image_input, np.ndarray):
            if image_input.shape[2] == 3:
                img = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            else:
                img = image_input
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input.convert("RGB"))
        else:
            raise ValueError(f"不支持的图像类型: {type(image_input)}")
        
        augmented = self.transform(image=img)
        tensor = augmented['image'].unsqueeze(0).to(self.device)
        return tensor
    
    @torch.no_grad()
    def extract_embedding(self, image_input):
        """提取嵌入向量"""
        tensor = self._preprocess_image(image_input)
        embedding = self.embedder(tensor)
        return embedding.cpu().numpy()[0]
    
    def recognize(self, image_input, topk=5):
        """识别试剂"""
        if self.index.ntotal == 0:
            return {
                "recognized": False,
                "message": "系统中尚无注册试剂",
                "candidates": []
            }
        
        # 提取嵌入
        embedding = self.extract_embedding(image_input)
        
        # FAISS检索
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
        
        # 构建候选列表
        candidates = [
            {
                "reagent_id": m.get("reagent_id", "unknown"),
                "reagent_name": m.get("reagent_name", "unknown"),
                "similarity": float(s),
                "confidence_pct": f"{float(s) * 100:.1f}%",
            }
            for s, m in zip(similarities, metadatas)
        ]
        
        # 判断是否识别成功
        recognized = best_score >= self.threshold
        
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


def main():
    parser = argparse.ArgumentParser(description="试剂识别推理")
    parser.add_argument("--image_path", type=str, required=True, help="图片路径")
    parser.add_argument("--topk", type=int, default=5, help="返回前K个候选")
    parser.add_argument("--model_dir", type=str, default=None, help="模型目录（默认为脚本所在目录）")
    args = parser.parse_args()
    
    # 初始化识别引擎
    print("[INFO] 初始化识别引擎...")
    
    # 如果指定了模型目录，使用指定的；否则使用脚本所在目录
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = Path(__file__).parent
    
    engine = ReagentRecognitionEngine(
        model_path=model_dir / "models" / "best_model.pth",
        index_path=model_dir / "embeddings" / "reagent.index",
        metadata_path=model_dir / "embeddings" / "metadata.json",
    )
    
    # 识别
    print(f"[INFO] 识别图片: {args.image_path}")
    result = engine.recognize(args.image_path, topk=args.topk)
    
    # 输出结果
    print("" + "="*50)
    print("识别结果:")
    print("="*50)
    
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
        
        script_path = output_path / "inference.py"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(inference_script)
        print(f"  推理脚本已保存: {script_path}")
    else:
        print(f"  跳过推理脚本生成 (--skip-inference-script)")

    # 5. 创建README
    readme_content = '''# 试剂识别模型部署包

## 包含内容

- `models/best_model.pth` - 训练好的模型权重
- `embeddings/reagent.index` - FAISS向量索引
- `embeddings/metadata.json` - 试剂元数据
- `config/class_mapping.json` - 类别映射
- `config/config.json` - 模型配置
- `inference.py` - 推理脚本

## 快速开始

### 1. 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 2. 运行推理

```bash
python inference.py --image_path path/to/image.jpg
```

### 3. 批量推理

```bash
for img in images/*.jpg; do
    python inference.py --image_path "$img"
done
```

## 输出格式

```
识别结果:
==================================================
✅ 识别成功!
   试剂ID: 乙醇001
   试剂名称: 乙醇
   置信度: 92.3%

候选结果:
  1. 乙醇001 (乙醇) - 92.3%
  2. 乙醇002 (乙醇) - 85.6%
  3. 乙醇003 (乙醇) - 78.2%
```

## 🔧 高级用法

### 自定义阈值

修改 `config/config.json` 中的 `similarity_threshold`:

```json
{{
  "inference_config": {{
    "similarity_threshold": 0.75
  }}
}}
```

### 批量推理

```python
from pathlib import Path
from backend.core.recognition_engine import ReagentRecognitionEngine

engine = ReagentRecognitionEngine()

for img_path in Path("images").glob("*.jpg"):
    result = engine.recognize(str(img_path))
    print(f"{{img_path.name}}: {{result['reagent_name']}}")
```

## 📝 注意事项

1. 确保图片清晰，试剂特征明显
2. 光照条件良好
3. 试剂在图片中居中
4. 如果识别率低，可以降低相似度阈值

---
打包时间: {package_time}
'''

    readme_path = output_path / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content.format(
            package_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
    print(f" README已保存: {readme_path}")

    # 6. 创建requirements.txt
    requirements = '''# 试剂识别模型依赖

--extra-index-url https://download.pytorch.org/whl/cu117

torch==2.0.1+cu117
torchvision==0.15.2+cu117
numpy==1.24.4
opencv-python==4.8.1.78
pillow==10.0.1
albumentations==1.3.1
timm==0.9.12
faiss-cpu==1.7.4
'''

    req_path = output_path / "requirements.txt"
    with open(req_path, "w", encoding="utf-8") as f:
        f.write(requirements)
    print(f"  requirements.txt已保存: {req_path}")

    # 7. 创建目录结构说明
    structure = '''deploy_package/
├── models/
│   └── best_model.pth          # 模型权重
├── embeddings/
│   ├── reagent.index           # FAISS索引
│   └── metadata.json           # 试剂元数据
├── config/
│   ├── class_mapping.json      # 类别映射
│   └── config.json            # 配置文件
├── inference.py                # 推理脚本
├── requirements.txt            # 依赖列表
└── README.md                 # 使用说明
'''
    print(f"\n目录结构:")
    print(structure)

    print(f"\n打包完成!")
    print(f"  输出目录: {output_path.absolute()}")
    print(f"\n下一步:")
    print(f"  1. 将 {output_path.name} 目录复制到目标机器")
    print(f"  2. 安装依赖: pip install -r requirements.txt")
    print(f"  3. 运行推理: python inference.py --image_path path/to/image.jpg")


def main():
    parser = argparse.ArgumentParser(description="打包试剂识别模型")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="deploy_package",
        help="输出目录"
    )
    parser.add_argument(
        "--skip-inference-script",
        action="store_true",
        help="跳过生成推理脚本（保留现有修改）"
    )
    args = parser.parse_args()
    
    package_model(args.output_dir, skip_inference_script=args.skip_inference_script)


if __name__ == "__main__":
    main()