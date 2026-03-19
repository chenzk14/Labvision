"""
scripts/train.py
使用方法：
  python scripts/train.py --data_dir data/images
  python scripts/train.py --data_dir data/images --no_small_sample
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.trainer import ReagentTrainer


SMALL_SAMPLE_OVERRIDES = {
    "val_split":             0.2,
    "epochs":                80,
    "lr":                    3e-4,
    "early_stop_patience":   15,
    "arcface_margin":        0.30,
    "arcface_scale":         64.0,
    "triplet_weight":        0.0,
    "warmup_epochs":         5,
    "batch_size":            4,
    "accumulation_steps":    4,
}


def main():
    parser = argparse.ArgumentParser(description="训练试剂识别模型")
    parser.add_argument("--data_dir", type=str, default="data/images")
    parser.add_argument("--no_small_sample", action="store_true",
                        help="禁用小样本模式，使用 config.py 默认参数")
    parser.add_argument("--build_index", action="store_true", default=True,
                        help="训练后自动构建 FAISS 索引")
    args = parser.parse_args()

    data_path = Path(args.data_dir).resolve()
    if not data_path.exists():
        print(f"❌ 数据目录不存在: {data_path}")
        sys.exit(1)

    classes = [d for d in data_path.iterdir() if d.is_dir()]
    for cls in sorted(classes)[:10]:
        imgs = list(cls.glob("*.[jJpP][pPnN][gG]"))
        print(f"  {cls.name}: {len(imgs)} 张图片")
    if len(classes) > 10:
        print(f"  ... 还有 {len(classes) - 10} 个类别")
    if len(classes) < 2:
        print("⚠️  至少需要 2 个类别才能训练！")
        sys.exit(1)

    overrides = {} if args.no_small_sample else SMALL_SAMPLE_OVERRIDES
    trainer = ReagentTrainer(data_dir=str(data_path), overrides=overrides)
    best_model = trainer.train()

    if args.build_index:
        trainer.build_faiss_index_after_training()

    print(f"\n✅ 完成！模型: {best_model}")
    print("下一步启动服务：")
    print("  python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()