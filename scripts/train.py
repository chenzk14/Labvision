# scripts/train.py
"""
训练启动脚本
使用方法：
  conda activate reagent-vision
  python scripts/train.py --data_dir backend/data/images

可选参数：
  --epochs 50
  --batch_size 16
  --lr 1e-4
"""

import argparse
from pathlib import Path
import sys

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from backend import config

def main():
    parser = argparse.ArgumentParser(description="训练试剂识别模型")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/images",
        help="训练数据目录（每个子文件夹为一个试剂ID）"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    # 更新配置
    # from backend import config
    # config.TRAIN_CONFIG["epochs"] = args.epochs
    # config.TRAIN_CONFIG["batch_size"] = args.batch_size
    # config.TRAIN_CONFIG["lr"] = args.lr
    # config.MODEL_CONFIG["img_size"] = args.img_size

    # 检查数据目录
    data_path = Path(args.data_dir).resolve()
    if not data_path.exists():
        print(f"❌ 数据目录不存在: {data_path}")
        print("请先创建数据目录并放入试剂图片：")
        print("  backend/data/images/")
        print("  ├── 乙醇/")
        print("  │   ├── img1.jpg")
        print("  │   └── img2.jpg")
        print("  └── 钠/")
        print("      └── img1.jpg")
        sys.exit(1)

    # 统计数据
    classes = [d for d in data_path.iterdir() if d.is_dir()]
    # print(f"\n📊 数据统计:")
    # print(f"  类别数: {len(classes)}")
    for cls in sorted(classes)[:10]:
        imgs = list(cls.glob("*.[jJpP][pPnN][gG]"))
        print(f"  {cls.name}: {len(imgs)} 张图片")
    if len(classes) > 10:
        print(f"  ... 还有 {len(classes) - 10} 个类别")

    if len(classes) < 2:
        print("\n⚠️  至少需要2个类别才能训练！")
        sys.exit(1)

    print(f"\n🚀 开始训练...")
    print(f"  数据目录: {data_path.absolute()}")
    # print(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(
        f"  Epochs: {config.TRAIN_CONFIG['epochs']} | Batch: {config.TRAIN_CONFIG['batch_size']} | LR: {config.TRAIN_CONFIG['lr']}")

    from backend.core.trainer import ReagentTrainer
    trainer = ReagentTrainer(data_dir=str(data_path))
    best_model = trainer.train()

    print(f"\n✅ 训练完成！")
    print(f"  模型路径: {best_model}")
    print(f"\n下一步：重建FAISS索引")
    print(f"  python scripts/build_index.py")


if __name__ == "__main__":
    main()