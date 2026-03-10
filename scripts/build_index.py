# scripts/build_index.py
"""
从图片目录重建FAISS索引
训练完成后运行此脚本，将所有试剂图片嵌入到索引中

使用方法：
  python scripts/build_index.py --data_dir data/images
"""

import sys
import argparse
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main():
    parser = argparse.ArgumentParser(description="构建FAISS识别索引")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/images",
    )
    args = parser.parse_args()

    print("📦 构建FAISS识别索引...")
    print(f"  数据目录: {args.data_dir}")

    from backend.core.recognition_engine import ReagentRecognitionEngine
    from backend.core.database import get_db

    engine = ReagentRecognitionEngine()
    
    # 获取数据库会话
    async for db in get_db():
        engine.rebuild_index_from_images(args.data_dir, db=db)
        break

    stats = engine.get_stats()
    print(f"\n✅ 索引构建完成！")
    print(f"  注册数量: {stats['total_registrations']}")
    print(f"  唯一试剂ID: {stats['unique_reagent_ids']}")
    print(f"  唯一试剂名称: {stats['unique_reagent_names']}")
    print(f"  向量总数: {stats['faiss_vectors']}")
    print(f"\n现在可以启动服务：")
    print(f"  python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    asyncio.run(main())