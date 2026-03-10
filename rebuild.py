# rebuild_index.py
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from backend.core.recognition_engine import get_engine

# 获取引擎实例
engine = get_engine()

# 从图片目录重建索引
# 默认目录是 data/images/
engine.rebuild_index_from_images("data/images/")

print("索引重建完成！")