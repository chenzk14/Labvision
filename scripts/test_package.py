# scripts/test_package.py
"""
测试打包后的模型

使用方法：
  python scripts/test_package.py --package_dir deploy_package
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_package_structure(package_dir: str):
    """测试打包目录结构"""
    print("\n" + "=" * 60)
    print("1️⃣ 测试目录结构")
    print("=" * 60)

    package_path = Path(package_dir)

    # 检查必需文件
    required_files = [
        "models/best_model.pth",
        "embeddings/reagent.index",
        "embeddings/metadata.json",
        "config/class_mapping.json",
        "config/config.json",
        "inference.py",
        "requirements.txt",
        "README.md",
    ]

    all_exist = True
    for file_path in required_files:
        full_path = package_path / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"  ✅ {file_path} ({size:,} bytes)")
        else:
            print(f"  ❌ {file_path} 不存在！")
            all_exist = False

    return all_exist


def test_config_files(package_dir: str):
    """测试配置文件"""
    print("\n" + "=" * 60)
    print("2️⃣ 测试配置文件")
    print("=" * 60)

    import json
    package_path = Path(package_dir)

    # 测试 config.json
    config_path = package_path / "config" / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"  ✅ config.json 加载成功")
        print(f"     - 模型: {config.get('model_config', {}).get('backbone', 'N/A')}")
        print(f"     - 设备: {config.get('device', 'N/A')}")
        print(f"     - 相似度阈值: {config.get('inference_config', {}).get('similarity_threshold', 'N/A')}")
    except Exception as e:
        print(f"  ❌ config.json 加载失败: {e}")
        return False

    # 测试 class_mapping.json
    mapping_path = package_path / "config" / "class_mapping.json"
    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        class_to_idx = mapping.get("class_to_idx", {})
        print(f"  ✅ class_mapping.json 加载成功")
        print(f"     - 类别数: {len(class_to_idx)}")
        if class_to_idx:
            print(f"     - 类别列表: {list(class_to_idx.keys())}")
    except Exception as e:
        print(f"  ❌ class_mapping.json 加载失败: {e}")
        return False

    # 测试 metadata.json
    metadata_path = package_path / "embeddings" / "metadata.json"
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"  ✅ metadata.json 加载成功")
        print(f"     - 向量数: {len(metadata)}")
        if metadata:
            reagent_ids = set(m.get("reagent_id", "") for m in metadata)
            print(f"     - 试剂数: {len(reagent_ids)}")
    except Exception as e:
        print(f"  ❌ metadata.json 加载失败: {e}")
        return False

    return True


def test_model_loading(package_dir: str):
    """测试模型加载"""
    print("\n" + "=" * 60)
    print("3️⃣ 测试模型加载")
    print("=" * 60)

    import torch
    package_path = Path(package_dir)

    model_path = package_path / "models" / "best_model.pth"

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"  ✅ 模型权重加载成功")
        print(f"     - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"     - 最佳准确率: {checkpoint.get('best_val_acc', 'N/A')}")
        print(f"     - 类别数: {len(checkpoint.get('class_to_idx', {}))}")
        return True
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        return False


def test_inference(package_dir: str, test_image: str = None):
    """测试推理功能"""
    print("\n" + "=" * 60)
    print("4️⃣ 测试推理功能")
    print("=" * 60)

    package_path = Path(package_dir)

    # 查找测试图片
    if test_image is None:
        # 在 data/images 中查找
        data_dir = Path("data/images")
        test_images = []
        for reagent_dir in data_dir.iterdir():
            if reagent_dir.is_dir():
                images = list(reagent_dir.glob("*.jpg")) + list(reagent_dir.glob("*.png"))
                if images:
                    test_images.append(images[0])

        if not test_images:
            print(f"  ⚠️  未找到测试图片")
            return False

        test_image = str(test_images[0])
        print(f"  📸 使用测试图片: {test_image}")

    # 运行推理
    inference_script = package_path / "inference.py"

    try:
        result = subprocess.run(
            [sys.executable, str(inference_script), "--image_path", test_image],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print(f"  ✅ 推理成功")
            print(f"\n  输出:")
            for line in result.stdout.split("\n"):
                if line.strip():
                    print(f"    {line}")
            return True
        else:
            print(f"  ❌ 推理失败")
            print(f"  错误: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  ❌ 推理超时")
        return False
    except Exception as e:
        print(f"  ❌ 推理异常: {e}")
        return False


def test_dependencies(package_dir: str):
    """测试依赖文件"""
    print("\n" + "=" * 60)
    print("5️⃣ 测试依赖文件")
    print("=" * 60)

    package_path = Path(package_dir)
    req_path = package_path / "requirements.txt"

    try:
        with open(req_path, "r", encoding="utf-8") as f:
            requirements = f.read()
        print(f"  ✅ requirements.txt 加载成功")
        print(f"\n  依赖列表:")
        for line in requirements.strip().split("\n"):
            if line.strip() and not line.startswith("#"):
                print(f"    - {line.strip()}")
        return True
    except Exception as e:
        print(f"  ❌ requirements.txt 加载失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="测试打包后的模型")
    parser.add_argument(
        "--package_dir",
        type=str,
        default="deploy_package",
        help="打包目录"
    )
    parser.add_argument(
        "--test_image",
        type=str,
        default=None,
        help="测试图片路径（可选）"
    )
    args = parser.parse_args()

    print("\n" + "🧪" * 30)
    print("测试打包模型")
    print("🧪" * 30)
    print(f"\n打包目录: {args.package_dir}")

    # 运行所有测试
    tests = [
        ("目录结构", lambda: test_package_structure(args.package_dir)),
        ("配置文件", lambda: test_config_files(args.package_dir)),
        ("模型加载", lambda: test_model_loading(args.package_dir)),
        ("依赖文件", lambda: test_dependencies(args.package_dir)),
        ("推理功能", lambda: test_inference(args.package_dir, args.test_image)),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n  测试 '{test_name}' 异常: {e}")
            results.append((test_name, False))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "通过" if result else "失败"
        print(f"  {test_name}: {status}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n所有测试通过！打包模型可以部署。")
    else:
        print("\n部分测试失败，请检查问题。")


if __name__ == "__main__":
    main()