# scripts/correction_manager.py
"""
纠错管理脚本
提供纠错提交、查看、应用、导出等功能

使用方法：
  # 查看纠错统计
  python scripts/correction_manager.py --action stats

  # 查看所有纠错记录
  python scripts/correction_manager.py --action list

  # 应用所有未应用的纠错
  python scripts/correction_manager.py --action apply --all

  # 应用指定纠错ID
  python scripts/correction_manager.py --action apply --id 1

  # 导出纠错样本用于训练
  python scripts/correction_manager.py --action export --output data/corrections

  # 验证纠错质量
  python scripts/correction_manager.py --action verify --reagent_id 乙醇001

  # 摄像头纠错模式
  python scripts/correction_manager.py --action camera --camera 0
"""

import sys
import cv2
import time
import argparse
import requests
import numpy as np
import base64
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

API_BASE = "http://localhost:8000"


def encode_image(img) -> str:
    """编码图像为base64"""
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')


def recognize_frame(img) -> dict:
    """调用API识别"""
    try:
        b64 = encode_image(img)
        resp = requests.post(
            f"{API_BASE}/api/recognize/base64",
            json={"image": b64},
            timeout=5,
        )
        return resp.json()
    except Exception as e:
        return {"recognized": False, "message": str(e), "candidates": []}


def submit_correction(
        img,
        corrected_reagent_id: str,
        corrected_reagent_name: str,
        original_recognition_id: str = None,
        original_confidence: float = None,
        notes: str = None,
        apply_immediately: bool = True,
) -> dict:
    """提交纠错"""
    _, buffer = cv2.imencode('.jpg', img)
    try:
        resp = requests.post(
            f"{API_BASE}/api/corrections/submit",
            files={"file": ("image.jpg", buffer.tobytes(), "image/jpeg")},
            data={
                "corrected_reagent_id": corrected_reagent_id,
                "corrected_reagent_name": corrected_reagent_name,
                "original_recognition_id": original_recognition_id,
                "original_confidence": original_confidence,
                "notes": notes,
                "apply_immediately": apply_immediately,
            },
            timeout=10,
        )
        return resp.json()
    except Exception as e:
        return {"success": False, "message": str(e)}


def action_stats():
    """查看纠错统计"""
    print("\n" + "=" * 60)
    print("📊 纠错统计信息")
    print("=" * 60)

    try:
        resp = requests.get(f"{API_BASE}/api/corrections/statistics", timeout=5)
        stats = resp.json()

        print(f"\n总向量数: {stats['total_vectors']}")
        print(f"纠错向量数: {stats['correction_count']}")
        print(f"纠错比例: {stats['correction_ratio']}")
        print(f"已纠正试剂数: {stats['unique_corrected_reagents']}")

        if stats.get('correction_sources'):
            print(f"\n纠错来源:")
            for source, count in stats['correction_sources'].items():
                print(f"  - {source}: {count}")

        print("\n✅ 统计信息获取成功")
        return True
    except Exception as e:
        print(f"\n❌ 获取统计信息失败: {e}")
        return False


def action_list(applied_only: bool = False, limit: int = 20):
    """查看纠错记录列表"""
    print("\n" + "=" * 60)
    print("📋 纠错记录列表")
    print("=" * 60)

    try:
        resp = requests.get(
            f"{API_BASE}/api/corrections",
            params={"applied_only": applied_only, "limit": limit},
            timeout=5,
        )
        corrections = resp.json()

        if not corrections:
            print("\n暂无纠错记录")
            return True

        print(f"\n共 {len(corrections)} 条记录:\n")

        for i, c in enumerate(corrections, 1):
            status = "✅已应用" if c['is_applied'] else "⏳未应用"
            exported = "📦已导出" if c['is_exported'] else ""

            print(f"{i}. [{status}] {exported}")
            print(f"   ID: {c['id']}")
            print(f"   时间: {c['timestamp'][:19] if c['timestamp'] else 'N/A'}")
            print(f"   原识别: {c['original_recognition_id'] or 'N/A'} (置信度: {c['original_confidence'] or 'N/A'})")
            print(f"   纠正为: {c['corrected_reagent_id']} ({c['corrected_reagent_name']})")
            print(f"   来源: {c['correction_source'] or 'N/A'}")
            if c['notes']:
                print(f"   备注: {c['notes']}")
            print()

        return True
    except Exception as e:
        print(f"\n❌ 获取纠错记录失败: {e}")
        return False


def action_apply(correction_id: int = None, apply_all: bool = False):
    """应用纠错"""
    print("\n" + "=" * 60)
    print("🔧 应用纠错")
    print("=" * 60)

    if apply_all:
        # 获取所有未应用的纠错
        try:
            resp = requests.get(
                f"{API_BASE}/api/corrections",
                params={"applied_only": False, "limit": 1000},
                timeout=5,
            )
            all_corrections = resp.json()
            unapplied = [c['id'] for c in all_corrections if not c['is_applied']]

            if not unapplied:
                print("\n没有未应用的纠错")
                return True

            print(f"\n找到 {len(unapplied)} 个未应用的纠错")
            confirm = input("是否全部应用？(y/n): ")
            if confirm.lower() != 'y':
                print("已取消")
                return False

            # 批量应用
            resp = requests.post(
                f"{API_BASE}/api/corrections/batch-apply",
                json={"correction_ids": unapplied},
                timeout=30,
            )
            result = resp.json()

            print(f"\n✅ 批量应用完成")
            print(f"  总数: {result['total']}")
            print(f"  成功: {result['success_count']}")

            if result['success_count'] < result['total']:
                print(f"\n失败的纠错:")
                for r in result['results']:
                    if not r['success']:
                        print(f"  - ID {r['correction_id']}: {r['message']}")

            return True
        except Exception as e:
            print(f"\n❌ 批量应用失败: {e}")
            return False

    elif correction_id is not None:
        # 应用单个纠错
        try:
            print(f"\n应用纠错 ID: {correction_id}")
            resp = requests.post(
                f"{API_BASE}/api/corrections/apply/{correction_id}",
                timeout=10,
            )
            result = resp.json()

            if result['success']:
                print(f"✅ 纠错已应用")
                print(f"  向量ID: {result.get('vector_id', 'N/A')}")
                return True
            else:
                print(f"❌ 应用失败: {result.get('message', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"\n❌ 应用纠错失败: {e}")
            return False

    else:
        print("\n请指定 --id 或 --all 参数")
        return False


def action_export(output_dir: str = None, include_original: bool = False):
    """导出纠错样本"""
    print("\n" + "=" * 60)
    print("📦 导出纠错样本")
    print("=" * 60)

    try:
        resp = requests.post(
            f"{API_BASE}/api/corrections/export",
            params={
                "output_dir": output_dir,
                "include_original": include_original,
            },
            timeout=30,
        )
        result = resp.json()

        if result['success']:
            print(f"\n✅ 导出成功")
            print(f"  导出数量: {result['exported_count']}")
            print(f"  试剂种类: {result['exported_reagents']}")
            print(f"  输出目录: {result['output_directory']}")
            print(f"  报告文件: {result['report_path']}")
            print(f"\n💡 提示: 可以使用导出的样本重新训练模型")
            print(f"   python scripts/train.py --data_dir {result['output_directory']}")
            return True
        else:
            print(f"\n❌ 导出失败: {result.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"\n❌ 导出失败: {e}")
        return False


def action_verify(reagent_id: str, min_samples: int = 3):
    """验证纠错质量"""
    print("\n" + "=" * 60)
    print("🔍 验证纠错质量")
    print("=" * 60)

    try:
        resp = requests.get(
            f"{API_BASE}/api/corrections/verify/{reagent_id}",
            params={"min_samples": min_samples},
            timeout=5,
        )
        result = resp.json()

        print(f"\n试剂ID: {result['reagent_id']}")
        print(f"总样本数: {result['total_samples']}")
        print(f"纠错样本数: {result['correction_samples']}")
        print(f"纠错比例: {result['correction_ratio']}")
        print(f"满足最小样本数: {'是' if result['meets_minimum'] else '否'}")
        print(f"可重新训练: {'✅是' if result['ready_for_retraining'] else '❌否'}")

        if result['ready_for_retraining']:
            print(f"\n💡 该试剂的纠错样本已足够，可用于模型重训")

        return True
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        return False


def action_camera(camera_idx: int):
    """摄像头纠错模式"""
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {camera_idx}")
        return False

    print("\n" + "=" * 60)
    print("🎥 摄像头纠错模式")
    print("=" * 60)
    print("\n操作说明:")
    print("  按 Space: 识别当前画面")
    print("  按 C: 提交纠错（识别错误时使用）")
    print("  按 Q: 退出")
    print("\n纠错流程:")
    print("  1. 按Space识别")
    print("  2. 如果识别错误，按C提交纠错")
    print("  3. 输入正确的试剂ID和名称")
    print("  4. 纠错将立即应用到识别系统")

    last_result = {"recognized": False, "message": "按Space识别", "candidates": []}
    last_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        last_frame = frame.copy()

        # 绘制识别结果
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        if last_result.get("recognized"):
            color = (0, 255, 0)
            text1 = f"✅ 识别: {last_result['reagent_id']}"
            text2 = f"置信度: {last_result.get('confidence_pct', 'N/A')}"
        else:
            color = (0, 0, 255)
            text1 = "❌ 未识别"
            text2 = last_result.get("message", "")[:30]

        cv2.putText(frame, text1, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, text2, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, "Space=识别 | C=纠错 | Q=退出", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # 候选列表
        candidates = last_result.get("candidates", [])
        for i, cand in enumerate(candidates[:3]):
            y = 100 + i * 20
            text = f"Top{i + 1}: {cand['reagent_id']} ({cand.get('confidence_pct', '')})"
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("试剂纠错系统", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            last_result = recognize_frame(frame)
            print(f"\n识别结果: {last_result.get('message')}")
        elif key == ord('c'):
            # 提交纠错
            if last_frame is None:
                print("❌ 没有可用的图像")
                continue

            print("\n" + "=" * 40)
            print("📝 提交纠错")
            print("=" * 40)

            original_id = last_result.get("reagent_id", "未识别")
            original_conf = last_result.get("confidence", 0)

            print(f"原识别: {original_id} (置信度: {original_conf:.2f})")

            corrected_id = input("请输入正确的试剂ID: ").strip()
            if not corrected_id:
                print("❌ 试剂ID不能为空")
                continue

            corrected_name = input("请输入试剂名称: ").strip() or corrected_id
            notes = input("备注 (可选): ").strip() or None

            print("\n正在提交纠错...")
            result = submit_correction(
                img=last_frame,
                corrected_reagent_id=corrected_id,
                corrected_reagent_name=corrected_name,
                original_recognition_id=original_id if original_id != "未识别" else None,
                original_confidence=original_conf if original_id != "未识别" else None,
                notes=notes,
                apply_immediately=True,
            )

            if result.get("success"):
                print(f"✅ 纠错提交成功！")
                print(f"   试剂ID: {result['reagent_id']}")
                print(f"   向量ID: {result.get('vector_id', 'N/A')}")
                print(f"   已应用: {'是' if result.get('is_applied') else '否'}")
            else:
                print(f"❌ 纠错提交失败: {result.get('message', 'Unknown error')}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ 纠错模式结束")
    return True


def main():
    parser = argparse.ArgumentParser(description="试剂识别系统 - 纠错管理工具")
    parser.add_argument(
        "--action",
        choices=["stats", "list", "apply", "export", "verify", "camera"],
        required=True,
        help="操作类型"
    )
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引")
    parser.add_argument("--id", type=int, help="纠错记录ID")
    parser.add_argument("--all", action="store_true", help="应用到所有未应用的纠错")
    parser.add_argument("--output", type=str, help="输出目录")
    parser.add_argument("--reagent_id", type=str, help="试剂ID")
    parser.add_argument("--min_samples", type=int, default=3, help="最小样本数")
    parser.add_argument("--limit", type=int, default=20, help="列表显示数量")
    parser.add_argument("--include_original", action="store_true", help="导出时包含原始错误图片")

    args = parser.parse_args()

    # 检查API是否可用
    try:
        resp = requests.get(f"{API_BASE}/api/status", timeout=2)
        print("✅ API服务连接成功")
    except Exception as e:
        print(f"❌ 无法连接到API服务 ({API_BASE})")
        print("请确保API服务正在运行: uvicorn backend.api.main:app")
        sys.exit(1)

    # 执行对应操作
    success = False

    if args.action == "stats":
        success = action_stats()
    elif args.action == "list":
        success = action_list(limit=args.limit)
    elif args.action == "apply":
        success = action_apply(correction_id=args.id, apply_all=args.all)
    elif args.action == "export":
        success = action_export(output_dir=args.output, include_original=args.include_original)
    elif args.action == "verify":
        if not args.reagent_id:
            print("❌ --reagent_id 参数是必需的")
            sys.exit(1)
        success = action_verify(reagent_id=args.reagent_id, min_samples=args.min_samples)
    elif args.action == "camera":
        success = action_camera(camera_idx=args.camera)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()