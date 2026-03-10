# scripts/camera_test.py
"""
摄像头实时识别测试
在有摄像头的环境中运行，实时显示识别结果

使用方法：
  python scripts/camera_test.py
  python scripts/camera_test.py --camera 0  # 摄像头索引
  python scripts/camera_test.py --mode register --reagent_id 乙醇001  # 注册模式
"""

import sys
import cv2
import time
import argparse
import requests
import numpy as np
import base64
from pathlib import Path

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


def draw_result(img, result: dict):
    """在图像上绘制识别结果"""
    h, w = img.shape[:2]

    # 背景框
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    if result.get("recognized"):
        color = (0, 255, 0)  # 绿色
        text1 = f"识别成功: {result['reagent_id']}"
        text2 = f"置信度: {result.get('confidence_pct', 'N/A')}"
    else:
        color = (0, 0, 255)  # 红色
        text1 = "未识别"
        text2 = result.get("message", "")[:40]

    cv2.putText(img, text1, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(img, text2, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # 候选列表
    candidates = result.get("candidates", [])
    for i, cand in enumerate(candidates[:3]):
        y = 100 + i * 20
        text = f"Top{i+1}: {cand['reagent_id']} ({cand.get('confidence_pct', '')})"
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return img


def run_recognize_mode(camera_idx: int):
    """识别模式：实时识别摄像头画面"""
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {camera_idx}")
        return

    print("🎥 识别模式启动")
    print("  按 Space：立即识别")
    print("  按 Q：退出")

    last_result = {"recognized": False, "message": "按Space识别", "candidates": []}
    fps_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每秒自动识别2次
        frame_count += 1
        if frame_count % 15 == 0:  # 假设30fps，每0.5秒识别一次
            last_result = recognize_frame(frame)

        # 绘制结果
        display = frame.copy()
        display = draw_result(display, last_result)

        # FPS
        elapsed = time.time() - fps_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(display, f"FPS: {fps:.1f}", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1) if False else None

        cv2.imshow("试剂识别系统", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            last_result = recognize_frame(frame)
            print(f"识别结果: {last_result.get('message')}")

    cap.release()
    cv2.destroyAllWindows()


def run_register_mode(camera_idx: int, reagent_id: str, reagent_name: str):
    """注册模式：拍摄试剂图片并注册"""
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {camera_idx}")
        return

    print(f"📸 注册模式: {reagent_id} ({reagent_name})")
    print("  按 S：拍照注册（正面）")
    print("  按 A：拍照注册（侧面）")
    print("  按 T：拍照注册（顶部）")
    print("  按 Q：退出")

    registered_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(display, f"注册: {reagent_id}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(display, f"已注册: {registered_count}张 | S=正面 A=侧面 T=顶部 Q=退出",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("试剂注册", display)
        key = cv2.waitKey(1) & 0xFF

        angle_map = {ord('s'): 'front', ord('a'): 'side', ord('t'): 'top'}

        if key in angle_map:
            angle = angle_map[key]
            _, buf = cv2.imencode('.jpg', frame)
            try:
                resp = requests.post(
                    f"{API_BASE}/api/reagents/{reagent_id}/register-image",
                    files={"file": ("image.jpg", buf.tobytes(), "image/jpeg")},
                    data={"angle": angle},
                )
                result = resp.json()
                if result.get("success"):
                    registered_count += 1
                    print(f"  ✅ 注册成功 ({angle})，第{registered_count}张")
                else:
                    print(f"  ❌ 注册失败: {result}")
            except Exception as e:
                print(f"  ❌ 请求失败: {e}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ 注册完成，共注册 {registered_count} 张图片")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mode", choices=["recognize", "register"], default="recognize")
    parser.add_argument("--reagent_id", type=str, default="")
    parser.add_argument("--reagent_name", type=str, default="")
    args = parser.parse_args()

    if args.mode == "register":
        if not args.reagent_id:
            print("注册模式需要 --reagent_id 参数")
            sys.exit(1)
        run_register_mode(args.camera, args.reagent_id, args.reagent_name or args.reagent_id)
    else:
        run_recognize_mode(args.camera)