# backend/api/main.py
"""
FastAPI 主应用
提供：
- 试剂录入API（含图片注册）
- 实时识别API
- 试剂管理CRUD
- 识别日志查询
- 摄像头图像接收
"""

import io
import os
import sys
import time
import base64
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from fastapi import (
    FastAPI, File, UploadFile, Form, Depends,
    HTTPException, BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import IMAGES_DIR, DEVICE
from backend.core.database import init_db, get_db, Reagent, ReagentImage, RecognitionLog
from backend.core.recognition_engine import get_engine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update


# ===================== 初始化 =====================
app = FastAPI(
    title="试剂视觉识别系统",
    description="基于Metric Learning + ArcFace的试剂精细化识别",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件（试剂图片）
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


@app.on_event("startup")
async def startup():
    await init_db()
    print("[API] 服务启动完成")


# ===================== Pydantic 模型 =====================
class ReagentCreate(BaseModel):
    reagent_id: Optional[str] = None  # 唯一ID，如'乙醇001'（留空则自动生成）
    reagent_name: str                 # 名称，如'乙醇'
    cas_number: Optional[str] = None
    manufacturer: Optional[str] = None
    batch_number: Optional[str] = None
    expiry_date: Optional[str] = None
    location: Optional[str] = None
    quantity: Optional[float] = None
    unit: Optional[str] = None
    notes: Optional[str] = None


class ReagentResponse(BaseModel):
    id: int
    reagent_id: str
    reagent_name: str
    cas_number: Optional[str]
    manufacturer: Optional[str]
    batch_number: Optional[str]
    expiry_date: Optional[str]
    location: Optional[str]
    quantity: Optional[float]
    unit: Optional[str]
    is_active: bool
    image_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class RecognitionResult(BaseModel):
    recognized: bool
    reagent_id: Optional[str]
    reagent_name: Optional[str]
    confidence: Optional[float]
    confidence_pct: Optional[str]
    candidates: List[dict]
    message: str


# ===================== 系统状态 =====================
@app.get("/api/status")
async def get_status():
    """系统状态"""
    engine = get_engine()
    stats = engine.get_stats()
    return {
        "status": "running",
        "device": DEVICE,
        "timestamp": datetime.now().isoformat(),
        **stats,
    }


# ===================== 试剂管理 =====================
@app.post("/api/reagents", response_model=dict)
async def create_reagent(
    reagent: ReagentCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    第一步：创建试剂记录（录入基本信息）
    在用户在一体机上录入信息后调用
    
    如果reagent_id为空，则自动生成（名称+序号）
    """
    # 如果没有提供reagent_id，自动生成
    if not reagent.reagent_id:
        # 查询该名称已有多少个试剂
        result = await db.execute(
            select(Reagent).where(Reagent.reagent_name == reagent.reagent_name)
        )
        existing_reagents = result.scalars().all()
        count = len(existing_reagents)
        
        # 生成ID：名称 + 三位序号
        reagent_id = f"{reagent.reagent_name}{count + 1:03d}"
        print(f"[API] 自动生成试剂ID: {reagent_id}")
    else:
        reagent_id = reagent.reagent_id
        # 检查是否已存在
        result = await db.execute(
            select(Reagent).where(Reagent.reagent_id == reagent_id)
        )
        existing = result.scalar_one_or_none()
        if existing:
            raise HTTPException(400, f"试剂ID '{reagent_id}' 已存在")

    # 创建试剂目录
    reagent_dir = IMAGES_DIR / reagent_id
    reagent_dir.mkdir(parents=True, exist_ok=True)

    # 写入数据库
    reagent_data = reagent.dict()
    reagent_data["reagent_id"] = reagent_id
    db_reagent = Reagent(**reagent_data)
    db.add(db_reagent)
    await db.commit()
    await db.refresh(db_reagent)

    return {
        "success": True,
        "id": db_reagent.id,
        "reagent_id": db_reagent.reagent_id,
        "message": f"试剂 {db_reagent.reagent_id} 创建成功，请上传图片完成注册",
    }


@app.post("/api/reagents/{reagent_id}/register-image")
async def register_reagent_image(
    reagent_id: str,
    file: UploadFile = File(...),
    angle: str = Form(default="front"),  # front/side/top
    db: AsyncSession = Depends(get_db),
):
    """
    第二步：注册试剂图片（摄像头拍摄后调用）
    支持多角度多张图片，每张都会提取嵌入加入FAISS索引
    """
    # 查询试剂是否存在
    result = await db.execute(
        select(Reagent).where(Reagent.reagent_id == reagent_id)
    )
    reagent = result.scalar_one_or_none()
    if not reagent:
        raise HTTPException(404, f"试剂 '{reagent_id}' 不存在，请先创建")

    # 保存图片
    timestamp = int(time.time() * 1000)
    filename = f"{timestamp}_{angle}.jpg"
    save_dir = IMAGES_DIR / reagent_id
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # 注册到识别引擎
    engine = get_engine()
    reg_result = engine.register_reagent(
        image_input=str(save_path),
        reagent_id=reagent_id,
        reagent_name=reagent.reagent_name,
        extra_info={
            "cas_number": reagent.cas_number,
            "angle": angle,
        },
        image_save_path=str(save_path),
    )

    # 更新数据库
    img_record = ReagentImage(
        reagent_id=reagent_id,
        image_path=str(save_path),
        vector_id=reg_result.get("vector_id"),
        angle=angle,
    )
    db.add(img_record)

    # 更新图片计数
    await db.execute(
        update(Reagent)
        .where(Reagent.reagent_id == reagent_id)
        .values(image_count=reagent.image_count + 1)
    )
    await db.commit()

    return {
        "success": True,
        "reagent_id": reagent_id,
        "image_path": f"/images/{reagent_id}/{filename}",
        "vector_id": reg_result.get("vector_id"),
        "total_images": reagent.image_count + 1,
        "message": f"图片注册成功，已加入识别索引",
    }


@app.post("/api/recognize", response_model=RecognitionResult)
async def recognize_reagent(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    log_result: bool = True,
):
    """
    实时识别试剂
    摄像头拍摄到试剂后调用此接口
    """
    # 读取图片
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(400, "无法解析图片")

    # 识别
    engine = get_engine()
    result = engine.recognize(img, topk=5)

    # 记录日志
    if log_result:
        log = RecognitionLog(
            recognized_id=result.get("reagent_id"),
            confidence=result.get("confidence"),
            top1_id=result["candidates"][0]["reagent_id"] if result["candidates"] else None,
            top1_score=result["candidates"][0]["similarity"] if result["candidates"] else None,
        )
        db.add(log)
        await db.commit()

    return RecognitionResult(
        recognized=result["recognized"],
        reagent_id=result.get("reagent_id"),
        reagent_name=result.get("reagent_name"),
        confidence=result.get("confidence"),
        confidence_pct=result.get("confidence_pct"),
        candidates=result.get("candidates", []),
        message=result.get("message", ""),
    )


@app.post("/api/recognize/base64")
async def recognize_base64(data: dict):
    """
    Base64图像识别（摄像头流/前端Canvas适用）
    body: {"image": "base64string", "log": true}
    """
    b64 = data.get("image", "")
    if "," in b64:
        b64 = b64.split(",")[1]

    img_bytes = base64.b64decode(b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(400, "无法解析Base64图像")

    engine = get_engine()
    result = engine.recognize(img)
    return result


@app.post("/api/recognize/multiple")
async def recognize_multiple_reagents(
    file: UploadFile = File(...),
    min_confidence: float = Form(default=0.5),
    topk: int = Form(default=5),
    db: AsyncSession = Depends(get_db),
):
    """
    多物体识别 - 识别图片中的多个试剂
    
    Args:
        file: 上传的图片文件
        min_confidence: 目标检测的最小置信度 (0-1)
        topk: 每个物体返回的候选数量
    
    Returns:
        {
            "total_objects": 3,
            "recognized_count": 2,
            "unrecognized_count": 1,
            "recognized_objects": [
                {
                    "bbox": [100, 50, 200, 300],
                    "detection_confidence": 0.95,
                    "reagent_id": "乙醇001",
                    "reagent_name": "乙醇",
                    "confidence": 0.92,
                    "confidence_pct": "92.0%"
                }
            ],
            "unrecognized_objects": [
                {
                    "bbox": [300, 100, 400, 350],
                    "detection_confidence": 0.88,
                    "best_candidate": "乙醇002",
                    "best_candidate_name": "乙醇",
                    "confidence": 0.68,
                    "confidence_pct": "68.0%"
                }
            ],
            "message": "检测到 3 个物体，识别成功 2 个"
        }
    """
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(400, "无法解析图片")

    engine = get_engine()
    result = engine.recognize_multiple(
        img,
        topk=topk,
        min_confidence=min_confidence,
    )

    # 记录日志（为每个识别到的物体记录）
    if result["recognized_objects"]:
        for obj in result["recognized_objects"]:
            log = RecognitionLog(
                recognized_id=obj.get("reagent_id"),
                confidence=obj.get("confidence"),
                action="multiple_recognition",
            )
            db.add(log)
        await db.commit()

    return result


@app.post("/api/recognize/multiple/base64")
async def recognize_multiple_base64(data: dict):
    """
    多物体识别 - Base64版本
    body: {
        "image": "base64string",
        "min_confidence": 0.5,
        "topk": 5
    }
    """
    b64 = data.get("image", "")
    if "," in b64:
        b64 = b64.split(",")[1]

    img_bytes = base64.b64decode(b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(400, "无法解析Base64图像")

    engine = get_engine()
    result = engine.recognize_multiple(
        img,
        topk=data.get("topk", 5),
        min_confidence=data.get("min_confidence", 0.5),
    )

    return result


@app.get("/api/reagents", response_model=List[dict])
async def list_reagents(
    db: AsyncSession = Depends(get_db),
    active_only: bool = True,
    skip: int = 0,
    limit: int = 100,
):
    """获取所有试剂列表"""
    query = select(Reagent)
    if active_only:
        query = query.where(Reagent.is_active == True)
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    reagents = result.scalars().all()

    return [
        {
            "id": r.id,
            "reagent_id": r.reagent_id,
            "reagent_name": r.reagent_name,
            "cas_number": r.cas_number,
            "manufacturer": r.manufacturer,
            "batch_number": r.batch_number,
            "expiry_date": r.expiry_date,
            "location": r.location,
            "quantity": r.quantity,
            "unit": r.unit,
            "is_active": r.is_active,
            "image_count": r.image_count,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in reagents
    ]


@app.get("/api/reagents/{reagent_id}")
async def get_reagent(reagent_id: str, db: AsyncSession = Depends(get_db)):
    """获取单个试剂详情"""
    result = await db.execute(
        select(Reagent).where(Reagent.reagent_id == reagent_id)
    )
    reagent = result.scalar_one_or_none()
    if not reagent:
        raise HTTPException(404, f"试剂 '{reagent_id}' 不存在")

    # 获取图片列表
    img_result = await db.execute(
        select(ReagentImage).where(ReagentImage.reagent_id == reagent_id)
    )
    images = img_result.scalars().all()

    return {
        "reagent": {
            "reagent_id": reagent.reagent_id,
            "reagent_name": reagent.reagent_name,
            "cas_number": reagent.cas_number,
            "manufacturer": reagent.manufacturer,
            "batch_number": reagent.batch_number,
            "expiry_date": reagent.expiry_date,
            "location": reagent.location,
            "quantity": reagent.quantity,
            "unit": reagent.unit,
            "image_count": reagent.image_count,
            "created_at": reagent.created_at.isoformat() if reagent.created_at else None,
            "notes": reagent.notes,
        },
        "images": [
            {
                "id": img.id,
                "path": f"/images/{reagent_id}/{Path(img.image_path).name}",
                "angle": img.angle,
                "created_at": img.created_at.isoformat() if img.created_at else None,
            }
            for img in images
        ],
    }


@app.delete("/api/reagents/{reagent_id}")
async def deactivate_reagent(reagent_id: str, db: AsyncSession = Depends(get_db)):
    """标记试剂为不在库（软删除）"""
    await db.execute(
        update(Reagent)
        .where(Reagent.reagent_id == reagent_id)
        .values(is_active=False)
    )
    await db.commit()
    return {"success": True, "message": f"试剂 {reagent_id} 已标记为取出"}



@app.delete("/api/reagents/{reagent_id}/permanent")
async def delete_reagent_permanent(reagent_id: str, db: AsyncSession = Depends(get_db)):
    """
    永久删除试剂（包括数据库记录、FAISS索引特征、图片文件）
    
    此操作不可恢复，请谨慎使用！
    """
    import shutil
    
    # 1. 查询试剂信息
    result = await db.execute(
        select(Reagent).where(Reagent.reagent_id == reagent_id)
    )
    reagent = result.scalar_one_or_none()
    if not reagent:
        raise HTTPException(404, f"试剂 '{reagent_id}' 不存在")
    
    # 2. 获取所有图片记录
    img_result = await db.execute(
        select(ReagentImage).where(ReagentImage.reagent_id == reagent_id)
    )
    images = img_result.scalars().all()
    
    # 3. 删除FAISS索引中的特征向量
    engine = get_engine()
    index_result = engine.delete_reagent(reagent_id)
    
    # 4. 删除图片文件
    deleted_files = []
    for img in images:
        img_path = Path(img.image_path)
        if img_path.exists():
            try:
                img_path.unlink()
                deleted_files.append(str(img_path))
            except Exception as e:
                print(f"[API] 删除图片文件失败: {img_path}, 错误: {e}")
    
    # 5. 删除图片目录（如果为空）
    reagent_dir = IMAGES_DIR / reagent_id
    if reagent_dir.exists():
        try:
            shutil.rmtree(reagent_dir)
            print(f"[API] 已删除试剂目录: {reagent_dir}")
        except Exception as e:
            print(f"[API] 删除试剂目录失败: {reagent_dir}, 错误: {e}")
    
    # 6. 删除数据库记录（试剂图片）
    await db.execute(
        ReagentImage.__table__.delete().where(ReagentImage.reagent_id == reagent_id)
    )
    
    # 7. 删除数据库记录（试剂）
    await db.execute(
        Reagent.__table__.delete().where(Reagent.reagent_id == reagent_id)
    )
    await db.commit()
    
    return {
        "success": True,
        "reagent_id": reagent_id,
        "deleted_vectors": index_result.get("deleted_count", 0),
        "deleted_files": len(deleted_files),
        "message": f"试剂 {reagent_id} 已永久删除（{index_result.get('deleted_count', 0)} 个特征，{len(deleted_files)} 个文件）",
    }


@app.get("/api/logs")
async def get_recognition_logs(
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
):
    """获取识别日志"""
    result = await db.execute(
        select(RecognitionLog)
        .order_by(RecognitionLog.timestamp.desc())
        .limit(limit)
    )
    logs = result.scalars().all()
    return [
        {
            "id": log.id,
            "timestamp": log.timestamp.isoformat() if log.timestamp else None,
            "recognized_id": log.recognized_id,
            "confidence": log.confidence,
            "top1_id": log.top1_id,
            "action": log.action,
        }
        for log in logs
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)