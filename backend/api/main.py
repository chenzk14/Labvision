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
from backend.core.database import init_db, get_db, Reagent, ReagentImage, RecognitionLog, CorrectionLog
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


class CorrectionSubmit(BaseModel):
    corrected_reagent_id: str
    corrected_reagent_name: str
    original_recognition_id: Optional[str] = None
    original_confidence: Optional[float] = None
    notes: Optional[str] = None
    apply_immediately: bool = True


class CorrectionResponse(BaseModel):
    id: int
    timestamp: datetime
    original_recognition_id: Optional[str]
    original_confidence: Optional[float]
    corrected_reagent_id: str
    corrected_reagent_name: str
    is_applied: bool
    is_exported: bool
    correction_source: Optional[str]
    notes: Optional[str]

    class Config:
        from_attributes = True


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


@app.post("/api/reagents/sync-image-counts")
async def sync_image_counts(db: AsyncSession = Depends(get_db)):
    """
    同步所有试剂的 image_count 与实际的 ReagentImage 记录数量
    用于修复图片数显示不正确的问题
    """
    # 获取所有试剂
    reagent_result = await db.execute(select(Reagent))
    reagents = reagent_result.scalars().all()
    
    # 获取所有图片记录
    img_result = await db.execute(
        select(ReagentImage.reagent_id, ReagentImage.id)
    )
    img_records = img_result.all()
    
    # 统计每个试剂的图片数量
    image_counts = {}
    for reagent_id, _ in img_records:
        image_counts[reagent_id] = image_counts.get(reagent_id, 0) + 1
    
    # 更新每个试剂的 image_count
    updated_count = 0
    mismatch_count = 0
    for reagent in reagents:
        actual_count = image_counts.get(reagent.reagent_id, 0)
        if reagent.image_count != actual_count:
            await db.execute(
                update(Reagent)
                .where(Reagent.reagent_id == reagent.reagent_id)
                .values(image_count=actual_count)
            )
            updated_count += 1
            mismatch_count += 1
            print(f"[Sync] {reagent.reagent_id}: {reagent.image_count} -> {actual_count}")
    
    await db.commit()
    
    return {
        "success": True,
        "message": f"已同步 {updated_count} 个试剂的图片数量",
        "total_reagents": len(reagents),
        "updated_count": updated_count,
        "mismatch_count": mismatch_count,
        "details": [
            {
                "reagent_id": r.reagent_id,
                "old_count": r.image_count,
                "new_count": image_counts.get(r.reagent_id, 0)
            }
            for r in reagents
            if r.image_count != image_counts.get(r.reagent_id, 0)
        ]
    }


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


# ===================== 纠错管理 =====================
@app.post("/api/corrections/submit")
async def submit_correction(
    file: UploadFile = File(...),
    corrected_reagent_id: str = Form(...),
    corrected_reagent_name: str = Form(...),
    crop_x1: Optional[int] = Form(None),
    crop_y1: Optional[int] = Form(None),
    crop_x2: Optional[int] = Form(None),
    crop_y2: Optional[int] = Form(None),
    original_recognition_id: Optional[str] = Form(None),
    original_confidence: Optional[float] = Form(None),
    notes: Optional[str] = Form(None),
    apply_immediately: bool = Form(True),
    correction_source: str = Form("web"),
    db: AsyncSession = Depends(get_db),
):
    """提交纠错
    
    当识别错误时，上传正确的图片并指定正确的试剂ID
    图片会保存到试剂专属文件夹，并创建ReagentImage记录
    """
    # 查询试剂是否存在
    result = await db.execute(
        select(Reagent).where(Reagent.reagent_id == corrected_reagent_id)
    )
    reagent = result.scalar_one_or_none()
    if not reagent:
        raise HTTPException(404, f"试剂 '{corrected_reagent_id}' 不存在，请先创建")

    # 保存图片到试剂专属文件夹
    timestamp = int(time.time() * 1000)
    save_dir = IMAGES_DIR / corrected_reagent_id
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{timestamp}_correction.jpg"
    save_path = save_dir / filename

    # 保存原图
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # 可选：裁剪后再用于纠错/注册（用于多物体纠错等场景）
    if all(v is not None for v in [crop_x1, crop_y1, crop_x2, crop_y2]):
        try:
            img0 = cv2.imread(str(save_path))
            if img0 is not None:
                h, w = img0.shape[:2]
                x1 = max(0, min(int(crop_x1), w - 1))
                y1 = max(0, min(int(crop_y1), h - 1))
                x2 = max(0, min(int(crop_x2), w))
                y2 = max(0, min(int(crop_y2), h))
                if x2 > x1 and y2 > y1:
                    cropped = img0[y1:y2, x1:x2]
                    cv2.imwrite(str(save_path), cropped)
        except Exception as e:
            print(f"[API] 裁剪纠错图片失败，将使用原图: {e}")

    # 创建纠错记录
    correction = CorrectionLog(
        original_recognition_id=original_recognition_id,
        original_confidence=original_confidence,
        corrected_reagent_id=corrected_reagent_id,
        corrected_reagent_name=corrected_reagent_name,
        corrected_image_path=str(save_path),
        correction_source=correction_source,
        notes=notes,
        is_applied=False,
    )
    db.add(correction)
    await db.commit()
    await db.refresh(correction)

    # 如果需要立即应用
    vector_id = None
    if apply_immediately:
        engine = get_engine()
        try:
            # 读取图片并注册到识别引擎
            img = cv2.imdecode(np.fromfile(str(save_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                reg_result = engine.register_reagent(
                    image_input=str(save_path),
                    reagent_id=corrected_reagent_id,
                    reagent_name=corrected_reagent_name,
                    extra_info={
                        "correction_id": correction.id,
                        "angle": "correction",
                    },
                    image_save_path=str(save_path),
                )
                vector_id = reg_result.get("vector_id")
                
                # 创建ReagentImage记录
                img_record = ReagentImage(
                    reagent_id=corrected_reagent_id,
                    image_path=str(save_path),
                    vector_id=vector_id,
                    angle="correction",
                )
                db.add(img_record)
                
                # 更新图片计数
                await db.execute(
                    update(Reagent)
                    .where(Reagent.reagent_id == corrected_reagent_id)
                    .values(image_count=reagent.image_count + 1)
                )
                
                # 更新纠错记录
                await db.execute(
                    update(CorrectionLog)
                    .where(CorrectionLog.id == correction.id)
                    .values(is_applied=True, vector_id=vector_id)
                )
                await db.commit()
        except Exception as e:
            print(f"[API] 应用纠错失败: {e}")

    return {
        "success": True,
        "id": correction.id,
        "reagent_id": corrected_reagent_id,
        "vector_id": vector_id,
        "is_applied": apply_immediately,
        "message": f"纠错提交成功，已{'应用' if apply_immediately else '保存'}到识别系统",
    }


@app.delete("/api/reagents/{reagent_id}/images/{image_id}")
async def delete_reagent_image(
    reagent_id: str,
    image_id: int,
    db: AsyncSession = Depends(get_db),
):
    """删除单张已注册图片，并自动重建检索索引（保证删除后立即生效）"""
    # 1) 查试剂
    result = await db.execute(select(Reagent).where(Reagent.reagent_id == reagent_id))
    reagent = result.scalar_one_or_none()
    if not reagent:
        raise HTTPException(404, f"试剂 '{reagent_id}' 不存在")

    # 2) 查图片记录
    img_res = await db.execute(
        select(ReagentImage).where(
            ReagentImage.id == image_id,
            ReagentImage.reagent_id == reagent_id,
        )
    )
    img = img_res.scalar_one_or_none()
    if not img:
        raise HTTPException(404, f"图片记录 {image_id} 不存在")

    # 3) 删除文件（如果存在）
    img_path = Path(img.image_path)
    if img_path.exists():
        try:
            img_path.unlink()
        except Exception as e:
            raise HTTPException(500, f"删除图片文件失败: {e}")

    # 4) 删除 DB 记录
    await db.execute(ReagentImage.__table__.delete().where(ReagentImage.id == image_id))
    await db.commit()

    # 5) 重建索引（同时会回写每个试剂的 image_count）
    try:
        engine = get_engine()
        await engine.rebuild_index_from_images(str(IMAGES_DIR), db=db)
    except Exception as e:
        # 索引重建失败不应该让接口整体失败（至少图片已删除）
        print(f"[API] 删除图片后重建索引失败: {e}")

    # 6) 返回最新详情
    return {"success": True, "message": f"图片 {image_id} 已删除并更新检索索引"}


@app.get("/api/corrections")
async def get_corrections(
    db: AsyncSession = Depends(get_db),
    applied_only: bool = False,
    limit: int = 50,
):
    """获取纠错记录列表"""
    query = select(CorrectionLog)
    if applied_only:
        query = query.where(CorrectionLog.is_applied == True)
    query = query.order_by(CorrectionLog.timestamp.desc()).limit(limit)
    
    result = await db.execute(query)
    corrections = result.scalars().all()
    
    return [
        {
            "id": c.id,
            "timestamp": c.timestamp.isoformat() if c.timestamp else None,
            "original_recognition_id": c.original_recognition_id,
            "original_confidence": c.original_confidence,
            "corrected_reagent_id": c.corrected_reagent_id,
            "corrected_reagent_name": c.corrected_reagent_name,
            "corrected_image_path": c.corrected_image_path,
            "is_applied": c.is_applied,
            "is_exported": c.is_exported,
            "vector_id": c.vector_id,
            "correction_source": c.correction_source,
            "notes": c.notes,
        }
        for c in corrections
    ]


@app.post("/api/corrections/apply/{correction_id}")
async def apply_correction(
    correction_id: int,
    db: AsyncSession = Depends(get_db),
):
    """应用单个纠错到识别系统"""
    result = await db.execute(
        select(CorrectionLog).where(CorrectionLog.id == correction_id)
    )
    correction = result.scalar_one_or_none()
    
    if not correction:
        raise HTTPException(404, f"纠错记录 {correction_id} 不存在")
    
    if correction.is_applied:
        return {"success": True, "message": "该纠错已应用", "vector_id": correction.vector_id}
    
    # 应用纠错
    engine = get_engine()
    try:
        if not correction.corrected_image_path:
            raise HTTPException(400, "纠错记录缺少图片路径，无法应用。请重新提交纠错。")
        
        # 检查文件是否存在
        from pathlib import Path
        img_path = Path(correction.corrected_image_path)
        if not img_path.exists():
            print(f"[API] 纠错图片文件不存在: {correction.corrected_image_path}")
            raise HTTPException(400, f"纠错图片文件不存在: {correction.corrected_image_path}")
        
        if not img_path.is_file():
            print(f"[API] 纠错路径不是文件: {correction.corrected_image_path}")
            raise HTTPException(400, f"纠错路径不是文件: {correction.corrected_image_path}")
        
        img = cv2.imdecode(np.fromfile(correction.corrected_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[API] cv2.imdecode返回None，文件可能损坏或格式不支持: {correction.corrected_image_path}")
            raise HTTPException(400, f"无法读取纠错图片，文件可能损坏或格式不支持: {correction.corrected_image_path}")
        
        reg_result = engine.register_reagent(
            image_input=correction.corrected_image_path,
            reagent_id=correction.corrected_reagent_id,
            reagent_name=correction.corrected_reagent_name,
            extra_info={"correction_id": correction.id},
            image_save_path=correction.corrected_image_path,
        )
        vector_id = reg_result.get("vector_id")
        
        # 在ReagentImage表中创建记录，使纠错图片显示在试剂详情中
        img_record = ReagentImage(
            reagent_id=correction.corrected_reagent_id,
            image_path=correction.corrected_image_path,
            vector_id=vector_id,
            angle="correction",
        )
        db.add(img_record)
        
        # 更新试剂的图片计数
        await db.execute(
            update(Reagent)
            .where(Reagent.reagent_id == correction.corrected_reagent_id)
            .values(image_count=Reagent.image_count + 1)
        )
        
        # 更新纠错状态
        await db.execute(
            update(CorrectionLog)
            .where(CorrectionLog.id == correction_id)
            .values(is_applied=True, vector_id=vector_id)
        )
        await db.commit()
        
        return {
            "success": True,
            "vector_id": vector_id,
            "message": "纠错已成功应用到识别系统",
        }
    except Exception as e:
        raise HTTPException(500, f"应用纠错失败: {str(e)}")


@app.post("/api/corrections/batch-apply")
async def batch_apply_corrections(
    correction_ids: List[int],
    db: AsyncSession = Depends(get_db),
):
    """批量应用纠错"""
    results = []
    success_count = 0
    
    for cid in correction_ids:
        try:
            result = await db.execute(
                select(CorrectionLog).where(CorrectionLog.id == cid)
            )
            correction = result.scalar_one_or_none()
            
            if not correction or correction.is_applied:
                results.append({"correction_id": cid, "success": False, "message": "不存在或已应用"})
                continue
            
            engine = get_engine()
            
            # 检查文件是否存在
            from pathlib import Path
            img_path = Path(correction.corrected_image_path)
            if not correction.corrected_image_path or not img_path.exists():
                results.append({"correction_id": cid, "success": False, "message": f"图片文件不存在: {correction.corrected_image_path}"})
                continue
            
            img = cv2.imdecode(np.fromfile(correction.corrected_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                results.append({"correction_id": cid, "success": False, "message": f"无法读取图片(可能损坏): {correction.corrected_image_path}"})
                continue
            
            reg_result = engine.register_reagent(
                image_input=correction.corrected_image_path,
                reagent_id=correction.corrected_reagent_id,
                reagent_name=correction.corrected_reagent_name,
                extra_info={"correction_id": correction.id},
                image_save_path=correction.corrected_image_path,
            )
            vector_id = reg_result.get("vector_id")
            
            # 在ReagentImage表中创建记录，使纠错图片显示在试剂详情中
            img_record = ReagentImage(
                reagent_id=correction.corrected_reagent_id,
                image_path=correction.corrected_image_path,
                vector_id=vector_id,
                angle="correction",
            )
            db.add(img_record)
            
            # 更新试剂的图片计数
            await db.execute(
                update(Reagent)
                .where(Reagent.reagent_id == correction.corrected_reagent_id)
                .values(image_count=Reagent.image_count + 1)
            )
            
            # 更新纠错状态
            await db.execute(
                update(CorrectionLog)
                .where(CorrectionLog.id == cid)
                .values(is_applied=True, vector_id=vector_id)
            )
            
            success_count += 1
            results.append({"correction_id": cid, "success": True, "message": "应用成功"})
        except Exception as e:
            results.append({"correction_id": cid, "success": False, "message": str(e)})
    
    await db.commit()
    
    return {
        "success": True,
        "total": len(correction_ids),
        "success_count": success_count,
        "results": results,
    }


@app.get("/api/corrections/statistics")
async def get_correction_statistics(
    db: AsyncSession = Depends(get_db),
):
    """获取纠错统计信息"""
    engine = get_engine()
    stats = engine.get_stats()
    
    # 获取纠错统计
    result = await db.execute(select(CorrectionLog))
    corrections = result.scalars().all()
    
    total_corrections = len(corrections)
    applied_corrections = len([c for c in corrections if c.is_applied])
    unique_reagents = len(set(c.corrected_reagent_id for c in corrections))
    
    # 统计来源
    sources = {}
    for c in corrections:
        source = c.correction_source or "unknown"
        sources[source] = sources.get(source, 0) + 1
    
    return {
        "total_vectors": stats.get("total_vectors", 0),
        "correction_count": total_corrections,
        "applied_count": applied_corrections,
        "correction_ratio": f"{total_corrections / stats.get('total_vectors', 1) * 100:.2f}%" if stats.get("total_vectors", 0) > 0 else "0%",
        "unique_corrected_reagents": unique_reagents,
        "correction_sources": sources,
    }


@app.delete("/api/corrections/{correction_id}")
async def delete_correction(
    correction_id: int,
    db: AsyncSession = Depends(get_db),
):
    """删除纠错记录"""
    result = await db.execute(
        select(CorrectionLog).where(CorrectionLog.id == correction_id)
    )
    correction = result.scalar_one_or_none()
    
    if not correction:
        raise HTTPException(404, f"纠错记录 {correction_id} 不存在")
    
    # 删除图片文件
    if correction.corrected_image_path:
        img_path = Path(correction.corrected_image_path)
        if img_path.exists():
            img_path.unlink()
    
    # 从FAISS中删除向量
    if correction.vector_id is not None:
        engine = get_engine()
        engine.delete_vector(correction.vector_id)
    
    # 删除ReagentImage表中的对应记录
    if correction.vector_id is not None:
        await db.execute(
            ReagentImage.__table__.delete().where(ReagentImage.vector_id == correction.vector_id)
        )
    
    # 更新试剂的图片计数
    await db.execute(
        update(Reagent)
        .where(Reagent.reagent_id == correction.corrected_reagent_id)
        .values(image_count=Reagent.image_count - 1)
    )
    
    # 删除纠错记录
    await db.execute(
        CorrectionLog.__table__.delete().where(CorrectionLog.id == correction_id)
    )
    await db.commit()
    
    return {"success": True, "message": f"纠错记录 {correction_id} 已删除"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)