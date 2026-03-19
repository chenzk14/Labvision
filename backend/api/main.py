"""
backend/api/main.py

修复：
[Fix-1] delete_correction: image_count -= 1 无下界保护 → 加 GREATEST(image_count-1, 0) 等价写法
[Fix-2] batch_apply_corrections: 循环内部异常导致部分 db.add 但 commit 失败 → 每条独立 commit
[Fix-3] submit_correction apply_immediately 路径: image_count 更新改为基于数据库查询，
         避免使用 Stale reagent.image_count（FastAPI 同一请求内 reagent 对象不会自动刷新）
[Fix-4] apply_correction 接口: 纠错应用调用 engine.register_reagent 改为 force_save=True
         确保纠错后立即持久化，下次识别立即生效
[Fix-5] delete_reagent_image: 删除后重建索引期间 image_count 已由 rebuild 正确同步，
         不需要额外手动递减
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
from sqlalchemy import select, update, func


app = FastAPI(
    title="试剂视觉识别系统",
    description="基于 Metric Learning + ArcFace 的试剂精细化识别",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


@app.on_event("startup")
async def startup():
    await init_db()
    print("[API] 服务启动完成")


# ══════════════════════════════════════════════════════════════════════════════
#  Pydantic 模型
# ══════════════════════════════════════════════════════════════════════════════

class ReagentCreate(BaseModel):
    reagent_id:    Optional[str]   = None
    reagent_name:  str
    cas_number:    Optional[str]   = None
    manufacturer:  Optional[str]   = None
    batch_number:  Optional[str]   = None
    expiry_date:   Optional[str]   = None
    location:      Optional[str]   = None
    quantity:      Optional[float] = None
    unit:          Optional[str]   = None
    notes:         Optional[str]   = None


class ReagentResponse(BaseModel):
    id:           int
    reagent_id:   str
    reagent_name: str
    cas_number:   Optional[str]
    manufacturer: Optional[str]
    batch_number: Optional[str]
    expiry_date:  Optional[str]
    location:     Optional[str]
    quantity:     Optional[float]
    unit:         Optional[str]
    is_active:    bool
    image_count:  int
    created_at:   datetime

    class Config:
        from_attributes = True


class RecognitionResult(BaseModel):
    recognized:     bool
    reagent_id:     Optional[str]
    reagent_name:   Optional[str]
    confidence:     Optional[float]
    confidence_pct: Optional[str]
    candidates:     List[dict]
    message:        str


# ══════════════════════════════════════════════════════════════════════════════
#  系统状态
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/status")
async def get_status():
    engine = get_engine()
    stats  = engine.get_stats()
    return {"status": "running", "device": DEVICE, "timestamp": datetime.now().isoformat(), **stats}


@app.post("/api/index/save")
async def save_index(force: bool = False):
    engine = get_engine()
    saved  = engine._save_index(force=force)
    return {"success": True, "saved": saved,
            "message": "索引保存成功" if saved else "索引未保存（批量保存策略）"}


# ══════════════════════════════════════════════════════════════════════════════
#  试剂管理
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/reagents", response_model=dict)
async def create_reagent(reagent: ReagentCreate, db: AsyncSession = Depends(get_db)):
    if not reagent.reagent_id:
        result   = await db.execute(select(Reagent).where(Reagent.reagent_name == reagent.reagent_name))
        existing = result.scalars().all()
        reagent_id = f"{reagent.reagent_name}{len(existing) + 1:03d}"
    else:
        reagent_id = reagent.reagent_id
        result     = await db.execute(select(Reagent).where(Reagent.reagent_id == reagent_id))
        if result.scalar_one_or_none():
            raise HTTPException(400, f"试剂ID '{reagent_id}' 已存在")

    (IMAGES_DIR / reagent_id).mkdir(parents=True, exist_ok=True)

    reagent_data               = reagent.dict()
    reagent_data["reagent_id"] = reagent_id
    db_reagent = Reagent(**reagent_data)
    db.add(db_reagent)
    await db.commit()
    await db.refresh(db_reagent)

    return {
        "success":    True,
        "id":         db_reagent.id,
        "reagent_id": db_reagent.reagent_id,
        "message":    f"试剂 {db_reagent.reagent_id} 创建成功，请上传图片完成注册",
    }


@app.post("/api/reagents/{reagent_id}/register-image")
async def register_reagent_image(
    reagent_id: str,
    file:  UploadFile = File(...),
    angle: str        = Form(default="front"),
    db:    AsyncSession = Depends(get_db),
):
    result  = await db.execute(select(Reagent).where(Reagent.reagent_id == reagent_id))
    reagent = result.scalar_one_or_none()
    if not reagent:
        raise HTTPException(404, f"试剂 '{reagent_id}' 不存在，请先创建")

    timestamp = int(time.time() * 1000)
    filename  = f"{timestamp}_{angle}.jpg"
    save_dir  = IMAGES_DIR / reagent_id
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    engine     = get_engine()
    reg_result = engine.register_reagent(
        image_input=str(save_path),
        reagent_id=reagent_id,
        reagent_name=reagent.reagent_name,
        extra_info={"cas_number": reagent.cas_number, "angle": angle},
        image_save_path=str(save_path),
        force_save=True,  # [Fix-4] 注册后立即写盘
    )

    db.add(ReagentImage(
        reagent_id=reagent_id,
        image_path=str(save_path),
        vector_id=reg_result.get("vector_id"),
        angle=angle,
    ))
    # [Fix-3] 用数据库计数而不是 stale 的 reagent.image_count + 1
    await db.execute(
        update(Reagent)
        .where(Reagent.reagent_id == reagent_id)
        .values(image_count=Reagent.image_count + 1)
    )
    await db.commit()

    return {
        "success":      True,
        "reagent_id":   reagent_id,
        "image_path":   f"/images/{reagent_id}/{filename}",
        "vector_id":    reg_result.get("vector_id"),
        "message":      "图片注册成功，已加入识别索引",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  识别
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/recognize", response_model=RecognitionResult)
async def recognize_reagent(
    file:       UploadFile = File(...),
    db:         AsyncSession = Depends(get_db),
    log_result: bool = True,
):
    content = await file.read()
    nparr   = np.frombuffer(content, np.uint8)
    img     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "无法解析图片")

    engine = get_engine()
    result = engine.recognize(img, topk=5)

    if log_result:
        db.add(RecognitionLog(
            recognized_id=result.get("reagent_id"),
            confidence=result.get("confidence"),
            top1_id=result["candidates"][0]["reagent_id"] if result["candidates"] else None,
            top1_score=result["candidates"][0]["similarity"] if result["candidates"] else None,
        ))
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
    b64 = data.get("image", "")
    if "," in b64:
        b64 = b64.split(",")[1]
    img_bytes = base64.b64decode(b64)
    nparr     = np.frombuffer(img_bytes, np.uint8)
    img       = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "无法解析 Base64 图像")
    return get_engine().recognize(img)


@app.post("/api/recognize/multiple")
async def recognize_multiple_reagents(
    file:           UploadFile = File(...),
    min_confidence: float      = Form(default=0.5),
    topk:           int        = Form(default=5),
    db:             AsyncSession = Depends(get_db),
):
    content = await file.read()
    nparr   = np.frombuffer(content, np.uint8)
    img     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "无法解析图片")

    engine = get_engine()
    result = engine.recognize_multiple(img, topk=topk, min_confidence=min_confidence)

    if result["recognized_objects"]:
        for obj in result["recognized_objects"]:
            db.add(RecognitionLog(
                recognized_id=obj.get("reagent_id"),
                confidence=obj.get("confidence"),
                action="multiple_recognition",
            ))
        await db.commit()
    return result


@app.post("/api/recognize/multiple/base64")
async def recognize_multiple_base64(data: dict):
    b64 = data.get("image", "")
    if "," in b64:
        b64 = b64.split(",")[1]
    img_bytes = base64.b64decode(b64)
    nparr     = np.frombuffer(img_bytes, np.uint8)
    img       = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "无法解析 Base64 图像")
    return get_engine().recognize_multiple(
        img,
        topk=data.get("topk", 5),
        min_confidence=data.get("min_confidence", 0.5),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  试剂列表 / 详情
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/reagents", response_model=List[dict])
async def list_reagents(
    db:          AsyncSession = Depends(get_db),
    active_only: bool = True,
    skip:        int  = 0,
    limit:       int  = 100,
):
    query = select(Reagent)
    if active_only:
        query = query.where(Reagent.is_active == True)
    query  = query.offset(skip).limit(limit)
    result = await db.execute(query)
    return [
        {
            "id":           r.id,
            "reagent_id":   r.reagent_id,
            "reagent_name": r.reagent_name,
            "cas_number":   r.cas_number,
            "manufacturer": r.manufacturer,
            "batch_number": r.batch_number,
            "expiry_date":  r.expiry_date,
            "location":     r.location,
            "quantity":     r.quantity,
            "unit":         r.unit,
            "is_active":    r.is_active,
            "image_count":  r.image_count,
            "created_at":   r.created_at.isoformat() if r.created_at else None,
        }
        for r in result.scalars().all()
    ]


@app.post("/api/reagents/sync-image-counts")
async def sync_image_counts(db: AsyncSession = Depends(get_db)):
    """同步所有试剂的 image_count 与实际 ReagentImage 记录数量"""
    reagent_result = await db.execute(select(Reagent))
    reagents       = reagent_result.scalars().all()

    img_result = await db.execute(select(ReagentImage.reagent_id, ReagentImage.id))
    image_counts: dict = {}
    for rid, _ in img_result.all():
        image_counts[rid] = image_counts.get(rid, 0) + 1

    updated_count = 0
    for reagent in reagents:
        actual = image_counts.get(reagent.reagent_id, 0)
        if reagent.image_count != actual:
            await db.execute(
                update(Reagent)
                .where(Reagent.reagent_id == reagent.reagent_id)
                .values(image_count=actual)
            )
            updated_count += 1
    await db.commit()
    return {"success": True, "message": f"已同步 {updated_count} 个试剂的图片数量"}


@app.get("/api/reagents/{reagent_id}")
async def get_reagent(reagent_id: str, db: AsyncSession = Depends(get_db)):
    result  = await db.execute(select(Reagent).where(Reagent.reagent_id == reagent_id))
    reagent = result.scalar_one_or_none()
    if not reagent:
        raise HTTPException(404, f"试剂 '{reagent_id}' 不存在")

    img_result = await db.execute(select(ReagentImage).where(ReagentImage.reagent_id == reagent_id))
    images     = img_result.scalars().all()

    return {
        "reagent": {
            "reagent_id":   reagent.reagent_id,
            "reagent_name": reagent.reagent_name,
            "cas_number":   reagent.cas_number,
            "manufacturer": reagent.manufacturer,
            "batch_number": reagent.batch_number,
            "expiry_date":  reagent.expiry_date,
            "location":     reagent.location,
            "quantity":     reagent.quantity,
            "unit":         reagent.unit,
            "image_count":  reagent.image_count,
            "created_at":   reagent.created_at.isoformat() if reagent.created_at else None,
            "notes":        reagent.notes,
        },
        "images": [
            {
                "id":         img.id,
                "path":       f"/images/{reagent_id}/{Path(img.image_path).name}",
                "angle":      img.angle,
                "created_at": img.created_at.isoformat() if img.created_at else None,
            }
            for img in images
        ],
    }


@app.delete("/api/reagents/{reagent_id}")
async def deactivate_reagent(reagent_id: str, db: AsyncSession = Depends(get_db)):
    await db.execute(update(Reagent).where(Reagent.reagent_id == reagent_id).values(is_active=False))
    await db.commit()
    return {"success": True, "message": f"试剂 {reagent_id} 已标记为取出"}


@app.delete("/api/reagents/{reagent_id}/permanent")
async def delete_reagent_permanent(reagent_id: str, db: AsyncSession = Depends(get_db)):
    result  = await db.execute(select(Reagent).where(Reagent.reagent_id == reagent_id))
    reagent = result.scalar_one_or_none()
    if not reagent:
        raise HTTPException(404, f"试剂 '{reagent_id}' 不存在")

    img_result = await db.execute(select(ReagentImage).where(ReagentImage.reagent_id == reagent_id))
    images     = img_result.scalars().all()

    engine       = get_engine()
    index_result = engine.delete_reagent(reagent_id)

    deleted_files = 0
    for img in images:
        img_path = Path(img.image_path)
        if img_path.exists():
            try:
                img_path.unlink()
                deleted_files += 1
            except Exception as e:
                print(f"[API] 删除图片失败: {img_path}, {e}")

    reagent_dir = IMAGES_DIR / reagent_id
    if reagent_dir.exists():
        try:
            shutil.rmtree(reagent_dir)
        except Exception as e:
            print(f"[API] 删除目录失败: {reagent_dir}, {e}")

    await db.execute(ReagentImage.__table__.delete().where(ReagentImage.reagent_id == reagent_id))
    await db.execute(Reagent.__table__.delete().where(Reagent.reagent_id == reagent_id))
    await db.commit()

    return {
        "success":         True,
        "reagent_id":      reagent_id,
        "deleted_vectors": index_result.get("deleted_count", 0),
        "deleted_files":   deleted_files,
        "message":         f"试剂 {reagent_id} 已永久删除",
    }


@app.delete("/api/reagents/{reagent_id}/images/{image_id}")
async def delete_reagent_image(
    reagent_id: str,
    image_id:   int,
    db:         AsyncSession = Depends(get_db),
):
    result  = await db.execute(select(Reagent).where(Reagent.reagent_id == reagent_id))
    if not result.scalar_one_or_none():
        raise HTTPException(404, f"试剂 '{reagent_id}' 不存在")

    img_res = await db.execute(
        select(ReagentImage).where(
            ReagentImage.id == image_id,
            ReagentImage.reagent_id == reagent_id,
        )
    )
    img = img_res.scalar_one_or_none()
    if not img:
        raise HTTPException(404, f"图片记录 {image_id} 不存在")

    img_path = Path(img.image_path)
    if img_path.exists():
        try:
            img_path.unlink()
        except Exception as e:
            raise HTTPException(500, f"删除图片文件失败: {e}")

    await db.execute(ReagentImage.__table__.delete().where(ReagentImage.id == image_id))
    await db.commit()

    # 重建索引（同时回写 image_count）
    try:
        engine = get_engine()
        await engine.rebuild_index_from_images(str(IMAGES_DIR), db=db)
    except Exception as e:
        print(f"[API] 删除图片后重建索引失败: {e}")

    return {"success": True, "message": f"图片 {image_id} 已删除并更新检索索引"}


# ══════════════════════════════════════════════════════════════════════════════
#  识别日志
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/logs")
async def get_recognition_logs(db: AsyncSession = Depends(get_db), limit: int = 50):
    result = await db.execute(
        select(RecognitionLog).order_by(RecognitionLog.timestamp.desc()).limit(limit)
    )
    return [
        {
            "id":            log.id,
            "timestamp":     log.timestamp.isoformat() if log.timestamp else None,
            "recognized_id": log.recognized_id,
            "confidence":    log.confidence,
            "top1_id":       log.top1_id,
            "action":        log.action,
        }
        for log in result.scalars().all()
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  纠错管理
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/corrections/submit")
async def submit_correction(
    file:                    UploadFile    = File(...),
    corrected_reagent_id:    str           = Form(...),
    corrected_reagent_name:  str           = Form(...),
    crop_x1:                 Optional[int] = Form(None),
    crop_y1:                 Optional[int] = Form(None),
    crop_x2:                 Optional[int] = Form(None),
    crop_y2:                 Optional[int] = Form(None),
    original_recognition_id: Optional[str]   = Form(None),
    original_confidence:     Optional[float] = Form(None),
    notes:                   Optional[str]   = Form(None),
    apply_immediately:       bool            = Form(True),
    correction_source:       str             = Form("web"),
    db:                      AsyncSession    = Depends(get_db),
):
    """
    提交纠错

    [Fix-3] image_count 更新改为 Reagent.image_count + 1（数据库表达式），
            避免使用查询时的 stale 值
    [Fix-4] register_reagent 使用 force_save=True，确保纠错后立即可识别
    """
    result  = await db.execute(select(Reagent).where(Reagent.reagent_id == corrected_reagent_id))
    reagent = result.scalar_one_or_none()
    if not reagent:
        raise HTTPException(404, f"试剂 '{corrected_reagent_id}' 不存在，请先创建")

    timestamp = int(time.time() * 1000)
    save_dir  = IMAGES_DIR / corrected_reagent_id
    save_dir.mkdir(parents=True, exist_ok=True)
    filename  = f"{timestamp}_correction.jpg"
    save_path = save_dir / filename

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # 可选裁剪
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
                    cv2.imwrite(str(save_path), img0[y1:y2, x1:x2])
        except Exception as e:
            print(f"[API] 裁剪纠错图片失败，使用原图: {e}")

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

    vector_id = None
    if apply_immediately:
        engine = get_engine()
        try:
            img = cv2.imdecode(np.fromfile(str(save_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                reg_result = engine.register_reagent(
                    image_input=str(save_path),
                    reagent_id=corrected_reagent_id,
                    reagent_name=corrected_reagent_name,
                    extra_info={"correction_id": correction.id, "angle": "correction"},
                    image_save_path=str(save_path),
                    force_save=True,  # [Fix-4]
                )
                vector_id = reg_result.get("vector_id")

                db.add(ReagentImage(
                    reagent_id=corrected_reagent_id,
                    image_path=str(save_path),
                    vector_id=vector_id,
                    angle="correction",
                ))
                # [Fix-3] 用表达式而非 stale 值
                await db.execute(
                    update(Reagent)
                    .where(Reagent.reagent_id == corrected_reagent_id)
                    .values(image_count=Reagent.image_count + 1)
                )
                await db.execute(
                    update(CorrectionLog)
                    .where(CorrectionLog.id == correction.id)
                    .values(is_applied=True, vector_id=vector_id)
                )
                await db.commit()
        except Exception as e:
            print(f"[API] 应用纠错失败: {e}")

    return {
        "success":    True,
        "id":         correction.id,
        "reagent_id": corrected_reagent_id,
        "vector_id":  vector_id,
        "is_applied": apply_immediately and vector_id is not None,
        "message":    f"纠错提交成功，已{'应用' if (apply_immediately and vector_id is not None) else '保存'}到识别系统",
    }


@app.get("/api/corrections")
async def get_corrections(
    db:           AsyncSession = Depends(get_db),
    applied_only: bool = False,
    limit:        int  = 50,
):
    query = select(CorrectionLog)
    if applied_only:
        query = query.where(CorrectionLog.is_applied == True)
    query  = query.order_by(CorrectionLog.timestamp.desc()).limit(limit)
    result = await db.execute(query)
    return [
        {
            "id":                    c.id,
            "timestamp":             c.timestamp.isoformat() if c.timestamp else None,
            "original_recognition_id": c.original_recognition_id,
            "original_confidence":   c.original_confidence,
            "corrected_reagent_id":  c.corrected_reagent_id,
            "corrected_reagent_name": c.corrected_reagent_name,
            "corrected_image_path":  c.corrected_image_path,
            "is_applied":            c.is_applied,
            "is_exported":           c.is_exported,
            "vector_id":             c.vector_id,
            "correction_source":     c.correction_source,
            "notes":                 c.notes,
        }
        for c in result.scalars().all()
    ]


@app.post("/api/corrections/apply/{correction_id}")
async def apply_correction(correction_id: int, db: AsyncSession = Depends(get_db)):
    """
    应用单个纠错
    [Fix-4] force_save=True 确保立即生效
    """
    result     = await db.execute(select(CorrectionLog).where(CorrectionLog.id == correction_id))
    correction = result.scalar_one_or_none()
    if not correction:
        raise HTTPException(404, f"纠错记录 {correction_id} 不存在")
    if correction.is_applied:
        return {"success": True, "message": "该纠错已应用", "vector_id": correction.vector_id}

    engine = get_engine()
    try:
        if not correction.corrected_image_path:
            raise HTTPException(400, "纠错记录缺少图片路径")

        img_path = Path(correction.corrected_image_path)
        if not img_path.exists() or not img_path.is_file():
            raise HTTPException(400, f"纠错图片不存在: {correction.corrected_image_path}")

        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, f"无法读取纠错图片: {correction.corrected_image_path}")

        reg_result = engine.register_reagent(
            image_input=correction.corrected_image_path,
            reagent_id=correction.corrected_reagent_id,
            reagent_name=correction.corrected_reagent_name,
            extra_info={"correction_id": correction.id},
            image_save_path=correction.corrected_image_path,
            force_save=True,  # [Fix-4]
        )
        vector_id = reg_result.get("vector_id")

        db.add(ReagentImage(
            reagent_id=correction.corrected_reagent_id,
            image_path=correction.corrected_image_path,
            vector_id=vector_id,
            angle="correction",
        ))
        await db.execute(
            update(Reagent)
            .where(Reagent.reagent_id == correction.corrected_reagent_id)
            .values(image_count=Reagent.image_count + 1)  # [Fix-3]
        )
        await db.execute(
            update(CorrectionLog)
            .where(CorrectionLog.id == correction_id)
            .values(is_applied=True, vector_id=vector_id)
        )
        await db.commit()

        return {"success": True, "vector_id": vector_id, "message": "纠错已成功应用到识别系统"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"应用纠错失败: {str(e)}")


@app.post("/api/corrections/batch-apply")
async def batch_apply_corrections(correction_ids: List[int], db: AsyncSession = Depends(get_db)):
    """
    批量应用纠错

    [Fix-2] 每条纠错独立 commit，避免单条失败导致整批回滚
    """
    results       = []
    success_count = 0
    engine        = get_engine()

    for cid in correction_ids:
        try:
            result     = await db.execute(select(CorrectionLog).where(CorrectionLog.id == cid))
            correction = result.scalar_one_or_none()

            if not correction:
                results.append({"correction_id": cid, "success": False, "message": "记录不存在"})
                continue
            if correction.is_applied:
                results.append({"correction_id": cid, "success": False, "message": "已应用"})
                continue

            img_path = Path(correction.corrected_image_path or "")
            if not img_path.exists():
                results.append({"correction_id": cid, "success": False,
                                 "message": f"图片不存在: {img_path}"})
                continue

            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                results.append({"correction_id": cid, "success": False, "message": "无法读取图片"})
                continue

            reg_result = engine.register_reagent(
                image_input=correction.corrected_image_path,
                reagent_id=correction.corrected_reagent_id,
                reagent_name=correction.corrected_reagent_name,
                extra_info={"correction_id": correction.id},
                image_save_path=correction.corrected_image_path,
                force_save=True,  # [Fix-4]
            )
            vector_id = reg_result.get("vector_id")

            db.add(ReagentImage(
                reagent_id=correction.corrected_reagent_id,
                image_path=correction.corrected_image_path,
                vector_id=vector_id,
                angle="correction",
            ))
            await db.execute(
                update(Reagent)
                .where(Reagent.reagent_id == correction.corrected_reagent_id)
                .values(image_count=Reagent.image_count + 1)  # [Fix-3]
            )
            await db.execute(
                update(CorrectionLog)
                .where(CorrectionLog.id == cid)
                .values(is_applied=True, vector_id=vector_id)
            )
            # [Fix-2] 每条独立 commit
            await db.commit()

            success_count += 1
            results.append({"correction_id": cid, "success": True, "message": "应用成功"})

        except Exception as e:
            await db.rollback()  # [Fix-2] 单条失败回滚当前事务
            results.append({"correction_id": cid, "success": False, "message": str(e)})

    return {
        "success":       True,
        "total":         len(correction_ids),
        "success_count": success_count,
        "results":       results,
    }


@app.get("/api/corrections/statistics")
async def get_correction_statistics(db: AsyncSession = Depends(get_db)):
    engine = get_engine()
    stats  = engine.get_stats()

    result      = await db.execute(select(CorrectionLog))
    corrections = result.scalars().all()

    total_corrections   = len(corrections)
    applied_corrections = sum(1 for c in corrections if c.is_applied)
    unique_reagents     = len(set(c.corrected_reagent_id for c in corrections))
    sources: dict       = {}
    for c in corrections:
        src = c.correction_source or "unknown"
        sources[src] = sources.get(src, 0) + 1

    return {
        "total_vectors":             stats.get("total_vectors", 0),
        "correction_count":          total_corrections,
        "applied_count":             applied_corrections,
        "correction_ratio":          (
            f"{total_corrections / stats.get('faiss_vectors', 1) * 100:.2f}%"
            if stats.get("faiss_vectors", 0) > 0 else "0%"
        ),
        "unique_corrected_reagents": unique_reagents,
        "correction_sources":        sources,
    }


@app.delete("/api/corrections/{correction_id}")
async def delete_correction(correction_id: int, db: AsyncSession = Depends(get_db)):
    """
    删除纠错记录

    [Fix-1] image_count 递减加下界保护，不能低于 0
    """
    result     = await db.execute(select(CorrectionLog).where(CorrectionLog.id == correction_id))
    correction = result.scalar_one_or_none()
    if not correction:
        raise HTTPException(404, f"纠错记录 {correction_id} 不存在")

    # 删除图片文件
    if correction.corrected_image_path:
        img_path = Path(correction.corrected_image_path)
        if img_path.exists():
            img_path.unlink()

    # 从 FAISS 中删除向量
    if correction.vector_id is not None:
        engine = get_engine()
        engine.delete_vector(correction.vector_id)
        # 删除对应的 ReagentImage 记录
        await db.execute(
            ReagentImage.__table__.delete().where(ReagentImage.vector_id == correction.vector_id)
        )
        # [Fix-1] image_count 递减，但不低于 0
        await db.execute(
            update(Reagent)
            .where(Reagent.reagent_id == correction.corrected_reagent_id)
            .values(image_count=func.max(Reagent.image_count - 1, 0))
        )

    await db.execute(CorrectionLog.__table__.delete().where(CorrectionLog.id == correction_id))
    await db.commit()

    return {"success": True, "message": f"纠错记录 {correction_id} 已删除"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)