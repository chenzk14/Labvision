# backend/core/database.py
"""
SQLite数据库模型
存储：试剂信息、录入记录、识别日志
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Float, DateTime,
    Boolean, Text, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import DATABASE_URL

Base = declarative_base()


class Reagent(Base):
    """试剂主表"""
    __tablename__ = "reagents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    reagent_id = Column(String(50), unique=True, nullable=False, index=True)  # 如'乙醇001'
    reagent_name = Column(String(100), nullable=False)    # 如'乙醇'
    cas_number = Column(String(50), nullable=True)        # CAS号
    manufacturer = Column(String(100), nullable=True)     # 厂商
    batch_number = Column(String(50), nullable=True)      # 批次
    expiry_date = Column(String(20), nullable=True)       # 保质期
    location = Column(String(50), nullable=True)          # 存放位置
    quantity = Column(Float, nullable=True)               # 数量
    unit = Column(String(20), nullable=True)              # 单位（mL, g等）
    is_active = Column(Boolean, default=True)             # 是否在库
    image_count = Column(Integer, default=0)              # 已注册图片数
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    notes = Column(Text, nullable=True)                   # 备注


class ReagentImage(Base):
    """试剂图片记录"""
    __tablename__ = "reagent_images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    reagent_id = Column(String(50), nullable=False, index=True)
    image_path = Column(String(500), nullable=False)
    vector_id = Column(Integer, nullable=True)     # FAISS向量索引位置
    angle = Column(String(20), nullable=True)      # 拍摄角度（正面/侧面/顶部）
    created_at = Column(DateTime, default=datetime.utcnow)


class RecognitionLog(Base):
    """识别日志"""
    __tablename__ = "recognition_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    recognized_id = Column(String(50), nullable=True)     # 识别结果
    confidence = Column(Float, nullable=True)             # 置信度
    is_correct = Column(Boolean, nullable=True)           # 人工标注是否正确
    image_path = Column(String(500), nullable=True)
    top1_id = Column(String(50), nullable=True)
    top1_score = Column(Float, nullable=True)
    action = Column(String(20), nullable=True)            # 'take_out' 取出 / 'put_in' 放入
    notes = Column(Text, nullable=True)


class CorrectionLog(Base):
    """纠错记录表 - 存储识别错误的纠错信息"""
    __tablename__ = "correction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # 原始识别信息
    original_recognition_id = Column(String(50), nullable=True)  # 原识别结果ID
    original_confidence = Column(Float, nullable=True)          # 原识别置信度
    original_image_path = Column(String(500), nullable=True)    # 原识别图片路径
    
    # 纠正信息
    corrected_reagent_id = Column(String(50), nullable=False, index=True)  # 纠正后的试剂ID
    corrected_reagent_name = Column(String(100), nullable=False)           # 纠正后的试剂名称
    corrected_image_path = Column(String(500), nullable=True)              # 纠正后的图片保存路径
    
    # 纠正状态
    is_applied = Column(Boolean, default=False)           # 是否已应用到FAISS索引
    is_exported = Column(Boolean, default=False)          # 是否已导出用于训练
    vector_id = Column(Integer, nullable=True)           # 应用后生成的向量ID
    
    # 元数据
    correction_source = Column(String(20), nullable=True)  # 纠正来源：'manual'人工 / 'auto'自动
    notes = Column(Text, nullable=True)                   # 纠正备注
    
    # 关联的识别日志ID
    recognition_log_id = Column(Integer, nullable=True)


# 数据库连接
async_engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db():
    """初始化数据库表"""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("[DB] 数据库初始化完成")


async def get_db():
    """FastAPI依赖注入"""
    async with AsyncSessionLocal() as session:
        yield session