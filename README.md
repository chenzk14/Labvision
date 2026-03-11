## 🛠️ 环境配置

### 1. 创建Conda环境
```bash
conda create -n reagent_vision python=3.10 -y
conda activate reagent_vision
```

### 2. 安装PyTorch（CUDA 11.7版本）
```bash
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

### 3. 安装其他依赖
```bash
# 核心依赖
pip install timm faiss-cpu fastapi uvicorn sqlalchemy aiosqlite opencv-python albumentations pydantic python-multipart tqdm scikit-learn

# 多物体识别依赖（YOLOv8）
pip install ultralytics

# 前端依赖（可选，如果需要开发前端）
cd forntend
npm install

# 后端启动
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000

```

### 4. 验证安装
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import ultralytics; print('YOLOv8 installed successfully')"
```

---

## 📁 数据准备

### 目录结构
```
data/images/
├── 乙醇001/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── 乙醇002/
│   ├── 1.jpg
│   └── ...
├── 钠/
│   ├── 1.png
│   └── ...
└── ...
```

### 数据要求
- 每个试剂类别至少10-20张图片
- 支持多角度拍摄（正面、侧面、顶部）
- 图片格式：jpg、png
- 建议图片尺寸：≥ 640x480

### 数据增强策略
系统会自动进行以下数据增强：
- 随机旋转、翻转
- 颜色抖动
- 高斯模糊
- 随机裁剪

---

## 🎯 模型训练

### 训练命令
```bash
python scripts/train.py
```

### 训练配置（backend/config.py）
| 参数 | 默认值 | 说明 |
|------|--------|------|
| batch_size | 16 | 批次大小（适配1050Ti 4GB显存） |
| epochs | 50 | 最大训练轮数 |
| lr | 1e-4 | 学习率 |
| val_split | 0.2 | 验证集比例 |
| early_stop_patience | 10 | 早停耐心值 |

### 训练输出
- 模型保存：`saved_models/best_model.pth`
- 训练日志：`logs/`
- TensorBoard：`tensorboard --logdir logs`

### 预计训练时间
- 25张图片，5个类别：10-30分钟
- 100张图片，10个类别：30-60分钟

---

## 🔍 构建索引

### 构建命令
```bash
python scripts/build_index.py
```

### 索引输出
- FAISS索引：`data/embeddings/reagent.index`
- 元数据：`data/embeddings/metadata.json`

### 增量注册
无需重新训练，可直接注册新试剂：
```bash
# 通过API注册新试剂
POST /api/reagents/{reagent_id}/register-image
```

---

## 🚀 启动服务

### 后端API服务
```bash
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

### 前端服务
```bash
cd forntend
npm start
```

### 访问地址
- 前端界面：http://localhost:3000
- API文档：http://localhost:8000/docs
- 系统状态：http://localhost:8000/api/status

---

## ✨ 功能说明

### 1. 试剂录入
- 录入试剂基本信息（名称、CAS号、批次等）
- 拍摄多角度图片并注册
- 支持多张图片注册

### 2. 实时识别（单物体）
- 摄像头实时识别
- 自动识别模式
- 识别历史记录

### 3. 多试剂检测 
- 同时识别图片中的多个试剂瓶
- 支持摄像头实时检测
- 支持图片上传批量检测
- 自动绘制检测框和标签
- 可调整检测阈值

### 4. 试剂库管理
- 查看所有已注册试剂
- 试剂详情和图片列表
- 标记试剂为取出状态

### 5. 识别日志
- 查看所有识别记录
- 识别统计信息

---

## 📡 API接口

### 单物体识别
```bash
# 文件上传
POST /api/recognize
Content-Type: multipart/form-data

# Base64编码
POST /api/recognize/base64
Content-Type: application/json
{"image": "base64string"}
```

### 多物体识别 ⭐ 新接口
```bash
# 文件上传
POST /api/recognize/multiple
Content-Type: multipart/form-data
参数：file, min_confidence, topk

# Base64编码
POST /api/recognize/multiple/base64
Content-Type: application/json
{"image": "base64string", "min_confidence": 0.5, "topk": 5}
```

### 试剂管理
```bash
# 创建试剂
POST /api/reagents

# 注册图片
POST /api/reagents/{reagent_id}/register-image

# 查询试剂
GET /api/reagents
GET /api/reagents/{reagent_id}

# 删除试剂
DELETE /api/reagents/{reagent_id}
```

---

## 📂 项目结构

```
reagent-vision/
├── backend/
│   ├── config.py                    ← 所有超参数配置
│   ├── models/
│   │   └── metric_model.py          ← EfficientNet + ArcFace 核心模型
│   ├── core/
│   │   ├── dataset.py               ← 数据集 + 增强策略
│   │   ├── trainer.py               ← 训练引擎（混合精度/早停）
│   │   ├── recognition_engine.py    ← FAISS识别引擎（注册/检索）
│   │   ├── object_detector.py       ← YOLOv8目标检测（多物体）
│   │   └── database.py              ← SQLite数据库
│   └── api/
│       └── main.py                  ← FastAPI全部接口
├── forntend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx        ← 系统概览
│   │   │   ├── ReagentRegister.jsx  ← 试剂录入界面
│   │   │   ├── ReagentRecognize.jsx ← 实时识别界面
│   │   │   ├── MultipleRecognize.jsx ← 多试剂检测界面 ⭐
│   │   │   └── ReagentList.jsx     ← 试剂库管理
│   │   ├── services/
│   │   │   └── api.js              ← API服务封装
│   │   ├── styles/
│   │   │   └── global.css          ← 全局样式
│   │   ├── App.jsx                  ← 主应用
│   │   └── index.js                ← 入口文件
│   └── package.json
├── scripts/
│   ├── train.py                    ← 训练启动
│   ├── build_index.py              ← 构建FAISS索引
│   └── camera_test.py              ← 摄像头测试
├── data/
│   ├── images/                     ← 试剂图片数据集
│   ├── embeddings/                 ← FAISS索引文件
│   ├── db/                         ← SQLite数据库
│   └── logs/                       ← 训练日志
├── saved_models/                   ← 训练好的模型
├── docs/
│   └── TUTORIAL.md                 ← 手把手教程
├── example.txt                      ← 本文件
├── Environment.yml                  ← Conda环境配置
└── start.bat                       ← Windows启动脚本
```

---

## 🔧 高级配置

### 更换模型
在backend/config.py中修改：
```python
MODEL_CONFIG = {
    "backbone": "swin_tiny_patch4_window7_224",  # 细粒度识别推荐
    # 或 "resnet50", "convnext_tiny", "mobilenetv3_large_100"
}
```

### 调整识别阈值
```python
INFERENCE_CONFIG = {
    "similarity_threshold": 0.75,  # 提高阈值减少误识别
    # 或降低到0.65增加召回率
}
```

### 自定义数据增强
在backend/core/dataset.py中修改增强策略。

---

## 📝 更新日志

### v2.0 (最新)
- ✨ 新增多试剂检测功能（YOLOv8）
- ✨ 新增多物体识别API接口
- ✨ 新增多试剂检测前端页面
- 🐛 修复中文路径图片读取问题
- 📝 完善文档和使用说明

### v1.0
- 🎉 初始版本发布
- ✅ 单物体识别功能
- ✅ 试剂录入和管理
- ✅ 实时识别界面