# 试剂识别系统 (Reagent Vision System)

基于深度学习的试剂智能识别系统，支持单物体识别、多物体检测、试剂库管理和纠错学习功能。

## ✨ 核心特性

- 🔍 **智能识别**：基于EfficientNet + ArcFace的细粒度识别
- 🎯 **多物体检测**：YOLOv8同时识别多个试剂瓶
- 📚 **试剂库管理**：完整的试剂录入、查询、删除功能
- 🔄 **纠错学习**：持续学习机制，识别错误可人工纠错
- 📊 **实时监控**：识别日志、统计分析、系统状态监控
- 🚀 **快速部署**：一键部署包，支持离线运行

---

## 🛠️ 环境配置

### 1. 创建Conda环境
```bash
conda env create -f environment.yml
conda activate reagent-vision
```

# 前端依赖（可选，如果需要开发前端）
cd forntend
npm install
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
data/
├── images/                     # 试剂图片数据集
│   ├── 乙醇001/
│   │   ├── 1773279184217_front.jpg
│   │   └── ...
│   ├── 乙醇002/
│   │   ├── 1773279742602_front.jpg
│   │   └── ...
├── embeddings/                 # FAISS索引文件
│   ├── reagent.index
│   └── metadata.json
├── db/                         # SQLite数据库
│   └── reagent.db
└── logs/                       # 训练日志
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

### 训练监控
```bash
tensorboard --logdir=logs
```

监控指标：
- Loss/train - 训练损失（应该持续下降）
- Loss/val - 验证损失（不应持续上升）
- Accuracy/train - 训练准确率（应该持续上升）
- Accuracy/val - 验证准确率（应该稳定上升）
- Learning Rate - 学习率变化

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

### 摄像头测试
```bash
python scripts/camera_test.py
```

---

## 🚀 启动服务

### 分别启动

**后端API服务**
```bash
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

**前端服务**
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
- 实时预览已注册图片

### 2. 多试剂检测 
- 同时识别图片中的多个试剂瓶
- 支持摄像头实时检测
- 支持图片上传批量检测
- 自动绘制检测框和标签
- 可调整检测阈值

### 3. 试剂库管理
- 查看所有已注册试剂
- 试剂详情和图片列表
- 添加/删除试剂图片
- 永久删除试剂
- 搜索过滤功能

### 4. 识别日志
- 查看所有识别记录
- 识别统计信息
- 按时间/试剂筛选

### 5. 纠错管理 ⭐ 新功能
- 摄像头纠错：实时拍摄并提交纠错
- 上传纠错：上传已有的正确图片
- 查看纠错记录
- 应用纠错：将纠错样本应用到识别系统
- 批量应用：一次性应用所有未应用的纠错
- 纠错统计：查看纠错效果

### 6. 系统监控
- 已注册试剂数量
- 图片向量数量
- 注册记录统计
- 实时数据刷新

---

## 📡 API接口

### 系统状态
```bash
GET /api/status
```

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

### 多物体识别
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

# 查询试剂列表
GET /api/reagents

# 查询试剂详情
GET /api/reagents/{reagent_id}

# 删除试剂（软删除）
DELETE /api/reagents/{reagent_id}

# 永久删除试剂
DELETE /api/reagents/{reagent_id}/permanent

# 删除试剂图片
DELETE /api/reagents/{reagent_id}/images/{image_id}
```

### 纠错管理
```bash
# 提交纠错
POST /api/corrections/submit
参数：file, corrected_reagent_id, corrected_reagent_name, notes, apply_immediately

# 获取纠错列表
GET /api/corrections

# 获取纠错统计
GET /api/corrections/statistics

# 应用单个纠错
POST /api/corrections/apply/{correction_id}

# 批量应用纠错
POST /api/corrections/batch-apply
参数：correction_ids (数组)

# 删除纠错
DELETE /api/corrections/{correction_id}
```

### 识别日志
```bash
# 获取识别日志
GET /api/recognitions

# 获取识别统计
GET /api/recognitions/statistics
```

---

## 📂 项目结构

```
reagent-vision/
├── backend/
│   ├── config.py                    ← 所有超参数配置
│   ├── models/
│   │   ├── metric_model.py          ← EfficientNet + ArcFace 核心模型
│   │   └── foundation_embedder.py   ← DINOv2/CLIP 基础模型
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
│   │   │   ├── ReagentRecognize.jsx ← 实时识别界面（单试剂）
│   │   │   ├── MultipleRecognize.jsx ← 多试剂检测界面
│   │   │   ├── ReagentList.jsx      ← 试剂库管理
│   │   │   └── CorrectionManage.jsx ← 纠错管理界面 ⭐
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
│   ├── camera_test.py              ← 摄像头测试
│   ├── correction_manager.py      ← 纠错管理脚本 ⭐
│   ├── package_model.py            ← 模型打包脚本
│   └── test_package.py             ← 部署包测试脚本
├── data/
│   ├── images/                     ← 试剂图片数据集
│   ├── embeddings/                 ← FAISS索引文件
│   ├── db/                         ← SQLite数据库
│   └── logs/                       ← 训练日志
├── saved_models/                   ← 训练好的模型
├── deploy_package/                 ← 部署包 ⭐
│   ├── models/
│   │   └── best_model.pth
│   ├── embeddings/
│   │   ├── reagent.index
│   │   └── metadata.json
│   ├── config/
│   │   ├── config.json
│   │   └── class_mapping.json
│   ├── inference.py                ← 推理脚本
│   ├── requirements.txt
│   └── README.md
├── docs/
│   ├── TUTORIAL.md                 ← 手把手教程
│   └── correction_workflow.md       ← 纠错系统使用指南 ⭐
├── Environment.yml                 ← Conda环境配置
└── README.md                       ← 本文件
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

## 📦 部署包使用

### 打包模型
```bash
python scripts/package_model.py
```

### 测试部署包
```bash
python scripts/test_package.py
```

### 部署包结构
```
deploy_package/
├── models/
│   └── best_model.pth
├── embeddings/
│   ├── reagent.index
│   └── metadata.json
├── config/
│   ├── config.json
│   └── class_mapping.json
├── inference.py
├── requirements.txt
└── README.md
```

### 离线部署
1. 将 `deploy_package` 目录复制到目标机器
2. 安装依赖：`pip install -r requirements.txt`
3. 运行推理：`python inference.py`

## 🛠️ 纠错系统

### 使用纠错管理脚本
```bash
# 查看纠错统计
python scripts/correction_manager.py --action stats

# 查看所有纠错记录
python scripts/correction_manager.py --action list

# 应用所有未应用的纠错
python scripts/correction_manager.py --action apply --all

# 应用指定纠错ID
python scripts/correction_manager.py --action apply --id 1

# 摄像头纠错模式
python scripts/correction_manager.py --action camera --camera 0

# 导出纠错样本用于训练
python scripts/correction_manager.py --action export --output data/corrections

# 验证纠错质量
python scripts/correction_manager.py --action verify --reagent_id 乙醇001
```

### 纠错工作流程
1. **发现识别错误**：在识别过程中发现系统错误识别
2. **提交纠错**：通过摄像头或上传方式提交正确的样本
3. **应用纠错**：将纠错样本应用到识别系统
4. **持续优化**：系统会自动学习，提升识别准确率

## 🔍 故障排查

### 常见问题

**1. CUDA不可用**
```bash
# 检查CUDA版本
nvidia-smi

# 重新安装PyTorch
pip uninstall torch torchvision
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

**2. 摄像头无法开启**
```bash
# 检查摄像头权限
# Windows：设置 > 隐闲 > 相机
# Linux：sudo chmod 666 /dev/video0
```

**3. FAISS索引构建失败**
```bash
# 检查图片路径是否正确
# 确保图片格式支持（jpg, png）
# 检查图片是否损坏
```

**4. 识别准确率低**
- 增加训练数据量
- 调整数据增强策略
- 尝试不同的模型backbone
- 使用纠错系统持续优化

**5. 纠错图片不显示**
- 确保纠错已成功应用
- 刷新试剂详情页面
- 检查数据库中的ReagentImage记录

### 日志查看
```bash
# 后端日志
# 直接查看终端输出

# 前端日志
# 打开浏览器开发者工具 > Console

# 训练日志
tensorboard --logdir logs
```

---

## 📝 更新日志

### v2.1 (最新)
- ✨ 新增纠错管理系统
  - 摄像头纠错功能
  - 上传纠错功能
  - 批量应用纠错
  - 纠错统计和验证
- 🐛 修复应用纠错后图片不显示在试剂详情的问题
- 🐛 修复删除纠错记录时的数据一致性问题
- 📝 完善纠错系统文档
- 📦 新增部署包功能
  - 模型打包脚本
  - 离线部署支持
  - 部署包测试工具

### v2.0
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