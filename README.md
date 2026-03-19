# 试剂识别系统 (Reagent Vision System)

基于深度学习的试剂智能识别系统，支持多物体检测、试剂库管理和纠错学习功能。

## ✨ 核心特性

- **智能识别**：基于EfficientNetV2-S + ArcFace的细粒度识别
- **多物体检测**：YOLOv11同时识别多个试剂瓶
- **试剂库管理**：完整的试剂录入、查询、删除功能
- **纠错学习**：持续学习机制，识别错误可人工纠错
- **实时监控**：识别日志、统计分析、系统状态监控
- **快速部署**：一键部署包，支持离线运行
- **显存优化**：混合精度训练 + 梯度累积，适配1050Ti 4GB显存

---

## 🛠️ 环境配置

### 创建Conda环境
```bash
conda env create -f environment.yml
conda activate reagent-vision
```

# 前端依赖
```bash
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
├── images/                     # 图片数据集
│   ├── 类别001/
│   │   ├── 1773279184217_front.jpg
│   │   └── ...
│   ├── 类别002/
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
- 每个类别至少10-20张图片
- 支持多角度拍摄（正面、侧面、顶部）
- 图片格式：jpg、png
- 建议图片尺寸：≥ 640x480
- 小样本模式：自动优化训练参数（lr=3e-4, epochs=80, val_split=0.2）

---

## 🎯 模型训练

### 训练命令
```bash
# 小样本模式（默认，自动覆盖参数，训练后自动建索引）
python scripts/train.py --data_dir data/images

# 想用 config.py 中的原始参数
python scripts/train.py --data_dir data/images --no_small_sample

# 禁用自动构建索引
python scripts/train.py --data_dir data/images --no-build-index
```

### 训练配置（backend/config.py）
| 参数 | 默认值 | 说明 |
|------|--------|------|
| backbone | efficientnetv2_s | 主干网络（EfficientNetV2-S） |
| img_size | 288 | 输入图像尺寸（EfficientNetV2-S推荐384，288适配1050Ti 4GB显存） |
| batch_size | 4 | 批次大小（配合梯度累积，有效batch=16） |
| epochs | 120 | 最大训练轮数 |
| lr | 3e-4 | 学习率（小样本优化） |
| val_split | 0.3 | 验证集比例 |
| early_stop_patience | 30 | 早停耐心值 |
| arcface_weight | 1.0 | ArcFace Loss 权重 |
| triplet_weight | 0.0 | TripletLoss 权重（小样本时关闭） |
| accumulation_steps | 4 | 梯度累积步数 |
| use_amp | True | 混合精度训练（节省显存） |

### 训练输出
- 模型保存：`saved_models/best_model.pth`
- 训练日志：`logs/`
- FAISS索引：`data/embeddings/reagent.index`（训练后自动构建）
- TensorBoard：`tensorboard --logdir logs`

### 预计训练时间
- 25张图片，3个类别：5-15分钟（小样本模式）
- 100张图片，10个类别：20-40分钟
- 1050Ti 4GB显存：显存占用约1500-2000M
- 梯度累积：有效batch_size=4×4=16

### 训练监控
```bash
tensorboard --logdir=logs
```
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

| 模块 | 功能 | 子功能 |
|------|------|--------|
| 试剂录入 | 基本信息管理 | 录入试剂名称、CAS号、批次等 |
|        | 图片采集 | 拍摄多角度图片并注册 |
|        | 图片管理 | 支持多张图片注册 |
|        | 预览 | 实时预览已注册图片 |
| 多试剂检测 | 目标检测 | 同时识别图片中的多个试剂瓶 |
|          | 实时检测 | 支持摄像头实时检测 |
|          | 批量检测 | 支持图片上传批量检测 |
|          | 结果展示 | 自动绘制检测框和标签 |
| 试剂库管理 | 数据查看 | 查看所有已注册试剂 |
|            | 详情管理 | 试剂详情和图片列表 |
|            | 图片维护 | 添加/删除试剂图片 |
|            | 数据删除 | 永久删除试剂 |
|            | 搜索功能 | 搜索过滤功能 |
| 纠错管理 | 数据采集 | 摄像头纠错：实时拍摄并提交纠错 |
|   ⭐ 新功能       | 记录管理 | 查看纠错记录 |
|          | 应用机制 | 应用纠错：将纠错样本应用到识别系统 |
|          | 批量处理 | 批量应用所有未应用的纠错 |

---

## 📂 项目结构

```
reagent-vision/
├── backend/
│   ├── config.py                    ← 所有超参数配置
│   ├── models/
│   │   └── metric_model.py          ← EfficientNetV2-S + ArcFace 核心模型
│   ├── core/
│   │   ├── dataset.py               ← 数据集 + 增强策略
│   │   ├── trainer.py               ← 训练引擎（混合精度/梯度累积/早停）
│   │   ├── recognition_engine.py    ← FAISS识别引擎（注册/检索）
│   │   ├── object_detector.py       ← YOLOv11目标检测（多物体）
│   │   └── database.py              ← SQLite数据库
│   └── api/
│       └── main.py                  ← FastAPI全部接口
├── forntend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ImageCropper.jsx     ← 图片裁剪组件
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx        ← 系统概览
│   │   │   ├── ReagentRegister.jsx  ← 试剂录入界面
│   │   │   ├── MultipleRecognize.jsx ← 多试剂检测界面
│   │   │   ├── ReagentList.jsx      ← 试剂库管理
│   │   │   └── CorrectionManage.jsx ← 纠错管理界面 ⭐
│   │   ├── services/
│   │   │   └── api.js              ← API服务封装
│   │   ├── styles/
│   │   │   └── global.css          ← 全局样式
│   │   ├── App.jsx                 ← 主应用
│   │   └── index.js                ← 入口文件
│   └── package.json
├── scripts/
│   ├── train.py                    ← 训练启动
│   ├── build_index.py              ← 构建FAISS索引
│   ├── correction_manager.py       ← 纠错管理脚本 ⭐
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
├── Environment.yml                 ← Conda环境配置
└── README.md                       ← 本文件
```

---

## 🔧 高级配置

### 更换模型
在backend/config.py中修改：
```python
MODEL_CONFIG = {
    "backbone": "efficientnetv2_s",  # 默认：EfficientNetV2-S
    "img_size": 288,  # 输入尺寸，V2-S推荐384
    # 其他选项：
    # "efficientnetv2_t" - 更小，显存占用更低
    # "efficientnetv2_m" - 更大，精度更高
}
```

### 显存优化配置
系统已针对1050Ti 4GB显存优化：
- ✅ 混合精度训练（AMP）：节省30%显存
- ✅ 梯度累积：有效batch_size=4×4=16
- ✅ 输入尺寸优化：288×288（平衡精度和显存，EfficientNetV2-S推荐384）
- ✅ TripletLoss默认关闭：小样本时节省显存
- ✅ 投影头优化：1280→512→256，避免直接映射到256维导致过拟合

如需进一步优化显存：
```python
# 减小输入尺寸
"img_size": 256  # 从288降低

# 减小batch size
"batch_size": 2  # 从4降低

# TripletLoss默认已关闭
"triplet_weight": 0.0

# 使用更小的模型
"backbone": "efficientnetv2_t"
```

### 调整识别阈值
```python
INFERENCE_CONFIG = {
    "similarity_threshold": 0.7,  # 默认0.7，提高阈值减少误识别
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
python scripts/correction_manager.py --action verify --reagent_id 类别001
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

**2. 识别准确率低**
- 增加训练数据量
- 调整数据增强策略
- 尝试不同的模型backbone
- 使用纠错系统持续优化

### 日志查看
```bash
# 训练日志
tensorboard --logdir logs
```

---

## 📝 更新日志

### v2.2 (最新)
| 版本 | 类型 | 内容 |
|------|------|------|
| v2.2 | ⚡ 性能 | 升级到EfficientNetV2-S（训练速度提升2-3倍） |
|      |        | 添加混合精度训练（节省30%显存） |
|      |        | 添加梯度累积（有效batch_size=16） |
|      |        | 优化输入尺寸为288（EfficientNetV2-S推荐384） |
|      | 🎯 模型 | 升级YOLOv8到YOLOv11（精度更高） |
|      |        | TripletLoss默认关闭（小样本优化） |
|      |        | 学习率优化为3e-4（小样本收敛更快） |
|      | 🔧 依赖 | 升级ultralytics到8.2.0+（支持YOLOv11） |
|      |        | 升级timm到0.9.14+（支持EfficientNetV2预训练） |
|      | 🐛 修复 | 修复学习率调度器警告 |
|      |        | 修复TripletLoss未生效问题 |
|      |        | 修复forward方法逻辑错误 |
|      | ✨ 新增 | 训练后自动构建FAISS索引 |
|------|------|------|
| v2.1 | ✨ 新增 | 新增纠错管理系统 |
|      |        | 摄像头纠错功能 |
|      |        | 批量应用纠错 |
|      |        | 纠错统计和验证 |
|      | 🐛 修复 | 修复应用纠错后图片不显示在试剂详情的问题 |
|      |        | 修复删除纠错记录时的数据一致性问题 |
|      | 📝 文档 | 完善纠错系统文档 |
|      | 📦 新增 | 新增部署包功能 |
|      |        | 模型打包脚本 |
|      |        | 离线部署支持 |
|      |        | 部署包测试工具 |
|------|------|------|
| v2.0 | ✨ 新增 | 新增多试剂检测功能（YOLOv8） |
|      |        | 新增多物体识别API接口 |
|      |        | 新增多试剂检测前端页面 |
|      | 🐛 修复 | 修复中文路径图片读取问题 |
|      | 📝 文档 | 完善文档和使用说明 |
|------|------|------|
| v1.0 | 🎉 发布 | 初始版本发布 |
|      | ✅ 功能 | 单物体识别功能 |
|      |        | 试剂录入和管理 |
|      |        | 实时识别界面 |