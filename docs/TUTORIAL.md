# 🧪 试剂视觉识别系统 - 完整手把手教程

> 环境：Windows 10 + GTX 1050Ti (4GB) + conda
> 技术：EfficientNet-B0 + ArcFace Metric Learning + FAISS + FastAPI + React

---


## 第一章：原理说明

### 为什么能区分乙醇001和乙醇002？

普通分类器：学习"这是乙醇"，区分不了两瓶一样的乙醇。

**Metric Learning + ArcFace** 的做法：
1. 把每张图片压缩成 **512维的特征向量**（像指纹一样）
2. **ArcFace Loss** 训练时，强制同一瓶试剂的图片向量**相互靠近**，不同瓶的向量**相互远离**
3. 即使乙醇001和乙醇002外观99%相似，盖子齿轮、标签褶皱等**极细微差别**会体现在向量的差异上
4. 识别时：新拍图片 → 提取512维向量 → FAISS找最近邻 → 返回最相似的试剂ID

```
训练阶段（学习细微差别）:
乙醇001_图A → 向量[0.12, -0.34, ..., 0.89]  ←→ 距离近
乙醇001_图B → 向量[0.11, -0.35, ..., 0.88]  ←→ 距离近
乙醇002_图A → 向量[0.45,  0.12, ..., -0.23] ←→ 距离远（即使名称相同）

推理阶段（快速检索）:
未知图片 → 向量[0.12, -0.34, ..., 0.88] → FAISS检索 → 乙醇001 (相似度0.97)
```

### 增量学习（不需要重训练！）

新来一瓶乙醇003？
- 直接拍照 → 提取向量 → 加入FAISS索引
- 全程不需要重新训练模型
- 系统立刻就能识别乙醇003

---

## 第二章：环境安装

### 2.1 确认你的CUDA版本

```cmd
nvidia-smi
```
找到右上角的 "CUDA Version: xx.x"，1050Ti一般支持CUDA 11.x。

### 2.2 安装Anaconda

下载地址：https://www.anaconda.com/download
选择 Windows 64-bit，安装时勾选 "Add to PATH"。

### 2.3 创建虚拟环境

打开 **Anaconda Prompt**（不是普通CMD），运行：

```bash
# 创建Python 3.10环境
conda create -n reagent-vision python=3.10 -y

# 激活环境
conda activate reagent-vision
```

### 2.4 安装PyTorch（适配1050Ti的CUDA 11.7版本）

```bash
# 安装PyTorch（CUDA 11.7，适配1050Ti）
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 验证GPU是否可用
python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

✅ 应该看到：`CUDA可用: True` 和 `GPU: NVIDIA GeForce GTX 1050 Ti`

### 2.5 安装其余依赖

```bash
# 进入项目目录（把路径替换为你的实际路径）
cd C:\你的路径\reagent-vision

# 安装依赖
pip install timm==0.9.12 faiss-cpu==1.7.4 numpy opencv-python Pillow albumentations
pip install fastapi uvicorn python-multipart sqlalchemy aiosqlite pydantic
pip install tqdm scikit-learn matplotlib tensorboard PyYAML requests python-dotenv

# 验证安装
python -c "import timm, faiss, cv2, fastapi; print('所有依赖安装成功！')"
```

---

## 第三章：数据准备

### 3.1 目录结构

每个试剂建一个
文件夹，文件夹名就是**试剂唯一ID**：

```
reagent-vision/
└── backend/
    └── data/
        └── images/
            ├── 乙醇001/          ← 一个文件夹 = 一瓶试剂
            │   ├── front_001.jpg  ← 正面照
            │   ├── side_001.jpg   ← 侧面照
            │   └── top_001.jpg    ← 顶部照
            ├── 乙醇002/          ← 另一瓶乙醇（不同瓶！）
            │   ├── front_001.jpg
            │   └── side_001.jpg
            ├── 盐酸001/
            │   └── ...
            └── 硫酸001/
                └── ...
```

### 3.2 每瓶建议拍几张？

| 阶段 | 最少 | 推荐 | 效果 |
|------|------|------|------|
| 快速测试 | 1张 | - | 基本可用，精度一般 |
| 正式使用 | 3张 | 5-10张 | 不同角度 + 不同光线 |
| 高精度 | 10张 | 15-20张 | 多角度 + 细微差异场景 |

### 3.3 拍摄建议

- **正面**：标签正对摄像头
- **侧面**：45度斜拍
- **顶部**：俯拍瓶盖（齿轮等细节都在这里！）
- 光线**均匀**，避免强阴影
- 不同时间/光线条件各拍一张更好

---

## 第四章：训练模型

> 注意：如果你只有1-2类试剂，建议先跳到第五章直接用（不训练），等积累了5类以上再训练。

### 4.1 检查数据

```bash
conda activate reagent-vision
cd C:\你的路径\reagent-vision

# 统计数据
python -c "
from pathlib import Path
data_dir = Path('backend/data/images')
for d in sorted(data_dir.iterdir()):
    if d.is_dir():
        imgs = list(d.glob('*.jpg')) + list(d.glob('*.png'))
        print(f'{d.name}: {len(imgs)}张图片')
"
```

### 4.2 开始训练

```bash
# 标准训练（适合1050Ti）
python scripts/train.py --data_dir backend/data/images --epochs 50 --batch_size 16

# 如果显存不足改为 batch_size 8
python scripts/train.py --data_dir backend/data/images --epochs 50 --batch_size 8
```

训练过程你会看到：
```
Epoch   1/50 | Train: 4.2341 | Val: 4.1892 | LR: 1.00e-04 | Time: 12.3s
Epoch   2/50 | Train: 3.8123 | Val: 3.7654 | LR: 9.99e-05 | Time: 11.8s
  ✅ 保存最优模型 → saved_models/best_model.pth
...
✅ 训练完成！最优Val Loss: 0.8234
```

Loss不断下降 = 模型在正常学习 ✅

### 4.3 用TensorBoard查看训练曲线

```bash
# 新开一个命令行窗口
conda activate reagent-vision
cd C:\你的路径\reagent-vision
tensorboard --logdir=logs

# 浏览器打开 http://localhost:6006
```

### 4.4 构建识别索引

训练完成后，把所有图片的特征提取并存入FAISS：

```bash
python scripts/build_index.py --data_dir backend/data/images
```

输出：
```
✅ 索引构建完成！
  注册数量: 150
  唯一试剂ID: 30
  向量总数: 150
```

---

## 第五章：启动服务

### 5.1 启动后端API

```bash
conda activate reagent-vision
cd C:\你的路径\reagent-vision

# 方式1：直接运行
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

# 方式2：双击bat文件
# start_backend.bat
```

看到这个就成功了：
```
INFO:     Started server process
INFO:     Waiting for application startup.
[DB] 数据库初始化完成
[API] 服务启动完成
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

验证：浏览器打开 http://localhost:8000/docs （Swagger UI）

### 5.2 启动前端

新开一个命令行：

```bash
cd C:\你的路径\reagent-vision\frontend

# 首次安装依赖
npm install

# 启动
npm start
```

浏览器自动打开 http://localhost:3000

### 5.3 验证API状态

```bash
curl http://localhost:8000/api/status
```

或浏览器打开 http://localhost:8000/api/status

---

## 第六章：录入试剂（完整流程）

### 方式A：通过前端界面（推荐）

1. 打开 http://localhost:3000
2. 点击左侧 **「试剂录入」**
3. **步骤1**：填写试剂信息
   - 试剂唯一ID：`乙醇001`（同名不同瓶要用序号区分！）
   - 试剂名称：`乙醇`
   - 其他信息可选填
   - 点击「下一步」
4. **步骤2**：拍摄/上传图片
   - 点击「开启摄像头」
   - 点击「拍照」→ 选择角度（正面/侧面/顶部）→ 点击对应按钮注册
   - 或点击「上传图片」选择已有图片
   - 重复拍3-5张不同角度
5. **步骤3**：点击「完成注册」

### 方式B：通过摄像头脚本

```bash
# 先创建试剂记录（API）
curl -X POST http://localhost:8000/api/reagents \
  -H "Content-Type: application/json" \
  -d '{"reagent_id":"乙醇001","reagent_name":"乙醇"}'

# 然后用摄像头注册图片
python scripts/camera_test.py --mode register --reagent_id 乙醇001 --camera 0
# 按 S=正面  A=侧面  T=顶部  Q=退出
```

### 方式C：批量从已有图片导入

如果你已经有整理好的图片目录：

```bash
python scripts/build_index.py --data_dir backend/data/images
```

---

## 第七章：实时识别

### 方式A：前端界面

1. 打开 http://localhost:3000
2. 点击「实时识别」
3. 点击「开启摄像头」
4. 把试剂对准摄像头，按「立即识别」
5. 或打开「自动识别」开关，每1.5秒自动识别一次

### 方式B：摄像头脚本

```bash
python scripts/camera_test.py --mode recognize --camera 0
# 按 Space 立即识别  Q 退出
```

### 方式C：API调用

```python
import requests

# 发送图片识别
with open("test.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/api/recognize",
        files={"file": f}
    )
    result = resp.json()
    print(result)
    # {"recognized": true, "reagent_id": "乙醇001", "confidence": 0.94, ...}
```

---

## 第八章：常见问题

### Q: 显存不足 CUDA out of memory

```bash
# 在 backend/config.py 中修改
TRAIN_CONFIG["batch_size"] = 8  # 从16改为8

# 或者训练时指定
python scripts/train.py --batch_size 8
```

### Q: 识别置信度偏低（< 75%）

原因和解决：
1. **图片太少** → 每个试剂至少拍5张以上
2. **角度单一** → 拍摄正面、侧面、顶部各2-3张
3. **模型未训练** → 运行 `python scripts/train.py` 
4. **光线差异太大** → 在相同光照条件下拍摄注册图片

### Q: 乙醇001和乙醇002混淆识别

这说明两者的图片特征太相似，解决方案：
1. 专门拍摄**最有区别的部分**（如瓶盖特写、标签细节）
2. 增加注册图片数量（每个试剂10张以上）
3. 重新训练模型（确保数据集中两者都有足够样本）

### Q: 新安装试剂不需要重训练吗？

**对！** 不需要重新训练。
直接录入新试剂（前端界面或API），系统立刻就能识别。
模型只需要训练一次，之后所有新试剂都是**增量注册**。

### Q: 训练时报错 "至少需要2个类别"

需要至少2种不同的试剂才能训练。如果只有1种，先用默认权重（ImageNet预训练），
等积累了多种试剂后再训练。

### Q: 摄像头打不开

```bash
# 测试摄像头
python -c "
import cv2
cap = cv2.VideoCapture(0)
print('摄像头0:', cap.isOpened())
cap.release()
cap = cv2.VideoCapture(1)
print('摄像头1:', cap.isOpened())
cap.release()
"
# 如果都是False，检查驱动或使用摄像头索引1/2
```

---

## 第九章：项目维护

### 查看识别日志

```bash
curl http://localhost:8000/api/logs
```

### 备份数据

重要文件：
```
backend/data/embeddings/reagent.index   ← FAISS索引（最重要！）
backend/data/embeddings/metadata.json  ← 试剂元数据
backend/data/db/reagent.db             ← SQLite数据库
backend/data/images/                   ← 所有图片
saved_models/best_model.pth            ← 训练好的模型
```

### 重置系统

```bash
# 删除索引重新构建（不删除图片和数据库）
del backend\data\embeddings\reagent.index
del backend\data\embeddings\metadata.json
python scripts/build_index.py
```

---

## 附录：API接口速查

| 功能 | 方法 | 路径 |
|------|------|------|
| 系统状态 | GET | /api/status |
| 创建试剂 | POST | /api/reagents |
| 注册图片 | POST | /api/reagents/{id}/register-image |
| 识别图片 | POST | /api/recognize |
| Base64识别 | POST | /api/recognize/base64 |
| 试剂列表 | GET | /api/reagents |
| 试剂详情 | GET | /api/reagents/{id} |
| 标记取出 | DELETE | /api/reagents/{id} |
| 识别日志 | GET | /api/logs |

完整API文档：http://localhost:8000/docs