# backend/models/metric_model.py
"""
核心识别模型:
EfficientNet-B0 特征提取器 + ArcFace Loss
实现细粒度试剂识别的Metric Learning

原理：
- ArcFace在超球面上施加角度边距，
  同类试剂的嵌入向量更聚拢，
  不同试剂(即使名称相同但包装不同)更分离
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from typing import Optional


class EfficientNetEmbedder(nn.Module):
    """
    EfficientNet-B2 特征提取器
    输出512维归一化嵌入向量
    """

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()

        # 加载EfficientNet-B2主干（ImageNet预训练）
        self.backbone = timm.create_model(
            "efficientnet_b2",
            pretrained=pretrained,
            num_classes=0,  # 去掉分类头，只要特征
            global_pool="avg",  # 全局平均池化
        )

        # backbone输出维度
        backbone_out_dim = self.backbone.num_features  # 1280

        # 投影头: 1280 → 512
        self.projector = nn.Sequential(
            nn.Linear(backbone_out_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 图像张量 [B, 3, H, W]
        Returns:
            embeddings: L2归一化嵌入向量 [B, 512]
        """
        # 提取backbone特征
        features = self.backbone(x)  # [B, 1280]

        # 投影到512维
        embeddings = self.projector(features)  # [B, 512]

        # L2归一化 - ArcFace必须在单位超球面上
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss (Additive Angular Margin Loss)

    核心思想：在角度空间添加边距m，
    使同类之间更紧凑、异类之间更分离

    公式: L = -log( e^(s*cos(θ+m)) / (e^(s*cos(θ+m)) + Σ e^(s*cos(θj))) )

    对于试剂识别：乙醇001 和 乙醇002 虽然都是乙醇，
    但通过ArcFace会学习到各自特有的细微特征，
    从而在嵌入空间中分开。
    """

    def __init__(
            self,
            embedding_dim: int = 512,
            num_classes: int = 100,
            margin: float = 0.5,
            scale: float = 64.0,
            easy_margin: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin

        # 可学习的类中心权重矩阵 [num_classes, embedding_dim]
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)

        # 预计算
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: L2归一化嵌入 [B, 512]
            labels: 类别标签 [B]
        Returns:
            loss: ArcFace损失值
        """
        # 归一化权重
        weight_norm = F.normalize(self.weight, p=2, dim=1)  # [num_classes, 512]

        # 计算余弦相似度
        cosine = F.linear(embeddings, weight_norm)  # [B, num_classes]
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        # 计算 sin(θ)
        sine = torch.sqrt(1.0 - cosine ** 2)

        # cos(θ + m) = cos(θ)*cos(m) - sin(θ)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # one-hot编码
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # 目标类用 phi，其他类用 cosine
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        loss = F.cross_entropy(output, labels.long())
        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss - 辅助ArcFace进一步约束特征空间

    思想：
    anchor(乙醇001图A) 与 positive(乙醇001图B) 的距离，
    应小于 anchor 与 negative(乙醇002) 的距离 + margin
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
            self,
            anchor: torch.Tensor,
            positive: torch.Tensor,
            negative: torch.Tensor
    ) -> torch.Tensor:
        dist_ap = F.pairwise_distance(anchor, positive, p=2)
        dist_an = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(dist_ap - dist_an + self.margin).mean()
        return loss


class ReagentRecognitionModel(nn.Module):
    """
    完整的试剂识别模型
    训练阶段：EfficientNet + ArcFace Loss (+ Triplet Loss)
    推理阶段：只用EfficientNet提取嵌入，FAISS检索
    """

    def __init__(
            self,
            num_classes: int,
            embedding_dim: int = 512,
            pretrained: bool = True,
            arcface_margin: float = 0.5,
            arcface_scale: float = 64.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedder = EfficientNetEmbedder(
            embedding_dim=embedding_dim,
            pretrained=pretrained
        )
        self.arcface = ArcFaceLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            margin=arcface_margin,
            scale=arcface_scale,
        )
        self.triplet_loss = TripletLoss(margin=0.3)

    def forward(
            self,
            x: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
    ):
        embeddings = self.embedder(x)

        if labels is not None:
            arc_loss = self.arcface(embeddings, labels)
            return embeddings, arc_loss
        return embeddings

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """推理阶段只提取嵌入"""
        self.eval()
        with torch.no_grad():
            return self.embedder(x)