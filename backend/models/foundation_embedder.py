"""
Foundation-model feature extractors (DINOv2 / CLIP).

This module is optional: it requires extra dependencies (transformers).
The rest of the system can keep using EfficientNetEmbedder unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DependencyMissing(RuntimeError):
    pass


def _require_transformers():
    try:
        from transformers import AutoImageProcessor, AutoModel  # noqa: F401
        from transformers import CLIPImageProcessor, CLIPModel  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise _DependencyMissing(
            "未安装 transformers 依赖，无法使用 DINOv2/CLIP 特征提取。\n"
            "请先安装：pip install transformers accelerate safetensors\n"
            f"原始错误: {e}"
        )


@dataclass(frozen=True)
class EmbedderBundle:
    model: nn.Module
    preprocess: Callable[[torch.Tensor], torch.Tensor]
    embedding_dim: int
    model_id: str


class DinoV2Embedder(nn.Module):
    """
    DINOv2 image encoder (transformers).
    Output: L2-normalized CLS token embedding.
    """

    def __init__(self, model_name: str):
        super().__init__()
        _require_transformers()
        from transformers import AutoModel

        self.model_name = model_name
        self.backbone = AutoModel.from_pretrained(model_name)

    @property
    def embedding_dim(self) -> int:
        return int(self.backbone.config.hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        cls = out.last_hidden_state[:, 0, :]  # [B, D]
        return F.normalize(cls, p=2, dim=1)


class CLIPVisionEmbedder(nn.Module):
    """
    CLIP vision encoder (transformers).
    Output: L2-normalized image features (already projected to projection_dim).
    """

    def __init__(self, model_name: str):
        super().__init__()
        _require_transformers()
        from transformers import CLIPModel

        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name)

    @property
    def embedding_dim(self) -> int:
        return int(self.model.config.projection_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.model.get_image_features(pixel_values=pixel_values)  # [B, D]
        return F.normalize(feats, p=2, dim=1)


def create_foundation_embedder(
    feature_extractor: str,
    *,
    dinov2_model_name: str = "facebook/dinov2-base",
    clip_model_name: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
) -> EmbedderBundle:
    """
    Create (model, preprocess, dim, id) for foundation extractors.

    preprocess(tensor) expects float tensor [B, 3, H, W] in range [0, 1] RGB,
    and returns pixel_values ready for the model.
    """
    _require_transformers()
    from transformers import AutoImageProcessor, CLIPImageProcessor

    fe = feature_extractor.lower().strip()

    if fe == "dinov2":
        processor = AutoImageProcessor.from_pretrained(dinov2_model_name)
        model = DinoV2Embedder(dinov2_model_name).to(device).eval()

        def preprocess(x: torch.Tensor) -> torch.Tensor:
            # transformers expects a list of PIL or numpy; but it also accepts torch via images=
            # To avoid CPU<->GPU copies in processor, we pass CPU tensors.
            x_cpu = x.detach().cpu()
            # Convert to list of CHW tensors
            images = [img for img in x_cpu]  # each [3,H,W] float in [0,1]
            # x 已经是 [0,1]，避免重复 rescale 触发警告
            inputs = processor(images=images, return_tensors="pt", do_rescale=False)
            return inputs["pixel_values"].to(device)

        return EmbedderBundle(
            model=model,
            preprocess=preprocess,
            embedding_dim=model.embedding_dim,
            model_id=f"dinov2:{dinov2_model_name}",
        )

    if fe == "clip":
        processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        model = CLIPVisionEmbedder(clip_model_name).to(device).eval()

        def preprocess(x: torch.Tensor) -> torch.Tensor:
            x_cpu = x.detach().cpu()
            images = [img for img in x_cpu]
            inputs = processor(images=images, return_tensors="pt", do_rescale=False)
            return inputs["pixel_values"].to(device)

        return EmbedderBundle(
            model=model,
            preprocess=preprocess,
            embedding_dim=model.embedding_dim,
            model_id=f"clip:{clip_model_name}",
        )

    raise ValueError(f"不支持的 feature_extractor: {feature_extractor}（可选：dinov2/clip）")

