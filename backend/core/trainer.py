# backend/core/trainer.py
"""
训练器实现
整合模型、数据集和训练逻辑
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import faiss
from tqdm import tqdm

from backend import config
from backend.core.dataset import create_dataloaders, TripletDataset
from backend.models.metric_model import ReagentRecognitionModel


class ReagentTrainer:
    """
    试剂识别模型训练器

    功能:
    - 数据加载与预处理
    - 模型训练与验证
    - 损失函数组合 (ArcFace + Triplet)
    - 模型保存与加载
    - 训练过程可视化
    """

    def __init__(
            self,
            data_dir: str,
            model_save_dir: Optional[str] = None,
            log_dir: Optional[str] = None,
    ):
        # [优化] 梯度累积步数：模拟 BatchSize = 8 * 8 = 64
        self.accumulation_steps = 8

        # [优化] 自动混合精度加速 (1050Ti支持)
        self.scaler = torch.cuda.amp.GradScaler()

        # 确保使用绝对路径
        self.data_dir = Path(data_dir).resolve()
        self.device = torch.device(config.DEVICE)
        self.model_save_dir = Path(model_save_dir or config.MODELS_DIR)
        self.log_dir = Path(log_dir or config.LOGS_DIR)

        # 创建目录
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 初始化TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # 检查数据目录是否存在
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        print(f"[Trainer] 数据目录: {self.data_dir}")

        # 加载数据
        self.train_loader, self.val_loader, self.full_dataset = create_dataloaders(
            data_dir=str(self.data_dir),
            img_size=config.MODEL_CONFIG["img_size"],
            batch_size=config.TRAIN_CONFIG["batch_size"],
            val_split=config.TRAIN_CONFIG["val_split"],
            num_workers=0,  # Windows下设为0避免多进程问题
        )

        # 创建Triplet数据集
        self.triplet_dataset = TripletDataset(
            base_dataset=self.full_dataset,
            transform=self.full_dataset.transform,
        )
        self.triplet_loader = DataLoader(
            self.triplet_dataset,
            batch_size=config.TRAIN_CONFIG["batch_size"] // 2,  # Triplet需要更多内存
            shuffle=True,
            num_workers=config.TRAIN_CONFIG["num_workers"],
            drop_last=True,
        )

        # 使用单流EfficientNet模型
        self.model = ReagentRecognitionModel(
            num_classes=self.full_dataset.num_classes,
            embedding_dim=config.MODEL_CONFIG["embedding_dim"],
            pretrained=True,
            arcface_margin=config.MODEL_CONFIG["arcface_margin"],
            arcface_scale=config.MODEL_CONFIG["arcface_scale"],
        ).to(self.device)
        print(f"[Trainer] 使用单流EfficientNet-B2模型")

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.TRAIN_CONFIG["lr"],
            weight_decay=config.TRAIN_CONFIG["weight_decay"],
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.TRAIN_CONFIG["epochs"],
            eta_min=1e-6,
        )

        # 训练状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        # 类别映射
        self.class_to_idx = self.full_dataset.class_to_idx
        self.idx_to_class = self.full_dataset.idx_to_class

        # 初始化FAISS索引用于验证
        self.faiss_index = faiss.IndexHNSWFlat(config.MODEL_CONFIG["embedding_dim"], 32)
        self.faiss_index.hnsw.efConstruction = 200
        self.faiss_index.hnsw.efSearch = 64
        self.faiss_labels = []

        # 保存类别映射
        self.save_class_mapping()

        print(f"[Trainer] 初始化完成")
        print(f"  设备: {self.device}")
        print(f"  类别数: {self.full_dataset.num_classes}")
        print(f"  训练样本: {len(self.train_loader.dataset)}")
        print(f"  验证样本: {len(self.val_loader.dataset)}")
        print(f"  模型参数: {sum(p.numel() for p in self.model.parameters()):,}")

    def save_class_mapping(self):
        """保存类别映射"""
        mapping_path = self.model_save_dir / "class_mapping.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump({
                "class_to_idx": self.class_to_idx,
                "idx_to_class": {str(k): v for k, v in self.idx_to_class.items()},
            }, f, ensure_ascii=False, indent=2)
        print(f"[Trainer] 类别映射已保存: {mapping_path}")

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_arc_loss = 0.0
        epoch_triplet_loss = 0.0
        num_batches = len(self.train_loader)

        # 进度条
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{config.TRAIN_CONFIG['epochs']}")

        # Triplet加载器迭代器
        triplet_iter = iter(self.triplet_loader)

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # ArcFace Loss
            embeddings, arc_loss = self.model(images, labels)

            # Triplet Loss (如果可用)
            triplet_loss_val = torch.tensor(0.0).to(self.device)
            if len(self.triplet_dataset.valid_classes) > 0:
                try:
                    anchor, positive, negative = next(triplet_iter)
                    anchor, positive, negative = (
                        anchor.to(self.device),
                        positive.to(self.device),
                        negative.to(self.device),
                    )

                    with torch.no_grad():
                        anchor_emb = self.model.embedder(anchor)
                        positive_emb = self.model.embedder(positive)
                        negative_emb = self.model.embedder(negative)

                    triplet_loss_val = self.model.triplet_loss(
                        anchor_emb, positive_emb, negative_emb
                    )
                except StopIteration:
                    triplet_iter = iter(self.triplet_loader)

            # 组合损失
            total_loss = arc_loss + 0.3 * triplet_loss_val  # Triplet权重较小

            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # 记录损失
            epoch_loss += total_loss.item()
            epoch_arc_loss += arc_loss.item()
            epoch_triplet_loss += triplet_loss_val.item()

            # 更新进度条
            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "arc": f"{arc_loss.item():.4f}",
                "triplet": f"{triplet_loss_val.item():.4f}",
            })

            # 记录到TensorBoard
            global_step = self.current_epoch * num_batches + batch_idx
            self.writer.add_scalar("Train/BatchLoss", total_loss.item(), global_step)
            self.writer.add_scalar("Train/ArcFaceLoss", arc_loss.item(), global_step)
            self.writer.add_scalar("Train/TripletLoss", triplet_loss_val.item(), global_step)

        # 返回平均损失
        avg_loss = epoch_loss / max(1, num_batches)  # 防止除零
        avg_arc_loss = epoch_arc_loss / max(1, num_batches)
        avg_triplet_loss = epoch_triplet_loss / max(1, num_batches)

        return {
            "loss": avg_loss,
            "arc_loss": avg_arc_loss,
            "triplet_loss": avg_triplet_loss,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def validate(self) -> Dict[str, float]:
        """验证模型 - 使用FAISS检索验证"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        # 清空FAISS索引
        self.faiss_index.reset()
        self.faiss_labels = []

        # 第一遍：提取验证集所有样本的嵌入，构建FAISS索引
        val_embeddings = []
        val_labels_list = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Building FAISS index"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                embeddings, loss = self.model(images, labels)
                val_loss += loss.item()

                # 收集嵌入和标签
                val_embeddings.append(embeddings.cpu().numpy())
                val_labels_list.append(labels.cpu().numpy())

        # 合并所有嵌入和标签
        all_embeddings = np.concatenate(val_embeddings, axis=0)
        all_labels = np.concatenate(val_labels_list, axis=0)

        # L2归一化
        faiss.normalize_L2(all_embeddings)

        # 添加到FAISS索引
        self.faiss_index.add(all_embeddings)
        self.faiss_labels = all_labels.tolist()

        # 第二遍：使用FAISS检索计算准确率
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="FAISS validating"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 提取嵌入
                embeddings, _ = self.model(images, labels)
                query_embeddings = embeddings.cpu().numpy()
                faiss.normalize_L2(query_embeddings)

                # FAISS检索（Top-1）
                k = 1
                distances, indices = self.faiss_index.search(query_embeddings, k)

                # 获取预测标签
                predicted_labels = np.array([self.faiss_labels[idx[0]] for idx in indices])
                true_labels = labels.cpu().numpy()

                # 计算准确率
                total += labels.size(0)
                correct += (predicted_labels == true_labels).sum().item()

        accuracy = correct / total
        avg_loss = val_loss / len(self.val_loader)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
        }

    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "class_to_idx": self.class_to_idx,
            "idx_to_class": self.idx_to_class,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "model_type": "single_efficientnet",
            "embedding_dim": config.MODEL_CONFIG["embedding_dim"],
        }

        # 保存最新检查点
        checkpoint_path = self.model_save_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_path = self.model_save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"[Trainer] 最佳模型已保存: {best_path}")

    def train(self) -> str:
        """完整训练流程"""
        print(f"\n[Trainer] 开始训练 {config.TRAIN_CONFIG['epochs']} 个epoch")
        print(f"  批次大小: {config.TRAIN_CONFIG['batch_size']}")
        print(f"  学习率: {config.TRAIN_CONFIG['lr']}")
        print(f"  早停耐心: {config.TRAIN_CONFIG['early_stop_patience']}")

        patience_counter = 0

        for epoch in range(config.TRAIN_CONFIG["epochs"]):
            self.current_epoch = epoch

            # 训练
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics["loss"])

            # 验证
            val_metrics = self.validate()
            self.val_losses.append(val_metrics["loss"])
            self.val_accuracies.append(val_metrics["accuracy"])

            # 打印结果
            print(f"\nEpoch {epoch+1}/{config.TRAIN_CONFIG['epochs']}:")
            print(f"  训练损失: {train_metrics['loss']:.4f} (Arc: {train_metrics['arc_loss']:.4f}, Triplet: {train_metrics['triplet_loss']:.4f})")
            print(f"  验证损失: {val_metrics['loss']:.4f}")
            print(f"  验证准确率: {val_metrics['accuracy']:.4f}")
            print(f"  学习率: {train_metrics['lr']:.6f}")

            # 记录到TensorBoard
            self.writer.add_scalar("Train/Loss", train_metrics["loss"], epoch)
            self.writer.add_scalar("Train/ArcFaceLoss", train_metrics["arc_loss"], epoch)
            self.writer.add_scalar("Train/TripletLoss", train_metrics["triplet_loss"], epoch)
            self.writer.add_scalar("Val/Loss", val_metrics["loss"], epoch)
            self.writer.add_scalar("Val/Accuracy", val_metrics["accuracy"], epoch)
            self.writer.add_scalar("LearningRate", train_metrics["lr"], epoch)

            # 保存检查点
            is_best = val_metrics["accuracy"] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics["accuracy"]
                patience_counter = 0
            else:
                patience_counter += 1

            # 定期保存
            if (epoch + 1) % config.TRAIN_CONFIG["save_every"] == 0 or is_best:
                self.save_checkpoint(is_best=is_best)

            # 更新学习率
            self.scheduler.step()

            # 更新学习率
            self.scheduler.step()

            # 早停检查
            if patience_counter >= config.TRAIN_CONFIG["early_stop_patience"]:
                print(f"\n[Trainer] 早停触发，已连续 {patience_counter} 个epoch无改善")
                break

        # 训练完成
        print(f"\n[Trainer] 训练完成!")
        print(f"  最佳验证准确率: {self.best_val_acc:.4f}")

        # 保存最终模型
        final_model_path = self.model_save_dir / "final_model.pth"
        self.save_checkpoint(is_best=True)

        # 关闭TensorBoard
        self.writer.close()

        return str(self.model_save_dir / "best_model.pth")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint["best_val_acc"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.val_accuracies = checkpoint.get("val_accuracies", [])

        print(f"[Trainer] 检查点已加载: {checkpoint_path}")
        print(f"  恢复到epoch {self.current_epoch}, 最佳准确率 {self.best_val_acc:.4f}")