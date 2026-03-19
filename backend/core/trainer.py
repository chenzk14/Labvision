"""
backend/core/trainer.py
试剂识别模型训练器（小样本 20-30张/类 + 1050Ti 4GB 优化）

通过 overrides 字典覆盖 config.py 默认参数，由 scripts/train.py 传入。
直接运行此文件也可触发训练（兼容旧用法）。
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import faiss
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend import config
from backend.core.dataset import create_dataloaders
from backend.models.metric_model import ReagentRecognitionModel


def get_linear_warmup_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(1e-6, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ReagentTrainer:

    def _get(self, key, config_dict, default=None):
        """优先从 overrides 取，其次从 config_dict 取，最后用 default"""
        return self._overrides.get(key, config_dict.get(key, default))

    def __init__(
        self,
        data_dir: str,
        model_save_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        overrides: Optional[Dict] = None,
    ):
        self._overrides = overrides or {}
        self.data_dir = Path(data_dir).resolve()
        self.device = torch.device(config.DEVICE)
        self.model_save_dir = Path(model_save_dir or config.MODELS_DIR)
        self.log_dir = Path(log_dir or config.LOGS_DIR)

        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        # ── 从 overrides / config 读取所有超参数 ──────────────────────────
        self.val_split           = self._get("val_split",           config.TRAIN_CONFIG)
        self.total_epochs        = self._get("epochs",              config.TRAIN_CONFIG)
        self.lr                  = self._get("lr",                  config.TRAIN_CONFIG)
        self.early_stop_patience = self._get("early_stop_patience", config.TRAIN_CONFIG)
        self.warmup_epochs       = self._get("warmup_epochs",       config.TRAIN_CONFIG, default=5)
        self.train_batch_size    = self._get("batch_size",          config.TRAIN_CONFIG)
        self.accumulation_steps  = self._get("accumulation_steps",  config.TRAIN_CONFIG, default=4)
        self.triplet_weight      = self._get("triplet_weight",      config.TRAIN_CONFIG, default=0.0)
        self.weight_decay        = self._get("weight_decay",        config.TRAIN_CONFIG)
        self.arcface_margin      = self._get("arcface_margin",      config.MODEL_CONFIG)
        self.arcface_scale       = self._get("arcface_scale",       config.MODEL_CONFIG)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # ── 数据 ─────────────────────────────────────────────────────────
        self.train_loader, self.val_loader, self.full_dataset = create_dataloaders(
            data_dir=str(self.data_dir),
            img_size=self._get("img_size", config.MODEL_CONFIG),
            batch_size=self.train_batch_size,
            val_split=self.val_split,
            num_workers=0,
        )

        num_classes = self.full_dataset.num_classes
        self.class_to_idx = self.full_dataset.class_to_idx
        self.idx_to_class = self.full_dataset.idx_to_class

        # ── 模型 ─────────────────────────────────────────────────────────
        self.model = ReagentRecognitionModel(
            num_classes=num_classes,
            embedding_dim=self._get("embedding_dim", config.MODEL_CONFIG),
            pretrained=True,
            arcface_margin=self.arcface_margin,
            arcface_scale=self.arcface_scale,
        ).to(self.device)

        # ── 优化器 & 调度器 ───────────────────────────────────────────────
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = get_linear_warmup_scheduler(
            self.optimizer,
            warmup_epochs=self.warmup_epochs,
            total_epochs=self.total_epochs,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

        # ── 训练状态 ─────────────────────────────────────────────────────
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses: List[float] = []
        self.val_accuracies: List[float] = []
        self.patience_counter = 0

        self.save_class_mapping()

        print("\n" + "═" * 60)
        print(f"  ReagentTrainer 初始化完成")
        print("═" * 60)
        print(f"  设备         : {self.device}")
        print(f"  类别数       : {num_classes}")
        print(f"  训练样本     : {len(self.train_loader.dataset)}")
        print(f"  验证样本     : {len(self.val_loader.dataset)}")
        print(f"  有效 batch   : {self.train_batch_size} × {self.accumulation_steps} = {self.train_batch_size * self.accumulation_steps}")
        print(f"  学习率       : {self.lr}")
        print(f"  ArcFace      : margin={self.arcface_margin}, scale={self.arcface_scale}")
        print(f"  Warmup       : {self.warmup_epochs} epochs")
        print(f"  Early stop   : patience={self.early_stop_patience}")
        print(f"  总 epoch     : {self.total_epochs}")
        print(f"  模型参数     : {sum(p.numel() for p in self.model.parameters()):,}")
        print("═" * 60 + "\n")

    def save_class_mapping(self):
        mapping_path = self.model_save_dir / "class_mapping.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "class_to_idx": self.class_to_idx,
                    "idx_to_class": {str(k): v for k, v in self.idx_to_class.items()},
                },
                f, ensure_ascii=False, indent=2,
            )

    # ──────────────────────────────────────────────────────────────────────────

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_arc_loss_sum = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1:3d}/{self.total_epochs}",
            ncols=90,
        )

        self.optimizer.zero_grad()

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                _, arc_loss = self.model(images, labels)
                loss_scaled = arc_loss / self.accumulation_steps

            self.scaler.scale(loss_scaled).backward()

            is_last_batch = (batch_idx == num_batches - 1)
            if (batch_idx + 1) % self.accumulation_steps == 0 or is_last_batch:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            epoch_arc_loss_sum += arc_loss.item()
            global_step = self.current_epoch * num_batches + batch_idx
            self.writer.add_scalar("Train/ArcFaceLoss_step", arc_loss.item(), global_step)
            pbar.set_postfix(arc=f"{arc_loss.item():.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")

        return {
            "arc_loss": epoch_arc_loss_sum / num_batches,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    # ──────────────────────────────────────────────────────────────────────────

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        val_embeddings, val_labels = [], []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                    emb = self.model.get_embedding(images)
                val_embeddings.append(emb.cpu().float().numpy())
                val_labels.append(labels.numpy())

        val_embeddings = np.concatenate(val_embeddings, axis=0).astype(np.float32)
        val_labels = np.concatenate(val_labels, axis=0)
        faiss.normalize_L2(val_embeddings)

        if len(val_labels) < 2:
            return self._validate_against_train(val_embeddings, val_labels)

        index = faiss.IndexFlatIP(val_embeddings.shape[1])
        index.add(val_embeddings)
        k = min(2, len(val_labels))
        _, indices = index.search(val_embeddings, k)

        correct = sum(
            val_labels[indices[i][1]] == val_labels[i]
            for i in range(len(val_labels))
        )
        return {"accuracy": correct / len(val_labels), "n_val": len(val_labels)}

    def _validate_against_train(self, val_embeddings, val_labels) -> Dict[str, float]:
        self.model.eval()
        train_embeddings, train_labels = [], []

        with torch.no_grad():
            for images, labels in self.train_loader:
                images = images.to(self.device)
                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                    emb = self.model.get_embedding(images)
                train_embeddings.append(emb.cpu().float().numpy())
                train_labels.append(labels.numpy())

        train_embeddings = np.concatenate(train_embeddings, axis=0).astype(np.float32)
        train_labels = np.concatenate(train_labels, axis=0)
        faiss.normalize_L2(train_embeddings)

        index = faiss.IndexFlatIP(train_embeddings.shape[1])
        index.add(train_embeddings)
        _, indices = index.search(val_embeddings, 1)
        accuracy = (train_labels[indices[:, 0]] == val_labels).mean()

        return {"accuracy": float(accuracy), "n_val": len(val_labels), "gallery": "train"}

    # ──────────────────────────────────────────────────────────────────────────

    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            "epoch":              self.current_epoch,
            "model_state_dict":   self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc":       self.best_val_acc,
            "class_to_idx":       self.class_to_idx,
            "idx_to_class":       self.idx_to_class,
            "train_losses":       self.train_losses,
            "val_accuracies":     self.val_accuracies,
            "model_type":         "single_efficientnet",
            "embedding_dim":      self._get("embedding_dim", config.MODEL_CONFIG),
        }
        torch.save(checkpoint, self.model_save_dir / "latest_checkpoint.pth")
        if is_best:
            torch.save(checkpoint, self.model_save_dir / "best_model.pth")
            print(f"  ✓ 最佳模型已保存 (acc={self.best_val_acc:.4f})")

    # ──────────────────────────────────────────────────────────────────────────

    def train(self) -> str:
        print(f"[Training] 开始训练，共 {self.total_epochs} epochs\n")

        for epoch in range(self.total_epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics["arc_loss"])

            do_val = (epoch % 3 == 0) or (epoch == self.total_epochs - 1)
            if do_val:
                val_metrics = self.validate()
                val_acc = val_metrics["accuracy"]
            else:
                val_acc = self.val_accuracies[-1] if self.val_accuracies else 0.0
                val_metrics = {"accuracy": val_acc}

            self.val_accuracies.append(val_acc)

            gallery_note = " [gallery=train]" if do_val and val_metrics.get("gallery") == "train" else ""
            print(
                f"  Epoch {epoch+1:3d}/{self.total_epochs}"
                f"  arc={train_metrics['arc_loss']:.4f}"
                f"  val_acc={val_acc:.4f}{gallery_note}"
                f"  lr={train_metrics['lr']:.2e}"
            )

            self.writer.add_scalar("Train/ArcFaceLoss", train_metrics["arc_loss"], epoch)
            self.writer.add_scalar("Val/Accuracy", val_acc, epoch)
            self.writer.add_scalar("LearningRate", train_metrics["lr"], epoch)

            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stop_patience:
                    print(f"\n  [Early stop] 连续 {self.patience_counter} 个 epoch 无提升，停止训练")
                    break

            self.scheduler.step()

        self.save_checkpoint(is_best=False)
        self.writer.close()

        best_model_path = str(self.model_save_dir / "best_model.pth")
        print(f"\n[Training] 完成！最佳验证准确率: {self.best_val_acc:.4f}")
        print(f"[Training] 最佳模型路径: {best_model_path}\n")
        return best_model_path

    # ──────────────────────────────────────────────────────────────────────────

    def build_faiss_index_after_training(self):
        print("[Index] 开始构建 FAISS 索引（使用全量数据）...")
        self.model.eval()

        from backend.core.dataset import ReagentDataset, get_val_transforms

        full_dataset = ReagentDataset(
            root_dir=str(self.data_dir),
            transform=get_val_transforms(self._get("img_size", config.MODEL_CONFIG)),
        )
        full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False, num_workers=0)

        all_embeddings, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(full_loader, desc="提取嵌入", ncols=60):
                images = images.to(self.device)
                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                    emb = self.model.get_embedding(images)
                all_embeddings.append(emb.cpu().float().numpy())
                all_labels.append(labels.numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        all_labels = np.concatenate(all_labels, axis=0)
        faiss.normalize_L2(all_embeddings)

        index = faiss.IndexFlatIP(all_embeddings.shape[1])
        index.add(all_embeddings)

        id_map = [
            {
                "reagent_id":   self.idx_to_class[int(label_idx)],
                "reagent_name": self.idx_to_class[int(label_idx)].rstrip("0123456789"),
                "vector_id":    i,
                "timestamp":    time.time(),
                "image_path":   "",
            }
            for i, label_idx in enumerate(all_labels)
        ]

        index_dir = Path(config.EMBEDDINGS_DIR)
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_dir / "reagent.index"))
        with open(str(index_dir / "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(id_map, f, ensure_ascii=False, indent=2)

        print(f"[Index] 完成！共索引 {len(id_map)} 个向量，{len(set(all_labels))} 个类别")
        print(f"[Index] 索引路径: {index_dir / 'reagent.index'}")


# ── 兼容直接运行 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    SMALL_SAMPLE_OVERRIDES = {
        "val_split": 0.2, "epochs": 80, "lr": 3e-4,
        "early_stop_patience": 15, "arcface_margin": 0.30,
        "arcface_scale": 64.0, "triplet_weight": 0.0,
        "warmup_epochs": 5, "batch_size": 4, "accumulation_steps": 4,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/images")
    parser.add_argument("--no_small_sample", action="store_true")
    parser.add_argument("--build_index", action="store_true", default=True)
    args = parser.parse_args()

    overrides = {} if args.no_small_sample else SMALL_SAMPLE_OVERRIDES
    trainer = ReagentTrainer(data_dir=args.data_dir, overrides=overrides)
    best = trainer.train()
    if args.build_index:
        trainer.build_faiss_index_after_training()
    print(f"完成：{best}")