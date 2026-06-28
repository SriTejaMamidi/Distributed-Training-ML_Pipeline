"""
Main distributed trainer class using PyTorch DDP.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class DistributedTrainer:
    """Distributed training manager with DDP backend."""

    def __init__(
            self,
            model: nn.Module,
            rank: int,
            world_size: int,
            device: torch.device,
            optimizer: torch.optim.Optimizer,
            gradient_checkpointing: bool = True,
            mixed_precision: bool = True,
    ):
        """Initialize distributed trainer.

        Args:
            model: PyTorch model
            rank: Rank of current process
            world_size: Total number of processes
            device: Device to use (cuda/cpu)
            optimizer: Optimizer instance
            gradient_checkpointing: Enable gradient checkpointing
            mixed_precision: Enable mixed precision training
        """
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.optimizer = optimizer

        # Initialize process group if not already done
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", timeout=600)

        # Wrap model with DDP
        self.model = DDP(
            model.to(device),
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
        )

        # Enable gradient checkpointing
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()

        # Mixed precision setup
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        self.mixed_precision = mixed_precision

        # Metrics tracking
        self.metrics = {
            "train_loss": [],
            "train_time": 0,
            "communication_time": 0,
        }

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.model.module, "gradient_checkpointing_enable"):
            self.model.module.gradient_checkpointing_enable()
            logger.info(f"Rank {self.rank}: Gradient checkpointing enabled")

    def train_epoch(self, train_loader, criterion, epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Distributed data loader
            criterion: Loss function
            epoch: Current epoch number

        Returns:
            Dictionary with metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Synchronize all processes
        dist.barrier()

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()

            if self.mixed_precision:
                # Mixed precision forward/backward
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = criterion(output, target)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward/backward
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if self.rank == 0 and batch_idx % 100 == 0:
                avg_loss = total_loss / num_batches
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {avg_loss:.4f}"
                )

        # Synchronize metrics across all ranks
        avg_loss = torch.tensor(total_loss / num_batches, device=self.device)
        dist.reduce(avg_loss, dst=0, op=dist.ReduceOp.AVG)

        if self.rank == 0:
            logger.info(f"Epoch {epoch} - Avg Loss: {avg_loss.item():.4f}")

        return {"loss": avg_loss.item()}

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint (rank 0 only).

        Args:
            epoch: Current epoch
            is_best: Whether this is best model
        """
        if self.rank != 0:
            return

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "world_size": self.world_size,
        }

        path = f"models/checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

        if is_best:
            torch.save(checkpoint, "models/best_model.pth")
            logger.info("Best model saved")

    def load_checkpoint(self, path: str):
        """Load checkpoint and resume training.

        Args:
            path: Path to checkpoint file

        Returns:
            Starting epoch
        """
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return 0

        checkpoint = torch.load(path, map_location=self.device)
        self.model.module.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Resumed from checkpoint: {path}")
        return checkpoint["epoch"]


def get_distributed_trainer(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        **kwargs
) -> DistributedTrainer:
    """Factory function to create distributed trainer.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        **kwargs: Additional arguments

    Returns:
        DistributedTrainer instance
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    return DistributedTrainer(
        model=model,
        rank=rank,
        world_size=world_size,
        device=device,
        optimizer=optimizer,
        **kwargs
    )
