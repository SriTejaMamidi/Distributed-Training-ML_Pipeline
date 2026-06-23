#!/usr/bin/env python3
"""
Main training script for DDP training.
Run with: python -m torch.distributed.launch --nproc_per_node=4 scripts/train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torchvision import models
import argparse
import yaml
import os
import logging
from pathlib import Path

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from distributed_trainer import DistributedTrainer
from dataloader import DistributedDataLoader
from utils import setup_logging, set_seed, get_rank_and_world_size, get_device

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    # Get rank and world size
    rank, world_size = get_rank_and_world_size()
    device = get_device(rank)

    # Setup logging
    setup_logging(rank)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config.get("device", {}).get("seed", 42))

    if rank == 0:
        logger.info(f"Training on {world_size} GPUs")
        logger.info(f"Config: {config}")

    # Create model
    model = models.resnet50(pretrained=config["model"]["pretrained"])
    model.fc = nn.Linear(2048, config["model"]["num_classes"])

    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        momentum=config["training"]["momentum"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Create trainer
    trainer = DistributedTrainer(
        model=model,
        rank=rank,
        world_size=world_size,
        device=device,
        optimizer=optimizer,
        gradient_checkpointing=config["optimization"]["gradient_checkpointing"],
        mixed_precision=config["optimization"]["mixed_precision"],
    )

    # Get data loaders
    train_loader = DistributedDataLoader.get_train_loader(
        batch_size=config["training"]["batch_size"],
        num_workers=config["optimization"]["num_workers"],
        rank=rank,
        world_size=world_size,
        pin_memory=config["optimization"]["pin_memory"],
    )

    # Training loop
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)

    for epoch in range(start_epoch, args.epochs):
        metrics = trainer.train_epoch(train_loader, criterion, epoch)

        # Save checkpoint
        if (epoch + 1) % config["checkpoint"]["save_frequency"] == 0:
            trainer.save_checkpoint(epoch, is_best=False)

        if rank == 0:
            logger.info(f"Epoch {epoch} completed - Metrics: {metrics}")

    if rank == 0:
        logger.info("Training completed!")
        trainer.save_checkpoint(args.epochs - 1, is_best=True)


if __name__ == "__main__":
    main()
