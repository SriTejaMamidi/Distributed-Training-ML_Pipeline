"""
Distributed data loading utilities.
"""

import torch
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import logging

logger = logging.getLogger(__name__)


class DistributedDataLoader:
    """Distributed data loading helper."""

    @staticmethod
    def get_train_loader(
            batch_size: int,
            num_workers: int,
            rank: int,
            world_size: int,
            pin_memory: bool = True,
    ) -> DataLoader:
        """Get distributed training data loader.

        Args:
            batch_size: Batch size per GPU
            num_workers: Number of workers
            rank: Current rank
            world_size: Total number of ranks
            pin_memory: Pin memory for faster transfer

        Returns:
            DataLoader
        """
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ])

        dataset = CIFAR10(
            root="data/",
            train=True,
            download=True,
            transform=transform
        )

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        logger.info(
            f"Rank {rank}: Train loader created "
            f"(batch_size={batch_size}, num_workers={num_workers})"
        )

        return loader

    @staticmethod
    def get_val_loader(
            batch_size: int,
            num_workers: int,
            rank: int,
            world_size: int,
    ) -> DataLoader:
        """Get distributed validation data loader."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ])

        dataset = CIFAR10(
            root="data/",
            train=False,
            download=True,
            transform=transform
        )

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
        )

        return loader
