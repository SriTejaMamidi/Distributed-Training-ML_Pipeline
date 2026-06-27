"""
Utility functions for distributed training.
"""

import os
import torch
import random
import numpy as np
import logging
import torch.distributed as dist


def setup_logging(rank: int):
    """Setup logging configuration."""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reduce_dict_across_processes(d: dict) -> dict:
    """Reduce dictionary values across all processes."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    reduced = {}
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            dist.reduce(value, dst=0, op=dist.ReduceOp.AVG)
            reduced[key] = value.item() if rank == 0 else None

    return reduced


def get_rank_and_world_size():
    """Get rank and world size from environment."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


def get_device(rank: int) -> torch.device:
    """Get device for current rank."""
    return torch.device(f"cuda:{rank % torch.cuda.device_count()}")
