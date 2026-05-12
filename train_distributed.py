#Importing the required modules
import os
import time
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models
import mlflow
import mlflow.pytorch
from dist_config import setup_distributed,cleanup
from data_utils import get_data_loaders


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
#Config
EPOCHS        = 20
BATCH_SIZE    = 128
LR            = 0.1
MOMENTUM      = 0.9
WEIGHT_DECAY  = 5e-4
DATA_DIR      = "./data"
MODEL_DIR     = "./models"
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT    = "distributed-resnet18-cifar10"

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
#Metrics helpers
def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    All-Reduce a tensor across all ranks and return the average.

    Why needed for metrics:
    Each rank computes accuracy on its own subset of validation data.
    To get the true global accuracy, we sum across all ranks and divide
    by world_size. dist.all_reduce uses the same All-Reduce algorithm
    as gradient synchronisation — efficient, no bottleneck.
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy for a batch."""
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    return correct / targets.size(0)


#Train one epoch
def train_epoch(
    model, loader, optimizer, criterion,
    scheduler, device, rank, world_size, epoch
) -> tuple[float, float]:
    """
    One full training pass.
    Returns (avg_loss, avg_accuracy) averaged across all ranks.
    """
    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss   = criterion(output, target)
        loss.backward()
        # Gradient synchronisation happens inside loss.backward() via DDP
        # DDP hooks into autograd and triggers All-Reduce automatically

        # Gradient clipping — prevents exploding gradients
        # Clip before optimizer.step() so we're clipping the synced gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        total_acc  += accuracy(output.detach(), target)
        n_batches  += 1

        # Only rank 0 prints — otherwise all ranks print and it's messy
        if rank == 0 and batch_idx % 50 == 0:
            logger.info(
                f"Epoch {epoch:3d} | Batch {batch_idx:4d}/{len(loader)} "
                f"| Loss: {loss.item():.4f}"
            )

    # Average metrics across all ranks via All-Reduce
    avg_loss = reduce_tensor(
        torch.tensor(total_loss / n_batches).to(device), world_size
    ).item()
    avg_acc = reduce_tensor(
        torch.tensor(total_acc / n_batches).to(device), world_size
    ).item()

    return avg_loss, avg_acc


#Validation
def validate(
    model, loader, criterion,
    device, rank, world_size
) -> tuple[float, float]:
    """
    Evaluate on validation set.
    Returns (avg_loss, avg_accuracy) averaged across all ranks.
    """
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss   = criterion(output, target)

            total_loss += loss.item()
            total_acc  += accuracy(output, target)
            n_batches  += 1

    avg_loss = reduce_tensor(
        torch.tensor(total_loss / n_batches).to(device), world_size
    ).item()
    avg_acc = reduce_tensor(
        torch.tensor(total_acc / n_batches).to(device), world_size
    ).item()

    return avg_loss, avg_acc


#Per-class accuracy
def test_per_class(model, loader, device, rank, world_size) -> dict:
    """
    Compute per-class accuracy on test set.
    Only runs on rank 0 — full test set evaluation.
    """
    model.eval()
    class_correct = torch.zeros(10).to(device)
    class_total   = torch.zeros(10).to(device)

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)

            for i in range(len(target)):
                label = target[i].item()
                class_correct[label] += predicted[i].eq(target[i]).item()
                class_total[label]   += 1

    # All-reduce so rank 0 gets global counts
    dist.all_reduce(class_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(class_total,   op=dist.ReduceOp.SUM)

    results = {}
    if rank == 0:
        for i, cls in enumerate(CLASSES):
            acc = (class_correct[i] / class_total[i]).item() if class_total[i] > 0 else 0.0
            results[cls] = round(acc, 4)
            logger.info(f"  {cls:12s}: {acc:.4f}")

    return results


#Main training function — called by each spawned process
def train(rank: int, world_size: int):
    """
    Full training pipeline for one rank.
    mp.spawn calls this once per GPU/process.

    rank 0 = master process:
    - Logs to MLflow
    - Saves checkpoints
    - Prints progress

    rank 1, 2, ... = worker processes:
    - Train in parallel
    - Sync gradients via All-Reduce
    - Participate in metric reduction
    - Stay silent (no print/log)
    """
    setup_distributed(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    #Data
    train_loader, val_loader, train_sampler = get_data_loaders(
        rank, world_size, BATCH_SIZE, DATA_DIR
    )

    #Model
    # ResNet-18 — your original choice, kept
    # pretrained=False — train from scratch on CIFAR-10
    # num_classes=10   — CIFAR-10 has 10 classes
    model = models.resnet18(weights=None, num_classes=10)

    # Adapt ResNet-18 for CIFAR-10 (32×32 images)
    # Original ResNet was designed for ImageNet (224×224)
    # First conv: 7×7 stride 2 → too aggressive for 32×32, replace with 3×3 stride 1
    # Remove MaxPool → keeps spatial resolution
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    model = model.to(device)

    # Wrap with DDP — this is where the distributed magic happens
    # DDP adds hooks to each parameter that trigger All-Reduce
    # during backward() to synchronise gradients across all ranks
    if torch.cuda.is_available():
        model = DDP(model, device_ids=[rank], output_device=rank)
    else:
        model = DDP(model)

    #Optimiser + Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # SGD with momentum — standard for ResNet training
    # momentum=0.9 + weight_decay=5e-4 is the canonical ResNet recipe
    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True   # Nesterov momentum — looks ahead before gradient step
    )

    # CosineAnnealingLR — smoothly reduces LR from LR to 0 over EPOCHS
    # Much better than step decay for ResNet — avoids sharp transitions
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-4
    )

    #MLflow setup (rank 0 only)
    if rank == 0:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT)
        os.makedirs(MODEL_DIR, exist_ok=True)

        run = mlflow.start_run(run_name=f"ddp_w{world_size}_b{BATCH_SIZE}")
        mlflow.log_params({
            "world_size":   world_size,
            "batch_size":   BATCH_SIZE,
            "effective_batch": BATCH_SIZE * world_size,
            "lr":           LR,
            "momentum":     MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "epochs":       EPOCHS,
            "model":        "resnet18-cifar10",
            "backend":      "nccl" if torch.cuda.is_available() else "gloo",
        })

    #Training timing — for honest speedup measurement
    t_start = time.time()
    best_val_acc = 0.0

    logger.info(f"[Rank {rank}] Starting training: {EPOCHS} epochs")

    for epoch in range(1, EPOCHS + 1):
        # CRITICAL: set_epoch so DistributedSampler reshuffles correctly
        # Without this, all epochs see data in the same order → no shuffle benefit
        train_sampler.set_epoch(epoch)

        # Train
        t_epoch = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion,
            scheduler, device, rank, world_size, epoch
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, rank, world_size
        )

        scheduler.step()
        epoch_time = time.time() - t_epoch

        # Rank 0 handles logging and checkpointing
        if rank == 0:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
            )

            mlflow.log_metrics({
                "train_loss":  train_loss,
                "train_acc":   train_acc,
                "val_loss":    val_loss,
                "val_acc":     val_acc,
                "lr":          current_lr,
                "epoch_time_s": epoch_time,
            }, step=epoch)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = os.path.join(MODEL_DIR, "best_model.pt")
                # Save only the module weights (unwrap DDP)
                torch.save(model.module.state_dict(), ckpt_path)
                logger.info(f"  → Saved best model (val_acc={val_acc:.4f})")
                mlflow.log_metric("best_val_acc", best_val_acc, step=epoch)

    #Final evaluation
    total_time = time.time() - t_start

    if rank == 0:
        logger.info(f"\nTraining complete in {total_time:.1f}s")
        logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
        logger.info("\nPer-class test accuracy:")

    per_class = test_per_class(model, val_loader, device, rank, world_size)

    if rank == 0:
        mlflow.log_metric("total_training_time_s", total_time)
        for cls, acc in per_class.items():
            mlflow.log_metric(f"test_acc_{cls}", acc)

        # Save timing info for speedup analysis
        timing = {
            "world_size":      world_size,
            "total_time_s":    round(total_time, 2),
            "avg_epoch_time_s": round(total_time / EPOCHS, 2),
            "best_val_acc":    round(best_val_acc, 4),
        }
        with open(os.path.join(MODEL_DIR, "timing.json"), "w") as f:
            json.dump(timing, f, indent=2)

        mlflow.log_artifact(os.path.join(MODEL_DIR, "timing.json"))
        mlflow.pytorch.log_model(
            model.module,
            artifact_path="model",
            registered_model_name="resnet18-cifar10-ddp"
        )
        mlflow.end_run()
        logger.info(f"Results logged to MLflow experiment: {EXPERIMENT}")

    cleanup()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Detect available GPUs
    # If none available, simulate 2 processes on CPU (for testing)
    n_gpus = torch.cuda.device_count()
    world_size = max(n_gpus, 2)  # minimum 2 processes

    logger.info("="*60)
    logger.info(f"Distributed ResNet-18 Training on CIFAR-10")
    logger.info(f"GPUs available: {n_gpus}")
    logger.info(f"World size:     {world_size}")
    logger.info(f"Effective batch size: {BATCH_SIZE * world_size}")
    logger.info("="*60)

    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
