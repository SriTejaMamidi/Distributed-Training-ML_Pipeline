# 🚀 Distributed High-Performance ML Training Pipeline

> A production-grade **Distributed Data Parallel (DDP)** backbone engineered to scale deep learning workloads across multi-GPU environments. Trains ResNet-18 on CIFAR-10 using true multi-process parallelism, **NCCL communication backend**, and full **MLflow experiment tracking** — with honest, reproducible benchmarks.

---

## 🚀 Live Demo

```bash
# Train distributed (2 processes)
python src/train_distributed.py

# Benchmark — measure actual speedup
python src/benchmark.py

# Serve trained model
uvicorn src.api:app --port 8002

# Classify an image
curl -X POST http://localhost:8002/predict \
  -F "file=@test_image.jpg"
```

```json
{
  "predicted_class": "automobile",
  "confidence": 0.9241,
  "top3": [
    {"class": "automobile", "confidence": 0.9241},
    {"class": "truck",      "confidence": 0.0534},
    {"class": "airplane",   "confidence": 0.0142}
  ],
  "latency_ms": 14.2
}
```

---

## 🏗️ System Architecture

```
CIFAR-10 (50,000 train / 10,000 test)
              │
              ▼
┌─────────────────────────────────────────────┐
│         mp.spawn (2 processes)              │
│                                             │
│  ┌──────────────┐    ┌──────────────────┐  │
│  │   Rank 0     │    │    Rank 1        │  │
│  │  (Master)    │    │   (Worker)       │  │
│  │              │    │                  │  │
│  │ Samples:     │    │ Samples:         │  │
│  │ [0,2,4,6...] │    │ [1,3,5,7...]    │  │
│  │              │    │                  │  │
│  │ Forward Pass │    │ Forward Pass     │  │
│  │ Loss Compute │    │ Loss Compute     │  │
│  │ Backward()   │    │ Backward()       │  │
│  └──────┬───────┘    └───────┬──────────┘  │
│         │                    │             │
│         └────────┬───────────┘             │
│                  │                         │
│         All-Reduce (NCCL)                  │
│         avg gradients across ranks         │
│                  │                         │
│         optimizer.step()                   │
│         (identical update on all ranks)    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
        MLflow Experiment Tracking
        params + metrics per epoch
                   │
                   ▼
        Best model checkpoint saved
        model.module.state_dict()
                   │
                   ▼
        FastAPI /predict endpoint
```

---

## 📁 Project Structure

```
Distributed-ML/
│
├── src/
│   ├── dist_config.py         # Process group init + NCCL/Gloo setup
│   │                          # setup_distributed() + cleanup()
│   │
│   ├── data_utils.py          # DistributedSampler + augmentations
│   │                          # Each rank gets unique data subset
│   │                          # RandomCrop, Flip, ColorJitter, Erasing
│   │
│   ├── train_distributed.py   # Core DDP training (your original extended)
│   │                          # Multi-epoch loop + validation
│   │                          # MLflow logging + checkpointing
│   │                          # LR scheduler + gradient clipping
│   │
│   ├── benchmark.py           # Honest speedup measurement
│   │                          # Single process vs DDP timing
│   │                          # Outputs real numbers for README
│   │
│   └── api.py                 # FastAPI production serving
│                              # /predict /metrics /health
│
├── models/                    # Checkpoints (generated)
├── data/                      # CIFAR-10 auto-downloaded
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works — Step by Step

### Step 1 — Process Spawning (`dist_config.py`)

```python
mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
```

`mp.spawn` launches `world_size` independent Python processes. Each process gets its own memory space — bypasses the Python GIL completely. Each process owns one GPU exclusively.

**Why this matters:** Python's GIL means only one thread runs Python code at a time. Threading-based parallelism can't truly parallelise CPU-bound work. `mp.spawn` launches separate processes — no GIL contention.

### Step 2 — Process Group Initialisation (`dist_config.py`)

```python
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
```

All processes find each other via `MASTER_ADDR:MASTER_PORT`. Once connected, they form a process group that can communicate via collective operations (All-Reduce, Broadcast, etc.).

**NCCL vs Gloo:**

| Backend | Use case | Why |
|---|---|---|
| NCCL | NVIDIA GPUs | Optimised for GPU memory, uses NVLink/PCIe |
| Gloo | CPU fallback | Works everywhere, slower |

### Step 3 — Data Sharding (`data_utils.py`)

```python
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
```

**Without DistributedSampler:** Every GPU trains on the same data → no parallelism benefit, just wasted compute.

**With DistributedSampler:**
```
50,000 training images
    Rank 0: images [0, 2, 4, 6, ...]  → 25,000 unique images
    Rank 1: images [1, 3, 5, 7, ...]  → 25,000 unique images
    Combined: all 50,000 images seen per epoch ✓
```

```python
sampler.set_epoch(epoch)  # CRITICAL — must call every epoch
```

Without `set_epoch`, the sampler uses the same random seed every epoch — all epochs see data in identical order. Calling `set_epoch(epoch)` changes the seed each epoch so shuffling works correctly across ranks.

### Step 4 — DDP Wrapping + Gradient Sync

```python
model = DDP(model, device_ids=[rank])
```

DDP adds hooks to every parameter. When `loss.backward()` runs, DDP automatically triggers **All-Reduce** to average gradients across all ranks before `optimizer.step()`. Every rank ends up with identical gradients → identical weight updates → models stay in sync.

```
Rank 0 gradient: [0.3, 0.7, 0.2]
Rank 1 gradient: [0.5, 0.3, 0.4]
                 ─────────────────
All-Reduce avg:  [0.4, 0.5, 0.3]  ← both ranks get this
```

### Step 5 — MLflow Tracking

Only rank 0 logs to MLflow — otherwise all ranks would write duplicate entries.

Per epoch logged:
- `train_loss`, `train_acc`
- `val_loss`, `val_acc`
- `lr` (learning rate schedule)
- `epoch_time_s`

Final run logged:
- `total_training_time_s`
- Per-class test accuracy (airplane, car, bird...)
- Model artifact registered in MLflow registry

### Step 6 — Honest Benchmarking (`benchmark.py`)

Run this after training to get real speedup numbers:

```bash
python src/benchmark.py
```

Output:
```
Single process:      XXX.Xs
DDP (2 processes):   YYY.Ys
Speedup:             Z.Zx
Scaling efficiency:  W.W%
```

**These are your real numbers for the README and interviews.**
Replace the placeholder table below with output from your actual run.

---

## 🛠️ Quickstart

### On Kaggle (Recommended — free T4 x2 GPU)

1. New notebook → Settings → Accelerator → **GPU T4 x2**
2. Upload `src/` as a dataset
3. Run cells:

```python
!pip install mlflow -q

import os, sys
sys.path.append('/kaggle/working/src')
os.chdir('/kaggle/working')

# Run training
exec(open('src/train_distributed.py').read())
```

### Locally (2 GPUs or CPU simulation)

```bash
pip install -r requirements.txt

# Train
python src/train_distributed.py

# Benchmark
python src/benchmark.py

# View MLflow
mlflow ui --port 5000

# Serve
uvicorn src.api:app --port 8002
```

### Docker

```bash
docker build -t distributed-ml .
docker run --gpus all -p 8002:8002 distributed-ml
```

---

## 📊 Solved Problems & Optimizations

| # | Challenge | Technical Solution |
|---|---|---|
| 1 | GIL Bottlenecks | `mp.spawn` launches separate processes — no GIL contention |
| 2 | Data Duplication | `DistributedSampler` shards dataset — each rank sees unique samples |
| 3 | Gradient Inconsistency | DDP All-Reduce averages gradients — all ranks stay in sync |
| 4 | Shuffle Correctness | `sampler.set_epoch(epoch)` ensures different shuffle each epoch |
| 5 | Memory Overhead | Save `model.module.state_dict()` not DDP wrapper |
| 6 | Metric Aggregation | `dist.all_reduce` averages metrics across ranks |
| 7 | Overconfident Model | `label_smoothing=0.1` in CrossEntropyLoss |

---

## 📈 Performance (replace with your benchmark.py output)

| Metric | Value |
|---|---|
| World size | 2 processes |
| Effective batch size | 256 (128 × 2) |
| Single process time | run benchmark.py |
| DDP time | run benchmark.py |
| Speedup | run benchmark.py |
| Scaling efficiency | run benchmark.py |
| Best val accuracy | ~82% (20 epochs) |

**Run `python src/benchmark.py` and replace these numbers with your real results.**

---

## 🎯 Interview Q&A

**Q: What is the difference between DataParallel and DistributedDataParallel?**
A: DataParallel runs on one machine, splits batches across GPUs, but has a bottleneck — one master GPU gathers outputs, computes loss, and distributes gradients. This master GPU becomes the bottleneck and gets more memory pressure than others. DistributedDataParallel gives each GPU its own process. Each process computes its own loss and gradients. Gradients are averaged via All-Reduce — no bottleneck, no single point of failure. DDP scales linearly. DataParallel does not.

**Q: Why do you call sampler.set_epoch(epoch)?**
A: DistributedSampler uses a random seed to decide which samples go to which rank. Without `set_epoch`, that seed never changes — every epoch the data is shuffled identically. Calling `set_epoch(epoch)` incorporates the epoch number into the seed, so each epoch has genuinely different shuffling. Critical for training quality.

**Q: How does gradient synchronisation work in DDP?**
A: DDP adds backward hooks to every parameter. When `loss.backward()` runs, as soon as a parameter's gradient is computed, DDP immediately fires an All-Reduce operation for that parameter across all ranks. The result is the average gradient. By the time `optimizer.step()` runs, every rank has identical averaged gradients — all models stay in sync without any explicit synchronisation call.

**Q: Why save model.module.state_dict() instead of model.state_dict()?**
A: DDP wraps the model — `model` is the DDP wrapper, `model.module` is the actual underlying model. If you save `model.state_dict()`, you save DDP wrapper state which can't be loaded without DDP. Saving `model.module.state_dict()` saves the raw model weights which can be loaded anywhere — inference, fine-tuning, serving.

**Q: What scaling efficiency did you achieve?**
A: Run `python src/benchmark.py` to get your real number. On CPU simulation (Gloo backend) expect 40-70% — CPU memory is shared, communication overhead is high. On real multi-GPU (NCCL) expect 80-95% — NVLink bandwidth is very high, communication overhead is minimal. I report the honest number from my actual run.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **PyTorch DDP** | Core distributed training framework |
| **NCCL** | GPU communication backend — All-Reduce |
| **Gloo** | CPU fallback backend |
| **MLflow** | Experiment tracking + model registry |
| **FastAPI** | Production serving |
| **Docker** | Containerised deployment |
| **CIFAR-10** | Standard image classification benchmark |
