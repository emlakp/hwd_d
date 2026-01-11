#!/usr/bin/env python3
"""Quick profiling checks."""
import torch
import time

print("="*70)
print("Quick System Check")
print("="*70)

# GPU info
print("\n[GPU Info]")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("No GPU available!")

# PyTorch config
print(f"\n[PyTorch Info]")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"cuDNN Version: {torch.backends.cudnn.version()}")
print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
print(f"cuDNN Benchmark: {torch.backends.cudnn.benchmark}")

# Test GPU speed
print(f"\n[GPU Speed Test]")
print("Testing matrix multiplication speed...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Warmup
for _ in range(10):
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    c = torch.matmul(a, b)
torch.cuda.synchronize()

# Actual test
times = []
for _ in range(100):
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    t0 = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    times.append(time.time() - t0)

print(f"Average time: {sum(times)/len(times)*1000:.3f}ms")
print(f"Operations/sec: {1.0/(sum(times)/len(times)):.1f}")

if sum(times)/len(times) > 0.01:
    print("⚠️  GPU seems slow - check if it's under load from other processes")
else:
    print("✓ GPU speed looks normal")

# Check dataset cache
from pathlib import Path
cache_path = Path("/home/akopyane/Desktop/rl/new_data/training/cached_data.pkl")
if cache_path.exists():
    size_gb = cache_path.stat().st_size / 1e9
    print(f"\n[Dataset Cache]")
    print(f"Cache file exists: {cache_path}")
    print(f"Cache size: {size_gb:.2f} GB")
else:
    print(f"\n[Dataset Cache]")
    print("⚠️  No cache file found - first run will be slow!")

# Suggestions
print("\n" + "="*70)
print("QUICK FIXES TO TRY:")
print("="*70)
print("1. Enable cuDNN benchmark:")
print("   Add to your script: torch.backends.cudnn.benchmark = True")
print("\n2. Check GPU utilization while training:")
print("   Run: watch -n 1 nvidia-smi")
print("\n3. Profile with PyTorch profiler:")
print("   Add torch.profiler.profile() to your training loop")
print("\n4. Reduce logging frequency:")
print("   Set log_every_n_steps to 50 or 100 instead of 1")

