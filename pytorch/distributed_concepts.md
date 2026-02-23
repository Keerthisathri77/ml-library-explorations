Data Parallel (DP) — 

Data Parallel replicates the same model across multiple devices and splits each batch of data between them. Each replica performs forward and backward passes independently on its shard of data. Gradients from all replicas are then gathered and averaged before the optimizer updates the parameters. In classic PyTorch DataParallel, gradient synchronization happens centrally, which creates overhead and bottlenecks due to Python’s GIL and single-process coordination. It is simpler but less efficient. It is generally considered outdated compared to Distributed Data Parallel for serious multi-GPU training.

Distributed Data Parallel (DDP) — 

Distributed Data Parallel (DDP) launches one process per device. Each process owns a full model replica and computes gradients on its local data shard. During backward propagation, gradients are synchronized using collective communication operations (typically all-reduce). Unlike DataParallel, synchronization is decentralized and happens in parallel using backends like NCCL (GPU) or Gloo (CPU). This avoids Python bottlenecks and scales significantly better. DDP is currently the standard approach for multi-GPU training in PyTorch because it offers near-linear scaling and efficient gradient synchronization.

All-Reduce — 

All-reduce is a distributed collective operation that aggregates values (such as gradients) across multiple processes and redistributes the result to all of them. In DDP, after each process computes gradients locally, all-reduce averages those gradients across devices. The result ensures that every replica updates with identical parameter values. Efficient implementations use ring-based or tree-based communication to minimize bandwidth usage. All-reduce is critical for synchronous distributed training because it guarantees consistent model state across all workers after every training step