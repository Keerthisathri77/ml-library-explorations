import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    model = nn.Linear(10, 10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    for _ in range(5):
        inputs = torch.randn(20, 10).to(rank)
        outputs = ddp_model(inputs)
        loss = outputs.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size)

if __name__ == "__main__":
    main()
