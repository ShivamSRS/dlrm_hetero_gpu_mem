import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from fakedata import MyTrainDataset
import time
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import mpu
import model
from mpu.layers import ColumnParallelLinear, RowParallelLinear, ParallelMLP, ParallelEmbedding

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"


def ddp_setup(world_size, model_parallel_size):
    init_process_group(backend="nccl", world_size=world_size, rank=int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    mpu.initialize_model_parallel(model_parallel_size)


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            save_every: int,
            snapshot_path: str,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train_epoch(self, epoch, max_epochs):
        start_time = time.time()  # Start timing the epoch
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        for batch_idx, (source, target) in enumerate(self.train_data):
            batch_start_time = time.time()  # Start timing the batch
            source = source.to(rank)
            target = target.to(rank)
            self._run_batch(source, target)
            batch_end_time = time.time()  # End timing the batch
            batch_time = batch_end_time - batch_start_time
            torch.distributed.barrier()

            # Gather batch timing information from all GPUs
            batch_times = torch.tensor([batch_time], device=torch.device('cuda'))
            torch.distributed.all_reduce(batch_times)
            total_batch_time = batch_times.item()
            # Print batch timing information for each GPU
            print(
                f"GPU: {rank} | Epoch: {epoch}/{max_epochs} | Batch: {batch_idx + 1}/{len(self.train_data)} | Batch Time: {total_batch_time:.4f}s")

        avg_batch_time = total_batch_time / len(self.train_data)  # Calculate average batch time
        print(f"avg batch time: {avg_batch_time:.6f}s")
        end_time = time.time()  # End timing the epoch
        epoch_time = end_time - start_time

        torch.distributed.barrier()

        # Gather epoch timing information from all GPUs
        epoch_times = torch.tensor([epoch_time], device=torch.device('cuda'))
        torch.distributed.all_reduce(epoch_times)
        total_epoch_time = epoch_times.item()
        # Print epoch timing information for each GPU
        if rank == 0:
            print(f"Epoch: {epoch + 1}/{max_epochs} | Epoch Time: {total_epoch_time:.4f}s")

        if rank == 0 and epoch % self.save_every == 0:
            self._save_snapshot(epoch)


def load_train_objs():
    train_set = MyTrainDataset(2048)

    model = torch.nn.Sequential(ParallelEmbedding(2000, 20), ParallelMLP(20, 0.5), ColumnParallelLinear(20, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    total_indices = 2000
    total_sparse_indices = sum(1 for data in train_set if torch.any(data[0] == 0))
    total_sparse_entries = total_sparse_indices / (train_set.size * total_indices)

    print("Total sparse entries:", total_sparse_entries)
    return train_set, model, optimizer, total_sparse_entries


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False
    )


def main(world_size, model_parallel_size, save_every: int, total_epochs: int, batch_size: int,
         snapshot_path: str = "snapshot.pt"):
    ddp_setup(world_size, model_parallel_size)
    dataset, model, optimizer, total_sparse_entries = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    for epoch in range(trainer.epochs_run, total_epochs):
        trainer.train_epoch(epoch + 1, total_epochs)

    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='horizontal distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--model_parallel_size', default=2, type=int,
                        help='Input batch size on each device (default: 32)')
    parser.add_argument('--world_size', default=2, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')

    args = parser.parse_args()

    main(args.world_size, args.model_parallel_size, args.save_every, args.total_epochs, args.batch_size)
