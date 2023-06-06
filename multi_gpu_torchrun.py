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
    # print("get rank working",torch.distributed.get_rank() )
    # print("is initialized workingv",torch.distributed.is_initialized())
    world_size = torch.distributed.get_world_size()
    device = int(os.environ["LOCAL_RANK"]) % torch.cuda.device_count()
    # print("device=",device)
    # print("world size wokring",world_size,"model prll size",model_parallel_size)

    # ranks = range(1, world_size, 1)
    # print(*ranks)
    # group = torch.distributed.new_group([0])
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
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        # if os.path.exists(snapshot_path):
        #     print("Loading snapshot")
        #     self._load_snapshot(snapshot_path)

        # self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        # print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        # print("about to pass data to model")
        output = self.model(source)
        # print(f"forward pass done, now loss.") #source: {source}, target: {targets}")
        # print("#############")
        # exit()
        loss = F.cross_entropy(output, targets)
        print(f"Loss:{loss}")
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        # self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            # print("SOURCE DATA SHAPOE",source.shape)
            targets = targets.to(self.gpu_id)
            # print("targets DATA SHAPOE",targets.shape)
            # print(targets)
            # exit()
            self._run_batch(source, targets)
            # exit()

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),  # .modules.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

    def train_epoch(self, epoch, max_epochs, trainer):
        start_time = time.time()  # Start timing the epoch
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        for batch_idx, (source, target) in enumerate(trainer.train_data):
            batch_start_time = time.time()  # Start timing the batch
            # print(f"batch start time: {batch_start_time:.4f}s")
            # Training code here
            source = source.to(rank)
            target = target.to(rank)
            # print("run batch called")
            self._run_batch(source, target)
            # print("run batch exited")
            batch_end_time = time.time()  # End timing the batch
            batch_time = batch_end_time - batch_start_time
            # print(f"batch time:{batch_time}")
            # Synchronize across GPUs
            torch.distributed.barrier()

            # Gather batch timing information from all GPUs
            batch_times = torch.tensor([batch_time], device=torch.device('cuda'))
            torch.distributed.all_reduce(batch_times)
            total_batch_time = batch_times.item()
            # Print batch timing information for each GPU
            print(
                f"GPU: {rank} | Epoch: {epoch}/{max_epochs} | Batch: {batch_idx + 1}/{len(trainer.train_data)} | Batch Time: {total_batch_time:.4f}s")

        avg_batch_time = total_batch_time / len(trainer.train_data)  # Calculate average batch time
        print(f"avg batch time: {avg_batch_time:.6f}s")
        end_time = time.time()  # End timing the epoch
        epoch_time = end_time - start_time

        # Synchronize across GPUs
        torch.distributed.barrier()

        # Gather epoch timing information from all GPUs
        epoch_times = torch.tensor([epoch_time], device=torch.device('cuda'))
        torch.distributed.all_reduce(epoch_times)
        total_epoch_time = epoch_times.item()
        # Print epoch timing information for each GPU
        if rank == 0:
            print(f"Epoch: {epoch + 1}/{max_epochs} | Epoch Time: {total_epoch_time:.4f}s")

        # Save snapshot
        if rank == 0 and epoch % trainer.save_every == 0:
            trainer._save_snapshot(epoch)


def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset

    model = torch.nn.Sequential(ParallelEmbedding(2000, 20), ParallelMLP(20, 0.5), ColumnParallelLinear(20, 2))
    # model = ColumnParallelLinear(20,2)
    print(model)
    # model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Compute the total sparse entries
    total_indices = 2000  # Total number of indices in the tensor
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
        # sampler=DistributedSampler(dataset)
    )


def main(world_size, model_parallel_size, save_every: int, total_epochs: int, batch_size: int,
         snapshot_path: str = "snapshot.pt"):
    ddp_setup(world_size, model_parallel_size)
    dataset, model, optimizer, total_sparse_entries = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    # trainer.train(total_epochs)
    for epoch in range(trainer.epochs_run, total_epochs):
        trainer.train_epoch(epoch + 1, total_epochs, trainer)  # Increment epoch by 1

    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--model_parallel_size', default=2, type=int,
                        help='Input batch size on each device (default: 32)')
    parser.add_argument('--world_size', default=2, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')

    args = parser.parse_args()

    main(args.world_size, args.model_parallel_size, args.save_every, args.total_epochs, args.batch_size)