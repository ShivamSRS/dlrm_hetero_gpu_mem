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


class VerticalParallelModel(nn.Module):
    def __init__(self, model_parallel_size):
        super(VerticalParallelModel, self).__init__()
        self.model_parallel_size = model_parallel_size
        self.embedding = ParallelEmbedding(2000, 20)
        self.mlp = ParallelMLP(20, 0.5)
        self.linear = ColumnParallelLinear(20, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = self.mlp(x)
        x = self.linear(x)
        return x


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
        self.model = nn.DataParallel(self.model)  # Wrap the model with DataParallel
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        print(f"Loss: {loss}")
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch, max_epochs, trainer, start_time):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        total_batch_time = 0.0

        for batch_idx, (source, targets) in enumerate(self.train_data):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

            if rank == 0:
                batch_start_time = time.time()
            torch.distributed.barrier()
            if rank == 0:
                batch_end_time = time.time()
                total_batch_time = batch_end_time - batch_start_time
                print(
                    f"GPU: {rank} | Epoch: {epoch}/{max_epochs} | Batch: {batch_idx + 1}/{len(self.train_data)} | Batch Time: {total_batch_time:.4f}s")
            torch.distributed.barrier()

        avg_batch_time = total_batch_time / len(self.train_data)
        print(f"avg batch time: {avg_batch_time:.6f}s")
        end_time = time.time()
        epoch_time = end_time - start_time
        torch.distributed.barrier()
        epoch_times = torch.tensor([epoch_time], device=torch.device('cuda'))
        torch.distributed.all_reduce(epoch_times)
        total_epoch_time = epoch_times.item()
        if rank == 0:
            print(f"Epoch: {epoch + 1}/{max_epochs} | Epoch Time: {total_epoch_time:.4f}s")
        if rank == 0 and epoch % trainer.save_every == 0:
            trainer._save_snapshot(epoch)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),  # Use model.module.state_dict() for DataParallel
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        start_time = time.time()
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch,max_epochs, start_time)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

    def train_epoch(self, epoch, max_epochs, trainer):
        start_time = time.time()
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        for batch_idx, (source, target) in enumerate(trainer.train_data):
            batch_start_time = time.time()
            source = source.to(rank)
            target = target.to(rank)
            self._run_batch(source, target)
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            torch.distributed.barrier()
            batch_times = torch.tensor([batch_time], device=torch.device('cuda'))
            torch.distributed.all_reduce(batch_times)
            total_batch_time = batch_times.item()
            if rank == 0:
                print(
                    f"GPU: {rank} | Epoch: {epoch}/{max_epochs} | Batch: {batch_idx + 1}/{len(trainer.train_data)} | Batch Time: {total_batch_time:.4f}s")
            torch.distributed.barrier()

        avg_batch_time = total_batch_time / len(trainer.train_data)
        print(f"avg batch time: {avg_batch_time:.6f}s")
        end_time = time.time()
        epoch_time = end_time - start_time
        torch.distributed.barrier()
        epoch_times = torch.tensor([epoch_time], device=torch.device('cuda'))
        torch.distributed.all_reduce(epoch_times)
        total_epoch_time = epoch_times.item()
        if rank == 0:
            print(f"Epoch: {epoch + 1}/{max_epochs} | Epoch Time: {total_epoch_time:.4f}s")
        if rank == 0 and epoch % trainer.save_every == 0:
            trainer._save_snapshot(epoch)


def load_train_objs():
    train_set = MyTrainDataset(2048)

    model = VerticalParallelModel(2)  # Use the custom VerticalParallelModel

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
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='vertical distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--model_parallel_size', default=2, type=int,
                        help='Input batch size on each device (default: 32)')
    parser.add_argument('--world_size', default=2, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')

    args = parser.parse_args()

    main(args.world_size, args.model_parallel_size, args.save_every, args.total_epochs, args.batch_size)
