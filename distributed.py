import math

import torch

from torch.utils.data.distributed import DistributedSampler


class DistributedRandomSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        num_samples=None,
        num_replicas=None,
        rank=None,
        seed=0,
    ):
        super().__init__(
            dataset,
            num_replicas,
            rank,
            seed,
        )
        self.total_size = num_samples
        self.num_samples_per_rank = math.ceil(self.total_size / self.num_replicas)
        assert self.total_size <= len(self.dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        indices = indices[: self.total_size]

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples_per_rank

        return iter(indices)

    def __len__(self):
        return self.num_samples_per_rank
