import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler


"""
Distributed semi sampler:
- Take one portion of data from labeled dataset and one portion from the unlabed dataset.
- It reads the "sample_ratio" argument, e.g. sample_ratio=[1,4].
    For example, if sample_per_gpu is 5, then sampler will take 1 labeled sample and 4 unlabeled samples.
"""


class DistributedSemiSampler(Sampler):
    def __init__(
        self,
        dataset,
        sample_ratio=None,
        samples_per_gpu=2,
        num_replicas=None,
        rank=None,
        **kwargs
    ):
        # check to avoid some problem
        assert samples_per_gpu > 1, "samples_per_gpu should be greater than 1."
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank

        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.num_samples = 0
        self.cumulative_sizes = dataset.cumulative_sizes
        # decide the frequency to sample each kind of datasets
        if not isinstance(sample_ratio, list):
            sample_ratio = [sample_ratio] * len(self.cumulative_sizes)
        self.sample_ratio = sample_ratio
        self.sample_ratio = [
            int(sr / min(self.sample_ratio)) for sr in self.sample_ratio
        ]
        self.size_of_dataset = []
        cumulative_sizes = [0] + self.cumulative_sizes
        norm_ratio = [x / sum(self.sample_ratio) for x in self.sample_ratio]
        self.epoch_length=max([ int(self.cumulative_sizes[-1] * r) for r in norm_ratio])
        size_of_dataset = 0
        for j in range(len(self.cumulative_sizes)):
            size_of_dataset = max(
                size_of_dataset, np.ceil((cumulative_sizes[j+1]-cumulative_sizes[j]) / self.sample_ratio[j])
            )

        self.size_of_dataset.append(
            int(np.ceil(size_of_dataset / self.samples_per_gpu / self.num_replicas))
            * self.samples_per_gpu
        )
        for j in range(len(self.cumulative_sizes)):
            self.num_samples += self.size_of_dataset[-1] * self.sample_ratio[j]

        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = []
        cumulative_sizes = [0] + self.cumulative_sizes

        indice_per_dataset = []

        for j in range(len(self.cumulative_sizes)):
            indice_per_dataset.append(
                np.arange(cumulative_sizes[j],cumulative_sizes[j+1])
            )

        shuffled_indice_per_dataset = [
            s[list(torch.randperm(int(s.shape[0]), generator=g).numpy())]
            for s in indice_per_dataset
        ]
        # split into
        total_indice = []
        batch_idx = 0
        # pdb.set_trace()
        while batch_idx < self.epoch_length * self.num_replicas:
            ratio = [x / sum(self.sample_ratio) for x in self.sample_ratio]

            # num of each dataset
            ratio = [int(r * self.samples_per_gpu) for r in ratio]

            ratio[-1] = self.samples_per_gpu - sum(ratio[:-1])
            selected = []
            # print(ratio)
            for j in range(len(shuffled_indice_per_dataset)):
                if len(shuffled_indice_per_dataset[j]) < ratio[j]:#如果最后pool剩下的不够了，就从头继续。
                    shuffled_indice_per_dataset[j] = np.concatenate(
                        (
                            shuffled_indice_per_dataset[j],
                            indice_per_dataset[j][
                                list(
                                    torch.randperm(
                                        int(indice_per_dataset[j].shape[0]),
                                        generator=g,
                                    ).numpy()
                                )
                            ],
                        )
                    )

                selected.append(shuffled_indice_per_dataset[j][: ratio[j]])
                shuffled_indice_per_dataset[j] = shuffled_indice_per_dataset[j][
                    ratio[j]:
                ]
                # 此处从pool中pop掉取出的元素
            selected = np.concatenate(selected)
            total_indice.append(selected)
            batch_idx += 1
            # print(self.size_of_dataset)
        indice = np.concatenate(total_indice)
        indices.append(indice)
        indices = np.concatenate(indices)  # k
        indices = [
            indices[j]
            for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu,
                    generator=g,
                )
            )
            for j in range(
                i * self.samples_per_gpu,
                (i + 1) * self.samples_per_gpu,
            )
        ]
        # 随机去掉多余样本

        offset = len(self) * self.rank
        indices = indices[offset : offset + len(self)]
        assert len(indices) == len(self)
        return iter(indices)

    def __len__(self):
        return self.epoch_length * self.samples_per_gpu

    def set_epoch(self, epoch):
        self.epoch = epoch