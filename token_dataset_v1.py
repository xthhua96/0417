import torch
import h5py
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import queue as Queue
import threading
from typing import List
from typing import Tuple
import numpy as np

from tool import load_config
from tool import get_paths


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self._preload()
        return self

    def _preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            self.iter = None
            return
        with torch.cuda.stream(self.stream):
            features, labels = self.batch
            features = features.to(self.local_rank, non_blocking=True)
            labels = labels.to(self.local_rank, non_blocking=True)
            self.batch = (features, labels)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if self.batch is None:
            raise StopIteration
        self._preload()
        return batch


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class TokenDatasetV1(Dataset):
    def __init__(
        self,
        local_rank: int = 0,
        vocab_peak: int = (256, 256),
    ):
        super(TokenDatasetV1, self).__init__()
        config = load_config()
        _, dataset_path, _ = get_paths(config)
        self.local_rank = local_rank
        self.vocab_peak = vocab_peak
        self.dataset_file = h5py.File(dataset_path, "r")
        self.tokens_list = self.dataset_file["records"]
        self.num = self.dataset_file.attrs["total_records"]

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.tokens_list[idx]


def preprocess_token_ids(token_ids: np.ndarray) -> torch.Tensor:
    tensor = torch.zeros(256 * 256, dtype=torch.float)
    valid_indices = token_ids[(token_ids >= 0) & (token_ids < 256 * 256)]

    tensor[valid_indices] = 1.0
    return tensor.view(256, 256).unsqueeze(0).repeat(3, 1, 1)


def padding(batch_input: List[List[Tuple[torch.Tensor, torch.Tensor]]]):
    max_len = max(len(l) for l in batch_input)
    for input_list in batch_input:
        while len(input_list) < max_len:
            input_list.append(
                (
                    torch.zeros((3, 256, 256), dtype=torch.float),
                    torch.tensor(-2025, dtype=torch.long),
                )
            )
    features = []
    labels = []
    for i in range(max_len):
        a_features = []
        a_labels = []
        for j in range(len(batch_input)):
            a_features.append(batch_input[j][i][0])
            a_labels.append(batch_input[j][i][1])
        features.append(torch.stack(a_features, dim=0))
        labels.append(torch.stack(a_labels, dim=0))
    features = torch.stack(features, dim=0)
    labels = torch.stack(labels, dim=0)
    return features, labels


def my_collate_fn(batch: List[np.ndarray]):
    features = []
    labels = []
    for token_ids in batch:
        for i in range(1, len(token_ids) - 1, 2):
            features.append(preprocess_token_ids(token_ids[:i]))
            labels.append(torch.tensor(token_ids[i], dtype=torch.long))
    features = torch.stack(features, dim=0)
    labels = torch.stack(labels, dim=0)
    return features, labels


if __name__ == "__main__":
    dataset = TokenDatasetV1()
    # print(dataset[0])
    # print(type(dataset[0]))
    dataloader = DataLoaderX(
        local_rank=0,
        dataset=dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        collate_fn=my_collate_fn,
    )
    for i, batch in enumerate(dataloader):
        # print(i)
        print(batch[0].shape)
        print(batch[1].shape)
        if i == 3:
            break

        # for i in range(batch[0].size(0)):
        #     print(batch[0][i].shape)
        #     print(batch[1][i].shape)
        # break
