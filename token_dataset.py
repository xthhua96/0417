import os
from random import sample
import threading
import queue as Queue
from enum import Enum
from typing import List
from typing import Tuple

import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


from tool import load_config
from tool import get_paths

system_name = os.uname().sysname
print(system_name)
DATA_DIR = ""
if system_name == "Darwin":
    DATA_DIR = "/Users/a58/MyProject/minimind/dataset/"
elif system_name == "Linux":
    DATA_DIR = "/workspace/dataset"
else:
    raise ValueError("Unsupported operating system")


class DatasetType(Enum):
    """
    Dataset Type
    """

    DPO = "dpo.jsonl"
    LORA_IDENTITY = "lora_identity.jsonl"
    LORA_MEDICAL = "lora_medical.jsonl"
    PRETRAIN_HQ = "pretrain_hq.jsonl"
    R1_MIX_1024 = "r1_mix_1024.jsonl"
    SFT_1024 = "sft_1024.jsonl"
    SFT_2048 = "sft_2048.jsonl"
    SFT_512 = "sft_512.jsonl"
    SFT_MINI_512 = "sft_mini_512.jsonl"


class TokenizerType(Enum):
    QWEN_QWQ_32B = "Qwen/QwQ-32B"


class TokenDataset(Dataset):
    def __init__(
        self,
        local_rank: int = 0,
        window_size: int = 1024,
        vocab_peak: int = (256, 256),
    ):
        super(TokenDataset, self).__init__()
        config = load_config()
        _, dataset_path, _ = get_paths(config)
        self.local_rank = local_rank
        self.window_size = window_size
        self.vocab_peak = vocab_peak
        self.dataset_file = h5py.File(dataset_path, "r")
        self.labels = self.dataset_file["labels"]
        self.num = self.dataset_file.attrs["number"]

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = (idx // self.window_size) * self.window_size
        if idx == 0:
            return self._preprocess_token_ids([-1]), torch.tensor(
                self.labels[idx], dtype=torch.long
            )
        return self._preprocess_token_ids(self.labels[start_idx:idx]), torch.tensor(
            self.labels[idx], dtype=torch.long
        )

    def _preprocess_token_ids(self, token_ids: List[int]) -> torch.Tensor:
        tensor = torch.full(
            (self.vocab_peak[0] * self.vocab_peak[1],), fill_value=0, dtype=torch.float
        )
        valid_len = min(len(token_ids), self.vocab_peak[0] * self.vocab_peak[1])
        tensor[:valid_len] = torch.tensor(token_ids[-valid_len:], dtype=torch.float)
        return (
            tensor.view(self.vocab_peak[0], self.vocab_peak[1])
            .unsqueeze(0)
            .repeat(3, 1, 1)
        )


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
            for k in range(len(self.batch)):
                features, labels = self.batch[k]
                
                self.batch[k] = self.batch[k].to(self.local_rank, non_blocking=True)

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


if __name__ == "__main__":
    tokenizer = TokenizerType.QWEN_QWQ_32B
    vocab_peak = 256 * 256
    data_type = DatasetType.PRETRAIN_HQ
    token_dataset = TokenDataset()
    # print(token_dataset[0])
    print(token_dataset.num)
    # print(f"样本数: {count_num}")
    # print(token_dataset[0][0].shape)
