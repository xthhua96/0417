import re
import os
import json
import threading
import queue as Queue
from enum import Enum
from typing import List
from typing import Tuple
from typing import Iterator
from typing import Any
from tqdm import tqdm

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from modelscope import AutoTokenizer

from stream_dataset import BackgroundGenerator
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


class StreamTokenDataset(Dataset):
    def __init__(
        self,
        tokenizer: "TokenizerType" = TokenizerType.QWEN_QWQ_32B,
        vocab_peak: int = 256 * 256,
        data_type: "DatasetType" = DatasetType.PRETRAIN_HQ,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer.value)
        self.vocab_peak = vocab_peak
        self.data_path = os.path.join(DATA_DIR, data_type.value)
        self._length = self._calculate_length()  # 预计算总样本数

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 通过生成器按需获取数据（实际使用时建议用DataLoader迭代）
        return next(self._data_generator(start_idx=idx, end_idx=idx + 1))

    def _calculate_length(self) -> int:
        """多进程并行计数"""
        word_token_ratio = 1.3  # 经验值：1单词≈1.3个token（需根据语料调整）
        count = 0
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="快速计数"):
                text = json.loads(line.strip())["text"]
                word_count = len(re.findall(r"\w+", text))  # 简单统计单词数
                count += max(1, int(word_count * word_token_ratio))  # 保证至少1个样本
        return count

    def _data_generator(
        self, start_idx: int = 0, end_idx: int = None
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """核心生成器：逐行流式处理数据"""
        current_idx = 0
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                token_ids = self.tokenizer.encode(
                    sample["text"], add_special_tokens=True
                )

                # 为每个token序列生成n-1个样本
                for i in range(1, len(token_ids)):
                    if current_idx >= start_idx:
                        if end_idx is not None and current_idx >= end_idx:
                            return
                        yield (
                            self._preprocess_token_ids(token_ids[:i]),
                            torch.tensor(token_ids[i], dtype=torch.int64),
                        )
                    current_idx += 1

    def _preprocess_token_ids(self, token_ids: List[int]) -> torch.Tensor:
        """单样本处理（优化版）"""
        tensor = torch.full(
            (self.vocab_peak,), fill_value=0, dtype=torch.float
        )  # 预填充0
        valid_len = min(len(token_ids), self.vocab_peak)
        tensor[:valid_len] = torch.tensor(
            token_ids[-valid_len:], dtype=torch.float
        )  # 保留最近的token

        return (
            tensor.view(256, 256).unsqueeze(0).repeat(3, 1, 1)
        )  # 调整为2D形状（根据需求修改）

    def get_stream_loader(
        self, batch_size: int = 32
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """获取批量的数据流"""
        batch = []
        for item in self._data_generator():
            batch.append(item)
            if len(batch) >= batch_size:
                inputs = torch.stack([x[0] for x in batch])
                labels = torch.stack([x[1] for x in batch])
                yield inputs, labels
                batch = []
        if batch:  # 处理剩余样本
            yield torch.stack([x[0] for x in batch]), torch.stack([x[1] for x in batch])

    def _local_count_samples(self) -> int:
        count = 0
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="本地计算样本数"):
                sample = json.loads(line)
                token_ids = self.tokenizer.encode(
                    sample["text"], add_special_tokens=True
                )
                # 一行直接加上 len(token_ids)-1，这里不再做内层循环
                count += max(0, len(token_ids) - 1)
        return count

    def _data_generator_mock(self, start_idx: int = 0, end_idx: int = None) -> Any:
        current_idx = 0
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                token_ids = self.tokenizer.encode(
                    sample["text"], add_special_tokens=True
                )

                for i in range(1, len(token_ids)):
                    if current_idx >= start_idx:
                        if end_idx is not None and current_idx >= end_idx:
                            return
                        yield 1
                    current_idx += 1


class TokenDataset(Dataset):
    def __init__(
        self,
        local_rank: int = 0,
        window_size: int = 1024,
        vocab_peak: int = (256, 265),
    ):
        super(TokenDataset, self).__init__()
        config = load_config()
        _, dataset_path, _ = get_paths(config)
        self.local_rank = local_rank
        self.window_size = window_size
        self.vocab_peak = vocab_peak
        self.dataset_file = h5py.File(dataset_path, "r")
        self.labels = self.dataset_file["labels"]
        self.num = self.dataset_file["number"]

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
    print(token_dataset[0][0].shape)
    # print(f"样本数: {count_num}")
    # print(token_dataset[0][0].shape)
