import re
import os
import json
from enum import Enum
import token
from typing import List
from typing import Tuple
from typing import Iterator
from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
from modelscope import AutoTokenizer


DATA_PATH = "./dataset"


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
        self.data_path = os.path.join(DATA_PATH, data_type.value)
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
            (self.vocab_peak,), fill_value=0, dtype=torch.long
        )  # 预填充0
        valid_len = min(len(token_ids), self.vocab_peak)
        tensor[:valid_len] = torch.tensor(
            token_ids[-valid_len:], dtype=torch.long
        )  # 保留最近的token
        return tensor.view(256, 256)  # 调整为2D形状（根据需求修改）

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


class OptimizedTokenDataset(Dataset):
    def __init__(
        self,
        tokenizer: "TokenizerType",
        vocab_peak: int,
        data_type: "DatasetType",
        num_workers: int = 4,
    ):
        self.tokenizer = tokenizer
        self.vocab_peak = vocab_peak
        self.data_path = os.path.join(DATA_PATH, data_type.value)
        self.num_workers = num_workers

        # 并行加载数据
        self.samples = self._parallel_load_data()

        # 预分配内存的批量处理
        self.input_label_pairs = self._batch_process_samples()

    def __len__(self):
        return len(self.input_label_pairs)

    def __getitem__(self, idx):
        return self.input_label_pairs[idx]

    def _parallel_load_data(self) -> List[dict]:
        """多进程并行加载JSON数据"""
        with open(self.data_path, "r", encoding="utf-8") as f:
            with Pool(self.num_workers) as pool:
                return pool.map(json.loads, f.readlines())

    def _batch_process_samples(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """批量处理样本（核心优化）"""
        # 预分配内存
        max_pairs = sum(len(self.tokenizer.encode(s["text"])) - 1 for s in self.samples)
        input_tensors = torch.zeros((max_pairs, self.vocab_peak), dtype=torch.long)
        label_tensors = torch.zeros(max_pairs, dtype=torch.long)

        idx = 0
        for sample in self.samples:
            token_ids = self.tokenizer.encode(sample["text"], add_special_tokens=True)
            n = len(token_ids)

            # 批量生成输入-标签对
            for i in range(1, n):
                # 输入处理（向量化操作）
                if i <= self.vocab_peak:
                    input_tensors[idx, :i] = torch.tensor(
                        token_ids[:i], dtype=torch.long
                    )
                else:
                    input_tensors[idx] = torch.tensor(
                        token_ids[i - self.vocab_peak : i], dtype=torch.long
                    )

                # 标签处理
                label_tensors[idx] = token_ids[i]
                idx += 1

        # 转换为内存共享的TensorDataset
        return list(
            zip(
                input_tensors[:idx].pin_memory(),  # 固定内存加速GPU传输
                label_tensors[:idx].pin_memory(),
            )
        )

    @staticmethod
    def collate_fn(batch):
        """自定义批处理函数"""
        inputs = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        return inputs, labels


if __name__ == "__main__":
    tokenizer = TokenizerType.QWEN_QWQ_32B
    vocab_peak = 256 * 256
    data_type = DatasetType.PRETRAIN_HQ
    token_dataset = StreamTokenDataset(tokenizer, vocab_peak, data_type)
    print(token_dataset[1])
