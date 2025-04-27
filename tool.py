import os
import yaml
import json
import h5py
from tqdm import tqdm

import numpy as np
from modelscope import AutoTokenizer


def load_config() -> dict:
    with open("./config.yaml", "r") as f:
        return yaml.safe_load(f)


def get_tokenizer(name: str):
    return AutoTokenizer.from_pretrained(name)


def get_paths(config: dict) -> tuple[str, str, str]:
    sys_name = os.uname().sysname
    if sys_name == "Darwin":
        raw_dir = config["raw_data"]["darwin_data_dir"]
        dataset_dir = config["dataset"]["darwin_data_dir"]
    elif sys_name == "Linux":
        raw_dir = config["raw_data"]["linux_data_dir"]
        dataset_dir = config["dataset"]["linux_data_dir"]
    else:
        raise ValueError(f"Unsupported OS: {sys_name}")
    return (
        os.path.join(raw_dir, config["raw_data"]["file_name"]),
        os.path.join(dataset_dir, config["dataset"]["file_name"]),
        config["raw_data"]["tokenizer"],
    )


def init_dataset():
    config = load_config()
    raw_path, dataset_path, tokenizer_name = get_paths(config)
    tokenizer = get_tokenizer(tokenizer_name)

    # 配置缓存批量写入参数
    BATCH_SIZE = 1000
    buffer_labels = []

    # 创建 HDF5 文件和动态数据集
    with h5py.File(dataset_path, "w") as h5file:
        label_ds = h5file.create_dataset(
            "labels", shape=(0,), maxshape=(None,), dtype="i4"
        )

        current_size = 0
        number = 0
        with open(raw_path, "r", encoding="utf-8") as dataset_file:
            for line in tqdm(dataset_file, desc="Processing"):
                sample = json.loads(line.strip())
                sample_token_ids = tokenizer.encode(
                    sample["text"], add_special_tokens=True
                )
                number += len(sample_token_ids) - 1

                for i in range(len(sample_token_ids)):
                    buffer_labels.append(sample_token_ids[i])

                    if len(buffer_labels) == BATCH_SIZE:
                        # 写入当前批次
                        new_size = current_size + BATCH_SIZE
                        label_ds.resize((new_size,))
                        label_ds[current_size:new_size] = buffer_labels
                        current_size = new_size
                        buffer_labels.clear()

            # 写入剩余未满一批的数据
            if buffer_labels:
                batch_len = len(buffer_labels)
                new_size = current_size + batch_len
                label_ds.resize((new_size,))
                label_ds[current_size:new_size] = buffer_labels

        h5file.attrs["number"] = number


def init_dataset_v1():
    config = load_config()  # 你的配置加载函数
    raw_path, dataset_path, tokenizer_name = get_paths(config)  # 获取数据路径
    tokenizer = get_tokenizer(tokenizer_name)  # 获取tokenizer

    # 配置缓存批量写入参数
    BATCH_SIZE = 1000
    buffer_records = []  # 用于缓存1000条记录
    total_records = 0  # 用于统计总记录条数

    # 创建 HDF5 文件和动态数据集
    with h5py.File(dataset_path, "w") as h5file:
        # 创建一个 "records" 数据集，用于存储token_ids，dtype='O' 表示存储对象类型
        records_ds = h5file.create_dataset(
            "records",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.vlen_dtype(np.dtype("int32")),
        )

        current_size = 0
        with open(raw_path, "r", encoding="utf-8") as dataset_file:
            for line in tqdm(dataset_file, desc="Processing"):
                sample = json.loads(line.strip())
                sample_token_ids = tokenizer.encode(
                    sample["text"], add_special_tokens=True
                )

                # 将 sample_token_ids 添加到 buffer_records
                buffer_records.append(sample_token_ids)
                total_records += 1  # 统计总记录数

                if len(buffer_records) >= BATCH_SIZE:
                    # 每处理1000条记录时，将数据批量写入 HDF5 文件
                    new_size = current_size + BATCH_SIZE
                    records_ds.resize((new_size,))
                    records_ds[current_size:new_size] = buffer_records[
                        :BATCH_SIZE
                    ]  # 写入前1000条数据
                    buffer_records = buffer_records[BATCH_SIZE:]  # 清空缓存

                    current_size = new_size  # 更新当前已保存数据的大小

        # 写入剩余未满一批的数据
        if buffer_records:
            new_size = current_size + len(buffer_records)
            records_ds.resize((new_size,))
            records_ds[current_size:new_size] = buffer_records

        # 将记录条数保存到文件
        h5file.attrs["total_records"] = total_records

    print(f"Total records processed: {total_records}")


if __name__ == "__main__":
    # init_dataset()
    init_dataset_v1()
