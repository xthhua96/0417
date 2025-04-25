import os
import yaml
import json
import h5py


def process_dump_h5py():
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)
        sys_name = os.uname().sysname
        data_dir = ""
        if sys_name == "Darwin":
            data_dir = config["dataset"]["DARWIN_DATA_DIR"]
        elif sys_name == "Linux":
            data_dir = config["dataset"]["LINUX_DATA_DIR"]
        else:
            raise ValueError("Unsupported operating system")
        file_name = config["dataset"]["file_name"]
        file_path = os.path.join(data_dir, file_name)

    with open(file_path, "r", encoding="utf-8") as dataset_file:
        for line in dataset_file:
            sample = json.loads(line.strip())
            token_ids = tokenizer.encode(sample["text"], add_special_tokens=True)

    pass
