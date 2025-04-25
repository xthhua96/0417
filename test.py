from email.policy import default
from math import fabs
from typing import List
from tqdm import tqdm

from sympy import false
from torchvision import models
import torch.multiprocessing as mp


def init_model(model_name: str):
    weights = None
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
    else:
        raise ValueError("Invalid model name")
    # 1. 加载预训练模型
    model = models.resnet50(weights=weights)
    model.eval()
    return model


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import wandb
import argparse


# 参数配置
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=false, help="Path to dataset")
    parser.add_argument(
        "--num_classes",
        type=int,
        required=false,
        help="Number of classes",
        default=152064,
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device selection",
    )
    return parser.parse_args()


# 自动选择设备
def get_device(device_type):
    if device_type == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():  # Apple Silicon
            return "mps"
        else:
            return "cpu"
    return device_type


# 灰度转三通道Dataset
class GrayscaleToRGBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.samples.append(
                    (os.path.join(cls_dir, img_name), self.class_to_idx[cls])
                )

        self.transform = transform or self.default_transform()

    @staticmethod
    def default_transform():
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),  # 关键：转为3通道
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        return self.transform(img), label


# 初始化wandb
def init_wandb(args):
    wandb.init(project="resnet50-grayscale-classification", config=vars(args))
    wandb.watch_called = False


def compute_batch_stats(args):
    """每个进程计算的函数"""
    batch, device = args
    inputs = batch[0].float().to(device)
    return (
        inputs.mean([0, 2, 3]).sum().item(),  # 均值
        inputs.std([0, 2, 3]).sum().item(),  # 标准差
        inputs.size(0),  # 样本数
    )


def compute_dataset_stats(dataset, num_batches=100, num_workers=4):
    """多进程计算数据集的均值和标准差"""
    # 初始化多进程
    mp.set_start_method("spawn", force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 准备数据批次
    loader = dataset.get_stream_loader(batch_size=32)
    batches = [(batch, device) for idx, batch in enumerate(loader) if idx < num_batches]

    # 多进程计算
    mean = 0.0
    std = 0.0
    total_samples = 0

    with mp.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(compute_batch_stats, batches),
                total=len(batches),
                desc="多进程计算统计量",
            )
        )

    # 汇总结果
    for batch_mean, batch_std, batch_samples in results:
        mean += batch_mean
        std += batch_std
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean, std


from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau


def train(model, dataloader, criterion, optimizer, device, epoch, scheduler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    batch_losses = []
    batch_accuracies = []

    # 手动标准化参数（假设已预计算）
    mean, std = compute_dataset_stats(dataloader.dataset, 4000)

    for batch_idx, (inputs, labels) in enumerate(
        dataloader.dataset.get_stream_loader(32)
    ):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = (inputs.float() - mean) / std  # 标准化

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 记录指标
        batch_loss = loss.item()
        total_loss += batch_loss * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        batch_correct = (preds == labels).sum().item()
        correct += batch_correct
        batch_accuracy = batch_correct / inputs.size(0)

        batch_losses.append(batch_loss)
        batch_accuracies.append(batch_accuracy)

        # 动态更新学习率（每个batch或每个epoch）
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(batch_loss)  # 基于损失更新
            else:
                scheduler.step()  # 基于步数更新

        # 记录学习率和指标
        wandb.log(
            {
                "batch_train_loss": batch_loss,
                "batch_train_accuracy": batch_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
                "batch": batch_idx,
            }
        )

        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch} Batch {batch_idx} | Loss: {batch_loss:.4f} | Acc: {batch_accuracy:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

    # Epoch总结
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)

    wandb.log({"train_loss": avg_loss, "train_accuracy": accuracy, "epoch": epoch})

    print(f"\nEpoch {epoch} Summary: Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")
    return avg_loss, accuracy, batch_losses, batch_accuracies


# 主函数
def main():
    args = parse_args()
    device = torch.device(get_device(args.device))
    print(f"Using device: {device}")

    # 初始化wandb
    init_wandb(args)

    from token_dataset import StreamTokenDataset

    # 数据加载
    train_dataset = StreamTokenDataset()
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    # 模型初始化
    model = init_model("resnet50")
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    model = model.to(device)

    # 监控模型
    wandb.watch(model, log="all")

    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        avg_loss, accuracy, _, _ = train(
            model, train_loader, criterion, optimizer, device, epoch, scheduler
        )

    # 保存模型
    torch.save(model.state_dict(), "token_resnet50.pth")
    wandb.save("token_resnet50.pth")


if __name__ == "__main__":
    main()
