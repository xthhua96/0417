from email.policy import default
from math import fabs
from typing import List

from sympy import false
from torchvision import models


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
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
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


# 训练函数
def train(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)

    wandb.log({"train_loss": avg_loss, "train_accuracy": accuracy, "epoch": epoch})

    print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


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

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, criterion, optimizer, device, epoch)

    # 保存模型
    torch.save(model.state_dict(), "grayscale_resnet50.pth")
    wandb.save("grayscale_resnet50.pth")


if __name__ == "__main__":
    main()
