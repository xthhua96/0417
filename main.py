import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import wandb
import math
import argparse
from sympy import false
from torchvision import models
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchmetrics.classification import Accuracy
from torchmetrics import MeanMetric


def init_model(model_name: str, default_weight: bool = True):
    model_zoo = {
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    }

    if model_name not in model_zoo:
        raise ValueError(f"Invalid model name: {model_name}")

    model_fn, weight = model_zoo[model_name]
    model = model_fn(weights=weight if default_weight else None)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=false, help="Path to dataset")
    parser.add_argument(
        "--num_classes", type=int, default=152064, help="Number of classes"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device selection",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )  # 🔥 新增
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory"
    )  # 🔥 新增
    return parser.parse_args()


def get_device(device_type):
    if device_type == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_type


def init_wandb(args):
    wandb.init(project="ChatCNN", config=vars(args))
    wandb.watch_called = False


def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        checkpoint_path,
    )
    print(f"✅ Saved checkpoint: {checkpoint_path}")
    wandb.save(checkpoint_path)


def load_latest_checkpoint(model, optimizer, scheduler, checkpoint_dir, device):
    if not os.path.exists(checkpoint_dir):
        print("No checkpoint found, starting fresh training.")
        return 0  # 从头开始

    checkpoints = [ckpt for ckpt in os.listdir(checkpoint_dir) if ckpt.endswith(".pth")]
    if not checkpoints:
        print("No checkpoint files in checkpoint directory.")
        return 0

    # 按 epoch 排序找最大的
    checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    latest_ckpt = checkpoints[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_ckpt)

    print(f"🔄 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    new_lr = 0.001
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["epoch"]  # 返回上次训练到的 epoch


def train(model, dataloader, criterion, optimizer, device, epoch, scheduler=None):
    model.train()

    loss_metric = MeanMetric().to(device)
    acc_metric = Accuracy(task="multiclass", num_classes=152064).to(device)
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 更新指标
        batch_size = inputs.size(0)
        preds = torch.argmax(outputs, dim=1)

        loss_metric.update(loss.detach(), weight=batch_size)
        acc_metric.update(preds, labels)

        total_samples += batch_size

        # Scheduler按batch调
        if scheduler is not None:
            if hasattr(scheduler, "step"):
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss)
                else:
                    scheduler.step()

        # logging per batch
        if batch_idx % 10 == 0:
            batch_loss = loss.item()
            batch_acc = (preds == labels).float().mean().item()

            wandb.log({
                "batch_train_loss": batch_loss,
                "batch_train_accuracy": batch_acc,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
                "batch": batch_idx,
            })

            print(
                f"Epoch {epoch} Batch {batch_idx} | Loss: {batch_loss:.4f} | Acc: {batch_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

    # 计算整体
    avg_loss = loss_metric.compute().item()
    avg_accuracy = acc_metric.compute().item()

    wandb.log({
        "train_loss": avg_loss,
        "train_accuracy": avg_accuracy,
        "epoch": epoch,
    })

    print(f"\n✅ Epoch {epoch} Summary: Loss: {avg_loss:.4f} | Acc: {avg_accuracy:.4f}")
    return avg_loss, avg_accuracy



def validate(model, dataloader, criterion, device, epoch):
    model.eval()

    loss_metric = MeanMetric().to(device)
    acc_metric = Accuracy(task="multiclass", num_classes=model.num_classes).to(device)
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs, labels = batch[0].to(device), batch[1].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = inputs.size(0)
            preds = torch.argmax(outputs, dim=1)

            loss_metric.update(loss, weight=batch_size)
            acc_metric.update(preds, labels)

            total_samples += batch_size

            if batch_idx % 10 == 0:
                batch_loss = loss.item()
                batch_acc = (preds == labels).float().mean().item()

                print(
                    f"[Val] Epoch {epoch} Batch {batch_idx} | Loss: {batch_loss:.4f} | Acc: {batch_acc:.4f}"
                )

    avg_loss = loss_metric.compute().item()
    avg_accuracy = acc_metric.compute().item()

    # logging
    wandb.log({
        "val_loss": avg_loss,
        "val_accuracy": avg_accuracy,
        "epoch": epoch,
    })

    print(f"\n🧪 Validation Epoch {epoch} Summary: Loss: {avg_loss:.4f} | Acc: {avg_accuracy:.4f}")
    return avg_loss, avg_accuracy

def main():
    args = parse_args()
    device = torch.device(get_device(args.device))
    print(f"Using device: {device}")

    init_wandb(args)

    from token_dataset_v1 import TokenDatasetV1
    from token_dataset_v1 import DataLoaderX
    from torch.utils.data import DataLoader
    from token_dataset_v1 import my_collate_fn

    # 数据加载
    train_dataset = TokenDatasetV1()
    train_loader = DataLoaderX(
        local_rank=0,
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=my_collate_fn
    )

    model = init_model(model_name="resnet18", default_weight=False)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    model = model.to(device)

    wandb.watch(model, log="all")

    criterion = nn.CrossEntropyLoss(ignore_index=-2025)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=3000000, gamma=0.1)

    start_epoch = 1

    if args.resume:
        start_epoch = (
            load_latest_checkpoint(
                model, optimizer, scheduler, args.checkpoint_dir, device
            )
            + 1
        )

    # 训练循环
    for epoch in range(start_epoch, args.epochs + 1):
        avg_loss, accuracy = train(
            model, train_loader, criterion, optimizer, device, epoch, scheduler
        )
        save_checkpoint(model, optimizer, scheduler, epoch, args.checkpoint_dir)

    # 最后保存一次
    final_model_path = os.path.join(args.checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Final model saved to {final_model_path}")
    wandb.save(final_model_path)


if __name__ == "__main__":
    main()