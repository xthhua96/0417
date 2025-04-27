import os
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
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
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
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["epoch"]  # 返回上次训练到的 epoch


def train(model, dataloader, criterion, optimizer, device, epoch, scheduler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    batch_losses = []
    batch_accuracies = []
    batch_ppls = []

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 记录指标
        batch_loss = loss.item()
        batch_ppl = (
            math.exp(batch_loss) if batch_loss < 20 else float("inf")
        )  # 防止爆掉
        total_loss += batch_loss * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        batch_correct = (preds == labels).sum().item()
        correct += batch_correct
        batch_accuracy = batch_correct / inputs.size(0)

        batch_losses.append(batch_loss)
        batch_accuracies.append(batch_accuracy)
        batch_ppls.append(batch_ppl)

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(batch_loss)
            else:
                scheduler.step()

        wandb.log(
            {
                "batch_train_loss": batch_loss,
                "batch_train_accuracy": batch_accuracy,
                "batch_train_ppl": batch_ppl,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
                "batch": batch_idx,
            }
        )

        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch} Batch {batch_idx} | Loss: {batch_loss:.4f} | PPL: {batch_ppl:.2f} | Acc: {batch_accuracy:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

    avg_loss = total_loss / len(dataloader.dataset)
    avg_accuracy = correct / len(dataloader.dataset)
    avg_ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    wandb.log(
        {
            "train_loss": avg_loss,
            "train_accuracy": avg_accuracy,
            "train_ppl": avg_ppl,
            "epoch": epoch,
        }
    )

    print(
        f"\n✅ Epoch {epoch} Summary: Loss: {avg_loss:.4f} | PPL: {avg_ppl:.2f} | Acc: {avg_accuracy:.4f}"
    )
    return avg_loss, avg_accuracy


def main():
    args = parse_args()
    device = torch.device(get_device(args.device))
    print(f"Using device: {device}")

    init_wandb(args)

    from token_dataset import TokenDataset
    from token_dataset import DataLoaderX

    # 数据加载
    train_dataset = TokenDataset()
    train_loader = DataLoaderX(
        local_rank=0,
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    model = init_model("resnet18")
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    model = model.to(device)

    wandb.watch(model, log="all")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=500000, gamma=0.1)

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
