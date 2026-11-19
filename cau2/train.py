"""Huấn luyện một cấu hình với ngân sách epoch cố định."""
import os
import random

import numpy as np
import torch
from torch import nn

from model import CNN

EPOCHS = 10
SEED = 42
CPU_THREADS = 2


def seed_everything(seed, deterministic):
    """Đặt lại seed trước mỗi lượt huấn luyện."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


@torch.inference_mode()
def measure(model, loader, criterion, device):
    """Đo loss trung bình mỗi ảnh và accuracy trên một loader sạch."""
    model.eval()
    totals = torch.zeros(2, device=device)
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        totals[0] += criterion(logits, labels) * labels.numel()
        totals[1] += (logits.argmax(dim=1) == labels).sum()
        count += labels.numel()
    return tuple((totals / count).tolist())


def fit(config, data, device, epochs, seed, deterministic, name, track_train=True):
    """Trả về trọng số ở epoch cuối và lịch sử Train/Validation."""
    seed_everything(seed, deterministic)
    model = CNN(config).to(device)
    if device.type == "cpu":
        model.to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train, train_clean, val = data.loaders(
        config.batch_size, config.augmentation, seed, device.type == "cuda",
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        if track_train:
            train_loss, train_accuracy = measure(model, train_clean, criterion, device)
        else:
            train_loss, train_accuracy = None, None
        val_loss, val_accuracy = measure(model, val, criterion, device)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
            "val_loss": val_loss, "val_accuracy": val_accuracy,
        })
        if track_train or epoch == epochs:
            print(
                f"{name} | epoch {epoch:02d}/{epochs:02d} | "
                f"val_loss={val_loss:.4f} | val_accuracy={val_accuracy:.4f}",
                flush=True,
            )
    return model, history
