"""Tính chỉ số Test, vẽ biểu đồ và minh họa suy luận."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import functional as TF

MAX_ERRORS = 12


@torch.inference_mode()
def evaluate_test(model, loader, device):
    """Duyệt Test đúng một lượt; dùng chung dự đoán cho mọi chỉ số."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    target, predicted, mistakes = [], [], []
    loss = 0.0
    for images, labels in loader:
        logits = model(images.to(device))
        loss += criterion(logits, labels.to(device)).item()
        guesses = logits.argmax(dim=1).cpu()
        target.extend(labels.tolist())
        predicted.extend(guesses.tolist())
        wrong = torch.where(guesses != labels)[0][:MAX_ERRORS - len(mistakes)]
        mistakes.extend((images[i].clone(), labels[i].item(), guesses[i].item()) for i in wrong)

    classes = list(range(10))
    precision, recall, f1, _ = precision_recall_fscore_support(
        target, predicted, labels=classes, average="macro", zero_division=0,
    )
    metrics = {
        "test_loss": loss / len(target), "test_accuracy": accuracy_score(target, predicted),
        "macro_precision": precision, "macro_recall": recall, "macro_f1": f1,
    }
    return metrics, confusion_matrix(target, predicted, labels=classes), mistakes


def save_figure(fig, destination):
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_history(histories, destination):
    """Ghép bốn cấu hình để so sánh khoảng cách Train và Validation."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for color, (name, history) in zip(plt.cm.tab10.colors, histories.items()):
        frame = pd.DataFrame(history)
        for ax, metric, title in zip(axes, ["loss", "accuracy"], ["Cross-entropy", "Accuracy"]):
            ax.plot(frame.epoch, frame[f"train_{metric}"], color=color, marker=".", label=name)
            ax.plot(frame.epoch, frame[f"val_{metric}"], color=color, marker=".", linestyle="--")
            ax.set(xlabel="Epoch", ylabel=title, title=title)
            ax.set_xticks(frame.epoch)
            ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, title="Cấu hình")
    fig.suptitle("Train (nét liền) và Validation (nét đứt)")
    save_figure(fig, destination)


def plot_confusion(matrix, classes, name, destination):
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    heatmap = ax.imshow(matrix, cmap="Blues")
    ticks = np.arange(len(classes))
    ax.set(xticks=ticks, yticks=ticks, yticklabels=classes,
           xlabel="Dự đoán", ylabel="Nhãn thật", title=name)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    for (row, col), count in np.ndenumerate(matrix):
        ax.text(col, row, str(count), ha="center", va="center", fontsize=7,
                color="white" if count > matrix.max() / 2 else "black")
    fig.colorbar(heatmap, ax=ax, label="Số ảnh")
    save_figure(fig, destination)


def plot_mistakes(mistakes, data, name, destination):
    fig, axes = plt.subplots(3, 4, figsize=(10, 8), layout="constrained")
    classes = data.test_source.classes
    for ax in axes.flat:
        ax.axis("off")
    for ax, (image, target, prediction) in zip(axes.flat, mistakes):
        pixels = (image * data.std + data.mean).clamp(0, 1)
        ax.imshow(pixels.squeeze(0), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Thật: {classes[target]}\nĐoán: {classes[prediction]}", fontsize=8)
    if not mistakes:
        axes.flat[0].text(0.5, 0.5, "Không có ảnh dự đoán sai", ha="center")
    fig.suptitle(f"Ảnh Test dự đoán sai · {name}")
    save_figure(fig, destination)


def plot_comparison(comparison, destination):
    """So sánh điểm Test và loss trên hai thang đo riêng."""
    fig, (scores, loss) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    colors = ["#4c78a8", "#f58518"]
    points = comparison.set_index("model")[["test_accuracy", "macro_f1"]]
    points.rename(columns={"test_accuracy": "Accuracy", "macro_f1": "Macro F1"}).T.plot.bar(
        ax=scores, rot=0, ylim=(0, 1), title="Accuracy và macro F1 trên Test", color=colors,
    )
    scores.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2)
    for bars in scores.containers:
        scores.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    losses = loss.bar(comparison["model"], comparison["test_loss"], color=colors)
    loss.set(title="Cross-entropy trên Test", ylabel="Loss")
    loss.set_ylim(0, comparison["test_loss"].max() * 1.15)
    loss.bar_label(losses, fmt="%.3f", padding=3, fontsize=8)
    save_figure(fig, destination)


@torch.inference_mode()
def predict_top3(model, image, data, device):
    """Trả về ba cặp (tên lớp, xác suất) cho ảnh PIL hoặc đường dẫn JPEG/PNG."""
    if isinstance(image, (str, Path)):
        image = TF.to_pil_image(read_image(str(image), mode=ImageReadMode.GRAY))
    image = TF.resize(TF.to_grayscale(image), [28, 28])
    batch = data.transform()(image).unsqueeze(0).to(device)
    model.eval()
    probabilities = model(batch).softmax(dim=1)[0]
    values, indices = probabilities.topk(3)
    classes = data.source.classes
    return list(zip((classes[i] for i in indices.tolist()), values.tolist()))
