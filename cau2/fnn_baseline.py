"""So sánh FNN NumPy và CNN cơ sở trên cùng phép chia Fashion-MNIST."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from torch import nn

from data import EVAL_BATCH_SIZE, TRAIN_SIZE, VAL_SIZE, load_data
from model import Config
from train import CPU_THREADS, EPOCHS, SEED, fit, measure

INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 384
CLASS_COUNT = 10
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
FIT_SCOPE = "Initialization, Adam updates, and clean Train/Validation evaluation each epoch"


class FNN:
    """Hai lớp tuyến tính, ReLU ở tầng ẩn, và logit ở tầng ra."""

    def __init__(self, rng):
        self.w1 = rng.standard_normal((INPUT_SIZE, HIDDEN_SIZE), dtype=np.float32)
        self.w1 *= np.sqrt(2.0 / INPUT_SIZE)
        self.b1 = np.zeros(HIDDEN_SIZE, dtype=np.float32)
        self.w2 = rng.standard_normal((HIDDEN_SIZE, CLASS_COUNT), dtype=np.float32)
        self.w2 *= np.sqrt(2.0 / HIDDEN_SIZE)
        self.b2 = np.zeros(CLASS_COUNT, dtype=np.float32)
        self.parameters = (self.w1, self.b1, self.w2, self.b2)

    def forward(self, images):
        hidden = np.maximum(images @ self.w1 + self.b1, 0)
        return hidden, hidden @ self.w2 + self.b2

    def gradients(self, images, labels):
        hidden, logits = self.forward(images)
        _, d_logits = cross_entropy(logits, labels)
        # dL/dlogits = (softmax - one_hot) / batch_size.
        d_logits[np.arange(len(labels)), labels] -= 1
        d_logits /= len(labels)
        d_hidden = (d_logits @ self.w2.T) * (hidden > 0)
        return images.T @ d_hidden, d_hidden.sum(axis=0), hidden.T @ d_logits, d_logits.sum(axis=0)


def cross_entropy(logits, labels):
    """Trả về loss từng ảnh và softmax sau khi dịch logit để tránh tràn số."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    sums = exponentials.sum(axis=1, keepdims=True)
    losses = np.log(sums[:, 0]) - shifted[np.arange(len(labels)), labels]
    return losses, exponentials / sums


def numpy_split(dataset):
    images, labels = dataset.tensors
    return images.numpy().reshape(-1, INPUT_SIZE), labels.numpy()


def measure_fnn(model, images, labels):
    total_loss = 0.0
    correct = 0
    for start in range(0, len(labels), EVAL_BATCH_SIZE):
        batch = slice(start, start + EVAL_BATCH_SIZE)
        _, logits = model.forward(images[batch])
        losses, _ = cross_entropy(logits, labels[batch])
        total_loss += losses.sum(dtype=np.float64)
        correct += np.count_nonzero(logits.argmax(axis=1) == labels[batch])
    return total_loss / len(labels), correct / len(labels)


def fit_fnn(config, training, validation, epochs, seed):
    rng = np.random.default_rng(seed)
    model = FNN(rng)
    first_moments = [np.zeros_like(param) for param in model.parameters]
    second_moments = [np.zeros_like(param) for param in model.parameters]
    images, labels = training
    step = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(labels))
        for start in range(0, len(order), config.batch_size):
            batch = order[start:start + config.batch_size]
            gradients = model.gradients(images[batch], labels[batch])
            step += 1
            correction1 = 1 - ADAM_BETA1 ** step
            correction2 = 1 - ADAM_BETA2 ** step
            for param, grad, first, second in zip(
                model.parameters, gradients, first_moments, second_moments,
            ):
                first *= ADAM_BETA1
                first += (1 - ADAM_BETA1) * grad
                second *= ADAM_BETA2
                second += (1 - ADAM_BETA2) * np.square(grad)
                first_hat = first / correction1
                second_hat = second / correction2
                param -= config.learning_rate * first_hat / (np.sqrt(second_hat) + ADAM_EPSILON)

        train_loss, train_accuracy = measure_fnn(model, *training)
        val_loss, val_accuracy = measure_fnn(model, *validation)
        print(
            f"FNN | epoch {epoch:02d}/{epochs:02d} | "
            f"val_loss={val_loss:.4f} | val_accuracy={val_accuracy:.4f}",
            flush=True,
        )

    metrics = {
        "train_loss": train_loss, "train_accuracy": train_accuracy,
        "val_loss": val_loss, "val_accuracy": val_accuracy,
    }
    return model, metrics


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("du-lieu"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fnn_comparison"))
    parser.add_argument("--offline", action="store_true", help="Đọc dữ liệu cục bộ.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--threads", type=int, default=CPU_THREADS)
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--val-size", type=int, default=VAL_SIZE)
    args = parser.parse_args()
    if min(args.epochs, args.threads) < 1:
        parser.error("--epochs và --threads phải là số nguyên dương.")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed phải thuộc [0, 2^32).")
    if min(args.train_size, args.val_size) < 10 or args.train_size + args.val_size > 59_990:
        parser.error("Train và Validation cần ít nhất 10 ảnh/tập và tổng tối đa 59.990.")
    return args


def main():
    args = arguments()
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    config = Config()
    data = load_data(args.data_dir, args.seed, args.offline, args.train_size, args.val_size)
    training, validation = numpy_split(data.train_clean), numpy_split(data.validation)

    with threadpool_limits(limits=args.threads, user_api="blas"):
        start = perf_counter()
        fnn, fnn_metrics = fit_fnn(config, training, validation, args.epochs, args.seed)
        fnn_seconds = perf_counter() - start

        start = perf_counter()
        cnn, cnn_history = fit(config, data, device, args.epochs, args.seed, True, "CNN")
        cnn_seconds = perf_counter() - start

        test = data.test_loader(pin_memory=False)
        fnn_test_loss, fnn_test_accuracy = measure_fnn(fnn, *numpy_split(test.dataset))
        cnn_test_loss, cnn_test_accuracy = measure(cnn, test, nn.CrossEntropyLoss(), device)

    common = {
        "dataset": "Fashion-MNIST", "seed": args.seed,
        "epochs": args.epochs, "threads": args.threads, "device": str(device),
        "train_size": len(data.train_clean), "val_size": len(data.validation),
        "test_size": len(test.dataset), "pixel_mean": data.mean, "pixel_std": data.std,
        "batch_size": config.batch_size, "optimizer": "Adam", "learning_rate": config.learning_rate,
        "dtype": "float32", "dropout": config.dropout, "augmentation": config.augmentation,
        "numpy_version": np.__version__, "torch_version": str(torch.__version__),
        "deterministic": True, "fit_scope": FIT_SCOPE,
    }
    results = pd.DataFrame([
        {
            "run": "fnn", "model": "FNN NumPy", **common,
            "parameters": sum(param.size for param in fnn.parameters), "fit_seconds": fnn_seconds,
            **fnn_metrics, "test_loss": fnn_test_loss, "test_accuracy": fnn_test_accuracy,
        },
        {
            "run": "cnn", "model": "CNN PyTorch", **common,
            "parameters": sum(param.numel() for param in cnn.parameters()),
            "fit_seconds": cnn_seconds,
            "train_loss": cnn_history[-1]["train_loss"],
            "train_accuracy": cnn_history[-1]["train_accuracy"],
            "val_loss": cnn_history[-1]["val_loss"], "val_accuracy": cnn_history[-1]["val_accuracy"],
            "test_loss": cnn_test_loss, "test_accuracy": cnn_test_accuracy,
        },
    ])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "fnn_comparison.csv"
    results.to_csv(destination, index=False, encoding="utf-8-sig")
    columns = ["model", "parameters", "fit_seconds", "train_accuracy", "val_accuracy", "test_accuracy"]
    print(results[columns].to_string(index=False, float_format=lambda number: f"{number:.4f}"))
    print(f"Đã lưu kết quả vào {destination.resolve()}")


if __name__ == "__main__":
    main()
