"""Chạy bốn thí nghiệm chính, Random Search, đánh giá và suy luận."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from data import TRAIN_SIZE, VAL_SIZE, load_data
from evaluate import (
    evaluate_test, plot_comparison, plot_confusion, plot_history, plot_mistakes, predict_top3,
)
from extension import TRIALS, random_search
from model import Config
from train import CPU_THREADS, EPOCHS, SEED, fit, seed_everything

EXPERIMENTS = (
    ("baseline", "Cơ sở", False, False),
    ("dropout", "Dropout", True, False),
    ("augmentation", "Augmentation", False, True),
    ("both", "Dropout + Augmentation", True, True),
)
EXTENSION = ("random_search", "Random Search")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Cần dùng số nguyên dương.")
    return number


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--offline", action="store_true", help="Đọc dữ liệu cục bộ.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=positive_int, default=EPOCHS)
    parser.add_argument("--train-size", type=positive_int, default=TRAIN_SIZE)
    parser.add_argument("--val-size", type=positive_int, default=VAL_SIZE)
    parser.add_argument("--trials", type=int, choices=range(8, 13), default=TRIALS)
    parser.add_argument("--threads", type=positive_int, default=CPU_THREADS)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dropout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--augmentation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--single", action="store_true", help="Chạy một cấu hình theo hai cờ.")
    parser.add_argument("--image", type=Path, help="Ảnh JPEG/PNG để in top 3.")
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed phải thuộc [0, 2^32).")
    if min(args.train_size, args.val_size) < 10 or args.train_size + args.val_size > 59_990:
        parser.error("Train và Validation cần ít nhất 10 ảnh/tập và tổng tối đa 59.990.")
    if (args.dropout or args.augmentation) and not args.single:
        parser.error("Dùng --single để bật/tắt riêng từng kỹ thuật.")
    if args.image is not None and not args.image.is_file():
        parser.error(f"Không tìm thấy ảnh {args.image}.")
    return args


def main():
    args = arguments()
    start = perf_counter()
    torch.set_num_threads(args.threads)
    seed_everything(args.seed, args.deterministic)
    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
    device_name = args.device
    if device_name == "auto":
        device_name = next(name for name, enabled in available.items() if enabled)
    if not available[device_name]:
        raise SystemExit(f"Thiết bị {device_name.upper()} không khả dụng.")
    device = torch.device(device_name)
    data = load_data(args.data_dir, args.seed, args.offline, args.train_size, args.val_size)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    npz = out / "split_indices.npz"
    np.savez_compressed(npz, train=data.train_indices, validation=data.val_indices)
    print(f"Thiết bị: {device} | seed: {args.seed} | deterministic: {args.deterministic}")
    print(f"Train={len(data.train_indices)} | Validation={len(data.val_indices)} | "
          f"Test={len(data.test_source)} | mean={data.mean:.6f} | std={data.std:.6f}")

    experiments = EXPERIMENTS
    if args.single:
        experiments = (("single", "Cấu hình riêng", args.dropout, args.augmentation),)
    base = Config()
    runs = []
    for key, name, dropout, augmentation in experiments:
        config = replace(base, dropout=dropout, augmentation=augmentation)
        model, history = fit(config, data, device, args.epochs, args.seed, args.deterministic, name)
        runs.append((key, name, config, model, history))

    if not args.single:
        best, ranking = random_search(
            data, device, args.epochs, args.seed, args.deterministic, args.trials,
        )
        ranking.to_csv(out / "random_search_trials.csv", index=False)
        print(f"Cấu hình được chọn trên Validation: {best}")
        key, name = EXTENSION
        model, history = fit(best, data, device, args.epochs, args.seed, args.deterministic, name)
        runs.append((key, name, best, model, history))

    # Chỉ duyệt Test sau khi mọi cấu hình và trọng số đã được chốt.
    loader = data.test_loader(device.type == "cuda")
    rows, long_history = [], []
    for key, name, config, model, history in runs:
        metrics, matrix, mistakes = evaluate_test(model, loader, device)
        plot_confusion(matrix, data.source.classes, name, out / f"confusion_{key}.png")
        plot_mistakes(mistakes, data, name, out / f"errors_{key}.png")
        final = history[-1]
        rows.append({
            "run": key, "model": name, **asdict(config), "epochs": args.epochs, "seed": args.seed,
            "deterministic": args.deterministic, "train_size": len(data.train_indices),
            "val_size": len(data.val_indices), "test_size": len(data.test_source),
            "train_loss": final["train_loss"], "train_accuracy": final["train_accuracy"],
            "val_loss": final["val_loss"], "val_accuracy": final["val_accuracy"],
            "accuracy_gap": final["train_accuracy"] - final["val_accuracy"], **metrics,
        })
        long_history.extend({"run": key, "model": name, **row} for row in history)

    results = pd.DataFrame(rows)
    results.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(long_history).to_csv(out / "history.csv", index=False, encoding="utf-8-sig")
    plot_history({name: history for _, name, _, _, history in runs[:4]}, out / "learning_curves.png")
    columns = ["model", "train_accuracy", "val_accuracy", "accuracy_gap", "test_loss",
               "test_accuracy", "macro_precision", "macro_recall", "macro_f1"]
    print("\nKết quả tổng hợp")
    print(results[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    if not args.single:
        comparison = results.loc[results["run"].isin(["baseline", "random_search"]), columns]
        comparison.to_csv(out / "extension_comparison.csv", index=False, encoding="utf-8-sig")
        plot_comparison(comparison, out / "extension_comparison.png")
        print("\nSo sánh cơ sở với Random Search")
        print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    # Mặc định lấy một ảnh gốc từ Validation để minh họa suy luận.
    demo_image = args.image
    if demo_image is None:
        demo_image, label = data.source[int(data.val_indices[0])]
        print(f"\nNhãn thật của ảnh Validation minh họa: {data.source.classes[label]}")
    print(f"Top 3 · {runs[-1][1]}")
    for label, probability in predict_top3(runs[-1][3], demo_image, data, device):
        print(f"  {label:<14} {probability:.2%}")
    print(f"\nĐã lưu kết quả vào {out.resolve()} | Thời gian: {perf_counter() - start:.1f} giây")


if __name__ == "__main__":
    main()
