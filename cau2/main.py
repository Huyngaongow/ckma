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
METRICS = [
    "train_loss", "train_accuracy", "val_loss", "val_accuracy", "accuracy_gap",
    "test_loss", "test_accuracy", "macro_precision", "macro_recall", "macro_f1",
]


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
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, default=SEED, help="Seed cho một lần chạy.")
    seeds.add_argument("--seeds", type=int, nargs="+", help="Ít nhất hai seed khác nhau.")
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
    if args.seeds is not None and (len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds)):
        parser.error("--seeds cần ít nhất hai seed và không được lặp.")
    if any(not 0 <= seed < 2 ** 32 for seed in (args.seeds or [args.seed])):
        parser.error("Mỗi seed phải thuộc [0, 2^32).")
    if min(args.train_size, args.val_size) < 10 or args.train_size + args.val_size > 59_990:
        parser.error("Train và Validation cần ít nhất 10 ảnh/tập và tổng tối đa 59.990.")
    if (args.dropout or args.augmentation) and not args.single:
        parser.error("Dùng --single để bật/tắt riêng từng kỹ thuật.")
    if args.image is not None and not args.image.is_file():
        parser.error(f"Không tìm thấy ảnh {args.image}.")
    return args


def main():
    args = arguments()
    torch.set_num_threads(args.threads)
    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
    device_name = args.device
    if device_name == "auto":
        device_name = next(name for name, enabled in available.items() if enabled)
    if not available[device_name]:
        raise SystemExit(f"Thiết bị {device_name.upper()} không khả dụng.")
    device = torch.device(device_name)
    if args.seeds is None:
        run_experiment(args, device, args.seed, args.output_dir)
        return

    results = pd.concat(
        [run_experiment(args, device, seed, args.output_dir / f"seed-{seed}") for seed in args.seeds],
        ignore_index=True,
    )
    grouped = results.groupby(["run", "model"], sort=False)
    aggregate = grouped[METRICS].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{statistic}" for metric, statistic in aggregate.columns]
    aggregate.insert(0, "n_seeds", grouped["seed"].nunique())
    destination = args.output_dir / "aggregate.csv"
    aggregate.to_csv(destination, encoding="utf-8-sig")
    print(f"\nKết quả trung bình và độ lệch chuẩn mẫu: {destination.resolve()}")


def run_experiment(args, device, seed, out):
    start = perf_counter()
    seed_everything(seed, args.deterministic)
    data = load_data(args.data_dir, seed, args.offline, args.train_size, args.val_size)
    out.mkdir(parents=True, exist_ok=True)
    npz = out / "split_indices.npz"
    np.savez_compressed(npz, train=data.train_indices, validation=data.val_indices)
    print(f"Thiết bị: {device} | seed: {seed} | deterministic: {args.deterministic}")
    print(f"Train={len(data.train_indices)} | Validation={len(data.val_indices)} | "
          f"Test={len(data.test_source)} | mean={data.mean:.6f} | std={data.std:.6f}")

    experiments = EXPERIMENTS
    if args.single:
        experiments = (("single", "Cấu hình riêng", args.dropout, args.augmentation),)
    base = Config()
    runs = []
    for key, name, dropout, augmentation in experiments:
        config = replace(base, dropout=dropout, augmentation=augmentation)
        model, history = fit(config, data, device, args.epochs, seed, args.deterministic, name)
        runs.append((key, name, config, model, history))

    if not args.single:
        best, ranking = random_search(
            data, device, args.epochs, seed, args.deterministic, args.trials,
        )
        ranking.to_csv(out / "random_search_trials.csv", index=False)
        print(f"Cấu hình được chọn trên Validation: {best}")
        key, name = EXTENSION
        model, history = fit(best, data, device, args.epochs, seed, args.deterministic, name)
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
            "run": key, "model": name, **asdict(config), "epochs": args.epochs, "seed": seed,
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
        comparison = results.loc[results["run"].isin(["baseline", "both", "random_search"]), columns]
        comparison.to_csv(out / "extension_comparison.csv", index=False, encoding="utf-8-sig")
        plot_comparison(comparison, out / "extension_comparison.png")
        print("\nSo sánh cơ sở, kết hợp hai kỹ thuật và Random Search")
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
    return results


if __name__ == "__main__":
    main()
