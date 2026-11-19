"""Hồi quy cường độ nén bê tông bằng FNN NumPy và dữ liệu UCI."""

import argparse
from copy import deepcopy
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 42
EXTRA_SEEDS = (7, 13, 21, 73)
HIDDEN_SIZES = (64, 32)
EPOCHS = 2500
BATCH_SIZE = 64
LEARNING_RATE = 0.035
PATIENCE = 150
MIN_DELTA = 1e-5
L2_CANDIDATES = (0.1, 1.0, 5.0, 20.0)
DATA_URL = (
    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/"
    "ff933238d9c30da179b7e3ad4b6ca938d67efc53/concrete.csv"
)
COLUMNS = (
    "cement", "slag", "ash", "water", "superplastic",
    "coarseagg", "fineagg", "age", "strength",
)


class FNN:
    """Mạng tanh nhiều tầng ẩn, đầu ra tuyến tính một chiều."""

    def __init__(self, input_size, hidden_sizes, seed):
        if len(hidden_sizes) < 2 or min(hidden_sizes) < 1:
            raise ValueError("Cần ít nhất hai tầng ẩn có kích thước dương.")
        sizes = (input_size, *hidden_sizes, 1)
        rng = np.random.default_rng(seed)
        # Xavier uniform cho tanh. Mọi bias bắt đầu từ 0.
        self.weights = [
            rng.uniform(-np.sqrt(6 / (p + q)), np.sqrt(6 / (p + q)), (p, q))
            for p, q in zip(sizes[:-1], sizes[1:])
        ]
        self.biases = [np.zeros((1, q), dtype=np.float64) for q in sizes[1:]]

    def forward(self, x):
        """Trả về dự đoán và cache (Z, A), với A[0] là đầu vào."""
        z_values, activations = [], [x]
        last = len(self.weights) - 1
        for layer, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ w + b
            a = np.tanh(z) if layer < last else z
            z_values.append(z)
            activations.append(a)
        return activations[-1], (z_values, activations)

    @staticmethod
    def loss(pred, y):
        """Tính nửa MSE thuần trên thang y đang dùng."""
        assert pred.shape == y.shape and y.ndim == 2 and y.shape[1] == 1
        return float(0.5 * np.mean(np.square(pred - y)))

    def backward(self, y, cache, l2, n_train):
        """Tính gradient của nửa MSE mini-batch cộng L2 chuẩn hóa theo n_train."""
        z_values, a = cache
        assert a[-1].shape == y.shape == (len(y), 1)
        # Đầu ra tuyến tính, d(1/2 * mean(error²))/dZ = error / B.
        dz = (a[-1] - y) / len(y)
        grad_w, grad_b = [], []
        for layer in reversed(range(len(self.weights))):
            w, b = self.weights[layer], self.biases[layer]
            assert dz.shape == z_values[layer].shape == (len(y), w.shape[1])
            assert a[layer].shape == (len(y), w.shape[0])
            # Phạt λ/(2N) * sum(||W||²), chỉ cộng λ/N * W vào dW.
            dw = a[layer].T @ dz + (l2 / n_train) * w
            db = dz.sum(axis=0, keepdims=True)
            assert dw.shape == w.shape and db.shape == b.shape
            grad_w.append(dw)
            grad_b.append(db)
            if layer > 0:
                dz = (dz @ w.T) * (1.0 - a[layer] ** 2)
        return grad_w[::-1], grad_b[::-1]

    def fit(self, x, y, x_val, y_val, *, l2, early_stop, seed,
            epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
            patience=PATIENCE, min_delta=MIN_DELTA):
        """Huấn luyện bằng mini-batch GD, trả lịch sử và các mốc epoch."""
        rng = np.random.default_rng(seed)
        n_train = len(x)
        history = np.empty((epochs, 2), dtype=np.float64)
        best_loss = patience_reference = np.inf
        best_epoch, stale, best_state = 0, 0, None
        for epoch in range(1, epochs + 1):
            order = rng.permutation(n_train)
            for start in range(0, n_train, batch_size):
                idx = order[start:start + batch_size]
                _, cache = self.forward(x[idx])
                dw, db = self.backward(y[idx], cache, l2, n_train)
                for w, b, d_w, d_b in zip(self.weights, self.biases, dw, db):
                    w -= lr * d_w
                    b -= lr * d_b

            train_loss = self.loss(self.forward(x)[0], y)
            val_loss = self.loss(self.forward(x_val)[0], y_val)
            if not np.isfinite([train_loss, val_loss]).all():
                raise FloatingPointError(f"Loss không hữu hạn tại epoch {epoch}. Giảm lr.")
            history[epoch - 1] = train_loss, val_loss

            # Luôn lưu cực tiểu thật. min_delta chỉ điều khiển patience.
            if val_loss < best_loss:
                best_epoch, best_loss = epoch, val_loss
                if early_stop:
                    best_state = deepcopy((self.weights, self.biases))
            if early_stop:
                if patience_reference - val_loss > min_delta:
                    patience_reference, stale = val_loss, 0
                else:
                    stale += 1
                if stale >= patience:
                    break

        if early_stop:
            self.weights, self.biases = best_state
        return history[:epoch], best_epoch, epoch


def gradient_check(x_train, y_train, seed):
    """So sánh gradient mạng tanh float64 bằng sai phân trung tâm."""
    # Lấy một lát nhỏ của dữ liệu thật, chỉ từ tập Train.
    x = np.asarray(x_train[:5, :3], dtype=np.float64)
    y = np.asarray(y_train[:5], dtype=np.float64)
    model = FNN(x.shape[1], (4, 3), seed)
    l2, n_train, eps = 0.7, len(x_train), 1e-5

    def objective():
        penalty = sum(np.sum(w ** 2) for w in model.weights)
        return model.loss(model.forward(x)[0], y) + l2 * penalty / (2 * n_train)

    _, cache = model.forward(x)
    dw, db = model.backward(y, cache, l2, n_train)
    analytic, numeric = [], []
    for param, grad in zip(model.weights + model.biases, dw + db):
        for idx in np.ndindex(param.shape):
            old = param[idx]
            param[idx] = old + eps
            plus = objective()
            param[idx] = old - eps
            minus = objective()
            param[idx] = old
            numeric.append((plus - minus) / (2 * eps))
            analytic.append(grad[idx])
    analytic, numeric = np.asarray(analytic), np.asarray(numeric)
    denominator = np.linalg.norm(analytic) + np.linalg.norm(numeric)
    error = np.linalg.norm(analytic - numeric) / max(denominator, 1e-15)
    print(f"Gradient check float64 (có L2) | sai số tương đối = {error:.3e}")
    if not error < 1e-7:
        raise RuntimeError("Gradient check không đạt ngưỡng 1e-7.")


def load_data(csv_path):
    """Đọc CSV UCI, báo cáo và loại bỏ giá trị thiếu hoặc dòng trùng."""
    if csv_path is None and Path("concrete.csv").is_file():
        csv_path = Path("concrete.csv")
    source = csv_path if csv_path is not None else DATA_URL
    try:
        df = pd.read_csv(source, sep=None, engine="python")
    except OSError as exc:
        raise SystemExit(f"Không đọc được {source}. Hãy dùng --data concrete.csv.") from exc
    if df.shape[1] != len(COLUMNS):
        raise ValueError("CSV cần có tiêu đề, 8 cột đầu vào và cường độ nén ở cột cuối.")
    df.columns = COLUMNS
    df = df.apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    print(f"Dữ liệu {source} | {len(df)} dòng | thiếu/không hữu hạn: "
          f"{df.isna().sum().sum()} ô | trùng: {df.duplicated().sum()} dòng")
    df = df.dropna().drop_duplicates()
    return df.iloc[:, :-1].to_numpy(dtype=np.float64), df[["strength"]].to_numpy(dtype=np.float64)


def split_data(x, y, seed):
    """Chia 70/15/15 và chỉ fit scaler trên tập Train."""
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(x, y, test_size=0.30, random_state=seed)
    x_va, x_te, y_va, y_te = train_test_split(x_tmp, y_tmp, test_size=0.50,
                                                 random_state=seed)
    x_scaler, y_scaler = StandardScaler().fit(x_tr), StandardScaler().fit(y_tr)
    print(f"\nSeed = {seed} | Train={len(x_tr)} Val={len(x_va)} Test={len(x_te)}")
    splits = {
        name: (x_scaler.transform(x_i), y_scaler.transform(y_i), y_i)
        for name, x_i, y_i in (("Train", x_tr, y_tr), ("Val", x_va, y_va), ("Test", x_te, y_te))
    }
    return splits, y_scaler


def plot_results(runs, scores, test_pairs, out):
    """Lưu ba hình PNG bằng Matplotlib, dùng lại dự đoán Test đã tính."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for ax, (name, (_, curve, best, _, early)) in zip(axes.flat, runs.items()):
        epochs = np.arange(1, len(curve) + 1)
        ax.plot(epochs, curve[:, 0], label="Train")
        ax.plot(epochs, curve[:, 1], label="Validation")
        if early:
            ax.axvline(best, ls="--", color="gray", label=f"Epoch tốt nhất = {best}")
        ax.set(title=name, xlabel="Epoch", ylabel="Nửa MSE (y chuẩn hóa)")
        ax.legend()
    fig.suptitle("Loss Train và Validation theo epoch")
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=160)
    plt.close(fig)

    test = scores.xs("Test", level="Tập")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = plt.get_cmap("tab10").colors
    for ax, metric, unit in zip(axes, ("MSE", "RMSE", "MAE"), ("MPa²", "MPa", "MPa")):
        for i, (name, value) in enumerate(test[metric].items()):
            ax.bar(i, value, color=colors[i], label=name)
        ax.set(title=f"Test {metric}", xlabel="Cấu hình", ylabel=unit, xticks=[])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out / "test_metrics.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    lo = min(min(a.min(), p.min()) for a, p in test_pairs.values())
    hi = max(max(a.max(), p.max()) for a, p in test_pairs.values())
    for ax, (name, (actual, pred)) in zip(axes.flat, test_pairs.items()):
        ax.scatter(actual.ravel(), pred.ravel(), s=13, alpha=0.55, label="Mẫu Test")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="Dự đoán = thực tế")
        ax.set(title=name, xlabel="Thực tế (MPa)", ylabel="Dự đoán (MPa)")
        ax.legend()
    fig.suptitle("Cường độ nén dự đoán và thực tế trên Test")
    fig.tight_layout()
    fig.savefig(out / "test_scatter.png", dpi=160)
    plt.close(fig)


def experiment(splits, y_scaler, seed, output=None):
    """Chọn λ trên Val và đánh giá bốn cấu hình với một seed."""
    np.random.seed(seed)
    fit_data = (*splits["Train"][:2], *splits["Val"][:2])

    def fit(l2, early):
        # Khởi tạo lại cùng seed, kể cả RNG shuffle, cho mọi lượt huấn luyện.
        model = FNN(fit_data[0].shape[1], HIDDEN_SIZES, seed)
        return model, *model.fit(*fit_data, l2=l2, early_stop=early, seed=seed), early

    # Chọn λ bằng Val của mô hình cuối sau EPOCHS cố định, rồi dùng chung λ.
    candidates = {}
    for l2 in L2_CANDIDATES:
        candidates[l2] = fit(l2, False)
        print(f"Chọn λ | {l2:g} | nửa MSE Val cuối = {candidates[l2][1][-1, 1]:.6f}")
    chosen = min(candidates, key=lambda l2: candidates[l2][1][-1, 1])
    print(f"λ được chọn từ Val = {chosen:g}")
    runs = {
        "FNN": fit(0.0, False),
        "FNN + ES": fit(0.0, True),
        "FNN + L2": candidates[chosen],
        "FNN + ES + L2": fit(chosen, True),
    }

    rows, test_pairs = [], {}
    for name, (model, curve, best, stop, early) in runs.items():
        used = best if early else stop
        train_loss, val_loss = curve[used - 1]
        gap = val_loss - train_loss
        print(f"{name} | tốt nhất={best}, dừng={stop}, dùng={used} | "
              f"Val−Train tại epoch dùng={gap:+.6f}")
        # Một lần dự đoán cho mỗi tập. Dùng lại dự đoán Test cho hình scatter.
        for split, (x, _, actual) in splits.items():
            pred = y_scaler.inverse_transform(model.forward(x)[0])
            mse = mean_squared_error(actual, pred)
            rows.append((name, split, mse, np.sqrt(mse), mean_absolute_error(actual, pred)))
            if split == "Test":
                test_pairs[name] = actual, pred

    scores = pd.DataFrame(rows, columns=["Cấu hình", "Tập", "MSE", "RMSE", "MAE"])
    scores = scores.set_index(["Cấu hình", "Tập"])
    print("\nMetric trên thang gốc | MSE (MPa²), RMSE và MAE (MPa)")
    print(scores.to_string(float_format=lambda value: f"{value:.4f}"))

    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        plot_results(runs, scores, test_pairs, output)
        print(f"\nĐã lưu ba biểu đồ PNG vào {output}")
    print("Val−Train dương lớn, hoặc Val tăng khi Train giảm, là dấu hiệu overfitting.")
    for left, right, technique in (
        ("FNN", "FNN + ES", "ES"),
        ("FNN", "FNN + L2", "L2"),
        ("FNN + L2", "FNN + ES + L2", "ES khi có L2"),
        ("FNN + ES", "FNN + ES + L2", "L2 khi có ES"),
    ):
        delta = scores.loc[(right, "Test"), "RMSE"] - scores.loc[(left, "Test"), "RMSE"]
        print(f"{technique} | {right} so với {left} | Δ RMSE Test = {delta:+.4f} MPa")
    print("Δ âm là sai số Test giảm trong lần chia này. Dùng Val để chọn siêu tham số.")
    return scores


def main():
    """Chạy gradient check và thí nghiệm một seed hoặc năm seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="CSV cục bộ của bộ Concrete UCI")
    parser.add_argument("--output", type=Path, default=Path("fnn_results"))
    parser.add_argument("--five-seeds", action="store_true", help="Báo cáo thêm thống kê 5 seed")
    args = parser.parse_args()
    x, y = load_data(args.data)
    splits, y_scaler = split_data(x, y, SEED)
    gradient_check(*splits["Train"][:2], SEED)
    scores = experiment(splits, y_scaler, SEED, args.output)
    if args.five_seeds:
        tables = [scores] + [experiment(*split_data(x, y, seed), seed) for seed in EXTRA_SEEDS]
        grouped = pd.concat(tables).groupby(level=["Cấu hình", "Tập"], sort=False)
        mean, std = grouped.mean(), grouped.std()
        summary = pd.DataFrame({
            metric: [f"{a:.4f} ± {b:.4f}" for a, b in zip(mean[metric], std[metric])]
            for metric in mean.columns
        }, index=mean.index)
        print("\n5 seed | Trung bình ± độ lệch chuẩn mẫu trên thang MPa/MPa²")
        print(summary.to_string())


if __name__ == "__main__":
    main()
