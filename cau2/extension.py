"""Random Search trên Validation với ngân sách cố định."""
from itertools import product
from random import Random

import pandas as pd

from model import Config
from train import fit

TRIALS = 8
LEARNING_RATES = (0.0005, 0.001, 0.002)
DROPOUT_RATES = (0.2, 0.4, 0.55)
FILTERS = (16, 24, 32)
BATCH_SIZES = (64, 96, 128)


def random_search(data, device, epochs, seed, deterministic, trials):
    """Lấy mẫu cấu hình không lặp và chọn theo Validation ở epoch cuối."""
    space = list(product(LEARNING_RATES, DROPOUT_RATES, FILTERS, BATCH_SIZES))
    candidates = Random(seed).sample(space, trials)
    records, configs = [], {}
    for trial, (lr, rate, filters, batch_size) in enumerate(candidates, 1):
        config = Config(lr, rate, filters, batch_size, dropout=True, augmentation=True)
        _, history = fit(
            config, data, device, epochs, seed, deterministic, f"Random Search {trial}/{trials}",
            track_train=False,
        )
        configs[trial] = config
        final = history[-1]
        records.append({
            "trial": trial, "learning_rate": lr, "dropout_rate": rate,
            "filters": filters, "batch_size": batch_size, "epochs": epochs,
            "val_accuracy": final["val_accuracy"], "val_loss": final["val_loss"],
        })
    # Ưu tiên accuracy cao, rồi loss thấp và số thứ tự lượt thử nhỏ.
    ranking = pd.DataFrame(records).sort_values(
        ["val_accuracy", "val_loss", "trial"], ascending=[False, True, True],
    )
    winning_trial = int(ranking.iloc[0]["trial"])
    ranking["selected"] = ranking["trial"] == winning_trial
    return configs[winning_trial], ranking
