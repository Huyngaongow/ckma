"""Chia dữ liệu cố định và tạo loader cho từng cấu hình."""
import gzip
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST

MIRROR = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
TRAIN_SIZE = 5_000
VAL_SIZE = 5_000
EVAL_BATCH_SIZE = 256
ROTATION = 10
TRANSLATION = (0.08, 0.08)


class FashionData(FashionMNIST):
    """Thêm mirror HTTPS và giữ cách đọc tệp cục bộ của torchvision."""

    mirrors = [MIRROR, *FashionMNIST.mirrors]

    @property
    def raw_folder(self):
        return str(Path(self.root) / "FashionMNIST" / "raw")

    @property
    def processed_folder(self):
        return str(Path(self.root) / "FashionMNIST" / "processed")


class ImageSubset(Dataset):
    """Đọc ảnh theo chỉ mục và áp dụng transform riêng cho mỗi tập."""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image, label = self.dataset[int(self.indices[position])]
        return self.transform(image), label


@dataclass
class Data:
    """Các tập dữ liệu và thống kê dùng chung giữa các thí nghiệm."""

    source: FashionData
    test_source: FashionData
    train_indices: np.ndarray
    val_indices: np.ndarray
    mean: float
    std: float
    train_clean: TensorDataset
    validation: TensorDataset

    def transform(self, augmentation=False):
        """Áp dụng augmentation ngẫu nhiên trước khi chuẩn hóa."""
        steps = []
        if augmentation:
            steps.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=ROTATION, translate=TRANSLATION),
            ])
        steps.extend([
            transforms.ToTensor(),
            transforms.Normalize((self.mean,), (self.std,)),
        ])
        return transforms.Compose(steps)

    def loaders(self, batch_size, augmentation, seed, pin_memory):
        """Tách loader cập nhật trọng số khỏi loader đo Train sạch."""
        training = self.train_clean
        if augmentation:
            training = ImageSubset(self.source, self.train_indices, self.transform(augmentation=True))
        train_loader = DataLoader(
            training, batch_size=batch_size, shuffle=True, pin_memory=pin_memory,
            generator=torch.Generator().manual_seed(seed),
        )
        return (
            train_loader,
            self.eval_loader(self.train_clean, pin_memory),
            self.eval_loader(self.validation, pin_memory),
        )

    @staticmethod
    def eval_loader(dataset, pin_memory):
        # DataLoader dùng RNG khi tạo iterator, kể cả với num_workers=0.
        # Tách RNG này để việc đo Train không đổi chuỗi Dropout và augmentation.
        return DataLoader(
            dataset, batch_size=EVAL_BATCH_SIZE, pin_memory=pin_memory,
            generator=torch.Generator().manual_seed(0),
        )

    def test_loader(self, pin_memory):
        pixels = self.test_source.data.to(torch.float32).div_(255)
        test = self.clean_dataset(pixels, self.test_source.targets, self.mean, self.std)
        return self.eval_loader(test, pin_memory)

    @staticmethod
    def clean_dataset(pixels, targets, mean, std):
        """Chuẩn hóa ảnh sạch một lần, rồi tái sử dụng qua các epoch."""
        return TensorDataset(pixels.sub_(mean).div_(std).unsqueeze(1), targets)


def load_data(root, seed, offline=False, train_size=TRAIN_SIZE, val_size=VAL_SIZE):
    """Chia Train/Validation theo tỷ lệ lớp và giữ nguyên Test chính thức."""
    raw = Path(root) / "FashionMNIST" / "raw"
    try:
        extract_local_archives(raw)
        source = FashionData(root, train=True, download=not offline)
        test = FashionData(root, train=False, download=not offline)
    except (RuntimeError, OSError) as error:
        raise RuntimeError(
            f"Không đọc được Fashion-MNIST tại {root}. "
            f"Đặt bốn tệp dữ liệu vào {raw} theo README, rồi chạy với --offline."
        ) from error

    labels = source.targets.numpy()
    train_indices, remaining = train_test_split(
        np.arange(len(source)), train_size=train_size, stratify=labels, random_state=seed,
    )
    val_indices, _ = train_test_split(
        remaining, train_size=val_size, stratify=labels[remaining], random_state=seed,
    )
    # Chỉ lấy pixel của Train; ToTensor cũng chia giá trị pixel cho 255.
    pixels = source.data[train_indices].to(torch.float32).div_(255)
    mean = pixels.mean().item()
    std = pixels.std(unbiased=False).item()
    if std == 0:
        raise ValueError("Tập Train có độ lệch chuẩn pixel bằng 0.")
    validation_pixels = source.data[val_indices].to(torch.float32).div_(255)
    train = Data.clean_dataset(pixels, source.targets[train_indices], mean, std)
    val = Data.clean_dataset(validation_pixels, source.targets[val_indices], mean, std)
    return Data(source, test, train_indices, val_indices, mean, std, train, val)


def extract_local_archives(raw):
    """Giải nén các tệp IDX .gz khi người dùng đã chép về máy."""
    for archive_name, _ in FashionMNIST.resources:
        archive = raw / archive_name
        target = archive.with_suffix("")
        if archive.is_file() and not target.exists():
            with TemporaryDirectory(dir=raw) as folder:
                temporary = Path(folder) / target.name
                with gzip.open(archive, "rb") as source, temporary.open("wb") as out:
                    shutil.copyfileobj(source, out)
                temporary.replace(target)
