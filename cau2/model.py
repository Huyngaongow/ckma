"""CNN nhỏ cho Fashion-MNIST."""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    """Siêu tham số của một lượt huấn luyện."""

    learning_rate: float = 0.001
    dropout_rate: float = 0.4
    filters: int = 24
    batch_size: int = 64
    dropout: bool = False
    augmentation: bool = False


class CNN(nn.Module):
    """Nhận tensor [B, 1, 28, 28] và trả về 10 logit mỗi ảnh."""

    def __init__(self, config: Config):
        super().__init__()
        f = config.filters
        self.layers = nn.Sequential(
            nn.Conv2d(1, f, kernel_size=3, padding=1),  # [B, f, 28, 28]
            nn.ReLU(),                               # [B, f, 28, 28]
            nn.MaxPool2d(2),                          # [B, f, 14, 14]
            nn.Conv2d(f, 2 * f, kernel_size=3, padding=1),  # [B, 2f, 14, 14]
            nn.ReLU(),                               # [B, 2f, 14, 14]
            nn.MaxPool2d(2),                          # [B, 2f, 7, 7]
            nn.Flatten(),                            # [B, 98f]
            nn.Linear(98 * f, 128),                    # [B, 128]
            nn.ReLU(),                               # [B, 128]
            # Dropout hoặc Identity giữ nguyên [B, 128].
            nn.Dropout(config.dropout_rate) if config.dropout else nn.Identity(),
            nn.Linear(128, 10),                        # [B, 10]
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)
