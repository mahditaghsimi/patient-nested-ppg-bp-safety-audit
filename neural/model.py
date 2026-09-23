from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """Architecture-compatible copy of the forecasting TCN residual block."""

    def __init__(self, channels: int = 64, dilation: int = 1, dropout: float = 0.15) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 5, padding=dilation * 2, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        z = self.dropout(F.gelu(self.norm2(self.conv2(z))))
        return x + z


class BPTransferTCN(nn.Module):
    """Forecast-pretrained morphology encoder with optional mathematical feature fusion."""

    def __init__(self, engineered_dim: int = 0, dropout: float = 0.15) -> None:
        super().__init__()
        hidden = 64
        self.engineered_dim = int(engineered_dim)
        self.stem = nn.Sequential(
            nn.Conv1d(3, hidden, 9, stride=5, padding=4),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(hidden, 1, dropout),
            ResidualBlock(hidden, 2, dropout),
            ResidualBlock(hidden, 4, dropout),
            ResidualBlock(hidden, 8, dropout),
        )
        # Last, mean, standard deviation and maximum preserve complementary morphology.
        representation_dim = hidden * 4 + self.engineered_dim
        self.bp_head = nn.Sequential(
            nn.Linear(representation_dim, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 2),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.blocks(self.stem(x))
        return torch.cat(
            [z[:, :, -1], z.mean(-1), z.std(-1, unbiased=False), z.amax(-1)], dim=1
        )

    def forward(self, x: torch.Tensor, engineered: torch.Tensor | None = None) -> torch.Tensor:
        representation = self.encode(x)
        if self.engineered_dim:
            if engineered is None:
                raise ValueError("engineered features are required for the hybrid model")
            representation = torch.cat([representation, engineered], dim=1)
        return self.bp_head(representation)

    def load_forecasting_encoder(self, checkpoint: dict) -> None:
        source = checkpoint["model_state"]
        compatible = {k: v for k, v in source.items() if k.startswith(("stem.", "blocks."))}
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        unexpected_encoder = [k for k in unexpected if k.startswith(("stem.", "blocks."))]
        missing_encoder = [k for k in missing if k.startswith(("stem.", "blocks."))]
        if unexpected_encoder or missing_encoder:
            raise RuntimeError(
                f"Forecast encoder is incompatible: missing={missing_encoder}, unexpected={unexpected_encoder}"
            )

    def freeze_encoder(self) -> None:
        for module in (self.stem, self.blocks):
            for parameter in module.parameters():
                parameter.requires_grad = False

