from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def portable_papagei_weights(value: str | Path) -> Path:
    """Resolve archived absolute checkpoint paths inside a cloned repository."""
    requested = Path(value).expanduser()
    if requested.exists():
        return requested
    bundled = PROJECT_ROOT / "data/pretraining" / requested.name
    if bundled.exists():
        return bundled
    raise FileNotFoundError(
        f"PaPaGei checkpoint not found at {requested} or bundled location {bundled}"
    )


class InputView(nn.Module):
    """Deterministic signal-view block shared by all backbones."""

    def __init__(self, name: str, duration_seconds: float, sampling_rate: int = 125) -> None:
        super().__init__()
        self.name = name
        self.samples = int(round(duration_seconds * sampling_rate))
        self.out_channels = {
            "ppg": 1,
            "ppg_zscore": 1,
            "ppg_bandpass": 1,
            "derivatives": 3,
            "derivatives_zscore": 3,
            "multiview": 5,
            "multiview_zscore": 5,
            "multiscale7": 7,
        }[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.samples > x.shape[-1]:
            raise ValueError(f"Requested {self.samples} samples from an input of {x.shape[-1]}")
        start = (x.shape[-1] - self.samples) // 2
        x = x[..., start : start + self.samples]
        ppg = x[:, :1]
        if self.name == "ppg":
            return ppg
        if self.name == "ppg_zscore":
            return (ppg - ppg.mean(-1, keepdim=True)) / ppg.std(-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        if self.name == "ppg_bandpass":
            smooth = F.avg_pool1d(ppg, 5, stride=1, padding=2)
            baseline = F.avg_pool1d(ppg, 125, stride=1, padding=62)
            return smooth - baseline
        if self.name == "derivatives":
            return x[:, :3]
        if self.name == "derivatives_zscore":
            z = x[:, :3]
            return (z - z.mean(-1, keepdim=True)) / z.std(-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        smooth = F.avg_pool1d(ppg, 5, stride=1, padding=2)
        baseline = F.avg_pool1d(ppg, 125, stride=1, padding=62)
        multiview = torch.cat([x[:, :3], smooth, smooth - baseline], dim=1)
        if self.name == "multiview":
            return multiview
        if self.name == "multiview_zscore":
            return (multiview - multiview.mean(-1, keepdim=True)) / multiview.std(-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        smooth15 = F.avg_pool1d(ppg, 15, stride=1, padding=7)
        baseline250 = F.avg_pool1d(ppg, 251, stride=1, padding=125)
        multiscale = torch.cat([ppg, x[:, 1:3], smooth, smooth15, smooth - baseline, smooth15 - baseline250], dim=1)
        return (multiscale - multiscale.mean(-1, keepdim=True)) / multiscale.std(-1, keepdim=True, unbiased=False).clamp_min(1e-4)


class ConvNormAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel: int, stride: int = 1, dilation: int = 1):
        padding = dilation * (kernel // 2)
        super().__init__(
            nn.Conv1d(cin, cout, kernel, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(min(8, cout), cout),
            nn.GELU(),
        )


class StatsPool(nn.Module):
    multiplier = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(-1), x.std(-1, unbiased=False), x.amax(-1)], dim=1)


class FCNBackbone(nn.Module):
    out_dim = 128 * StatsPool.multiplier

    def __init__(self, cin: int):
        super().__init__()
        self.net = nn.Sequential(ConvNormAct(cin, 64, 9, 2), ConvNormAct(64, 128, 7, 2), ConvNormAct(128, 128, 5, 2))
        self.pool = StatsPool()

    def forward(self, x):
        return self.pool(self.net(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, stride: int = 1):
        super().__init__()
        self.a = ConvNormAct(channels, channels, 5, stride, dilation)
        self.b = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, channels),
        )
        # ConvNormAct uses odd-kernel "same" padding, which yields ceil(L/stride).
        self.skip = nn.AvgPool1d(3, stride, 1) if stride > 1 else nn.Identity()

    def forward(self, x):
        return F.gelu(self.b(self.a(x)) + self.skip(x))


class ResNetBackbone(nn.Module):
    out_dim = 128 * StatsPool.multiplier

    def __init__(self, cin: int):
        super().__init__()
        self.stem = ConvNormAct(cin, 128, 11, 4)
        self.net = nn.Sequential(*[ResidualBlock(128, 2 ** (i % 4), 2 if i in (2, 5) else 1) for i in range(8)])
        self.pool = StatsPool()

    def forward(self, x):
        return self.pool(self.net(self.stem(x)))


class TCNBackbone(nn.Module):
    out_dim = 96 * StatsPool.multiplier

    def __init__(self, cin: int):
        super().__init__()
        self.stem = ConvNormAct(cin, 96, 9, 3)
        self.net = nn.Sequential(*[ResidualBlock(96, 2 ** i) for i in range(6)])
        self.pool = StatsPool()

    def forward(self, x):
        return self.pool(self.net(self.stem(x)))


class GatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels * 2, 5, padding=2 * dilation, dilation=dilation)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        a, b = self.conv(x).chunk(2, 1)
        return self.norm(x + self.proj(torch.tanh(a) * torch.sigmoid(b)))


class GatedTCNBackbone(nn.Module):
    out_dim = 96 * StatsPool.multiplier

    def __init__(self, cin: int):
        super().__init__()
        self.stem = ConvNormAct(cin, 96, 11, 4)
        self.net = nn.Sequential(*[GatedBlock(96, 2 ** i) for i in range(6)])
        self.pool = StatsPool()

    def forward(self, x):
        return self.pool(self.net(self.stem(x)))


class InceptionBlock(nn.Module):
    def __init__(self, cin: int, branch: int = 32):
        super().__init__()
        self.reduce = nn.Conv1d(cin, branch, 1)
        self.branches = nn.ModuleList([ConvNormAct(branch, branch, k) for k in (9, 19, 39)])
        self.pool = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(cin, branch, 1))
        self.norm = nn.GroupNorm(8, branch * 4)
        self.skip = nn.Conv1d(cin, branch * 4, 1) if cin != branch * 4 else nn.Identity()

    def forward(self, x):
        z = self.reduce(x)
        joined = torch.cat([b(z) for b in self.branches] + [self.pool(x)], 1)
        return F.gelu(self.norm(joined) + self.skip(x))


class InceptionBackbone(nn.Module):
    out_dim = 128 * StatsPool.multiplier

    def __init__(self, cin: int):
        super().__init__()
        self.down = ConvNormAct(cin, 64, 9, 4)
        self.net = nn.Sequential(InceptionBlock(64), InceptionBlock(128), InceptionBlock(128))
        self.pool = StatsPool()

    def forward(self, x):
        return self.pool(self.net(self.down(x)))


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.pw1 = nn.Conv1d(channels, channels * 4, 1)
        self.pw2 = nn.Conv1d(channels * 4, channels, 1)
        self.scale = nn.Parameter(torch.ones(channels) * 1e-4)

    def forward(self, x):
        z = self.pw2(F.gelu(self.pw1(self.norm(self.dw(x)))))
        return x + z * self.scale.view(1, -1, 1)


class ConvNeXtBackbone(nn.Module):
    out_dim = 128 * StatsPool.multiplier

    def __init__(self, cin: int):
        super().__init__()
        self.stem = ConvNormAct(cin, 128, 15, 5)
        self.net = nn.Sequential(*[ConvNeXtBlock(128) for _ in range(8)])
        self.pool = StatsPool()

    def forward(self, x):
        return self.pool(self.net(self.stem(x)))


class BiGRUBackbone(nn.Module):
    out_dim = 256

    def __init__(self, cin: int):
        super().__init__()
        self.front = nn.Sequential(ConvNormAct(cin, 64, 11, 5), ConvNormAct(64, 96, 7, 3))
        self.rnn = nn.GRU(96, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15)

    def forward(self, x):
        z = self.front(x).transpose(1, 2)
        z, _ = self.rnn(z)
        return z.mean(1)


class PatchTransformerBackbone(nn.Module):
    out_dim = 192 * 2

    def __init__(self, cin: int, patch: int = 25):
        super().__init__()
        self.embed = nn.Conv1d(cin, 192, patch, stride=patch // 2, padding=patch // 4)
        layer = nn.TransformerEncoderLayer(192, 6, 384, 0.1, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(192)

    def forward(self, x):
        z = self.embed(x).transpose(1, 2)
        positions = torch.arange(z.shape[1], device=z.device, dtype=z.dtype).unsqueeze(1)
        dims = torch.arange(z.shape[2], device=z.device, dtype=z.dtype).unsqueeze(0)
        z = z + 0.02 * torch.sin(positions / torch.pow(10000.0, (dims // 2 * 2) / z.shape[2]))
        z = self.norm(self.encoder(z))
        return torch.cat([z.mean(1), z.amax(1)], 1)


class MixerBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.token = nn.Sequential(nn.Conv1d(channels, channels, 9, padding=4, groups=channels), nn.GELU(), nn.Conv1d(channels, channels, 1))
        self.channel = nn.Sequential(nn.Conv1d(channels, channels * 4, 1), nn.GELU(), nn.Conv1d(channels * 4, channels, 1))
        self.n1 = nn.GroupNorm(1, channels)
        self.n2 = nn.GroupNorm(1, channels)

    def forward(self, x):
        x = x + self.token(self.n1(x))
        return x + self.channel(self.n2(x))


class PatchMixerBackbone(nn.Module):
    out_dim = 192 * StatsPool.multiplier

    def __init__(self, cin: int):
        super().__init__()
        self.embed = nn.Conv1d(cin, 192, 25, stride=12, padding=6)
        self.net = nn.Sequential(*[MixerBlock(192) for _ in range(6)])
        self.pool = StatsPool()

    def forward(self, x):
        return self.pool(self.net(self.embed(x)))


class MultiScaleBackbone(nn.Module):
    out_dim = 96 * 3 * StatsPool.multiplier

    def __init__(self, cin: int):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(ConvNormAct(cin, 64, k, 4), ConvNormAct(64, 96, max(3, k // 2), 2))
            for k in (7, 15, 31)
        ])
        self.pool = StatsPool()

    def forward(self, x):
        return torch.cat([self.pool(branch(x)) for branch in self.branches], 1)


class SpectralFusionBackbone(nn.Module):
    out_dim = 256 + 128

    def __init__(self, cin: int):
        super().__init__()
        self.temporal = nn.Sequential(ConvNormAct(cin, 96, 11, 4), ConvNormAct(96, 128, 7, 3))
        self.freq = nn.Sequential(nn.Conv1d(cin, 64, 7, 3, 3), nn.GELU(), nn.Conv1d(64, 128, 5, 2, 2), nn.GELU())

    def forward(self, x):
        t = self.temporal(x)
        t = torch.cat([t.mean(-1), t.amax(-1)], 1)
        f = torch.log1p(torch.abs(torch.fft.rfft(x, dim=-1)))
        f = self.freq(f).mean(-1)
        return torch.cat([t, f], 1)


class ECA1D(nn.Module):
    """Efficient channel attention with negligible parameter overhead."""

    def __init__(self, channels: int, kernel: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel, padding=kernel // 2, bias=False)

    def forward(self, x):
        weights = self.conv(x.mean(-1).unsqueeze(1)).squeeze(1).sigmoid().unsqueeze(-1)
        return x * weights


class SE1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels), nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.net(x.mean(-1)).unsqueeze(-1)


class AttentiveStatsPool(nn.Module):
    multiplier = 3

    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(channels, max(channels // 4, 16), 1),
            nn.Tanh(),
            nn.Conv1d(max(channels // 4, 16), 1, 1),
        )

    def forward(self, x):
        weight = torch.softmax(self.score(x), dim=-1)
        mean = (weight * x).sum(-1)
        variance = (weight * (x - mean.unsqueeze(-1)).square()).sum(-1).clamp_min(1e-6)
        return torch.cat([mean, variance.sqrt(), x.amax(-1)], dim=1)


class GeMStatsPool(nn.Module):
    multiplier = 3

    def __init__(self):
        super().__init__()
        self.raw_p = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        p = 1.0 + F.softplus(self.raw_p)
        gem = x.mean(-1).sign() * x.abs().clamp_min(1e-6).pow(p).mean(-1).pow(1.0 / p)
        return torch.cat([gem, x.std(-1, unbiased=False), x.amax(-1)], dim=1)


class AdvancedResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, attention: str, dropout: float):
        super().__init__()
        padding = dilation * (kernel // 2)
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=padding, dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=padding, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.attention = {"none": nn.Identity(), "eca": ECA1D(channels), "se": SE1D(channels)}[attention]
        self.dropout = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        z = self.conv1(x)
        z = F.gelu(self.norm1(z))
        z = self.norm2(self.conv2(self.dropout(z)))
        z = self.attention(z)
        return F.gelu(x + self.scale * z)


def make_pool(name: str, channels: int) -> nn.Module:
    if name == "stats":
        return StatsPool()
    if name == "attention":
        return AttentiveStatsPool(channels)
    if name == "gem":
        return GeMStatsPool()
    raise ValueError(name)


class AdaptiveResNetBackbone(nn.Module):
    """Configurable residual morphology encoder used by the 200-run search."""

    def __init__(self, cin: int, config: dict, morphology_stem: bool = False):
        super().__init__()
        width = int(config.get("model_width", 96))
        depth = int(config.get("model_depth", 6))
        kernel = int(config.get("model_kernel", 5))
        stride = int(config.get("stem_stride", 4))
        attention = config.get("channel_attention", "eca")
        block_dropout = float(config.get("block_dropout", 0.0))
        dilation_mode = config.get("dilation_mode", "cycle")
        if morphology_stem:
            self.stem = nn.Sequential(
                nn.Conv1d(cin, cin * 8, kernel, stride=stride, padding=kernel // 2, groups=cin, bias=False),
                nn.GroupNorm(min(8, cin * 8), cin * 8), nn.GELU(),
                nn.Conv1d(cin * 8, width, 1, bias=False), nn.GroupNorm(min(8, width), width), nn.GELU(),
            )
        else:
            self.stem = ConvNormAct(cin, width, int(config.get("stem_kernel", 11)), stride)
        if dilation_mode == "flat":
            dilations = [1] * depth
        elif dilation_mode == "wide":
            dilations = [1, 2, 4, 8, 16, 8, 4, 2, 1, 2][:depth]
        else:
            dilations = [2 ** (i % 4) for i in range(depth)]
        self.blocks = nn.Sequential(*[
            AdvancedResidualBlock(width, kernel, dilation, attention, block_dropout)
            for dilation in dilations
        ])
        self.down = nn.Sequential(
            ConvNormAct(width, width, 5, 2),
            ConvNormAct(width, width, 3, 2),
        )
        pool_name = config.get("pooling", "stats")
        self.pool = make_pool(pool_name, width)
        self.out_dim = width * self.pool.multiplier

    def forward(self, x):
        return self.pool(self.down(self.blocks(self.stem(x))))


class LongConvBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int):
        super().__init__()
        padding = dilation * (kernel // 2)
        self.norm = nn.GroupNorm(1, channels)
        self.filter = nn.Conv1d(channels, channels * 2, kernel, padding=padding, dilation=dilation, groups=channels)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        value, gate = self.filter(self.norm(x)).chunk(2, dim=1)
        return x + self.scale * self.proj(F.gelu(value) * torch.sigmoid(gate))


class LongConvBackbone(nn.Module):
    """Long-receptive-field gated convolution; a dependency-free state-space proxy."""

    def __init__(self, cin: int, config: dict):
        super().__init__()
        width = int(config.get("model_width", 96))
        depth = int(config.get("model_depth", 6))
        kernel = int(config.get("model_kernel", 15))
        self.stem = ConvNormAct(cin, width, 11, int(config.get("stem_stride", 4)))
        self.blocks = nn.Sequential(*[LongConvBlock(width, kernel, 2 ** (i % 4)) for i in range(depth)])
        self.down = ConvNormAct(width, width, 5, 2)
        self.pool = make_pool(config.get("pooling", "attention"), width)
        self.out_dim = width * self.pool.multiplier

    def forward(self, x):
        return self.pool(self.down(self.blocks(self.stem(x))))


class ConvTransformerLiteBackbone(nn.Module):
    def __init__(self, cin: int, config: dict):
        super().__init__()
        width = int(config.get("model_width", 96))
        heads = int(config.get("attention_heads", 4))
        depth = int(config.get("transformer_depth", 2))
        patch = int(config.get("patch_size", 15))
        self.front = nn.Sequential(
            ConvNormAct(cin, width, 11, 3),
            nn.Conv1d(width, width, patch, stride=max(patch // 2, 2), padding=patch // 2),
        )
        layer = nn.TransformerEncoderLayer(
            width, heads, width * 2, float(config.get("block_dropout", 0.05)),
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.out_dim = width * 3

    def forward(self, x):
        z = self.front(x).transpose(1, 2)
        z = self.norm(self.encoder(z)).transpose(1, 2)
        return torch.cat([z.mean(-1), z.std(-1, unbiased=False), z.amax(-1)], dim=1)


class PaPaGeiBackbone(nn.Module):
    out_dim = 512

    def __init__(self, weights: str, train_mode: str = "frozen"):
        super().__init__()
        weights = portable_papagei_weights(weights)
        repo = PROJECT_ROOT / "vendor/papagei"
        sys.path.insert(0, str(repo))
        from models.resnet import ResNet1DMoE

        self.model = ResNet1DMoE(1, 32, 3, 2, 1, 18, 512, n_experts=3)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
        if train_mode == "frozen":
            self.model.requires_grad_(False)
        elif train_mode == "last_blocks":
            self.model.requires_grad_(False)
            for name, parameter in self.model.named_parameters():
                if "basicblock_list.17" in name or "dense" in name or "experts" in name:
                    parameter.requires_grad = True
        elif train_mode != "full":
            raise ValueError(train_mode)
        self.train_mode = train_mode

    def train(self, mode: bool = True):
        super().train(mode)
        if self.train_mode == "frozen":
            self.model.eval()
        return self

    def forward(self, x):
        if x.shape[1] != 1:
            x = x[:, :1]
        output = self.model(x)
        return output[3]


BACKBONES = {
    "fcn": FCNBackbone,
    "resnet": ResNetBackbone,
    "tcn": TCNBackbone,
    "gated_tcn": GatedTCNBackbone,
    "inception": InceptionBackbone,
    "convnext": ConvNeXtBackbone,
    "bigru": BiGRUBackbone,
    "patch_transformer": PatchTransformerBackbone,
    "patch_mixer": PatchMixerBackbone,
    "multiscale": MultiScaleBackbone,
    "spectral_fusion": SpectralFusionBackbone,
}


class BPModel(nn.Module):
    def __init__(self, config: dict, engineered_dim: int = 0):
        super().__init__()
        self.config = config
        self.view = InputView(config["view"], config["duration_seconds"])
        if config["backbone"] == "papagei":
            self.backbone = PaPaGeiBackbone(config["papagei_weights"], config.get("foundation_mode", "frozen"))
        elif config["backbone"] == "adaptive_resnet":
            self.backbone = AdaptiveResNetBackbone(self.view.out_channels, config)
        elif config["backbone"] == "morphology_resnet":
            self.backbone = AdaptiveResNetBackbone(self.view.out_channels, config, morphology_stem=True)
        elif config["backbone"] == "longconv":
            self.backbone = LongConvBackbone(self.view.out_channels, config)
        elif config["backbone"] == "conv_transformer_lite":
            self.backbone = ConvTransformerLiteBackbone(self.view.out_channels, config)
        else:
            self.backbone = BACKBONES[config["backbone"]](self.view.out_channels)
        self.engineered_dim = engineered_dim
        hidden = int(config.get("head_hidden", 192))
        out_dim = {
            "direct": 2,
            "multitask": 4,
            "gaussian": 4,
            "quantile": 6,
            # Two regression outputs followed by four SBP and four DBP
            # clinical-range logits. These auxiliary logits are never used to
            # manufacture the reported continuous BP estimate.
            "clinical": 10,
            # Two regression outputs followed by cumulative (ordinal) logits
            # at four prespecified thresholds for each BP target.
            "ordinal": 10,
        }[config.get("head", "direct")]
        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_dim + engineered_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(float(config.get("dropout", 0.15))),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x, engineered=None):
        z = self.backbone(self.view(x))
        if self.engineered_dim:
            z = torch.cat([z, engineered], 1)
        return self.head(z)

    def point_prediction(self, output):
        head = self.config.get("head", "direct")
        if head in ("direct", "multitask", "gaussian", "clinical", "ordinal"):
            return output[:, :2]
        return output.view(-1, 2, 3)[:, :, 1]
