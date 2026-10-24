"""MRCCFrormer for raw I/Q modulation classification."""

from __future__ import annotations
import torch
from torch import nn
from torch.nn.init import trunc_normal_


def _init_conv_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv1d):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm1d):
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
        if module.weight is not None:
            nn.init.constant_(module.weight, 1.0)


def stemIQ(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(
            in_channels,
            out_channels // 2,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=in_channels,
        ),
        nn.BatchNorm1d(out_channels // 2),
    )


class SELayer(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.avg_pool(x).flatten(1)).unsqueeze(-1)
        return x * scale


class ConvEncoder(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.norm = nn.BatchNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(hidden_dim, dim, 1)
        self.layer_scale = nn.Parameter(torch.ones(dim, 1))
        self.apply(_init_conv_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(x)))))
        return x + self.layer_scale * residual


class FCN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim)
        self.fc1 = nn.Conv1d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv1d(hidden_dim, dim, 1)
        self.drop = nn.Dropout(drop)
        self.apply(_init_conv_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(self.norm1(x))))))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class LocalRepresentation(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.norm = nn.BatchNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(dim, dim, 1)
        self.layer_scale = nn.Parameter(torch.ones(dim, 1))
        self.apply(_init_conv_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(x)))))
        return x + self.layer_scale * residual


class MultiReceptiveWaveformExtractor(nn.Module):
    """Three depthwise temporal paths with learned group-wise fusion."""

    def __init__(self, channels=8):
        super().__init__()
        self.paths = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel,
                        padding=(kernel // 2) * dilation,
                        dilation=dilation,
                        groups=channels,
                        bias=False,
                    ),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                )
                for kernel, dilation in ((5, 1), (15, 2), (31, 4))
            ]
        )
        self.weight_estimator = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, 8, 1),
            nn.GELU(),
            nn.Conv1d(8, 12, 1),
        )
        self.project = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x):
        batch, channels, length = x.shape
        candidates = torch.stack([path(x) for path in self.paths], dim=1)
        weights = self.weight_estimator(x).view(batch, 4, 3, 1, 1).softmax(dim=2)
        grouped = candidates.view(batch, 3, 4, channels // 4, length).permute(
            0, 2, 1, 3, 4
        )
        fused = (grouped * weights).sum(dim=2).reshape_as(x)
        return self.project(fused)


class FeatureEnhancementEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = MultiReceptiveWaveformExtractor(8)
        self.se = SELayer(8)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        return x + self.drop(self.se(self.conv(x)))


class CumulantAttention(nn.Module):
    """Context aggregation through centered key-value covariance."""

    def __init__(self, dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

    def forward(self, x):
        batch, length, channels = x.shape
        head_dim = channels // 4
        q, k, value = (
            self.qkv(self.norm(x))
            .reshape(batch, length, 3, 4, head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        q = q.float() * head_dim**-0.5
        k, value = k.float(), value.float()
        k_centered = k - k.mean(dim=-2, keepdim=True)
        value_mean = value.mean(dim=-2, keepdim=True)
        value_centered = value - value_mean
        covariance = (k_centered.transpose(-2, -1) @ value_centered) / float(length)
        output = value_mean + q @ covariance
        return output.to(x.dtype).transpose(1, 2).reshape(batch, length, channels)


class ContextBlock(nn.Module):
    def __init__(self, signal_length):
        super().__init__()
        self.input_resolution = signal_length
        self.cpe1 = nn.Conv1d(64, 64, 3, padding=1, groups=64)
        self.norm1 = nn.LayerNorm(64)
        self.in_proj = nn.Linear(64, 64)
        self.act_proj = nn.Linear(64, 64)
        self.dwc = nn.Conv1d(64, 64, 3, padding=1, groups=64)
        self.act = nn.SiLU()
        self.attn = CumulantAttention()
        self.out_proj = nn.Linear(64, 64)
        self.cpe2 = nn.Conv1d(64, 64, 3, padding=1, groups=64)
        self.norm2 = nn.LayerNorm(64)
        self.mlp = Mlp(64, 256, 0.0)

    def forward(self, x):
        x = x + self.cpe1(x.transpose(1, 2)).transpose(1, 2)
        shortcut = x
        normalized = self.norm1(x)
        gate = self.act(self.act_proj(normalized))
        context = self.act(
            self.dwc(self.in_proj(normalized).transpose(1, 2))
        ).transpose(1, 2)
        x = shortcut + self.out_proj(self.attn(context) * gate)
        x = x + self.cpe2(x.transpose(1, 2)).transpose(1, 2)
        return x + self.mlp(self.norm2(x))


class RadioFormerEncoder(nn.Module):
    def __init__(self, signal_length):
        super().__init__()
        self.local_representation = LocalRepresentation(64)
        self.attn = ContextBlock(signal_length)
        self.linear = FCN(64, 256, 0.0)

    def forward(self, x):
        x = self.local_representation(x)
        x = x + self.attn(x.transpose(1, 2)).transpose(1, 2)
        return x + self.linear(x)


class Stage(nn.Module):
    def __init__(self, depth, signal_length):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                *[ConvEncoder(64, 256) for _ in range(depth - 1)],
                RadioFormerEncoder(signal_length),
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class MRPureCumFormer(nn.Module):
    def __init__(self, num_classes=11, signal_length=128):
        super().__init__()
        if signal_length not in (128, 1024) or num_classes < 2:
            raise ValueError(
                "Expected signal_length 128 or 1024 and at least two classes"
            )
        self.signal_length = signal_length
        depths = (2, 3, 2) if signal_length == 128 else (3, 3, 3)
        self.BN = nn.BatchNorm1d(2)
        self.patch_embedIQ = stemIQ(2, 16)
        self.fee = FeatureEnhancementEmbedding()
        self.patch_LSTM = nn.LSTM(
            input_size=8,
            hidden_size=32,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,
        )
        self.network = nn.ModuleList([Stage(depth, signal_length) for depth in depths])
        self.norm = nn.BatchNorm1d(64)
        self.globalavgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(64, num_classes)
        self.apply(_init_conv_weights)

    def forward(self, iq):
        if iq.ndim != 3 or iq.shape[1:] != (2, self.signal_length):
            raise ValueError(
                f"Expected [B, 2, {self.signal_length}], got {tuple(iq.shape)}"
            )
        x = self.fee(self.patch_embedIQ(self.BN(iq)))
        x, _ = self.patch_LSTM(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for stage in self.network:
            x = stage(x)
        return self.head(self.globalavgpool(self.norm(x)).flatten(1))
