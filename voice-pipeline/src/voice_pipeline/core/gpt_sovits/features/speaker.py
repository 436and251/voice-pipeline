# Copyright 3D-Speaker contributors.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.compliance import kaldi
from torchaudio.transforms import Resample


class _ReLU(nn.Hardtanh):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(0, 20, inplace)


class _AFF(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        inter_channels = channels // 4
        self.local_att = nn.Sequential(
            nn.Conv2d(channels * 2, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor, downsampled: torch.Tensor) -> torch.Tensor:
        attention = 1.0 + torch.tanh(self.local_att(torch.cat((x, downsampled), dim=1)))
        return x * attention + downsampled * (2.0 - attention)


class _BasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        *,
        fused: bool,
        base_width: int = 24,
        scale: int = 4,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        width = math.floor(planes * base_width / 64.0)
        self.conv1 = nn.Conv2d(in_planes, width * scale, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.convs = nn.ModuleList(
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False) for _ in range(scale)
        )
        self.bns = nn.ModuleList(nn.BatchNorm2d(width) for _ in range(scale))
        self.fuse_models = nn.ModuleList(_AFF(width) for _ in range(scale - 1)) if fused else nn.ModuleList()
        self.relu = _ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width * scale, planes * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(expansion * planes),
            )
        self.width = width
        self.scale = scale
        self.fused = fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        split = torch.split(self.relu(self.bn1(self.conv1(x))), self.width, dim=1)
        outputs = []
        current = split[0]
        for index in range(self.scale):
            if index:
                current = self.fuse_models[index - 1](current, split[index]) if self.fused else current + split[index]
            current = self.relu(self.bns[index](self.convs[index](current)))
            outputs.append(current)
        result = self.bn3(self.conv3(torch.cat(outputs, dim=1)))
        return self.relu(result + residual)


class _ERes2NetV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 3, stride=1, fused=False)
        self.layer2 = self._make_layer(128, 4, stride=2, fused=False)
        self.layer3 = self._make_layer(256, 6, stride=2, fused=True)
        self.layer4 = self._make_layer(512, 3, stride=2, fused=True)
        self.layer3_ds = nn.Conv2d(1024, 2048, kernel_size=3, padding=1, stride=2, bias=False)
        self.fuse34 = _AFF(2048)
        self.seg_1 = nn.Linear(40_960, 192)

    def _make_layer(self, planes: int, blocks: int, *, stride: int, fused: bool) -> nn.Sequential:
        layers = []
        for block_index in range(blocks):
            layers.append(
                _BasicBlock(
                    self.in_planes,
                    planes,
                    stride if block_index == 0 else 1,
                    fused=fused,
                )
            )
            self.in_planes = planes * 4
        return nn.Sequential(*layers)

    def forward3(self, features: torch.Tensor) -> torch.Tensor:
        output = F.relu(self.bn1(self.conv1(features.permute(0, 2, 1).unsqueeze(1))))
        output1 = self.layer1(output)
        output2 = self.layer2(output1)
        output3 = self.layer3(output2)
        output4 = self.layer4(output3)
        fused = self.fuse34(output4, self.layer3_ds(output3))
        return fused.flatten(start_dim=1, end_dim=2).mean(-1)


class SpeakerEncoder:
    def __init__(self, checkpoint_path: str | Path, device: str | torch.device = "cpu") -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        self.device = torch.device(device)
        self.model = _ERes2NetV2()
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)
        self.resample = Resample(32_000, 16_000).to(self.device)

    def extract(self, wav_32k: torch.Tensor) -> torch.Tensor:
        waveform = wav_32k.to(device=self.device, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        with torch.inference_mode():
            waveform = self.resample(waveform)
            features = torch.stack(
                [
                    kaldi.fbank(
                        item.unsqueeze(0),
                        num_mel_bins=80,
                        sample_frequency=16_000,
                        dither=0,
                    )
                    for item in waveform
                ]
            )
            return self.model.forward3(features)
