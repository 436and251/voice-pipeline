from __future__ import annotations

from pathlib import Path

import torch


class CNHubertExtractor:
    def __init__(
        self,
        model_path: str | Path,
        device: str | torch.device = "cpu",
        precision: str = "fp32",
    ) -> None:
        from transformers import HubertModel, Wav2Vec2FeatureExtractor

        if precision not in {"fp16", "fp32"}:
            raise ValueError(f"unsupported HuBERT precision: {precision}")
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(path)
        self.device = torch.device(device)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(path, local_files_only=True)
        self.model = HubertModel.from_pretrained(path, local_files_only=True).eval().to(self.device)
        self.precision = precision
        if precision == "fp16":
            self.model.half()

    def extract(self, wav_16k: torch.Tensor) -> torch.Tensor:
        waveform = wav_16k.detach().float().cpu().numpy()
        inputs = self.feature_extractor(waveform, return_tensors="pt", sampling_rate=16_000).input_values
        model_dtype = next(self.model.parameters()).dtype
        with torch.inference_mode():
            hidden = self.model(inputs.to(device=self.device, dtype=model_dtype)).last_hidden_state
        return hidden.transpose(1, 2)

    def to_float(self) -> None:
        self.model.float()
        self.precision = "fp32"
