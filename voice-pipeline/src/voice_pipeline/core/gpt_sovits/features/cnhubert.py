from __future__ import annotations

from pathlib import Path

import torch


class CNHubertExtractor:
    def __init__(self, model_path: str | Path, device: str | torch.device = "cpu") -> None:
        from transformers import HubertModel, Wav2Vec2FeatureExtractor

        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(path)
        self.device = torch.device(device)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(path, local_files_only=True)
        self.model = HubertModel.from_pretrained(path, local_files_only=True).eval().to(self.device)

    def extract(self, wav_16k: torch.Tensor) -> torch.Tensor:
        waveform = wav_16k.detach().float().cpu().numpy()
        inputs = self.feature_extractor(waveform, return_tensors="pt", sampling_rate=16_000).input_values
        with torch.inference_mode():
            hidden = self.model(inputs.to(self.device)).last_hidden_state
        return hidden.transpose(1, 2)
