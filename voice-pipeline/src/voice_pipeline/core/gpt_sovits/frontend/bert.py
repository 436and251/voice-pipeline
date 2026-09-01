from __future__ import annotations

from pathlib import Path

import torch


def expand_character_features(features: torch.Tensor, word2ph: list[int]) -> torch.Tensor:
    if features.ndim != 2 or features.shape[0] != len(word2ph):
        raise ValueError("character features and word2ph lengths differ")
    return torch.cat(
        [features[index].repeat(count, 1) for index, count in enumerate(word2ph)],
        dim=0,
    ).transpose(0, 1)


class BertAligner:
    def __init__(self, model_path: str | Path, device: str | torch.device = "cpu") -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(path)
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForMaskedLM.from_pretrained(path, local_files_only=True).eval().to(self.device)

    def extract(self, text: str, word2ph: list[int]) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            output = self.model(**inputs, output_hidden_states=True)
        character_features = output.hidden_states[-3][0, 1:-1]
        return expand_character_features(character_features, word2ph)
