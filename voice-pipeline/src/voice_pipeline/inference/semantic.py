from __future__ import annotations

from dataclasses import dataclass

import torch

from .reference import ReferenceCondition


@dataclass(frozen=True, slots=True)
class SemanticResult:
    codes: torch.Tensor
    target_phones: torch.Tensor


def generate_semantic(
    text: str,
    language: str,
    *,
    frontend,
    s1,
    reference: ReferenceCondition,
    device: torch.device,
    dtype: torch.dtype,
    top_k: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    early_stop_num: int,
) -> SemanticResult:
    text = text.strip()
    if text[-1] not in "，。？！,.?!~:：—…":
        text += "." if language == "en" else "。"
    target = frontend.process(text, language)
    target_phones = torch.tensor(target.phone_ids, dtype=torch.long, device=device).unsqueeze(0)
    phones = target_phones
    bert = target.bert_features.to(device=device, dtype=dtype)
    prompt = None

    if reference.prompt_frontend is not None:
        source = reference.prompt_frontend
        prompt_phones = torch.tensor(source.phone_ids, dtype=torch.long, device=device).unsqueeze(0)
        phones = torch.cat((prompt_phones, target_phones), dim=1)
        bert = torch.cat((source.bert_features.to(device=device, dtype=dtype), bert), dim=1)
        prompt = reference.prompt_semantic.to(device=device, dtype=torch.long).unsqueeze(0)

    with torch.inference_mode():
        generated, generated_length = s1.infer_panel(
            phones,
            torch.tensor([phones.shape[1]], dtype=torch.long, device=device),
            prompt,
            bert.unsqueeze(0),
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            early_stop_num=early_stop_num,
            repetition_penalty=repetition_penalty,
        )
    if isinstance(generated_length, torch.Tensor):
        generated_length = int(generated_length.item())
    if generated_length == 0 and reference.prompt_frontend is None:
        generated_length = generated.shape[-1]
    if not isinstance(generated_length, int) or generated_length <= 0 or generated_length > generated.shape[-1]:
        raise RuntimeError("S1 returned an invalid semantic length")
    codes = generated[:, -generated_length:].unsqueeze(0)
    return SemanticResult(codes, target_phones)


__all__ = ["SemanticResult", "generate_semantic"]
