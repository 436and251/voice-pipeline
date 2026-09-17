from __future__ import annotations

from voice_pipeline import __version__


def _labels(zh_cn: str, en: str, ja: str) -> dict[str, str]:
    return {"zh_CN": zh_cn, "en": en, "ja": ja}


def build_descriptor() -> dict[str, object]:
    """Return the immutable, side-effect-free module handshake payload."""
    return {
        "protocol_version": 2,
        "module_id": "gpt-sovits-v2proplus",
        "module_version": __version__,
        "frameworks": [
            {
                "id": "v2ProPlus",
                "display_name": "GPT-SoVITS v2ProPlus",
                "capabilities": [
                    "preprocess",
                    "train",
                    "evaluate",
                    "listen",
                    "promote",
                    "infer",
                ],
                "training_data": {
                    "kind": "file",
                    "extensions": [".list"],
                },
                "fields": [
                    {
                        "key": "preprocess.resume",
                        "kind": "boolean",
                        "default": True,
                        "constraints": {},
                        "labels": _labels("复用预处理缓存", "Reuse preprocessing cache", "前処理キャッシュを再利用"),
                    },
                    {
                        "key": "s2.batch_size",
                        "kind": "integer",
                        "default": 2,
                        "constraints": {"minimum": 1},
                        "labels": _labels("S2 批大小", "S2 batch size", "S2 バッチサイズ"),
                    },
                    {
                        "key": "s2.target_steps",
                        "kind": "integer",
                        "default": 800,
                        "constraints": {"minimum": 1},
                        "labels": _labels("S2 优化步数", "S2 optimizer steps", "S2 最適化ステップ数"),
                    },
                    {
                        "key": "s2.learning_rate",
                        "kind": "number",
                        "default": 0.0001,
                        "constraints": {"exclusive_minimum": 0},
                        "labels": _labels("S2 学习率", "S2 learning rate", "S2 学習率"),
                    },
                    {
                        "key": "s1.batch_size",
                        "kind": "integer",
                        "default": 2,
                        "constraints": {"minimum": 1},
                        "labels": _labels("S1 批大小", "S1 batch size", "S1 バッチサイズ"),
                    },
                    {
                        "key": "s1.target_optimizer_steps",
                        "kind": "integer",
                        "default": 500,
                        "constraints": {"minimum": 1},
                        "labels": _labels("S1 优化步数", "S1 optimizer steps", "S1 最適化ステップ数"),
                    },
                    {
                        "key": "evaluation.shortlist_size",
                        "kind": "integer",
                        "default": 3,
                        "constraints": {"minimum": 1, "maximum": 26},
                        "labels": _labels("试听候选数", "Listening candidates", "試聴候補数"),
                    },
                ],
            }
        ],
    }


__all__ = ["build_descriptor"]
