# Third-Party Notices

GPT-SoVITS-derived source and resources retain the upstream MIT license and exact
provenance under `src/voice_pipeline/core/gpt_sovits/UPSTREAM.md`.

The adapted ERes2NetV2 and AFF speaker-encoder layers originate from
3D-Speaker, Copyright 3D-Speaker contributors, licensed under Apache-2.0.
Their exact GPT-SoVITS source paths and pinned revision are recorded in
`src/voice_pipeline/core/gpt_sovits/UPSTREAM.md`.

The Chinese text normalization, ToneSandhi, and G2PW adapter files are derived
through GPT-SoVITS from PaddleSpeech/G2PW and retain their Apache License 2.0
copyright headers in the copied source files.

The Japanese full-context-label prosody algorithm is derived through GPT-SoVITS
from ESPnet. GPT-SoVITS records that source in `text/japanese.py`; the internal
provenance document identifies the exact pinned path.

`frontend/resources/cmudict.rep` contains CMUdict 0.7b, Copyright 1993-2015
Carnegie Mellon University. Its permissive redistribution conditions and warranty
disclaimer are retained at the beginning of that file. The remaining English
lexical caches and overrides are copied byte-for-byte from the pinned GPT-SoVITS
revision.

Model weights and runtime language assets are external and are not redistributed
as source: S1/S2, CN-HuBERT, ERes2NetV2, Chinese RoBERTa, G2PW ONNX,
pyopenjtalk's dictionary, NLTK tagger data, and fast-langdetect `lid.176.bin`.
