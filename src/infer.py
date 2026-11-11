"""저장한 최종 모델로 통화 하나의 증상을 예측한다(시연용).

- `save_model`: 학습한 모델 + 라벨 목록·설정을 .pt 하나로 저장
- `load_model`: .pt → (모델, 토크나이저, 설정)
- `predict_dialogue`: 발화 목록(utterances) 하나 → 증상별 확률 표
"""
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from .model import ChunkMaxClassifier, encode_chunks
from .text import dialogue_text
from .trainer import get_device, predict


def save_model(model, path, labels, cfg):
    torch.save({"state_dict": model.state_dict(), "labels": list(labels), "plm_name": cfg.plm_name,
                "max_len": cfg.max_len, "stride": cfg.stride}, path)


def load_model(path, device=None):
    device = device or get_device()
    ck = torch.load(path, map_location=device)
    model = ChunkMaxClassifier(ck["plm_name"], len(ck["labels"]))
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(ck["plm_name"])
    return model, tokenizer, ck


def predict_dialogue(utts, model, tokenizer, meta, threshold=0.5):
    """발화 목록 하나 → 증상, 확률, 판정(확률 ≥ threshold면 '있음') 표. 확률 높은 순."""
    text = dialogue_text(utts, tokenizer.sep_token)
    chunks = [encode_chunks(text, tokenizer, meta["max_len"], meta["stride"])]
    p = predict(model, chunks, tokenizer, batch=1, desc="예측")[0]
    out = pd.DataFrame({"증상": meta["labels"], "확률": np.round(p, 3)})
    out["판정"] = np.where(out["확률"] >= threshold, "있음", "")
    return out.sort_values("확률", ascending=False).reset_index(drop=True)
