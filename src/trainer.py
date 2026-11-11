"""학습·예측 루프 (인프 프로젝트 trainer와 같은 방식).

- `masked_bce`: 마스크가 0인 칸(오류로 뺀 칸)은 loss에서 빼는 BCE
- `train_model`: 모델 하나 학습
- `predict`: 증상별 확률(sigmoid)
"""
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .model import ChunkMaxClassifier, make_collate


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(s):
    """재현성을 위해 모든 난수 시드를 고정한다."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _free(device):
    if device == "cuda":
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────
# loss
# ─────────────────────────────────────────────
def compute_pos_weight(Y, M, power=0.75):
    """증상마다 (음성 수/양성 수)^power. 남긴 칸(M=1)만 센다."""
    pos = (Y * M).sum(0)
    neg = ((1 - Y) * M).sum(0)
    return (neg / np.maximum(pos, 1)) ** power


def masked_bce(logits, labels, mask, pos_weight):
    """칸별 BCE를 구한 뒤 마스크가 1인 칸만 평균 낸다. 마스크가 모두 1이면 보통 BCE와 같다."""
    loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


# ─────────────────────────────────────────────
# 학습 / 예측
# ─────────────────────────────────────────────
def _forward(model, b, device):
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
        return model(b["input_ids"].to(device), b["attention_mask"].to(device), b["doc_index"], b["num_docs"])


def train_model(chunks, Y, cfg, tokenizer, M=None, desc="", device=None, loss_csv=None):
    """조각 목록 `chunks`, 라벨 `Y`(대화 수, 증상 수), 마스크 `M`(1=사용, 0=제외)으로 모델 하나를 학습한다.

    loss_csv: 경로를 주면 에폭별 평균 loss를 에폭이 끝날 때마다 CSV로 저장한다(끊겨도 남음).
    """
    device = device or get_device()
    M = np.ones_like(Y) if M is None else M
    pos_weight = torch.tensor(compute_pos_weight(Y, M, cfg.pos_weight_power), dtype=torch.float32, device=device)
    model = ChunkMaxClassifier(cfg.plm_name, Y.shape[1]).to(device)
    loader = DataLoader(list(zip(chunks, Y, M)), batch_size=cfg.doc_batch, shuffle=True,
                        collate_fn=make_collate(tokenizer.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.epochs * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total), total)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    history = []

    for ep in range(cfg.epochs):
        model.train()
        run = 0.0
        for b in tqdm(loader, desc=f"{desc} ep{ep}"):
            loss = masked_bce(_forward(model, b, device), b["labels"].to(device), b["mask"].to(device), pos_weight)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
        print(f"{desc} ep{ep} loss {run / len(loader):.4f}")
        history.append({"epoch": ep, "loss": run / len(loader)})
        if loss_csv:
            pd.DataFrame(history).to_csv(loss_csv, index=False, encoding="utf-8-sig")
    return model


@torch.no_grad()
def predict(model, chunks, tokenizer, batch=16, desc="predict", device=None):
    """조각 목록마다 증상별 확률(sigmoid)을 예측한다. 반환 모양: (대화 수, 증상 수)"""
    device = device or get_device()
    model.eval()
    dummy = np.zeros(model.head.out_features, dtype=np.float32)
    loader = DataLoader([(c, dummy) for c in chunks], batch_size=batch, shuffle=False,
                        collate_fn=make_collate(tokenizer.pad_token_id))
    probs = [torch.sigmoid(_forward(model, b, device).float()).cpu().numpy() for b in tqdm(loader, desc=desc)]
    return np.concatenate(probs)
