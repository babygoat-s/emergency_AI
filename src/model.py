"""모델과 입력 만들기 (대회 때 `mission3_model.py`와 같은 구조).

긴 대화는 512토큰에 안 들어가므로 여러 조각으로 자른다.
조각마다 [CLS] 벡터로 점수를 내고, 같은 대화의 조각 점수 중 최댓값을 대화 점수로 쓴다.
→ 증상 말이 대화 어디에 나오든 그 조각이 점수를 끌어올린다.
"""
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel


# ─────────────────────────────────────────────
# 1. 텍스트 → 조각
# ─────────────────────────────────────────────
def encode_chunks(text, tokenizer, max_len=512, stride=128):
    """텍스트를 토큰화해 `max_len` 길이 조각 목록으로 자른다.

    각 조각 = [CLS] + 본문(max_len-2개) + [SEP]. 이웃 조각은 `stride` 토큰씩 겹친다.
    짧은 텍스트는 조각 1개.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    body = max_len - 2
    step = body - stride
    starts = range(0, max(len(ids) - stride, 1), step)
    return [[tokenizer.cls_token_id] + ids[s:s + body] + [tokenizer.sep_token_id] for s in starts]


# ─────────────────────────────────────────────
# 2. 배치 만들기
# ─────────────────────────────────────────────
def make_collate(pad_id):
    """DataLoader에 넣을 collate 함수를 만든다.

    입력 배치의 원소: (조각 목록, 라벨 벡터) 또는 (조각 목록, 라벨, 마스크)
    출력: 모든 조각을 한 텐서로 펼치고, `doc_index`로 각 조각이 몇 번째 대화인지 표시한다.
    """
    def collate(batch):
        flat, doc_index, labels, masks = [], [], [], []
        for i, item in enumerate(batch):
            chunks, y = item[0], item[1]
            flat.extend(chunks)
            doc_index.extend([i] * len(chunks))
            labels.append(y)
            masks.append(item[2] if len(item) > 2 else np.ones_like(y))
        width = max(len(c) for c in flat)
        ids = torch.full((len(flat), width), pad_id, dtype=torch.long)
        att = torch.zeros((len(flat), width), dtype=torch.long)
        for j, c in enumerate(flat):
            ids[j, :len(c)] = torch.tensor(c)
            att[j, :len(c)] = 1
        return {
            "input_ids": ids,
            "attention_mask": att,
            "doc_index": torch.tensor(doc_index),
            "num_docs": len(batch),
            "labels": torch.tensor(np.stack(labels)),
            "mask": torch.tensor(np.stack(masks)),
        }
    return collate


# ─────────────────────────────────────────────
# 3. 모델
# ─────────────────────────────────────────────
class ChunkMaxClassifier(nn.Module):
    """PLM 인코더 + 선형층. 조각 점수를 대화별 최댓값으로 모은다."""

    def __init__(self, plm_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(plm_name)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, doc_index, num_docs):
        # 조각마다 [CLS] 벡터 → 라벨별 점수
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        chunk_logits = self.head(self.dropout(cls)).float()
        # 같은 대화(doc_index가 같은 조각)끼리 최댓값
        out = torch.full((num_docs, chunk_logits.size(1)), float("-inf"), device=chunk_logits.device)
        idx = doc_index.to(chunk_logits.device).unsqueeze(1).expand_as(chunk_logits)
        return out.scatter_reduce(0, idx, chunk_logits, reduce="amax", include_self=True)
