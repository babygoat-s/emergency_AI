"""데이터 읽기: 라벨 표 → 모델 입력 문자열 + 8개 증상 0/1 행렬.

- `load_labels`: (train_df, val_df)
- `prep`: 대화 → 입력 문자열 열(`text`)을 붙인 표
- `symptom_matrix`: symptom 리스트 → (대화 수, 증상 수) 0/1 행렬. 구토·오심은 OR로 묶는다
"""
import os

import numpy as np
import pandas as pd

from . import config
from .text import dialogue_text


def load_labels(path=config.LABELS_PKL):
    """labels_df.pkl을 읽는다. 이 파일은 고치지 않는다."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 없음. data_in 폴더에 labels_df.pkl을 넣어 주세요.")
    d = pd.read_pickle(path)
    print("train", d["train"].shape, "/ val", d["val"].shape)
    return d["train"], d["val"]


def prep(df, sep, drop_empty):
    """file_name, utterances, symptom, text 열만 남긴다. drop_empty=True면 발화가 없는 대화를 뺀다(Train만)."""
    d = df[["file_name", "utterances", "symptom"]].copy()
    d["text"] = d["utterances"].apply(lambda u: dialogue_text(u, sep))
    if drop_empty:
        d = d[d["text"].str.len() > 0]
    return d.reset_index(drop=True)


def symptom_matrix(symptom_lists, labels=config.LABELS, merge=config.MERGE):
    """대화별 symptom 리스트를 (대화 수, 라벨 수) 0/1 float32 행렬로. 묶은 라벨은 원래 라벨 중 하나라도 있으면 1."""
    members = [merge.get(s, [s]) for s in labels]
    return np.array([[int(any(m in (ss or []) for m in ms)) for ms in members] for ss in symptom_lists],
                    dtype=np.float32)


def dialogue_ids(df):
    """file_name에서 .json을 뗀 대화ID."""
    return df["file_name"].str.replace(".json", "", regex=False).values
