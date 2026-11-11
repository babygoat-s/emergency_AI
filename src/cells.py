"""판정할 칸(대화 × 증상) 만들기와 Qwen 판정 실행.

모델 예측은 보지 않는다. Train·Val 모든 대화에 같은 규칙을 쓴다.
- 판정 대상: 원래 라벨이 1인 칸 + 라벨 0이지만 대화에 그 증상 관련 단어가 있는 칸
  (라벨 0이고 관련 단어도 없는 칸은 '언급 없음'이 확실하므로 묻지 않는다 → 시간 절약)
- 구토·오심은 합친 라벨(둘 중 하나라도 붙어 있으면 1)로 한 번만 묻는다
- 발화가 없는 대화는 판정할 수 없어 뺀다(Train은 학습에서도 빠짐)

- `build_cells`: 판정할 칸 표
- `judge_cells`: 칸마다 A/B/C/D 확률. 블록마다 드라이브에 저장 → 끊겨도 다시 실행하면 이어서
- `load_judged`: 저장한 판정 결과 읽기
"""
import os
import time

import numpy as np
import pandas as pd

from . import config
from .data import symptom_matrix
from .judge_llm import score_prompts
from .judge_prompt import CHOICE_NAMES, CHOICES, build_prompt
from .text import clean_utterances, show_dialogue


def build_cells(df, labels=config.LABELS, keywords=config.SYMPTOM_KEYWORDS):
    """열: 대화ID, 증상, 라벨(원래, 구토·오심은 합침), 단어(관련 단어 있음 1), 대화(읽기용 녹취록)."""
    has_text = df["utterances"].apply(lambda u: len(clean_utterances(u)) > 0)
    d = df[has_text]
    ids = d["file_name"].str.replace(".json", "", regex=False).values
    dia = d["utterances"].apply(show_dialogue)
    Y = symptom_matrix(d["symptom"], labels)
    rows = []
    for j, s in enumerate(labels):
        kw = dia.str.contains(keywords[s], regex=True).values.astype(int)
        rows.append(pd.DataFrame({"대화ID": ids, "증상": s, "라벨": Y[:, j].astype(int), "단어": kw,
                                  "대화": dia.values}))
    cells = pd.concat(rows, ignore_index=True)
    n_all = len(cells)
    cells = cells[(cells["라벨"] == 1) | (cells["단어"] == 1)].reset_index(drop=True)
    print(f"판정할 칸 {len(cells):,} / 전체 {n_all:,} (대화 {len(d):,} × 증상 {len(labels)}), "
          f"발화 없는 대화 {int((~has_text).sum())}개 제외")
    print(cells.groupby("증상")["라벨"].agg(라벨1="sum", 칸수="size").T.to_string())
    return cells


def judge_cells(cells, tokenizer, model, cache_dir, jcfg, max_blocks=None):
    """칸마다 A/B/C/D 확률 (칸 수, 4). `cache_dir/part{k}.npz`가 있으면 그 블록은 건너뛴다.

    max_blocks: 이번 실행에서 새로 판정할 블록 수 상한(시간 재기용). None이면 끝까지.
    반환: (확률 배열, 다 끝났는지)
    """
    os.makedirs(cache_dir, exist_ok=True)
    keys = (cells["대화ID"] + "|" + cells["증상"]).values
    out = np.full((len(cells), len(CHOICES)), np.nan, dtype=np.float32)
    n_block = -(-len(cells) // jcfg.block)
    done_new = 0
    for k, s in enumerate(range(0, len(cells), jcfg.block)):
        path = f"{cache_dir}/part{k:03d}.npz"
        sl = slice(s, s + jcfg.block)
        if os.path.exists(path):
            z = np.load(path, allow_pickle=True)
            assert list(z["keys"]) == list(keys[sl]), f"{path}의 칸이 지금 대상과 다름. 폴더를 지우고 다시 실행"
            assert z["p"].shape[1] == len(CHOICES), f"{path}는 예전 선택지({z['p'].shape[1]}개)로 판정됨. 폴더를 지우고 다시 실행"
            out[sl] = z["p"]
            continue
        if max_blocks is not None and done_new >= max_blocks:
            break
        sub = cells.iloc[sl]
        prompts = [build_prompt(tokenizer, d, sy) for d, sy in zip(sub["대화"], sub["증상"])]
        t0 = time.time()
        out[sl] = score_prompts(prompts, tokenizer, model, CHOICES, jcfg.batch_size, jcfg.max_tokens,
                                desc=f"블록 {k + 1}/{n_block}")
        np.savez(path, keys=keys[sl], p=out[sl])
        done_new += 1
        sec = time.time() - t0
        left = n_block - k - 1
        print(f"블록 {k + 1}/{n_block} 저장: {sec / 60:.1f}분 → 남은 {left}블록 약 {left * sec / 60:.0f}분")
    finished = not np.isnan(out).any()
    print("판정 완료" if finished else f"아직 {int(np.isnan(out[:, 0]).sum()):,}칸 남음. 다시 실행하면 이어서 함")
    return out, finished


def save_judged(cells, probs, path):
    """판정 결과 표(대화 본문 제외)를 저장한다: 대화ID, 증상, 라벨, 단어, p_호소, p_대원질문, p_없음, p_부정."""
    t = cells.drop(columns=["대화"]).copy()
    for k, c in enumerate(CHOICES):
        t[f"p_{CHOICE_NAMES[c]}"] = probs[:, k]
    t.to_csv(path, index=False, encoding="utf-8-sig")
    print("저장:", path, len(t))
    return t


def load_judged(path):
    return pd.read_csv(path, dtype={"대화ID": str})
