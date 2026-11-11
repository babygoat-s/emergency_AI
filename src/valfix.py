"""고친 Val 만들기: 대회 때 Claude가 Val 오류 칸 4,021개를 대화를 읽고 판정한 표를 적용한다.

인프 프로젝트 05와 같은 규칙(그래서 05의 '고친 Val 0.790'이 그대로 다시 나온다).
- 놓친 칸(FN) + 판정 '없음'(대화에 단서 없음) → mask: 채점에서 뺀다
- 잘못 잡은 칸(FP) + 판정 '호소'(신고자가 실제로 말함) → add: 라벨을 1로

- `load_fix_table`: 엑셀(시트 '전체판정') → (대화ID, 증상, 처리) 표
- `apply_fix_table`: 라벨 행렬에 적용 → (고친 라벨, 마스크)
- `merge_fixed`: 9개 기준 고친 Val → 묶은 8개(구토·오심) 기준
"""
import numpy as np
import pandas as pd


def load_fix_table(path):
    v = pd.read_excel(path, sheet_name="전체판정")
    t = v["유형"].astype(str).str[:2]
    act = np.where((t == "FN") & (v["판정"] == "없음"), "mask",
                   np.where((t == "FP") & (v["판정"] == "호소"), "add", "keep"))
    out = v.assign(처리=act)
    return out.loc[out["처리"] != "keep", ["대화ID", "증상", "처리"]].reset_index(drop=True)


def apply_fix_table(Y, file_names, fix, labels):
    """add → 라벨 1, mask → 마스크 0(채점에서 뺌). 마스크는 1이면 사용."""
    Y = Y.copy()
    M = np.ones_like(Y)
    row_of = {str(f).replace(".json", ""): i for i, f in enumerate(file_names)}
    col_of = {s: j for j, s in enumerate(labels)}
    n_skip = 0
    for did, sym, act in fix[["대화ID", "증상", "처리"]].itertuples(index=False):
        i, j = row_of.get(str(did).replace(".json", "")), col_of.get(sym)
        if i is None or j is None:
            n_skip += 1
            continue
        if act == "add":
            Y[i, j] = 1
        else:
            M[i, j] = 0
    if n_skip:
        print(f"수정표 {n_skip}행은 대화ID나 증상이 맞지 않아 건너뜀")
    return Y, M


def merge_fixed(Y_orig, Y_fix, M_fix, labels9, labels8, merge):
    """9개 기준 고친 Val을 묶은 라벨(8개) 기준으로 바꾼다.

    묶은 칸(예: 구토·오심) 규칙
    - 라벨: 묶음 안 증상 중 하나라도 '남은 양성'(고친 라벨 1이고 빼지 않음)이면 1
    - 빼기: 양성이 하나도 안 남았는데, 원래 라벨이 붙어 있던 증상이 빠진(mask) 경우에만 뺀다
      (원래 라벨이 대화에 근거가 없어서 뺀 것이므로 묶은 칸도 채점하지 않는다)
    반환: (Y8, M8)
    """
    Y8 = np.zeros((len(Y_orig), len(labels8)), dtype=np.float32)
    M8 = np.ones_like(Y8)
    for k, s in enumerate(labels8):
        idx = [labels9.index(m) for m in merge.get(s, [s])]
        pos = (Y_fix[:, idx] == 1) & (M_fix[:, idx] == 1)
        y = pos.any(1)
        masked = ~y & ((Y_orig[:, idx] == 1) & (M_fix[:, idx] == 0)).any(1)
        Y8[:, k], M8[:, k] = y, ~masked
    return Y8, M8
