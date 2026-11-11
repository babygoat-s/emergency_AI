"""Qwen 판정으로 라벨을 고치거나(수정) 애매한 칸을 뺀다(삭제). Train·Val에 같은 규칙을 쓴다.

칸 = 대화 1건 × 증상 1개. 대화는 지우지 않고 그 칸만 처리한다.

처리 규칙 (기준값은 config.JudgeConfig)
| 원래 라벨 | Qwen 확률                              | 처리          |
|-----------|----------------------------------------|---------------|
| 있음      | p_호소 ≥ 0.5                           | 그대로        |
| 있음      | p_없음 + p_부정 ≥ 0.9 (언급 없음·부정) | '없음'으로 수정 |
| 있음      | 그 사이 (대원만 물음, 반반 등)         | 삭제          |
| 없음      | p_호소 ≥ 0.9                           | '있음'으로 수정 |
| 없음      | 0.5 ≤ p_호소 < 0.9                     | 삭제          |
| 없음      | p_호소 < 0.5 (또는 판정 안 한 칸)      | 그대로        |
구토·오심 사이의 혼동은 합쳤기 때문에 오류로 세지 않는다.

검수에서 못 믿을 증상은 증상별로 낮출 수 있다 (`modes`)
- "수정": 위 규칙 그대로 (기본)
- "삭제만": 수정할 칸도 삭제로 돌린다 (라벨을 바꾸지 않음)
- "안 함": 그 증상은 원래 라벨 그대로

- `decide_actions`: 판정 표에 '처리' 열(그대로 / 없음으로수정 / 있음으로수정 / 삭제) 붙이기
- `action_summary`: 증상별 분포표(원래 있음, 수정 수, 삭제 수, 최종 있음, 남은 비율, 경고)
- `build_fixed` / `save_fixed` / `load_fixed`: 고친 라벨 Y와 마스크 M(1=사용, 0=삭제) 만들기·저장·읽기
- `review_sample` / `write_review_txt`: Claude·사람 검수용 표본(처리 종류별)과 읽기용 텍스트
"""
import numpy as np
import pandas as pd

from . import config

ACTIONS = ["그대로", "없음으로수정", "있음으로수정", "삭제"]
MODES = ["수정", "삭제만", "안 함"]


def decide_actions(judged, jcfg, modes=None):
    """modes: {증상: "수정" | "삭제만" | "안 함"}. 없는 증상은 "수정"."""
    modes = modes or {}
    bad = {s: m for s, m in modes.items() if m not in MODES}
    assert not bad, f"modes 값은 {MODES} 중 하나: {bad}"
    t = judged.copy()
    pos = t["라벨"] == 1
    p_say, p_not = t["p_호소"], t["p_없음"] + t["p_부정"]
    act = np.select(
        [pos & (p_say >= jcfg.thr_keep), pos & (p_not >= jcfg.thr_fix), pos,
         ~pos & (p_say >= jcfg.thr_fix), ~pos & (p_say >= jcfg.thr_keep)],
        ["그대로", "없음으로수정", "삭제", "있음으로수정", "삭제"], "그대로")
    mode = t["증상"].map(lambda s: modes.get(s, "수정"))
    act = np.where((mode == "삭제만") & np.isin(act, ["없음으로수정", "있음으로수정"]), "삭제", act)
    act = np.where(mode == "안 함", "그대로", act)
    t["처리"] = act
    return t


def action_summary(t, df, min_pos, jcfg):
    """증상별 분포표. df: 원래 대화 표(Train 또는 Val 전체). 원래 있음은 판정 안 한 대화(발화 없음)까지 센다.

    최종 있음 = 원래 있음 - '없음'으로 수정 - 있음 칸 삭제 + '있음'으로 수정
    경고: 최종 있음 < min_pos 또는 남은 비율 < jcfg.warn_keep_ratio
    """
    from .data import symptom_matrix
    g = t.groupby("증상")
    s = pd.DataFrame({
        "원래 있음": pd.Series(symptom_matrix(df["symptom"]).sum(0), index=config.LABELS),
        "있음→없음 수정": g.apply(lambda x: ((x["라벨"] == 1) & (x["처리"] == "없음으로수정")).sum()),
        "없음→있음 수정": g.apply(lambda x: ((x["라벨"] == 0) & (x["처리"] == "있음으로수정")).sum()),
        "있음 칸 삭제": g.apply(lambda x: ((x["라벨"] == 1) & (x["처리"] == "삭제")).sum()),
        "없음 칸 삭제": g.apply(lambda x: ((x["라벨"] == 0) & (x["처리"] == "삭제")).sum()),
    }).reindex(config.LABELS).fillna(0).astype(int)
    s["최종 있음"] = s["원래 있음"] - s["있음→없음 수정"] - s["있음 칸 삭제"] + s["없음→있음 수정"]
    s["남은 비율"] = (s["최종 있음"] / s["원래 있음"].clip(lower=1)).round(3)
    s["최종 있음 비율(대화 중)"] = (s["최종 있음"] / max(len(df), 1)).round(3)
    s["경고"] = np.where(s["최종 있음"] < min_pos, f"있음 {min_pos}개 미만",
                         np.where(s["남은 비율"] < jcfg.warn_keep_ratio,
                                  f"원래의 {jcfg.warn_keep_ratio:.0%} 미만", ""))
    return s


def build_fixed(df, t, labels=config.LABELS):
    """df(prep한 대화 표) 행 순서에 맞춘 (고친 라벨 Y, 마스크 M). 판정 안 한 칸은 원래 라벨, M=1."""
    from .data import symptom_matrix
    Y = symptom_matrix(df["symptom"], labels)
    M = np.ones_like(Y)
    row_of = {f.replace(".json", ""): i for i, f in enumerate(df["file_name"])}
    col_of = {s: j for j, s in enumerate(labels)}
    ch = t[t["처리"] != "그대로"]
    rows = ch["대화ID"].map(row_of)
    n_miss = int(rows.isna().sum())
    ch, rows = ch[rows.notna()], rows.dropna().astype(int).values
    cols = ch["증상"].map(col_of).values
    act = ch["처리"].values
    Y[rows[act == "없음으로수정"], cols[act == "없음으로수정"]] = 0
    Y[rows[act == "있음으로수정"], cols[act == "있음으로수정"]] = 1
    M[rows[act == "삭제"], cols[act == "삭제"]] = 0
    if n_miss:
        print(f"대화 표에 없는 대화ID {n_miss}개는 건너뜀")
    n = {a: int((act == a).sum()) for a in ACTIONS[1:]}
    print(f"수정 {n['없음으로수정'] + n['있음으로수정']:,}칸 (있음→없음 {n['없음으로수정']:,}, 없음→있음 {n['있음으로수정']:,})"
          f", 삭제 {n['삭제']:,}칸 / 전체 {Y.size:,}칸")
    return Y, M


def save_fixed(path, df, Y, M, labels=config.LABELS):
    np.savez(path, file_name=df["file_name"].values, Y=Y, M=M, labels=np.array(labels))
    print("저장:", path)


def load_fixed(path, df, labels=config.LABELS):
    """저장한 (Y, M)을 df 행 순서에 맞춰 읽는다. 파일에 없는 대화는 원래 라벨, M=1."""
    from .data import symptom_matrix
    z = np.load(path, allow_pickle=True)
    assert list(z["labels"]) == list(labels), "파일의 증상 목록이 다름"
    pos = {f: i for i, f in enumerate(z["file_name"])}
    Y = symptom_matrix(df["symptom"], labels)
    M = np.ones_like(Y)
    for i, f in enumerate(df["file_name"]):
        if f in pos:
            Y[i], M[i] = z["Y"][pos[f]], z["M"][pos[f]]
    return Y, M


REVIEW_PLAN = {"없음으로수정": 40, "있음으로수정": 40, "삭제": 30, "그대로": 40}


def review_sample(t, plan=REVIEW_PLAN, seed=42):
    """검수 표본: 처리 종류별로 plan 개수만큼(증상별 고르게). '그대로'는 라벨 있음 칸에서 뽑는다."""
    rng = np.random.RandomState(seed)
    parts = []
    for act, n in plan.items():
        pool = t[t["처리"] == act]
        if act == "그대로":
            pool = pool[pool["라벨"] == 1]
        if pool.empty:
            continue
        per = -(-n // pool["증상"].nunique())
        got = pd.concat([g.sample(min(len(g), per), random_state=rng) for _, g in pool.groupby("증상")])
        parts.append(got.sample(frac=1, random_state=rng).head(n))
    out = pd.concat(parts).sample(frac=1, random_state=rng).reset_index(drop=True)
    out.insert(0, "번호", range(1, len(out) + 1))
    out["확인(O/X)"], out["메모"] = "", ""
    return out


def write_review_txt(sample, dialogues, path):
    """읽기용 텍스트. dialogues: 대화ID → 녹취록 문자열."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("각 칸: Qwen 처리가 맞으면 O, 틀리면 X\n"
                "- 없음으로수정: 라벨 있음 → 없음. 신고자가 그 증상을 말하지 않았거나 부정했으면 O\n"
                "- 있음으로수정: 라벨 없음 → 있음. 신고자가 그 증상을 분명히 말했으면 O\n"
                "- 삭제: 애매해서 뺐다. 대화만으로 있다/없다를 확실히 정하기 어려우면 O\n"
                "- 그대로: 라벨 있음을 그대로 뒀다. 신고자가 그 증상을 말했으면 O\n\n")
        for r in sample.itertuples(index=False):
            f.write(f"==== #{r.번호} | {r.대화ID} | {r.증상} | 원래 라벨 {'있음' if r.라벨 == 1 else '없음'} | "
                    f"처리 {r.처리} | 호소 {r.p_호소:.2f} 대원질문 {r.p_대원질문:.2f} 없음 {r.p_없음:.2f} 부정 {r.p_부정:.2f}\n")
            f.write(dialogues[r.대화ID] + "\n\n")
    print("저장:", path, len(sample))


def review_accuracy(sample):
    """검수 엑셀(확인 O/X 채운 것)로 처리 종류 × 증상별 정확도. O/X가 빈 칸은 뺀다."""
    s = sample[sample["확인(O/X)"].astype(str).str.upper().isin(["O", "X"])].copy()
    s["맞음"] = s["확인(O/X)"].astype(str).str.upper() == "O"
    tab = s.pivot_table(index="증상", columns="처리", values="맞음", aggfunc=["mean", "size"])
    return tab.round(3)
