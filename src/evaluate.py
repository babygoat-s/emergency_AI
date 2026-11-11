"""채점: 여러 Val 기준으로 점수표를 만들고, Qwen 판정이 Claude 판정과 얼마나 맞는지 본다.

Val 기준 세 가지
- 원본 Val: 원래 라벨 그대로(구토·오심만 합침). 남들과 비교되는 점수
- 정리 Val: Qwen 판정으로 Train과 같은 규칙으로 고친 Val(수정 + 애매한 칸 삭제). 이 프로젝트의 대표 점수
- 고친 Val: 대회 때 Claude가 모델이 틀린 칸만 읽고 고친 Val(파일이 있을 때만). 예전 0.815와 비교용

- `claude_fixed_val`: 고친 Val(8개 기준) 라벨·마스크
- `score_table`: 모델별 × Val 기준별 Macro F1 + 증상별 F1
- `log_experiment`: 실험기록.csv에 실험마다 한 줄씩 추가(같은 이름이면 덮어씀)
- `agree_with_claude`: 고친 Val 표의 칸에서 Qwen(p_호소 ≥ 0.5)과 Claude('호소') 일치율
"""
import os

import numpy as np
import pandas as pd

from . import config
from .data import symptom_matrix
from .metrics import f1_report
from .valfix import apply_fix_table, load_fix_table, merge_fixed


def claude_fixed_val(val_df, path=config.VAL_CLAUDE_JUDGE):
    """(Y8, M8) 또는 파일이 없으면 None."""
    if not os.path.exists(path):
        print("고친 Val 파일 없음 → 고친 Val 점수는 생략")
        return None
    fix = load_fix_table(path)
    Y9 = symptom_matrix(val_df["symptom"], config.TARGET9, {})
    Y9f, M9f = apply_fix_table(Y9, val_df["file_name"].values, fix, config.TARGET9)
    return merge_fixed(Y9, Y9f, M9f, config.TARGET9, config.LABELS, config.MERGE)


def score_table(probs, val_sets, labels=config.LABELS, threshold=0.5):
    """probs: {모델 이름: Val 확률}, val_sets: {기준 이름: (Y, M)}. 반환 (Macro 표, 증상별 표)."""
    macro, detail = {}, []
    for m, p in probs.items():
        for v, (Y, M) in val_sets.items():
            r = f1_report(Y, p, labels, threshold, M)
            macro[(m, v)] = r["Macro"]
            detail.append(r.rename(f"{m} / {v}"))
    mt = pd.Series(macro).unstack().loc[list(probs), list(val_sets)]
    return mt, pd.concat(detail, axis=1)


def agree_with_claude(judged_val, path=config.VAL_CLAUDE_JUDGE, thr=0.5):
    """Claude가 읽고 판정한 칸(구토·오심은 둘 중 하나라도 '호소'면 호소)과 Qwen 판정 비교. 파일 없으면 None."""
    if not os.path.exists(path):
        return None
    v = pd.read_excel(path, sheet_name="전체판정")
    to8 = {m: s for s, ms in config.MERGE.items() for m in ms}
    v["증상8"] = v["증상"].map(lambda s: to8.get(s, s))
    v["대화ID"] = v["대화ID"].astype(str).str.replace(".json", "", regex=False)
    c = v.groupby(["대화ID", "증상8"])["판정"].apply(lambda x: (x == "호소").any()).rename("Claude호소").reset_index()
    j = judged_val.merge(c, left_on=["대화ID", "증상"], right_on=["대화ID", "증상8"])
    j["Qwen호소"] = j["p_호소"] >= thr

    def stats(x):
        q, c = x["Qwen호소"], x["Claude호소"]
        return pd.Series({"칸 수": len(x), "일치율": (q == c).mean(),
                          "Qwen '없음'이 맞은 비율": (~c[~q]).mean() if (~q).any() else np.nan,
                          "Qwen '호소'가 맞은 비율": c[q].mean() if q.any() else np.nan})

    out = pd.DataFrame({s: stats(g) for s, g in j.groupby("증상")}).T
    out.loc["전체"] = stats(j)
    out["칸 수"] = out["칸 수"].astype(int)
    return out.round(3)


def log_experiment(path, name, settings, macro, details, labels=config.LABELS):
    """실험 하나를 실험기록.csv에 한 줄로 남긴다. 같은 실험 이름이 있으면 그 줄을 바꾼다.

    settings: 바꾼 설정 설명(문자열)
    macro:    {Val 기준 이름: Macro F1}
    details:  {Val 기준 이름: pr_report 표(재현율·정밀도) 에 F1 열을 붙인 표}
    """
    from datetime import datetime
    row = {"실험": name, "바꾼 설정": settings, "기록 시각": datetime.now().strftime("%Y-%m-%d %H:%M")}
    for v, m in macro.items():
        row[f"{v} Macro F1"] = round(float(m), 4)
    for v, t in details.items():
        for s in labels:
            for col in ["F1", "재현율", "정밀도"]:
                row[f"{v} {s} {col}"] = t.loc[s, col]
    log = pd.read_csv(path, encoding="utf-8-sig") if os.path.exists(path) else pd.DataFrame()
    if len(log) and "실험" in log:
        log = log[log["실험"] != name]
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    log.to_csv(path, index=False, encoding="utf-8-sig")
    print("기록:", path, f"({len(log)}개 실험)")
    return log
