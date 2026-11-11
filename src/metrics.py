"""평가 지표: 증상별 F1과 Macro F1. 마스크가 0인 칸(오류로 뺀 칸)은 계산에서 뺀다."""
import numpy as np
import pandas as pd


def f1_per_label(Y, pred, M=None):
    """라벨(열)별 F1. M(마스크)이 0인 칸은 TP/FP/FN 계산에서 뺀다."""
    M = np.ones_like(Y) if M is None else M
    tp = ((Y == 1) & (pred == 1) & (M == 1)).sum(0)
    fp = ((Y == 0) & (pred == 1) & (M == 1)).sum(0)
    fn = ((Y == 1) & (pred == 0) & (M == 1)).sum(0)
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def f1_report(Y, prob, labels, threshold=0.5, M=None):
    """라벨별 F1 표 + 마지막 행에 Macro 평균."""
    f1 = f1_per_label(Y, (prob >= threshold).astype(int), M)
    s = pd.Series(f1, index=labels).round(4)
    s["Macro"] = round(f1.mean(), 4)
    return s


def micro_accuracy(Y, prob, threshold=0.5, M=None):
    """칸 단위 정확도(맞힌 칸 / 남은 칸). 공식 '정확도'와 비슷한 보조 지표."""
    M = np.ones_like(Y) if M is None else M
    hit = ((prob >= threshold).astype(int) == Y) & (M == 1)
    return hit.sum() / max(M.sum(), 1)


def pr_report(Y, prob, labels, threshold=0.5, M=None):
    """라벨별 재현율(있는 증상 중 잡은 비율)·정밀도(있다고 한 것 중 맞은 비율)·남은 있음 수. 마스크 0인 칸은 뺀다."""
    M = np.ones_like(Y) if M is None else M
    pred = (prob >= threshold).astype(int)
    tp = ((Y == 1) & (pred == 1) & (M == 1)).sum(0)
    fp = ((Y == 0) & (pred == 1) & (M == 1)).sum(0)
    fn = ((Y == 1) & (pred == 0) & (M == 1)).sum(0)
    t = pd.DataFrame({"재현율": tp / np.maximum(tp + fn, 1), "정밀도": tp / np.maximum(tp + fp, 1),
                      "있음 수": tp + fn}, index=labels)
    t.loc["평균"] = [t["재현율"].mean(), t["정밀도"].mean(), t["있음 수"].sum()]
    t["있음 수"] = t["있음 수"].astype(int)
    return t.round(3)
