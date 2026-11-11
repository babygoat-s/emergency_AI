"""대화(utterances 리스트)를 문자열로 바꾸는 함수.

- `dialogue_text`: 모델 입력용. 발화 텍스트만 [SEP]으로 이어 붙인다(대회·인프 프로젝트와 같은 방식).
- `show_dialogue`: 사람이 읽는 용도. 줄마다 번호와 화자를 붙인다.
"""
from .config import SPEAKER_NAMES


def clean_utterances(utts):
    """텍스트가 비어 있는 발화를 빼고 (원래 순번, 발화 dict, 정리된 텍스트) 목록을 돌려준다."""
    out = []
    for i, u in enumerate(utts or []):
        t = str(u.get("text") or "").strip()
        if t:
            out.append((i, u, t))
    return out


def dialogue_text(utts, sep="[SEP]"):
    """모델 입력 문자열. 예: '네 119입니다 [SEP] 배가 아파요 [SEP] ...'"""
    return f" {sep} ".join(t for _, _, t in clean_utterances(utts))


def show_dialogue(utts):
    """읽기용 문자열. 예: '[0] 대원: 네 119입니다\\n[1] 신고자: 배가 아파요'"""
    return "\n".join(
        f"[{i}] {SPEAKER_NAMES.get(u.get('speaker'), u.get('speaker'))}: {t}"
        for i, u, t in clean_utterances(utts)
    )
