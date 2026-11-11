"""백엔드(emergency_backend)에 연결하는 증상 분류기.

백엔드 계약 (`emergency_backend/models/README.md` 4번)
- `SymptomModel.predict(text) -> list[Symptom] | None`
- 입력: 누적 STT 텍스트(최대 20,000자). 화자 구분 없음
- 출력: 증상 목록. score는 모델 확률(검증된 임상 확률 아님), evidence는 원문에 실제 있는 문장
- `[]`는 "모델을 돌렸지만 증상 없음"이라는 뜻이다(이때 백엔드는 Gemini 증상으로 덮어쓰지 않는다)

학습 때 입력은 발화 텍스트를 [SEP]으로 이은 문자열이었다(`src/text.py`의 `dialogue_text`, 화자 표시 없음).
STT 텍스트에는 발화 경계가 없으므로 문장 끝(. ? ! 줄바꿈)에서 끊어 [SEP]으로 잇는다.

- `KoElectraSymptomModel`: .pt 하나 + 증상별 기준값(thresholds.json)으로 예측
- `to_model_text`: STT 텍스트 → 모델 입력 문자열
- `find_evidence`: 증상 관련 단어가 들어 있는 원문 문장 하나
"""
import json
import os
import re

from src import config

# 백엔드 `app/schemas.py`의 SymptomCode에 추가해야 하는 코드 (백엔드 담당과 맞출 것)
CODES = {
    "고열": "fever",
    "구토·오심": "vomiting_nausea",
    "두통": "headache",
    "복통": "abdominal_pain",
    "어지러움": "dizziness",
    "열상": "laceration",
    "전신쇠약": "general_weakness",
    "호흡곤란": "breathing_difficulty",   # 백엔드에 이미 있는 코드
}

HERE = os.path.dirname(os.path.abspath(__file__))
THRESHOLDS_JSON = os.path.join(HERE, "thresholds.json")   # 08 기준값 조정 결과
SENT_END = re.compile(r"(?<=[.?!])\s+|\n+")


def split_sentences(text):
    """문장 끝(. ? ! 뒤 공백, 줄바꿈)에서 끊고 빈 문장은 뺀다."""
    return [s.strip() for s in SENT_END.split(text or "") if s.strip()]


def to_model_text(text, sep="[SEP]"):
    """STT 텍스트 → 학습 때와 같은 꼴의 입력. 예: '배가 아파요. 토했어요' → '배가 아파요. [SEP] 토했어요'"""
    return f" {sep} ".join(split_sentences(text))


def find_evidence(text, label, max_len=1000):
    """그 증상의 관련 단어(`config.SYMPTOM_KEYWORDS`)가 처음 나오는 문장. 없으면 None."""
    pattern = re.compile(config.SYMPTOM_KEYWORDS[label])
    for s in split_sentences(text):
        if pattern.search(s):
            return s[:max_len]
    return None


class KoElectraSymptomModel:
    """학습한 KoELECTRA(.pt)로 STT 텍스트의 증상을 예측한다. torch는 만들 때 불러온다."""
    provider = "koelectra-symptom-v1"

    def __init__(self, model_path, thresholds_path=THRESHOLDS_JSON, device=None):
        from src.infer import load_model
        self.model, self.tokenizer, self.meta = load_model(model_path, device)
        with open(thresholds_path, encoding="utf-8") as f:
            self.thresholds = json.load(f)

    def probs(self, text):
        """증상 → 확률. 빈 텍스트면 빈 dict."""
        from src.model import encode_chunks
        from src.trainer import predict
        model_text = to_model_text(text, self.tokenizer.sep_token)
        if not model_text:
            return {}
        chunks = [encode_chunks(model_text, self.tokenizer, self.meta["max_len"], self.meta["stride"])]
        p = predict(self.model, chunks, self.tokenizer, batch=1, desc="증상")[0]
        return dict(zip(self.meta["labels"], map(float, p)))

    def predict(self, text):
        """기준값을 넘은 증상만, 확률 높은 순. 백엔드에서 `Symptom(**d)`로 감싸면 된다."""
        out = []
        for label, p in sorted(self.probs(text).items(), key=lambda x: -x[1]):
            if p >= self.thresholds.get(label, 0.5):
                out.append({"code": CODES[label], "label": label, "score": round(p, 4),
                            "evidence": find_evidence(text, label)})
        return out
