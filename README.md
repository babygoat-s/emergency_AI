# emergency_AI: 119 신고 대화 증상 인식 모델

119 상담원 보조 서비스의 AI 부분입니다. 신고 대화(STT 텍스트)를 읽고 **신고자가 말한 증상 8개**를 찾아냅니다.
백엔드는 [emergency_backend](https://github.com/babygoat-s/emergency_backend), 화면은 [emergency_frontend](https://github.com/babygoat-s/emergency_frontend)에 있습니다.

## 한눈에 보기

| 항목 | 내용 |
|---|---|
| 문제 | 대화 텍스트 → 증상 다중 라벨 분류 |
| 증상 8개 | 고열, 구토·오심, 두통, 복통, 어지러움, 열상, 전신쇠약, 호흡곤란 |
| 모델 | `monologg/koelectra-base-v3-discriminator` + 선형층, 512토큰 조각(겹침 128)의 점수 최댓값 |
| 데이터 | AI Hub 119 신고 대화(서울 구급), Train 29,200 / Val 3,640 대화 (저장소에 포함하지 않음) |
| 평가 | Macro F1 (증상별 F1의 평균) |
| 최종 모델 | 3개 증상(구토·오심, 두통, 어지러움) 라벨을 고쳐 학습 + 증상별 기준값 |

## 왜 라벨부터 고쳤나

처음 모델(Macro F1 0.637)이 틀린 Val 칸을 대화와 하나씩 대조해 보니, 모델보다 **정답 라벨 문제**가 컸습니다.

- 모델이 "없음"이라 했는데 라벨은 "있음"인 1,521칸 중 **91.7%는 대화에 그 증상 이야기가 아예 없음**
- 모델이 "있음"이라 했는데 라벨은 "없음"인 2,500칸 중 **29.7%는 신고자가 실제로 말함** (라벨 누락)
- "구토면 오심도" 식으로 연관 증상을 습관적으로 같이 붙인 라벨이 많음 → **구토·오심은 하나로 합침** (9개 → 8개)

그래서 과제를 **"신고자 쪽이 대화에서 말한 증상 찾기"**로 정의했습니다.
대화에서 확인되지 않는 라벨은 "오류"라기보다 "대화 텍스트로 확인되지 않는 라벨"로 봅니다. 라벨러가 음성이나 현장 정보를 보고 붙였을 수 있기 때문입니다.

## 방법

1. **Qwen3-8B 판정 (01, 02):** 모델 예측은 보지 않고, 대화와 증상 정의만 주고 칸마다 묻습니다.
   "신고자가 이 증상을 말했나?" → A 호소 / B 대원이 묻기만 함 / C 언급 없음 / D 부정
2. **라벨 수정·삭제 (03):** Train과 Val에 같은 규칙을 씁니다 (`src/cleaning.py`).
   - 라벨 있음: p_호소 ≥ 0.5면 유지, p_없음+p_부정 ≥ 0.9면 '없음'으로 수정, 그 사이면 그 칸만 학습·채점에서 뺌
   - 라벨 없음: p_호소 ≥ 0.9면 '있음'으로 수정, 0.5 이상이면 뺌
3. **신뢰도 확인:** Qwen 판정과 대회 때 Claude 판정의 일치율은 전체 0.847입니다(`results/01_Claude_일치율.csv`).
   - Qwen이 '없음'이라고 한 것은 96.8%가 맞았습니다.
   - Qwen이 '호소'라고 한 것은 61.5%만 맞았습니다. 특히 구토·오심은 42.8%로 낮았습니다.
4. **고칠 증상만 고르기 (07):** 8개를 전부 고치면 열상·전신쇠약이 크게 나빠졌습니다(06). 라벨 습관 문제가 뚜렷한 **구토·오심, 두통, 어지러움 3개만** 고쳤습니다(`results/07_처리_분포표.csv`).
5. **증상별 기준값 (08):** 고친 Val을 반으로 나눠, 절반 A에서 기준값을 고르고 절반 B에서 채점했습니다. 고친 3개 증상만 기준값을 바꿨습니다.

## 결과 (Macro F1)

| 모델 | 원본 Val | 고친 Val | 정리 Val (Qwen 기준, 참고) |
|---|---|---|---|
| 원본 라벨로 학습 | **0.671** | 0.818 | 0.702 |
| 8개 전부 고쳐 학습 (06) | 0.560 | 0.735 | 0.707 |
| **3개만 고쳐 학습 (07, 최종)** | 0.663 | **0.841** | **0.810** |

- **원본 Val:** AI Hub 라벨 그대로 채점합니다. 다른 연구와 비교할 때 씁니다.
- **고친 Val:** 대회 때 Claude가 고친 Val로 채점합니다.
- **정리 Val:** Qwen 규칙으로 고친 Val로 채점합니다. 같은 판정관이 Train도 고쳤기 때문에 점수가 부풀 수 있어 참고용입니다.
- 기준값 조정(08, 고친 Val 절반 B): 기준값 0.5일 때 0.835 → 증상별 기준값을 쓰면 0.842
- 증상별 점수: `results/07_점수_증상별.csv`, `results/08_B절반_요약.csv`

## 폴더 구조

```
src/          기능별 코드 (config, data, text, judge_prompt, judge_llm, cells, cleaning,
              model, trainer, metrics, valfix, evaluate, infer)
notebooks/    00 준비 확인 → 01 Qwen Val 판정 → 02 Train 판정 → 03 라벨 수정·삭제
              → 04 학습 → 05 시연 → 06 전부 수정 학습 → 07 3개 수정 재학습 → 08 기준값 조정
serving/      백엔드 연결용 증상 분류기 (symptom_model.py, thresholds.json)
results/      점수표·분포표 (데이터 원문 없음)
docs/         계획과 미리 정한 판단 규칙
```

경로와 숫자는 `src/config.py`에서만 바꿉니다. 노트북은 Colab에서 Google Drive를 연결해 실행하고, 단계 사이 데이터는 Drive 파일로 넘깁니다.

## 실행

```bash
pip install -r requirements.txt
```

- Colab: 노트북 첫 셀이 Drive를 연결하고 `SYMPTOM_FINAL_ROOT`를 프로젝트 폴더로 정합니다.
- 내 PC: 환경변수 `SYMPTOM_FINAL_ROOT`에 프로젝트 폴더를 넣고, 첫 셀의 `drive.mount` 두 줄은 건너뜁니다.
- 필요한 입력 파일 `data_in/labels_df.pkl`은 AI Hub 데이터로 만든 것이라 저장소에 없습니다.
- GPU 기준 시간(A100): 01은 15~30분, 02는 1~2시간, 모델 학습은 하나에 약 40분입니다.

## 백엔드에 연결하기

백엔드의 증상 자리(`SymptomModel.predict(text) -> list[Symptom] | None`)에 `serving/symptom_model.py`를 연결합니다.

```python
import sys; sys.path.insert(0, "<emergency_AI 경로>")     # src, serving을 import하기 위해
from serving.symptom_model import KoElectraSymptomModel

clf = KoElectraSymptomModel("<경로>/정리모델_3개.pt")      # CPU면 자동으로 CPU
clf.predict("배가 너무 아파요. 아까 두 번 토했어요.")
# [{'code': 'abdominal_pain', 'label': '복통', 'score': 0.97, 'evidence': '배가 너무 아파요.'}, ...]
```

- 결과 dict는 백엔드 `Symptom`의 필드(code, label, score, evidence)와 같습니다.
- 모델 입력은 STT 텍스트를 문장 끝에서 끊고 `[SEP]`으로 이어 붙인 문자열입니다. 학습 때 발화를 `[SEP]`으로 이은 것과 같은 꼴로 맞추려는 것입니다.
- `evidence`는 그 증상 관련 단어가 들어 있는 원문 문장입니다. 없으면 `None`입니다.
- **백엔드에서 맞춰야 할 것:** `app/schemas.py`의 `SymptomCode`에 우리 증상 코드 7개를 추가해야 합니다(`serving/symptom_model.py`의 `CODES`). `breathing_difficulty`는 이미 있습니다.
- **주의:** 학습 데이터는 사람이 받아 적은 녹취록이고 서비스 입력은 STT 결과입니다. 인식 오류가 있으면 점수가 위 표보다 낮을 수 있습니다.

## 모델 가중치

`정리모델_3개.pt`는 약 450MB라 GitHub에 올리지 않습니다. 배포 위치는 정해지는 대로 여기에 링크를 답니다.

## 데이터 출처

AI Hub "119 지능형 신고접수 음성 인식 데이터"를 사용했습니다. 원본 데이터와 그 데이터에서 만든 파일(라벨 표, 판정 결과, 검수 파일)은 AI Hub 이용 조건 때문에 저장소에 넣지 않습니다.
