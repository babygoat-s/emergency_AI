"""프로젝트 전체 설정: 경로, 증상 목록, 판정(Qwen)·학습(KoELECTRA) 설정.

경로나 숫자를 바꿀 일이 생기면 이 파일만 고친다. 노트북과 다른 모듈은 여기 값을 가져다 쓴다.
다른 계정에서 돌릴 때: 이 폴더를 그 계정 '내 드라이브' 바로 아래에 두면 그대로 돌아간다.
다른 곳에 두었으면 노트북 0단계의 PROJECT_ROOT만 바꾼다(환경변수 SYMPTOM_FINAL_ROOT로 전달됨).
"""
import os
from dataclasses import dataclass

# ─────────────────────────────────────────────
# 1. 경로
# ─────────────────────────────────────────────
PROJECT_ROOT = os.environ.get("SYMPTOM_FINAL_ROOT", "/content/drive/MyDrive/증상인식_최종_프로젝트")

# 입력 데이터(읽기만). 이 폴더에 두 파일을 넣어 둔다.
DATA_IN = f"{PROJECT_ROOT}/data_in"
LABELS_PKL = f"{DATA_IN}/labels_df.pkl"              # 인프 01이 만든 라벨 표 {"train": df, "val": df}
VAL_CLAUDE_JUDGE = f"{DATA_IN}/val_claude_judge.xlsx"  # (선택) 대회 때 Claude가 고친 Val. 없으면 '고친 Val' 점수만 생략

# 만들어지는 것
JUDGE_DIR = f"{PROJECT_ROOT}/judge"       # Qwen 판정 결과(블록마다 저장 → 끊겨도 이어서)
DATA_DIR = f"{PROJECT_ROOT}/data"         # 고친 라벨 + 삭제 마스크(03이 만듦)
RESULTS_DIR = f"{PROJECT_ROOT}/results"   # 점수표, 요약, 검수 파일
MODEL_DIR = f"{PROJECT_ROOT}/models"      # 학습한 KoELECTRA(시연용)

# ─────────────────────────────────────────────
# 2. 증상
# ─────────────────────────────────────────────
# 대회 9개 증상. 구토·오심은 라벨러가 섞어 붙여서(구토만 말해도 오심을 붙이는 등) 하나로 합친다 → 8개
TARGET9 = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
MERGE = {"구토·오심": ["구토", "오심"]}
LABELS = ["고열", "구토·오심", "두통", "복통", "어지러움", "열상", "전신쇠약", "호흡곤란"]

# Qwen에게 알려 주는 증상 뜻
SYMPTOM_DEF = {
    "고열": "열이 남, 체온이 높음, 몸이 뜨거움",
    "구토·오심": "실제로 토함·게움, 또는 메스꺼움·속이 울렁거림·토할 것 같음 (둘 중 하나라도 해당)",
    "두통": "머리가 아픔",
    "복통": "배가 아픔, 명치·아랫배 통증",
    "어지러움": "어지러움, 핑 돎, 빙빙 돎",
    "열상": "피부가 찢어지거나 베여서 피가 남",
    "전신쇠약": "기운이 없음, 힘이 빠져 못 움직임, 축 처짐",
    "호흡곤란": "숨이 참, 숨을 잘 못 쉼, 호흡이 힘듦",
}

# 증상 관련 단어(넓게). 라벨이 0인 칸은 이 단어가 대화에 있을 때만 Qwen에게 묻는다(나머지는 라벨 0이 맞다고 봄)
SYMPTOM_KEYWORDS = {
    "고열": r"열|체온|뜨거|뜨겁|[34]\d(?:\.\d)? ?도",
    "구토·오심": r"토[했하해한할]|토를|토 했|게[웠워우]|구토|올렸|토사|토가|메스|울렁|속이 안 ?좋|토할 ?것|구역|느글|미식|토하려|토할라",
    "두통": r"머리|두통|골치",
    "복통": r"배[가를]? |배가|복통|명치|아랫배|윗배|복부|배 아|뱃속|배꼽",
    "어지러움": r"어지|어질|핑 ?돌|핑 ?도|빙빙|현기|어찔",
    "열상": r"찢|베[였여이었인어]|열상|피[가를]|피 ?나|출혈|꿰매|봉합",
    "전신쇠약": r"기운|힘이 ?없|힘 ?없|축 ?처|못 ?일어|처져|처지|기력|쇠약|늘어져|맥이 ?없|힘을 못|힘이 빠",
    "호흡곤란": r"숨|호흡|헐떡|가쁘|가빠|헉헉|헥헥",
}

# speaker 값의 의미 (샘플 확인 결과 0이 119 대원)
SPEAKER_NAMES = {0: "대원", 1: "신고자"}


# ─────────────────────────────────────────────
# 3. Qwen 판정 설정
# ─────────────────────────────────────────────
@dataclass
class JudgeConfig:
    """라벨 오류 칸을 찾는 판정관(공개 LLM) 설정."""
    model_name: str = "Qwen/Qwen3-8B"
    load_in_4bit: bool = False  # A100이면 False. T4처럼 메모리가 작으면 True
    batch_size: int = 8
    max_tokens: int = 6000      # 이보다 긴 녹취록은 앞쪽을 자른다
    block: int = 2000           # 이 칸 수마다 드라이브에 저장
    # 라벨 처리 기준 (cleaning.decide_actions)
    # 라벨 있음: p_호소 ≥ thr_keep → 그대로 / p_없음+p_부정 ≥ thr_fix → '없음'으로 수정 / 그 사이 → 삭제
    # 라벨 없음: p_호소 ≥ thr_fix → '있음'으로 수정 / thr_keep 이상 → 삭제 / 그 미만 → 그대로
    thr_keep: float = 0.5
    thr_fix: float = 0.9
    # 정리 후 증상별 '있음' 수가 이보다 적거나, 원래보다 이 비율 아래로 줄면 경고
    warn_min_pos_train: int = 500
    warn_min_pos_val: int = 60
    warn_keep_ratio: float = 0.5


# ─────────────────────────────────────────────
# 4. 학습 설정
# ─────────────────────────────────────────────
@dataclass
class TrainConfig:
    """KoELECTRA 학습 설정(인프 프로젝트와 같은 값 → 예전 점수와 바로 비교된다)."""
    plm_name: str = "monologg/koelectra-base-v3-discriminator"
    max_len: int = 512          # 조각 하나의 최대 토큰 수([CLS], [SEP] 포함)
    stride: int = 128           # 이웃 조각끼리 겹치는 토큰 수
    epochs: int = 3
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1   # 전체 스텝 중 학습률을 0에서 올리는 구간 비율
    doc_batch: int = 8          # 한 배치에 넣는 대화 수(조각 수가 아님)
    pos_weight_power: float = 0.75  # pos_weight = (음성/양성)^power
    grad_clip: float = 1.0
    seed: int = 42
