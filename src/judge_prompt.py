"""Qwen에게 보낼 질문(프롬프트). 인프 프로젝트 04와 같은 질문, 구토·오심만 합친 뜻을 쓴다.

질문: 대화와 증상 하나를 주고 "신고자 쪽이 이 증상을 말했나?"를 A/B/C/D 중 하나로 묻는다.
  A = 호소함(직접 말했거나 같은 뜻의 표현. 대원 질문에 "네"라고 답한 경우 포함)
  B = 대원이 묻기만 했고, 신고자의 답이 없거나 분명하지 않음
  C = 언급 없음
  D = 신고자가 "없어요"라고 분명히 부정함
B와 D를 나눈 이유: 라벨을 고칠 때 D(부정)는 '없음'으로 고치고, B(대원 질문만)는 애매해서 그 칸을 뺀다.

- `build_prompt`: 대화·증상 → 모델에 넣을 문자열(Qwen3 생각 모드는 끔)
"""
from .config import SYMPTOM_DEF

CHOICES = ["A", "B", "C", "D"]
CHOICE_NAMES = {"A": "호소", "B": "대원질문", "C": "없음", "D": "부정"}

SYSTEM = "당신은 119 신고 전화 녹취록을 읽고 환자의 증상을 판정하는 응급의료 전문가입니다."

USER_TMPL = """다음은 119 신고 전화 녹취록입니다. '대원'은 119 상황요원, '신고자'는 전화를 건 사람(환자 본인 또는 보호자)입니다.

[녹취록]
{dialogue}

[질문]
이 통화에서 환자에게 '{symptom}' 증상이 있다고 신고자 쪽이 말했습니까?
'{symptom}'의 뜻: {definition}

A. 그렇다. 신고자가 그 증상을 직접 말했거나, 같은 뜻의 표현을 했다. (대원 질문에 "네"라고 답한 경우 포함)
B. 알 수 없다. 대원이 물어보기만 했고, 신고자의 답이 없거나 분명하지 않다.
C. 아니다. 그 증상에 대한 언급이 없다.
D. 아니다. 신고자가 그 증상이 없다고 분명히 부정했다.

A, B, C, D 중 하나만 답하세요."""


def build_user_prompt(dialogue, symptom):
    return USER_TMPL.format(dialogue=dialogue, symptom=symptom, definition=SYMPTOM_DEF.get(symptom, symptom))


def build_prompt(tokenizer, dialogue, symptom):
    """채팅 형식으로 감싼 최종 문자열. 답 첫 글자 자리에서 끝난다."""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user_prompt(dialogue, symptom)}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
