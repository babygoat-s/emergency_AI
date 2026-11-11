"""판정관 LLM(Qwen3-8B)을 불러오고 A/B/C/D 확률을 구한다.

글을 생성하지 않고, 답 첫 글자 자리에서 A/B/C/D 확률만 읽는다(forward 한 번 → 빠르고, 같은 입력이면 같은 답).

- `load_judge`: 모델 + 토크나이저
- `score_prompts`: 프롬프트 목록 → (프롬프트 수, 4) 확률
"""
import numpy as np
import torch
from tqdm.auto import tqdm


def load_judge(model_name, load_in_4bit=False):
    """판정용 LLM과 토크나이저. GPU 메모리가 작으면(T4 등) load_in_4bit=True."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"   # 마지막 토큰 위치를 맞추려고 왼쪽에 패딩
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"dtype": torch.bfloat16}
    if torch.cuda.is_available():
        kw["device_map"] = "auto"
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw).eval()
    return tok, model


@torch.no_grad()
def score_prompts(prompts, tokenizer, model, letters, batch_size=8, max_tokens=6000, desc="Qwen 판정"):
    """프롬프트마다 답 글자(letters)의 확률. 반환 (프롬프트 수, 글자 수), 원래 순서.

    길이순으로 묶어 패딩을 줄인다. max_tokens보다 긴 프롬프트는 앞쪽을 잘라 끝의 질문이 남게 한다.
    """
    ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in letters]
    order = np.argsort([len(p) for p in prompts])
    out = np.zeros((len(prompts), len(letters)), dtype=np.float32)
    tokenizer.truncation_side = "left"
    for k in tqdm(range(0, len(order), batch_size), desc=desc):
        idx = order[k:k + batch_size]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True,
                        truncation=True, max_length=max_tokens).to(model.device)
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :]  # 답 첫 토큰 자리만 계산(메모리 절약)
        out[idx] = torch.softmax(logits[:, ids].float(), -1).cpu().numpy()  # 답 글자끼리만 정규화
    return out
