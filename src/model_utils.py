# model_utils.py
# 임베딩 + LLM 호출 유틸리티 함수들 정의 


import json
from sentence_transformers import SentenceTransformer
from settings import client

OPENAI_MODEL_TEXT = "gpt-4.1-mini"
OPENAI_MODEL_JSON = "gpt-4.1-mini"

# BGE 임베딩 모델 로드
bge_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def get_embedding(text: str) -> list[float]:
    """텍스트를 1024차원 임베딩으로 변환"""
    return bge_model.encode(text, normalize_embeddings=True).tolist()

def call_llm_text(system_prompt: str, user_prompt: str, model: str | None = None) -> str:
    """LLM을 호출해 자연어 응답을 생성"""
    model = model or OPENAI_MODEL_TEXT
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content or ""

def call_llm_json(system_prompt: str, user_prompt: str, model: str | None = None) -> dict:
    """LLM을 호출해 JSON 응답을 생성"""
    model = model or OPENAI_MODEL_JSON
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}
