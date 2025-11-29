# evaluation.py
# “전체 문장 한 방 비교” → “문장 단위 + 키워드 기준으로 쪼개서 평가”로 바꿔서 “주제만 비슷한 경우에는 페널티 부여




"""
relevance 평가 로직 요약

1) 문장 단위 semantic similarity:
   - 답변을 문장 단위로 분리한 뒤, 각 문장별로 질문과의 임베딩 유사도를 계산한다.
   - 질문과 관련 없는 문장이 섞여 있으면 평균 점수가 자연스럽게 떨어지도록 설계.

2) keyword precision:
   - 질문에서 핵심 키워드(의미 있는 단어들)를 추출하고,
     해당 키워드들이 답변에 실제로 얼마나 포함되어 있는지 비율로 계산한다.

3) Agentic weight decision:
   - utterance_type, 질의 내용, 답변 길이/형태를 LLM에 넘겨서
     semantic vs keyword 가중치를 동적으로 조정한다.
   - 예) KEYWORD_TOPIC이면 키워드 비중↑, NL_TOPIC이면 semantic 비중↑ 등,
     기본 규칙을 LLM이 상황에 따라 보정하도록 한다.

최종 점수:
    final_score = w_sem * semantic_score + w_kw * keyword_precision
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from model_utils import get_embedding, call_llm_json
import re
import nltk

# punkt 데이터 자동 다운로드 (한 번만 실행됨)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# ---------------------------
# 1️ 문장 단위 의미 유사도 평가
# ---------------------------
def evaluate_sentence_level(query: str, answer: str) -> float:
    """
    답변을 문장 단위로 분리 후 각 문장의 의미 유사도를 계산.
    평균값으로 전체 답변의 관련도 점수를 산출.
    """
    if not query or not answer:
        return 0.0

    try:
        sents = nltk.sent_tokenize(answer)
        if not sents:
            return 0.0

        q_vec = np.array(get_embedding(query)).reshape(1, -1)
        sent_scores = []
        for s in sents:
            a_vec = np.array(get_embedding(s)).reshape(1, -1)
            score = cosine_similarity(q_vec, a_vec)[0][0]
            sent_scores.append(score)

        if not sent_scores:
            return 0.0
        return float(np.mean(sent_scores))
    except Exception as e:
        print(f"[evaluate_sentence_level] Error: {e}")
        return 0.0


# ---------------------------
# 2️ 키워드 기반 정밀도 평가
# ---------------------------
def keyword_precision(query: str, answer: str, top_k: int = 5) -> float:
    """
    질문에서 주요 키워드를 추출하고,
    답변에 얼마나 포함되어 있는지 비율로 측정.
    단순한 bag-of-words precision 느낌.
    """
    # 알파벳/한글 기준 토큰화
    q_tokens = re.findall(r"[A-Za-z가-힣]+", query.lower())
    a_tokens = re.findall(r"[A-Za-z가-힣]+", answer.lower())

    # stopword 제거 (간단 버전)
    stopwords = {
        "the", "is", "are", "was", "were", "and", "or",
        "a", "an", "to", "of", "in", "for"
    }
    q_keywords = [t for t in q_tokens if t not in stopwords and len(t) > 1]
    a_keywords = [t for t in a_tokens if len(t) > 1]

    if not q_keywords or not a_keywords:
        return 0.0

    matched = sum(1 for kw in q_keywords if kw in a_keywords)
    return matched / len(q_keywords)


# ---------------------------
# 3️ Agentic 가중치 결정
# ---------------------------
def get_dynamic_weights_agentic(
    query: str,
    answer: str,
    utterance_type: str,
) -> tuple[float, float]:
    """
    LLM에게 질의/답변/질문 유형을 넘겨서
    semantic vs keyword 가중치를 동적으로 결정하게 함.

    - 항상 w_sem + w_kw = 1.0 이 되도록 normalize
    - 실패 시에는 안전한 기본값(기존 룰)을 사용
    """
    # 1) 기본값 (예전 하드코딩 규칙)
    if utterance_type == "KEYWORD_TOPIC":
        base_w_sem, base_w_kw = 0.3, 0.7
    elif utterance_type == "SPECIFIC_PAPER":
        base_w_sem, base_w_kw = 0.5, 0.5
    else:  # NL_TOPIC, MEMORY_QUERY 등
        base_w_sem, base_w_kw = 0.8, 0.2

    # MEMORY_QUERY 같은 경우는 사실 평가 의미가 거의 없으니,
    # 그냥 기본값 그대로 리턴하는 게 낫다.
    if utterance_type == "MEMORY_QUERY":
        return base_w_sem, base_w_kw

    # 2) LLM에게 context 전달
    try:
        answer_len = len(answer or "")
        sents_cnt = len(nltk.sent_tokenize(answer)) if answer else 0

        system_prompt = """
너는 QA 시스템의 '채점 기준 튜너'이다.
질문과 답변, 그리고 질문 유형(utterance_type)을 보고
다음 두 점수의 가중치를 어떻게 줄지 결정해라.

- semantic_score : 질문과 답변 문장의 의미적 유사도
- keyword_precision : 질문의 핵심 키워드들이 답변에 얼마나 잘 포함되어 있는지

규칙:
1. w_sem + w_kw 는 반드시 1.0 이 되어야 한다.
2. 각 가중치는 0.1 이상 0.9 이하 범위에서만 선택한다.
3. 아래 기준을 기본으로 하되, 상황에 따라 미세 조정해라.
   - KEYWORD_TOPIC : keyword 중심 → keyword 비중을 좀 더 높인다.
   - SPECIFIC_PAPER : 특정 논문/저자 → semantic 과 keyword 균형을 맞춘다.
   - NL_TOPIC : 자연어 질문 → semantic 비중을 좀 더 높인다.
4. 답변 길이가 매우 짧아 keyword 정보가 적으면 semantic 비중을 상대적으로 더 준다.
5. 질문과 답변이 너무 포괄적인 주제(예: "AI 개요")일 경우 keyword 비중을 조금 더 준다.

JSON 형식으로만 답하라. 예:
{
  "w_sem": 0.8,
  "w_kw": 0.2,
  "note": "자연어 설명 위주라 semantic 비중을 높였음"
}
"""
        user_prompt = f"""
[질문 유형]
{utterance_type}

[질문]
{query}

[답변(일부, 최대 500자)]
{(answer or '')[:500]}

[답변 길이]
chars={answer_len}, sentences={sents_cnt}
"""

        resp = call_llm_json(system_prompt, user_prompt)
        w_sem = float(resp.get("w_sem", base_w_sem))
        w_kw = float(resp.get("w_kw", base_w_kw))

        # 3) sanity check & normalize
        if w_sem < 0.1 or w_kw < 0.1:
            raise ValueError("weight too small")
        total = w_sem + w_kw
        if total <= 0:
            raise ValueError("weight sum <= 0")

        w_sem /= total
        w_kw /= total

        return w_sem, w_kw

    except Exception as e:
        print(f"[get_dynamic_weights_agentic] Error: {e}")
        # 실패 시 기존 규칙으로 fallback
        total = base_w_sem + base_w_kw
        return base_w_sem / total, base_w_kw / total


# ---------------------------
# 4️ 종합 평가 (Agentic Weight 적용)
# ---------------------------
def evaluate_answer_relevance(
    query: str,
    answer: str,
    utterance_type: str = "NL_TOPIC"
) -> float:
    """
    질문 유형 + LLM 기반 agentic decision 으로
    semantic / keyword 가중치를 동적으로 조절해 최종 점수 계산.
    점수 스케일을 완화해서, '어느 정도 관련 있는 답변'은
    너무 0 근처로 떨어지지 않도록 보정한다.
    """
    # 답변이 거의 없으면 그냥 0
    if not answer or not answer.strip():
        return 0.0

    # 기본 점수 계산
    sem_score = evaluate_sentence_level(query, answer)
    kw_score = keyword_precision(query, answer)

    # LLM 기반 가중치 (실패 시 기본값)
    w_sem, w_kw = get_dynamic_weights_agentic(query, answer, utterance_type)

    raw = w_sem * sem_score + w_kw * kw_score
    # 혹시라도 범위 나가면 [0, 1]로 클램프
    raw = max(0.0, min(raw, 1.0))

    # -------------------------
    # 점수 완화 로직
    # -------------------------
    # 1) semantic/keyword 둘 다 거의 0이면 → 진짜 무관한 답변이므로
    #    0~0.2 사이에서만 움직이도록 유지
    if sem_score < 0.05 and kw_score < 0.05:
        final = 0.2 * raw      # 0 ~ 0.2

    else:
        # 2) 그 외 "조금이라도 관련 있음" → 최소 0.3은 보장
        #    raw(0~1)를 0.3~1.0으로 선형 맵핑
        #    raw=0  -> 0.3
        #    raw=1  -> 1.0
        final = 0.3 + 0.7 * raw

    return round(final, 4)
