# evaluation.py
# “전체 문장 한 방 비교” → “문장 단위 + 키워드 기준으로 쪼개서 평가”로 바꿔서 “주제만 비슷한 경우에는 페널티 부여

"""
relevance 평가 로직 요약

기존에는 `질문 전체 vs 답변 전체`를 한 번에 임베딩해서
코사인 유사도로만 점수를 계산했기 때문에,
'추천 시스템'을 물어봤는데도 단순히 'AI / 논문 / 모델' 같은
넓은 토픽만 비슷해도 높은 점수가 나오는 한계 존재

보완로직

1) 문장 단위 semantic similarity:
   - 답변을 문장 단위로 분리한 뒤, 각 문장별로 질문과의 임베딩 유사도를 계산한다.
   - 질문과 관련 없는 문장이 섞여 있으면 평균 점수가 자연스럽게 떨어지도록 설계.

2) keyword precision:
   - 질문에서 핵심 키워드(의미 있는 단어들)를 추출하고,
     해당 키워드들이 답변에 실제로 얼마나 포함되어 있는지 비율로 계산한다.
   - 예를 들어 '추천 시스템'을 물었는데 답변에 '추천', 'recsys' 등이 전혀 없으면
     keyword precision이 0에 가까워지고 최종 점수가 낮아진다.

최종 점수는
    final_score = 0.7 * semantic_score + 0.3 * keyword_precision
으로 계산하여,
'주제만 대충 비슷한 헛소리'는 점수가 내려가고,
'질문의 핵심 키워드를 실제로 다루는 답변'이 더 높은 점수를 받도록 조정.
"""




import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from model_utils import get_embedding
import re
import nltk

# punkt 데이터 자동 다운로드 (한 번만 실행됨)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# nltk 문장 분리기 처음 실행 시 필요
# nltk.download('punkt', quiet=True)

# ---------------------------
# 1️⃣ 문장 단위 의미 유사도 평가
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
# 2️⃣ 키워드 기반 정밀도 평가
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
    stopwords = {"the", "is", "are", "was", "were", "and", "or", "a", "an", "to", "of", "in", "for"}
    q_keywords = [t for t in q_tokens if t not in stopwords and len(t) > 1]
    a_keywords = [t for t in a_tokens if len(t) > 1]

    if not q_keywords or not a_keywords:
        return 0.0

    matched = sum(1 for kw in q_keywords if kw in a_keywords)
    return matched / len(q_keywords)


# ---------------------------
# 3️⃣ 종합 평가
# ---------------------------
def evaluate_answer_relevance(query: str, answer: str) -> float:
    """
    문장단위 유사도 + 키워드 정밀도를 혼합해 최종 점수 산출.
    기본적으로 semantic 70% + keyword precision 30% 가중.
    """
    sem_score = evaluate_sentence_level(query, answer)
    kw_score = keyword_precision(query, answer)
    final_score = 0.7 * sem_score + 0.3 * kw_score
    return round(final_score, 4)
