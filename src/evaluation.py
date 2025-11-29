# evaluation.py
# “전체 문장 한 방 비교” → “문장 단위 + 키워드 기준으로 쪼개서 평가”로 바꿔서 “주제만 비슷한 경우에는 페널티 부여

# 보완 로직 ①: 문장 단위 의미 유사도
# 효과
# 답변이 길어도 질문과 관련 없는 문장들은 낮은 점수에 기여
# 예를 들어:
# 질문: “추천 시스템”
# 답변 문장들:
# “이 논문은 대형 언어모델을 다룬다.” → 질문과 살짝만 관련 → 중간 점수
# “음성 복제 모델을 제안한다.” → 거의 무관 → 낮은 점수
# “비전-언어 멀티모달 분석을 다룬다.” → 또 무관 → 낮은 점수
# → 예전엔 이 전체를 한 덩어리로 비교해서 “AI 연구 관련 텍스트”로 높게 잡음 -> 관련 없는 문장들이 평균을 깎아먹어서 최종 점수가 내려가도록 개선 



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
