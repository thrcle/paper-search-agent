# 외부 API 보완 모듈
# 평가 점수가 낮으면 외부 검색 API를 사용하여 추가 정보를 가져옴 


# external_search.py
"""
Agentic External Search Module
------------------------------
질문(query)의 의도와 관련도 점수(relevance_score)에 따라
외부 학술 API를 agentic하게 선택하여 보완 정보를 제공

- Semantic Scholar:
    정리된 학술 논문 데이터, 리뷰/이론 중심 보완에 적합.
- OpenAlex:
    최신 논문 커버리지 및 citation network 강점, 트렌드형 질문에 적합.

전략 예시:
- "최신", "recent" → OpenAlex (최신 논문 위주)
- "리뷰", "survey" → Semantic Scholar
- "인용", "citations" → OpenAlex
- score < 0.5 → 두 API 병행
- score 높음 → 보완 생략
"""

import requests

# -----------------------------
# API Endpoint 정의
# -----------------------------
SEMANTIC_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_URL = "https://api.openalex.org/works"


# -----------------------------
# Semantic Scholar
# -----------------------------
def fetch_external_answer_from_semantic(query: str) -> str:
    """Semantic Scholar에서 관련 논문 3개 가져오기"""
    try:
        params = {
            "query": query,
            "limit": 3,
            "fields": "title,year,abstract,url,citationCount",
        }
        resp = requests.get(SEMANTIC_URL, params=params, timeout=8)
        if resp.status_code != 200:
            print(f"[Semantic] HTTP {resp.status_code}")
            return ""

        papers = resp.json().get("data", [])
        if not papers:
            return ""

        lines = []
        for i, p in enumerate(papers, start=1):
            title = p.get("title") or "제목 없음"
            year = p.get("year") or "연도 미상"
            citations = p.get("citationCount", 0)
            abstract = (p.get("abstract") or "").strip()
            url = p.get("url") or ""

            line = f"{i}. {title} ({year}, 인용수: {citations})"
            if abstract:
                line += f"\n   {abstract[:220]}..."
            if url:
                line += f"\n   링크: {url}"

            lines.append(line)

        return "🔍 [보완정보 - Semantic Scholar]\n" + "\n\n".join(lines)

    except Exception as e:
        print(f"[fetch_external_answer_from_semantic] Error: {e}")
        return ""


# -----------------------------
# OpenAlex
# -----------------------------
def fetch_external_answer_from_openalex(query: str) -> str:
    """OpenAlex API에서 관련 논문 3개 가져오기"""
    try:
        params = {"search": query, "per-page": 3}
        resp = requests.get(OPENALEX_URL, params=params, timeout=8)
        if resp.status_code != 200:
            print(f"[OpenAlex] HTTP {resp.status_code}")
            return ""

        results = resp.json().get("results", [])
        if not results:
            return ""

        lines = []
        for i, r in enumerate(results, start=1):
            title = r.get("title") or "제목 없음"
            year = r.get("publication_year") or "연도 미상"
            url = r.get("id", "").replace("https://openalex.org/", "https://doi.org/")
            cited_by = r.get("cited_by_count", 0)

            authors = [a["author"]["display_name"] for a in r.get("authorships", [])]
            author_str = ", ".join(authors[:3]) + (" 외" if len(authors) > 3 else "")

            line = f"{i}. {title} ({year}, 인용수: {cited_by})"
            if author_str:
                line += f"\n   저자: {author_str}"
            if url:
                line += f"\n   링크: {url}"

            lines.append(line)

        return "🔍 [보완정보 - OpenAlex]\n" + "\n\n".join(lines)

    except Exception as e:
        print(f"[fetch_external_answer_from_openalex] Error: {e}")
        return ""


# -----------------------------
# Agentic 전략 결정 함수
# -----------------------------
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def choose_external_strategy(query: str, utter_type: str, relevance_score: float) -> str:
    """
    LLM 기반 agentic 판단:
    질문의 의도, 유형, 관련도 점수를 LLM에게 요약 전달하고
    어떤 외부 API를 활용해야 할지 reasoning하도록 함.
    """
    prompt = f"""
    당신은 논문 검색 보조 에이전트입니다.
    사용자의 질문에 따라 어떤 외부 학술 API를 사용해야 할지 판단하세요.

    사용할 수 있는 옵션은 다음 중 하나입니다:
    - "semantic": 이론/정리형 질문 → Semantic Scholar 사용
    - "openalex": 최신/트렌드형 질문 → OpenAlex 사용
    - "both": 보완이 많이 필요하거나 불확실할 때 두 API 모두 사용
    - "none": 이미 충분한 정보가 있을 때 외부 보완 불필요

    아래 정보를 바탕으로 판단하세요:
    - 질문: {query}
    - 질문 유형(utter_type): {utter_type}
    - 현재 관련도 점수(relevance_score): {relevance_score:.3f}

    응답 형식은 JSON으로만 주세요. 예시:
    {{ "strategy": "openalex", "reason": "최신 연구 중심 질문" }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 외부 학술 API 선택 에이전트야."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content.strip()
        # 안전하게 파싱
        import json
        data = json.loads(content)

        strategy = data.get("strategy", "none").lower()
        reason = data.get("reason", "")
        print(f"🤖 [LLM Decision] strategy={strategy}, reason={reason}")
        return strategy

    except Exception as e:
        print(f"[choose_external_strategy LLM Error] {e}")
        # LLM 실패 시 fallback
        return "both" if relevance_score < 0.5 else "semantic"



# -----------------------------
# Agentic 외부 검색 통합 함수
# -----------------------------
def agentic_fetch_external_answer(query: str, utter_type: str, relevance_score: float) -> str:
    """
    Agentic 판단 기반 외부 검색.
    - 전략 선택: choose_external_strategy()
    - 선택된 API 호출 후 결과 반환
    """
    strategy = choose_external_strategy(query, utter_type, relevance_score)
    print(f"🤖 External Search Strategy: {strategy}")

    if strategy == "semantic":
        return fetch_external_answer_from_semantic(query)
    elif strategy == "openalex":
        return fetch_external_answer_from_openalex(query)
    elif strategy == "both":
        s = fetch_external_answer_from_semantic(query)
        o = fetch_external_answer_from_openalex(query)
        return (s + "\n\n" + o) if (s or o) else "🔍 [보완정보] 두 API 모두 결과 없음"
    else:
        return "🔍 [보완정보] 별도의 보완 검색 불필요."
