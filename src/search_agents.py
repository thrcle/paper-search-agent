# 검색 로직 (BM25 + Dense + RRF Fusion + 요약)



from collections import defaultdict
from typing import List, Dict, Any
from state import AgentState
from settings import (
    es,
    ES_INDEX,
    TITLE_FIELD,
    CONTENT_FIELD,
    YEAR_FIELD,
    CITATION_FIELD,
    EMBED_FIELD,
    TOP_K_SPARSE,
    TOP_K_DENSE,
    TOP_K_FINAL,
)
from model_utils import get_embedding, call_llm_json, call_llm_text


# --- 1. 인텐트 분류 ---
def classify_utterance_agent(state: AgentState) -> AgentState:
    query = state["query_text"].strip()

    system_prompt = """
너는 논문 검색 에이전트의 인텐트 분류기다.
유저의 질의를 아래 중 하나로 분류하라.
- KEYWORD_TOPIC
- NL_TOPIC
- SPECIFIC_PAPER
반드시 JSON으로 {"utterance_type": "..."} 형식으로 출력.
"""
    result = call_llm_json(system_prompt, f"질문: {query}")
    return {"utterance_type": result.get("utterance_type", "NL_TOPIC")}


# --- 2. 검색 전략 결정 ---
def strategy_agent(state: AgentState) -> AgentState:
    utter = state["utterance_type"]
    if utter == "KEYWORD_TOPIC":
        return {"search_strategy": "sparse"}
    elif utter == "SPECIFIC_PAPER":
        return {"search_strategy": "hybrid"}
    else:
        return {"search_strategy": "hybrid"}


# --- 3. BM25 검색 ---

"""
이 BM25 검색에서는 Elasticsearch의 function_score를 이용해
논문 인용수(citations)에 가중치를 부여함.

- 기존 BM25 점수: 텍스트 유사도 기반
- field_value_factor: 인용수 필드 값으로 추가 점수 부여
  - factor: 0.001 → 인용수 1000당 약 +1점
  - modifier: "log1p" → 인용수 증가에 따라 점수는 완만하게 상승(log scale)
- boost_mode: "sum" → BM25 점수 + 인용수 점수를 합산

즉, 유사도가 비슷한 두 논문 중에서는 인용수가 높은 논문이
조금 더 상위로 랭크되도록 조정한 구조.
"""


# def keyword_search_agent(state: AgentState) -> AgentState:
#     query = state["query_text"]
#     strategy = state.get("search_strategy", "hybrid")
#     if strategy not in ("sparse", "hybrid"):
#         return {"keyword_hits": []}

#     body = {
#         "size": TOP_K_SPARSE,
#         "query": {"multi_match": {"query": query, "fields": [TITLE_FIELD, CONTENT_FIELD]}},
#     }
#     resp = es.search(index=ES_INDEX, body=body)
#     hits = resp["hits"]["hits"]

#     return {
#         "keyword_hits": [
#             {
#                 "id": h["_id"],
#                 "score": h["_score"],
#                 "title": h["_source"].get(TITLE_FIELD),
#                 "content": h["_source"].get(CONTENT_FIELD),
#                 "year": h["_source"].get(YEAR_FIELD),
#                 "citations": h["_source"].get(CITATION_FIELD, 0),
#             }
#             for h in hits
#         ]
#     }
# --- 3. BM25 검색 (인용수 가중치 반영) ---
def keyword_search_agent(state: AgentState) -> AgentState:
    query = state["query_text"]
    strategy = state.get("search_strategy", "hybrid")
    if strategy not in ("sparse", "hybrid"):
        return {"keyword_hits": []}

    # citations 가중치를 주는 function_score 쿼리
    body = {
        "size": TOP_K_SPARSE,
        "query": {
            "function_score": {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": [TITLE_FIELD, CONTENT_FIELD],
                        "type": "best_fields",
                    }
                },
                "boost_mode": "sum",      # 텍스트 점수 + 가중치 합산
                "score_mode": "sum",
                "functions": [
                    {
                        "field_value_factor": {
                            "field": CITATION_FIELD,
                            "factor": 0.001,        # 인용수 1000 = +1점 정도
                            "modifier": "log1p",    # 로그 스케일로 완화
                            "missing": 0
                        }
                    }
                ],
            }
        },
    }

    resp = es.search(index=ES_INDEX, body=body)
    hits = resp["hits"]["hits"]

    return {
        "keyword_hits": [
            {
                "id": h["_id"],
                "score": h["_score"],
                "title": h["_source"].get(TITLE_FIELD),
                "content": h["_source"].get(CONTENT_FIELD),
                "year": h["_source"].get(YEAR_FIELD),
                "citations": h["_source"].get(CITATION_FIELD, 0),
            }
            for h in hits
        ]
    }



# --- 4. Dense 검색 ---
def semantic_search_agent(state: AgentState) -> AgentState:
    query = state["query_text"]
    strategy = state.get("search_strategy", "hybrid")
    if strategy not in ("dense", "hybrid"):
        return {"semantic_hits": []}

    q_vec = get_embedding(query)
    body = {
        "size": TOP_K_DENSE,
        "knn": {"field": EMBED_FIELD, "query_vector": q_vec, "k": TOP_K_DENSE},
    }
    resp = es.search(index=ES_INDEX, body=body)
    hits = resp["hits"]["hits"]

    return {
        "semantic_hits": [
            {
                "id": h["_id"],
                "score": h["_score"],
                "title": h["_source"].get(TITLE_FIELD),
                "content": h["_source"].get(CONTENT_FIELD),
                "year": h["_source"].get(YEAR_FIELD),
                "citations": h["_source"].get(CITATION_FIELD, 0),
            }
            for h in hits
        ]
    }


# --- 5. RRF Fusion ---
def rrf_fusion(keyword_hits, semantic_hits, k=TOP_K_FINAL, k_rrf=60):
    scores = defaultdict(float)
    docs = {}
    for rank, d in enumerate(keyword_hits):
        scores[d["id"]] += 1 / (k_rrf + rank + 1)
        docs[d["id"]] = d
    for rank, d in enumerate(semantic_hits):
        scores[d["id"]] += 1 / (k_rrf + rank + 1)
        docs[d["id"]] = d

    ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
    return [docs[doc_id] for doc_id in ranked_ids]


# --- 6. 결과 요약 ---
def merge_and_select_agent(state: AgentState) -> AgentState:
    keyword_hits = state.get("keyword_hits") or []
    semantic_hits = state.get("semantic_hits") or []
    fused = rrf_fusion(keyword_hits, semantic_hits, k=TOP_K_FINAL)
    query = state.get("query_text", "")

    if not fused:
        return {"top_papers": [], "answer": "관련 논문을 찾지 못했어요."}

    papers_text = "\n\n".join(
        [
            f"{i+1}. 제목: {p['title']}\n   연도: {p['year']}, 인용수: {p['citations']}\n   내용: {p['content'][:300]}..."
            for i, p in enumerate(fused)
        ]
    )

    system_prompt = """
너는 논문 검색 어시스턴트다.
사용자의 질문과 후보 논문 목록을 보고 핵심 논문을 요약해라.
"""
    answer = call_llm_text(system_prompt, f"질문: {query}\n\n후보 논문:\n{papers_text}")
    return {"top_papers": fused, "answer": answer}
