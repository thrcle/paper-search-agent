# 검색 로직 (BM25 + Dense + RRF Fusion + 요약)



from collections import defaultdict
from typing import List, Dict, Any
from state import AgentState, append_trace
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
    URL_FIELD,   
)
from model_utils import get_embedding, call_llm_json, call_llm_text


# --- 1. 인텐트 분류 ---
def classify_utterance_agent(state: dict) -> dict:
    query = state["query_text"].strip()

    system_prompt = """
    너는 논문 검색 에이전트의 인텐트 분류기다.
    유저의 질의를 아래 타입 중 하나로 분류하라.

    - KEYWORD_TOPIC    : 키워드 위주의 짧은 검색어 (예: "RAG retrieval 비교", "BERT 추천 시스템")
    - NL_TOPIC         : 자연어 문장 형태의 정보 탐색 (예: "RAG 성능 향상 최신 연구 알려줘")
    - SPECIFIC_PAPER   : 특정 논문/저자/연도 등을 지목하는 경우 (예: "Attention is All You Need 논문")
    - MEMORY_QUERY     : 이전 대화/질문/추천 내역을 물어보는 경우
                        예) "내 첫번째 질문이 뭐였지?"
                            "방금 추천해 준 첫번째 논문 다시 알려줘"
                            "이전에 추천해준 논문들 요약해줘"

    반드시 JSON으로만 답하라.
    예시: {"utterance_type": "NL_TOPIC"}
"""
    result = call_llm_json(system_prompt, f"질문: {query}")
    # return {"utterance_type": result.get("utterance_type", "NL_TOPIC")}
    utter_type = result.get("utterance_type", "NL_TOPIC")

    # 판단 로그 남기기
    append_trace(state, f"[classify] query='{query[:40]}...', type={utter_type}")

    # 상태 갱신 후 리턴
    state["utterance_type"] = utter_type
    return state

# --- 2. 검색 전략 결정 ---
# def strategy_agent(state: AgentState) -> AgentState:
#     utter = state["utterance_type"]
#     if utter == "KEYWORD_TOPIC":
#         return {"search_strategy": "sparse"}
#     elif utter == "SPECIFIC_PAPER":
#         return {"search_strategy": "hybrid"}
#     else:
#         return {"search_strategy": "hybrid"}
def strategy_agent(state: AgentState) -> AgentState:
    utter = state["utterance_type"]

    if utter == "KEYWORD_TOPIC":
        strategy = "sparse"
    elif utter == "SPECIFIC_PAPER":
        strategy = "hybrid"
    else:
        strategy = "hybrid"

    append_trace(
        state,
        f"[strategy] utter_type={utter}, search_strategy={strategy}",
    )

    return {
        "search_strategy": strategy,
        "reasoning_trace": state.get("reasoning_trace"),
    }


# --- 3. BM25 검색 ---

"""
이 BM25 검색에서는 Elasticsearch의 function_score를 이용해
논문 인용수(citations)에 가중치를 부여함.

- 기존 BM25 점수: 텍스트 유사도 기반
- field_value_factor: 인용수 필드 값으로 추가 점수 부여
  - factor: 0.001 → 인용수 1000당 약 +1점
  - modifier: "log1p" → 인용수 증가에 따라 점수는 완만하게 상승(log scale)
- boost_mode: "sum" → BM25 점수 + 인용수 점수를 합산

유사도가 비슷한 두 논문 중에서는 인용수가 높은 논문이 조금 더 상위로 랭크되도록 조정
"""


# def keyword_search_agent(state: AgentState) -> AgentState:
#     query = state["query_text"]
#     strategy = state.get("search_strategy", "hybrid")
#     if strategy not in ("sparse", "hybrid"):
#         return {"keyword_hits": []}

#     # citations 가중치를 주는 function_score 쿼리
#     body = {
#         "size": TOP_K_SPARSE,
#         "query": {
#             "function_score": {
#                 "query": {
#                     "multi_match": {
#                         "query": query,
#                         "fields": [TITLE_FIELD, CONTENT_FIELD],
#                         "type": "best_fields",
#                     }
#                 },
#                 "boost_mode": "sum",      # 텍스트 점수 + 가중치 합산
#                 "score_mode": "sum",
#                 "functions": [
#                     {
#                         "field_value_factor": {
#                             "field": CITATION_FIELD,
#                             "factor": 0.001,        # 인용수 1000 = +1점 정도
#                             "modifier": "log1p",    # 로그 스케일로 완화
#                             "missing": 0
#                         }
#                     }
#                 ],
#             }
#         },
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
#                  "url": h["_source"].get(URL_FIELD, ""), 
#             }
#             for h in hits
#         ]
#     }

# --- 3. BM25 검색 (인용수 가중치 반영) ---
def keyword_search_agent(state: AgentState) -> AgentState:
    query = state["query_text"]
    strategy = state.get("search_strategy", "hybrid")

    if strategy not in ("sparse", "hybrid"):
        append_trace(
            state,
            f"[keyword_search] skipped (strategy={strategy})",
        )
        return {
            "keyword_hits": [],
            "reasoning_trace": state.get("reasoning_trace"),
        }

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
                "boost_mode": "sum",
                "score_mode": "sum",
                "functions": [
                    {
                        "field_value_factor": {
                            "field": CITATION_FIELD,
                            "factor": 0.001,
                            "modifier": "log1p",
                            "missing": 0,
                        }
                    }
                ],
            }
        },
    }

    resp = es.search(index=ES_INDEX, body=body)
    hits = resp["hits"]["hits"]

    keyword_hits = [
        {
            "id": h["_id"],
            "score": h["_score"],
            "title": h["_source"].get(TITLE_FIELD),
            "content": h["_source"].get(CONTENT_FIELD),
            "year": h["_source"].get(YEAR_FIELD),
            "citations": h["_source"].get(CITATION_FIELD, 0),
            "url": h["_source"].get(URL_FIELD, ""),
        }
        for h in hits
    ]

    append_trace(
        state,
        f"[keyword_search] strategy={strategy}, hits={len(keyword_hits)}",
    )

    return {
        "keyword_hits": keyword_hits,
        "reasoning_trace": state.get("reasoning_trace"),
    }




# --- 4. Dense 검색 ---
# def semantic_search_agent(state: AgentState) -> AgentState:
#     query = state["query_text"]
#     strategy = state.get("search_strategy", "hybrid")
#     if strategy not in ("dense", "hybrid"):
#         return {"semantic_hits": []}

#     q_vec = get_embedding(query)
#     body = {
#         "size": TOP_K_DENSE,
#         "knn": {"field": EMBED_FIELD, "query_vector": q_vec, "k": TOP_K_DENSE},
#     }
#     resp = es.search(index=ES_INDEX, body=body)
#     hits = resp["hits"]["hits"]

#     return {
#         "semantic_hits": [
#             {
#                 "id": h["_id"],
#                 "score": h["_score"],
#                 "title": h["_source"].get(TITLE_FIELD),
#                 "content": h["_source"].get(CONTENT_FIELD),
#                 "year": h["_source"].get(YEAR_FIELD),
#                 "citations": h["_source"].get(CITATION_FIELD, 0),
#                  "url": h["_source"].get(URL_FIELD, ""),
#             }
#             for h in hits
#         ]
#     }
# --- 4. Dense 검색 ---
def semantic_search_agent(state: AgentState) -> AgentState:
    query = state["query_text"]
    strategy = state.get("search_strategy", "hybrid")

    if strategy not in ("dense", "hybrid"):
        append_trace(
            state,
            f"[semantic_search] skipped (strategy={strategy})",
        )
        return {
            "semantic_hits": [],
            "reasoning_trace": state.get("reasoning_trace"),
        }

    q_vec = get_embedding(query)
    body = {
        "size": TOP_K_DENSE,
        "knn": {"field": EMBED_FIELD, "query_vector": q_vec, "k": TOP_K_DENSE},
    }
    resp = es.search(index=ES_INDEX, body=body)
    hits = resp["hits"]["hits"]

    semantic_hits = [
        {
            "id": h["_id"],
            "score": h["_score"],
            "title": h["_source"].get(TITLE_FIELD),
            "content": h["_source"].get(CONTENT_FIELD),
            "year": h["_source"].get(YEAR_FIELD),
            "citations": h["_source"].get(CITATION_FIELD, 0),
            "url": h["_source"].get(URL_FIELD, ""),
        }
        for h in hits
    ]

    append_trace(
        state,
        f"[semantic_search] strategy={strategy}, hits={len(semantic_hits)}",
    )

    return {
        "semantic_hits": semantic_hits,
        "reasoning_trace": state.get("reasoning_trace"),
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
# def merge_and_select_agent(state: AgentState) -> AgentState:
#     keyword_hits = state.get("keyword_hits") or []
#     semantic_hits = state.get("semantic_hits") or []
#     fused = rrf_fusion(keyword_hits, semantic_hits, k=TOP_K_FINAL)
#     query = state.get("query_text", "")

#     if not fused:
#         return {"top_papers": [], "answer": "관련 논문을 찾지 못했어요."}

#     papers_text = "\n\n".join(
#         [
#             f"{i+1}. 제목: {p['title']}\n   연도: {p['year']}, 인용수: {p['citations']}\n   내용: {p['content'][:300]}..."
#             for i, p in enumerate(fused)
#         ]
#     )

#     system_prompt = """
# 너는 논문 검색 어시스턴트다.
# 사용자의 질문과 후보 논문 목록을 보고 핵심 논문을 요약해라.
# """
#     answer = call_llm_text(system_prompt, f"질문: {query}\n\n후보 논문:\n{papers_text}")
#     return {"top_papers": fused, "answer": answer}
# --- 6. 결과 요약 ---
def merge_and_select_agent(state: AgentState) -> AgentState:
    keyword_hits = state.get("keyword_hits") or []
    semantic_hits = state.get("semantic_hits") or []
    fused = rrf_fusion(keyword_hits, semantic_hits, k=TOP_K_FINAL)
    query = state.get("query_text", "")

    if not fused:
        append_trace(
            state,
            "[merge] no fused results → answer='관련 논문을 찾지 못했어요.'",
        )
        return {
            "top_papers": [],
            "answer": "관련 논문을 찾지 못했어요.",
            "reasoning_trace": state.get("reasoning_trace"),
        }

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

    append_trace(
        state,
        f"[merge] fused={len(fused)}, answer_len={len(answer)}",
    )

    return {
        "top_papers": fused,
        "answer": answer,
        "reasoning_trace": state.get("reasoning_trace"),
    }
