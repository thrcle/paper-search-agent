# main_agent.py
from state import AgentState
from search_agents import (
    classify_utterance_agent,
    strategy_agent,
    keyword_search_agent,
    semantic_search_agent,
    merge_and_select_agent,
)
from memory_agent import memory_update_agent
from evaluation import evaluate_answer_relevance
# from external_search import fetch_external_answer
from external_search import agentic_fetch_external_answer

def run_query(query: str, memory=None, threshold: float = 0.6) -> AgentState:
    """검색 → 요약 → 평가 → 보완 → 메모리 저장 파이프라인"""
    if memory is None:
        memory = []

    state: AgentState = {"query_text": query, "memory": memory}

    # 기본 검색 파이프라인
    state.update(classify_utterance_agent(state))
    state.update(strategy_agent(state))
    state.update(keyword_search_agent(state))
    state.update(semantic_search_agent(state))
    state.update(merge_and_select_agent(state))
    state.update(memory_update_agent(state))

    # # ---- 품질 평가 & 외부 보완 ----
    # answer = state.get("answer", "")
    # score = evaluate_answer_relevance(query, answer)
    # state["relevance_score"] = score

    # print(f"📊 Relevance Score: {score}")

    # if score < threshold:
    #     print("⚠️ 관련도 낮음 → 외부 검색 보완 실행")
    #     extra_info = fetch_external_answer(query)
    #     state["answer"] = f"{answer}\n\n{extra_info}"

    # ---- 품질 평가 & 외부 보완 ----
    # answer = state.get("answer", "")
    # utter_type = state.get("utterance_type", "NL_TOPIC")  # 👈 분류 결과 반영
    # score = evaluate_answer_relevance(query, answer, utter_type)  # 👈 동적 가중치 적용
    # state["relevance_score"] = score

    answer = state.get("answer", "")
    utter_type = state.get("utterance_type", "NL_TOPIC")
    strategy = state.get("search_strategy", "unknown")

    score = evaluate_answer_relevance(query, answer, utter_type)
    state["relevance_score"] = score

    print("─────────────────────────────────────────────")
    print(f"🧭 Search Strategy : {strategy}")
    print(f"💬 Utterance Type  : {utter_type}")
    print(f"📊 Relevance Score : {score:.4f}")
    print("─────────────────────────────────────────────")

    # if score < threshold:
    #     print("관련도 낮음 → 외부 검색 보완 실행")
    #     extra_info = fetch_external_answer(query)
    #     state["answer"] = f"{answer}\n\n{extra_info}"
    if score < threshold:
        print("⚠️ 관련도 낮음 → 외부 검색 보완 판단 중...")
        extra_info = agentic_fetch_external_answer(query, utter_type, score)
        state["answer"] = f"{answer}\n\n{extra_info}"



    return state


if __name__ == "__main__":
    # q = "최근 RAG retriever 성능 향상 관련 주요 논문 알려줘"  -- hybrid
    q = "추천 시스템의 최근 연구 경향은?"
    # q = "self-attention 에 대한 논문 알려줘"
    # q= "Transformer 모델의 한계점과 극복 방안에 대한 최신 연구 동향은?"
    # q="추천 시스템에 대한 최신 연구 동향과 주요 논문들을 알려줘"
    result = run_query(q)
    print(f"질문: {q}")
    print("\n[답변]\n", result["answer"])
    # print(f"\n📊 관련도 점수: {result.get('relevance_score')}")
    print("\n[상위 논문 목록]")
    for d in result.get("top_papers", []):
        print(f"- {d['title']} ({d['year']})")
