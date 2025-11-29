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
from langgraph.graph import StateGraph, END

# ──────────────────────────────────────────────
# LangGraph용 래퍼 노드들 (시각화/추상 구조용)
# 실제 실행은 run_query 가 담당
# ──────────────────────────────────────────────

def evaluate_answer_agent(state: AgentState) -> AgentState:
    """evaluate_answer_relevance를 그래프 노드처럼 감싼 버전 (실제 실행은 안 됨)"""
    # 여기서 진짜 평가까지 해도 되지만,
    # 우리는 시각화만 쓸 거라 구조 표현만 중요함.
    return state

def external_search_agent(state: AgentState) -> AgentState:
    """외부 검색 노드 자리 표시용 (실제로는 run_query 안에서 호출)"""
    return state


def build_graph():
    """
    ⚙️ main_agent.run_query의 논리 흐름을 LangGraph로 표현한 버전.
    - 실행용이 아니라 '구조 시각화용' 이라고 생각하면 됨.
    - 노드 이름/엣지 방향만 실제 파이프라인과 맞추는 게 핵심.
    """
    workflow = StateGraph(AgentState)

    # 1) 노드 등록 — 이름은 실제 에이전트 함수와 맞춰줌
    workflow.add_node("classify_utterance_agent", classify_utterance_agent)
    workflow.add_node("strategy_agent", strategy_agent)
    workflow.add_node("keyword_search_agent", keyword_search_agent)
    workflow.add_node("semantic_search_agent", semantic_search_agent)
    workflow.add_node("merge_and_select_agent", merge_and_select_agent)
    workflow.add_node("evaluate_answer_agent", evaluate_answer_agent)
    workflow.add_node("external_search_agent", external_search_agent)
    workflow.add_node("memory_update_agent", memory_update_agent)

    # 2) 진입 노드
    workflow.set_entry_point("classify_utterance_agent")

    # 3) 공통 흐름
    workflow.add_edge("classify_utterance_agent", "strategy_agent")

    # 4) 전략에 따른 분기
    #   - KEYWORD_TOPIC  → sparse → keyword만
    #   - NL_TOPIC       → hybrid → keyword+semantic+merge
    #   - SPECIFIC_PAPER → hybrid → 동일
    # 실제 run_query 구조에 최대한 맞춰서 적어줘
    workflow.add_conditional_edges(
        "strategy_agent",
        lambda s: s.get("search_strategy", "hybrid"),
        {
            "sparse": "keyword_search_agent",
            "dense": "semantic_search_agent",
            "hybrid": "merge_and_select_agent",  # 개념상 keyword+semantic 둘 다 쓴다는 의미
        },
    )

    # hybrid 전략에서 keyword/semantic 둘 다 거쳐서 merge되는 흐름 표현
    workflow.add_edge("keyword_search_agent", "merge_and_select_agent")
    workflow.add_edge("semantic_search_agent", "merge_and_select_agent")

    # 5) 병합 → 평가
    workflow.add_edge("merge_and_select_agent", "evaluate_answer_agent")

    # 6) 평가 결과에 따라 외부 검색 여부
    workflow.add_conditional_edges(
        "evaluate_answer_agent",
        lambda s: s.get("relevance_score", 1.0) < 0.6,
        {
            True: "external_search_agent",
            False: "memory_update_agent",
        },
    )

    # 7) 외부 검색 한 뒤에는 메모리 업데이트로 수렴
    workflow.add_edge("external_search_agent", "memory_update_agent")

    # 8) 마지막 노드에서 종료
    workflow.add_edge("memory_update_agent", END)

    return workflow.compile()




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



    # ---- 품질 평가 & 외부 보완 ----
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
