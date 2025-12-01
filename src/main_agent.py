
# main_agent.py
from langgraph.graph import StateGraph, END
from state import AgentState, append_trace
from search_agents import (
    classify_utterance_agent,
    strategy_agent,
    keyword_search_agent,
    semantic_search_agent,
    merge_and_select_agent,
)
from memory_agent import memory_update_agent, memory_llm_agent
from evaluation import evaluate_answer_relevance
from external_search import agentic_fetch_external_answer


# ───────────────────────────────
# 개별 노드 함수 정의
# ───────────────────────────────

def evaluate_answer_agent(state: AgentState) -> AgentState:
    """검색 결과 품질 평가 노드"""
    query = state["query_text"]
    answer = state.get("answer", "")
    utter_type = state.get("utterance_type", "NL_TOPIC")

    score = evaluate_answer_relevance(query, answer, utter_type)
    # print("⚠️ 답변 관련도 평가 완료:", score)
    # state["relevance_score"] = score
    # return state

    # 판단 - 로그
    append_trace(
        state,
        f"[evaluate_answer] type={utter_type}, score={score:.4f}, "
        f"query={query[:40]!r}, answer_snippet={answer[:60]!r}",
    )
    return {
        "relevance_score": score,
        "reasoning_trace": state.get("reasoning_trace")  # 추가

    }

def external_search_agent(state: AgentState) -> AgentState:
    """검색 결과가 부족할 때 외부 API(Tavily/OpenAlex) 호출"""
    query = state["query_text"]
    utter_type = state.get("utterance_type", "NL_TOPIC")
    score = state.get("relevance_score", 0.0)

    print("⚠️ 관련도 낮음 → 외부 검색 보완 판단 중...")
    extra_info = agentic_fetch_external_answer(query, utter_type, score)
    state["answer"] = f"{state.get('answer', '')}\n\n{extra_info}"
    return state


# ───────────────────────────────
# LangGraph Workflow 정의
# ───────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # 1️⃣ 노드 등록
    graph.add_node("classify_utterance_agent", classify_utterance_agent)
    graph.add_node("memory_llm_agent", memory_llm_agent)
    graph.add_node("strategy_agent", strategy_agent)
    graph.add_node("keyword_search_agent", keyword_search_agent)
    graph.add_node("semantic_search_agent", semantic_search_agent)
    graph.add_node("merge_and_select_agent", merge_and_select_agent)
    graph.add_node("evaluate_answer_agent", evaluate_answer_agent)
    graph.add_node("external_search_agent", external_search_agent)
    graph.add_node("memory_update_agent", memory_update_agent)

    # 2️⃣ 시작 노드
    graph.set_entry_point("classify_utterance_agent")

    # 3️⃣ 인텐트 기반 1차 분기
    graph.add_conditional_edges(
        "classify_utterance_agent",
        lambda s: s.get("utterance_type", "NL_TOPIC"),
        {
            # 메모리 질의면 검색 건너뛰고 메모리 LLM으로
            "MEMORY_QUERY": "memory_llm_agent",

            # 나머지 유형은 다 검색 파이프라인 태움
            "FOLLOWUP_RECOMMEND": "strategy_agent",
            "NL_TOPIC": "strategy_agent",
            "KEYWORD_TOPIC": "strategy_agent",
            "SPECIFIC_PAPER": "strategy_agent",
        },
    )

    # 4️⃣ 공통 검색 파이프라인
    #    - strategy_agent가 search_strategy만 설정
    #    - 각 search_agent 내부에서 strategy 보고 동작/무동작 결정
    graph.add_edge("strategy_agent", "keyword_search_agent")
    graph.add_edge("keyword_search_agent", "semantic_search_agent")
    graph.add_edge("semantic_search_agent", "merge_and_select_agent")

    # 5️⃣ merge → 평가
    graph.add_edge("merge_and_select_agent", "evaluate_answer_agent")

    # 6️⃣ 평가 결과에 따라 외부 검색 여부
    graph.add_conditional_edges(
        "evaluate_answer_agent",
        lambda s: s.get("relevance_score", 1.0) < 0.6,
        {
            True: "external_search_agent",
            False: "memory_update_agent",
        },
    )

    # 7️⃣ 외부 검색 후에도 메모리 업데이트
    graph.add_edge("external_search_agent", "memory_update_agent")

    # 8️⃣ 메모리 답변도 업데이트 후 종료
    graph.add_edge("memory_llm_agent", "memory_update_agent")

    # 9️⃣ 종료점
    graph.add_edge("memory_update_agent", END)

    return graph.compile()


# ───────────────────────────────
# 실행 함수 (Multi-Turn용)
# ───────────────────────────────
def run_query(query: str, memory=None, threshold: float = 0.6) -> AgentState:
    if memory is None:
        memory = []

    # 초기 상태
    state: AgentState = {"query_text": query, "memory": memory}

    # LangGraph 실행
    graph = build_graph()
    final_state: AgentState = graph.invoke(state) or {}

    # print("🧪 [run_query] === graph.invoke 결과 ===")
    # print(f"  - final_state keys      : {list(final_state.keys())}")
    # print(f"  - raw relevance_score   : {final_state.get('relevance_score')}")
    # print(f"  - raw utterance_type    : {final_state.get('utterance_type')}")
    # print(f"  - raw search_strategy   : {final_state.get('search_strategy')}")
    # print()

    # 기본 정보 추출
    answer = final_state.get("answer", "")
    utter_type = final_state.get("utterance_type", "NL_TOPIC")
    strategy = final_state.get("search_strategy", "unknown")
    score = final_state.get("relevance_score", 0.0)

    # ───────────────────────────────────
    # ✅ LLM 판단 요약 / 외부 검색 전략 주입
    # ───────────────────────────────────
    # 혹시 그래프 안에서 이미 넣어준 값이 있으면 우선 사용
    llm_decision = final_state.get("llm_decision")
    external_strategy = (
        final_state.get("external_strategy")
        or final_state.get("external_search_strategy")
        or strategy  # 없으면 검색 전략이랑 동일하게
    )

    # 없으면 간단한 reason 자동 생성
    if not llm_decision:
        if strategy == "sparse":
            reason = "짧은 키워드 중심 질의라 BM25 기반 키워드 검색을 우선 사용함"
        elif strategy == "dense":
            reason = "의미 기반 유사도가 중요하다고 판단해 dense vector 검색을 우선 사용함"
        elif strategy == "hybrid":
            reason = "키워드와 의미 정보를 모두 활용하는 것이 적합하다고 판단해 hybrid 검색을 사용함"
        else:
            reason = "기본 설정에 따라 검색 전략을 선택함"

        llm_decision = f"[LLM Decision] strategy={strategy}, reason={reason}"

    # 최종 state에 넣어주기 → app.py에서 사용
    final_state["llm_decision"] = llm_decision
    final_state["external_strategy"] = external_strategy

    # 콘솔용 로그
    print("─────────────────────────────────────────────")
    print(f"💬 Utterance Type  : {utter_type}")
    print(f"🧭 Search Strategy : {strategy}")
    print(f"📊 Relevance Score : {score:.4f}")
    print(f"🤖 LLM Decision    : {llm_decision}")
    print(f"🌐 External Search : {external_strategy}")
    print("─────────────────────────────────────────────")

    return final_state



# ───────────────────────────────
# 단독 실행 테스트
# ───────────────────────────────
if __name__ == "__main__":
    memory = []
    q1 = "RAG 관련 논문 추천"
    q2 = "llm 성능을 높이기 위한 방법 알려줘" # NL_TOPIC, hybrid 
    q3 = "내 첫번째 질문이 뭐였지?"       # 메모리 확인용 
    q4 = "RAG survey 논문"           # SPECIFIC_PAPER,hybrid
    q5 = "transformer "             # KEYWORD_TOPIC, sparse
    q6 ="RAG 성능 향상 최신 연구 알려줘"
    

    for q in [q1, q2, q3,q4, q5, q6]:
    # for q in [q1]:
        print(f"\n\n===== 🧠 질문: {q} =====")
        result = run_query(q, memory=memory)
        memory = result["memory"]
        print(f"\n[답변]\n{result['answer']}")
