# main_agent.py
# from state import AgentState
# from search_agents import (
#     classify_utterance_agent,
#     strategy_agent,
#     keyword_search_agent,
#     semantic_search_agent,
#     merge_and_select_agent,
# )
# from memory_agent import memory_update_agent
# from evaluation import evaluate_answer_relevance
# # from external_search import fetch_external_answer
# from external_search import agentic_fetch_external_answer
# from langgraph.graph import StateGraph, END

# # ──────────────────────────────────────────────
# # LangGraph용 래퍼 노드들 (시각화/추상 구조용)
# # 실제 실행은 run_query 가 담당
# # ──────────────────────────────────────────────

# def evaluate_answer_agent(state: AgentState) -> AgentState:
#     """evaluate_answer_relevance를 그래프 노드처럼 감싼 버전 (실제 실행은 안 됨)"""
#     # 여기서 진짜 평가까지 해도 되지만,
#     # 우리는 시각화만 쓸 거라 구조 표현만 중요함.
#     return state

# def external_search_agent(state: AgentState) -> AgentState:
#     """외부 검색 노드 자리 표시용 (실제로는 run_query 안에서 호출)"""
#     return state


# def build_graph():
#     """
#     ⚙️ main_agent.run_query의 논리 흐름을 LangGraph로 표현한 버전.
#     - 실행용이 아니라 '구조 시각화용' 이라고 생각하면 됨.
#     - 노드 이름/엣지 방향만 실제 파이프라인과 맞추는 게 핵심.
#     """
#     workflow = StateGraph(AgentState)

#     # 1) 노드 등록 — 이름은 실제 에이전트 함수와 맞춰줌
#     workflow.add_node("classify_utterance_agent", classify_utterance_agent)
#     workflow.add_node("strategy_agent", strategy_agent)
#     workflow.add_node("keyword_search_agent", keyword_search_agent)
#     workflow.add_node("semantic_search_agent", semantic_search_agent)
#     workflow.add_node("merge_and_select_agent", merge_and_select_agent)
#     workflow.add_node("evaluate_answer_agent", evaluate_answer_agent)
#     workflow.add_node("external_search_agent", external_search_agent)
#     workflow.add_node("memory_update_agent", memory_update_agent)

#     # 2) 진입 노드
#     workflow.set_entry_point("classify_utterance_agent")

#     # 3) 공통 흐름
#     workflow.add_edge("classify_utterance_agent", "strategy_agent")

#     # 4) 전략에 따른 분기
#     #   - KEYWORD_TOPIC  → sparse → keyword만
#     #   - NL_TOPIC       → hybrid → keyword+semantic+merge
#     #   - SPECIFIC_PAPER → hybrid → 동일
#     # 실제 run_query 구조에 최대한 맞춰서 적어줘
#     workflow.add_conditional_edges(
#         "strategy_agent",
#         lambda s: s.get("search_strategy", "hybrid"),
#         {
#             "sparse": "keyword_search_agent",
#             "dense": "semantic_search_agent",
#             "hybrid": "merge_and_select_agent",  # 개념상 keyword+semantic 둘 다 쓴다는 의미
#         },
#     )

#     # hybrid 전략에서 keyword/semantic 둘 다 거쳐서 merge되는 흐름 표현
#     workflow.add_edge("keyword_search_agent", "merge_and_select_agent")
#     workflow.add_edge("semantic_search_agent", "merge_and_select_agent")

#     # 5) 병합 → 평가
#     workflow.add_edge("merge_and_select_agent", "evaluate_answer_agent")

#     # 6) 평가 결과에 따라 외부 검색 여부
#     workflow.add_conditional_edges(
#         "evaluate_answer_agent",
#         lambda s: s.get("relevance_score", 1.0) < 0.6,
#         {
#             True: "external_search_agent",
#             False: "memory_update_agent",
#         },
#     )

#     # 7) 외부 검색 한 뒤에는 메모리 업데이트로 수렴
#     workflow.add_edge("external_search_agent", "memory_update_agent")

#     # 8) 마지막 노드에서 종료
#     workflow.add_edge("memory_update_agent", END)

#     return workflow.compile()




# def run_query(query: str, memory=None, threshold: float = 0.6) -> AgentState:
#     if memory is None:
#         memory = []

#     state: AgentState = {"query_text": query, "memory": memory}

#     # 1️⃣ 인텐트 분류
#     state.update(classify_utterance_agent(state))
#     utter_type = state.get("utterance_type", "NL_TOPIC")

#     # 2️⃣ 메모리 질의면 → 검색 건너뛰고 메모리 LLM만 호출
#     if utter_type == "MEMORY_QUERY":
#         state = memory_llm_agent(state)
#         # 원하면 이 질의도 history에 남기고 싶으면 아래 추가
#         state = memory_update_agent(state)
#         return state

#     # 3️⃣ RAG 파이프라인
#     state.update(strategy_agent(state))
#     state.update(keyword_search_agent(state))
#     state.update(semantic_search_agent(state))
#     state.update(merge_and_select_agent(state))

#     answer = state.get("answer", "")
#     strategy = state.get("search_strategy", "unknown")

#     # 4️⃣ 품질 평가
#     score = evaluate_answer_relevance(query, answer, utter_type)
#     state["relevance_score"] = score

#     print("─────────────────────────────────────────────")
#     print(f"🧭 Search Strategy : {strategy}")
#     print(f"💬 Utterance Type  : {utter_type}")
#     print(f"📊 Relevance Score : {score:.4f}")
#     print("─────────────────────────────────────────────")

#     # 5️⃣ 관련도 부족하면 외부 보완
#     if score < threshold:
#         print("⚠️ 관련도 낮음 → 외부 검색 보완 판단 중...")
#         extra_info = agentic_fetch_external_answer(query, utter_type, score)
#         state["answer"] = f"{answer}\n\n{extra_info}"

#     # 6️⃣ 마지막에 메모리 업데이트
#     state = memory_update_agent(state)
#     return state


# if __name__ == "__main__":
#     # q = "최근 RAG retriever 성능 향상 관련 주요 논문 알려줘"  -- hybrid
#     q = "추천 시스템의 최근 연구 경향은?"
#     # q = "self-attention 에 대한 논문 알려줘"
#     # q= "Transformer 모델의 한계점과 극복 방안에 대한 최신 연구 동향은?"
#     # q="추천 시스템에 대한 최신 연구 동향과 주요 논문들을 알려줘"
#     result = run_query(q)
#     print(f"질문: {q}")
#     print("\n[답변]\n", result["answer"])
#     # print(f"\n📊 관련도 점수: {result.get('relevance_score')}")
#     print("\n[상위 논문 목록]")
#     for d in result.get("top_papers", []):
#         print(f"- {d['title']} ({d['year']})")


# -----------------------------
#### ver 2
# # main_agent.py
# from state import AgentState
# from search_agents import (
#     classify_utterance_agent,
#     strategy_agent,
#     keyword_search_agent,
#     semantic_search_agent,
#     merge_and_select_agent,
# )
# from memory_agent import memory_update_agent
# from evaluation import evaluate_answer_relevance
# from external_search import agentic_fetch_external_answer
# from langgraph.graph import StateGraph, END


# # ──────────────────────────────────────────────
# # 1. LangGraph용 래퍼 노드들
# #    - 이제는 진짜 동작도 하게 바꿈 (이전엔 dummy)
# # ──────────────────────────────────────────────

# def evaluate_answer_agent(state: AgentState) -> AgentState:
#     query = state.get("query_text", "")
#     answer = state.get("answer", "")
#     utter_type = state.get("utterance_type", "NL_TOPIC")

#     print("\n[DEBUG] evaluate_answer_agent 호출됨")
#     print(f" - answer 존재 여부: {bool(answer)}")
#     print(f" - answer 길이: {len(answer)}")
#     print(f" - 일부 내용: {answer[:200]!r}")

#     score = evaluate_answer_relevance(query, answer, utter_type)
#     print(f" - relevance score 계산 결과: {score}")

#     state["relevance_score"] = score
#     return state



# def external_search_agent(state: AgentState) -> AgentState:
#     query = state.get("query_text", "")
#     utter_type = state.get("utterance_type", "NL_TOPIC")
#     relevance_score = state.get("relevance_score", 1.0)
#     extra_info = agentic_fetch_external_answer(query, utter_type, relevance_score)
#     answer = state.get("answer", "")

#     if extra_info:
#         state["answer"] = f"{answer}\n\n{extra_info}"

#     return state  # ✅ 반드시 state 전체 반환



# # ──────────────────────────────────────────────
# # 2. 그래프 정의: run_query의 흐름을 그대로 옮긴 버전
# # ──────────────────────────────────────────────

# def build_graph(threshold: float = 0.6):
#     """
#     run_query()의 실제 실행 순서를 LangGraph로 그대로 옮긴 그래프.

#     순서:
#     1) classify_utterance_agent
#     2) strategy_agent
#     3) keyword_search_agent
#     4) semantic_search_agent
#     5) merge_and_select_agent
#     6) memory_update_agent        ← run_query와 맞춘 포인트
#     7) evaluate_answer_agent
#     8) (relevance_score < threshold) → external_search_agent
#        (그 외) → END
#     """
#     workflow = StateGraph(AgentState)

#     # 노드 등록
#     workflow.add_node("classify_utterance_agent", classify_utterance_agent)
#     workflow.add_node("strategy_agent", strategy_agent)
#     workflow.add_node("keyword_search_agent", keyword_search_agent)
#     workflow.add_node("semantic_search_agent", semantic_search_agent)
#     workflow.add_node("merge_and_select_agent", merge_and_select_agent)
#     workflow.add_node("memory_update_agent", memory_update_agent)
#     workflow.add_node("evaluate_answer_agent", evaluate_answer_agent)
#     workflow.add_node("external_search_agent", external_search_agent)

#     # 진입 노드
#     workflow.set_entry_point("classify_utterance_agent")

#     # run_query 순서 그대로 엣지 연결
#     workflow.add_edge("classify_utterance_agent", "strategy_agent")
#     workflow.add_edge("strategy_agent", "keyword_search_agent")
#     workflow.add_edge("keyword_search_agent", "semantic_search_agent")
#     workflow.add_edge("semantic_search_agent", "merge_and_select_agent")
#     workflow.add_edge("merge_and_select_agent", "memory_update_agent")
#     workflow.add_edge("memory_update_agent", "evaluate_answer_agent")

#     # 평가 결과에 따라 외부 검색 여부 분기
#     def low_relevance(state: AgentState) -> bool:
#         return state.get("relevance_score", 1.0) < threshold

#     workflow.add_conditional_edges(
#         "evaluate_answer_agent",
#         low_relevance,
#         {
#             True: "external_search_agent",
#             False: END,  # 바로 종료
#         },
#     )

#     # 외부 검색 후 종료
#     workflow.add_edge("external_search_agent", END)

#     return workflow.compile()



# # ──────────────────────────────────────────────
# # 3. 기존 함수형 파이프라인 실행: run_query (그대로 유지)
# # ──────────────────────────────────────────────

# def run_query(query: str, memory=None, threshold: float = 0.6) -> AgentState:
#     """검색 → 요약 → 평가 → 보완 → 메모리 저장 파이프라인"""
#     if memory is None:
#         memory = []

#     state: AgentState = {"query_text": query, "memory": memory}

#     # 기본 검색 파이프라인
#     state.update(classify_utterance_agent(state))
#     state.update(strategy_agent(state))
#     state.update(keyword_search_agent(state))
#     state.update(semantic_search_agent(state))
#     state.update(merge_and_select_agent(state))
#     state.update(memory_update_agent(state))

#     # ---- 품질 평가 & 외부 보완 ----
#     answer = state.get("answer", "")
#     utter_type = state.get("utterance_type", "NL_TOPIC")
#     strategy = state.get("search_strategy", "unknown")

#     score = evaluate_answer_relevance(query, answer, utter_type)
#     state["relevance_score"] = score

#     print("─────────────────────────────────────────────")
#     print(f"🧭 Search Strategy : {strategy}")
#     print(f"💬 Utterance Type  : {utter_type}")
#     print(f"📊 Relevance Score : {score:.4f}")
#     print("─────────────────────────────────────────────")

#     if score < threshold:
#         print("⚠️ 관련도 낮음 → 외부 검색 보완 판단 중...")
#         extra_info = agentic_fetch_external_answer(query, utter_type, score)
#         state["answer"] = f"{answer}\n\n{extra_info}"

#     return state


# # ──────────────────────────────────────────────
# # 4. LangGraph 기반 실행 버전: run_query_graph (테스트용)
# # ──────────────────────────────────────────────

# def run_query_graph(query: str, memory=None, threshold: float = 0.6) -> AgentState:
#     """
#     LangGraph로 orchestrate하는 버전.
#     - build_graph(threshold)로 그래프를 만들고
#     - graph.invoke(initial_state) 한 번으로 전체 에이전트 흐름 실행.
#     """
#     if memory is None:
#         memory = []

#     graph = build_graph(threshold=threshold)
#     initial_state: AgentState = {"query_text": query, "memory": memory}

#     final_state: AgentState = graph.invoke(initial_state)

#     # ✅ 디버깅용 추가
#     print("\n[DEBUG] 최종 state keys:", final_state.keys())
#     print("[DEBUG] relevance_score:", final_state.get("relevance_score"))
#     if "_output" in final_state:
#         print("[DEBUG] _output keys:", final_state["_output"].keys())
#         print("[DEBUG] _output.relevance_score:", final_state["_output"].get("relevance_score"))


#     strategy = final_state.get("search_strategy", "unknown")
#     utter_type = final_state.get("utterance_type", "NL_TOPIC")
#     score = final_state.get("relevance_score", 0.0)

#     print("─────────────────────────────────────────────")
#     print(f"[GRAPH] 🧭 Search Strategy : {strategy}")
#     print(f"[GRAPH] 💬 Utterance Type  : {utter_type}")
#     print(f"[GRAPH] 📊 Relevance Score : {score:.4f}")
#     print("─────────────────────────────────────────────")

#     return final_state


# # ──────────────────────────────────────────────
# # 5. 직접 실행 테스트
# # ──────────────────────────────────────────────

# if __name__ == "__main__":
#     q = "추천 시스템의 최근 연구 경향은?"

#     # print("\n=== 기존 run_query 실행 ===")
#     # result = run_query(q)
#     # print(f"질문: {q}")
#     # print("\n[답변]\n", result["answer"])
#     # print("\n[상위 논문 목록]")
#     # for d in result.get("top_papers", []):
#     #     print(f"- {d['title']} ({d['year']})")

#     print("\n=== LangGraph run_query_graph 실행 ===")
#     result_g = run_query_graph(q)
#     print("\n[GRAPH 답변]\n", result_g.get("answer", ""))
#     print("\n[GRAPH 상위 논문 목록]")
#     for d in result_g.get("top_papers", []):
#         print(f"- {d['title']} ({d['year']})")

# -----------------------------
# main_agent.py
from langgraph.graph import StateGraph, END
from state import AgentState
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
    state["relevance_score"] = score
    return state


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

    # 3️⃣ 인텐트 기반 분기
    graph.add_conditional_edges(
        "classify_utterance_agent",
        lambda s: s.get("utterance_type", "NL_TOPIC"),
        {
            "MEMORY_QUERY": "memory_llm_agent",
            "FOLLOWUP_RECOMMEND": "semantic_search_agent",  # 확장 대비
            "NL_TOPIC": "strategy_agent",
            "KEYWORD_TOPIC": "strategy_agent",
            "SPECIFIC_PAPER": "strategy_agent",
        },
    )

    # 4️⃣ 검색 전략 분기
    graph.add_conditional_edges(
        "strategy_agent",
        lambda s: s.get("search_strategy", "hybrid"),
        {
            "sparse": "keyword_search_agent",
            "dense": "semantic_search_agent",
            "hybrid": "merge_and_select_agent",
        },
    )

    # 5️⃣ hybrid 병합
    graph.add_edge("keyword_search_agent", "merge_and_select_agent")
    graph.add_edge("semantic_search_agent", "merge_and_select_agent")

    # 6️⃣ 평가 및 외부 검색 조건 분기
    graph.add_edge("merge_and_select_agent", "evaluate_answer_agent")
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

    # 8️⃣ 메모리 답변 노드는 바로 종료
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

    state: AgentState = {"query_text": query, "memory": memory}

    # LangGraph 실행
    graph = build_graph()
    final_state = graph.invoke(state)

    # 콘솔용 로그
    answer = final_state.get("answer", "")
    utter_type = final_state.get("utterance_type", "NL_TOPIC")
    strategy = final_state.get("search_strategy", "unknown")
    score = final_state.get("relevance_score", 0.0)

    print("─────────────────────────────────────────────")
    print(f"💬 Utterance Type  : {utter_type}")
    print(f"🧭 Search Strategy : {strategy}")
    print(f"📊 Relevance Score : {score:.4f}")
    print("─────────────────────────────────────────────")

    return final_state


# ───────────────────────────────
# 단독 실행 테스트
# ───────────────────────────────
if __name__ == "__main__":
    memory = []
    q1 = "RAG 관련 논문 추천"
    q2 = "첫번째 논문이랑 비슷한 거 추천"
    q3 = "내 첫번째 질문이 뭐였지?"

    for q in [q1, q2, q3]:
        print(f"\n\n===== 🧠 질문: {q} =====")
        result = run_query(q, memory=memory)
        memory = result["memory"]
        print(f"\n[답변]\n{result['answer']}")
