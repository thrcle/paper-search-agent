
#  대화 기반 논문 검색 에이전트에서 "대화 메모리" 관리
#  사용자의 이전 질의(query_text), 답변(answer), 추천 논문(top_papers)을 저장하고 다음 턴에서 "과거 대화 참조형 질의"가 들어오면, 검색 없이 메모리만으로 응답 생성
#
# 핵심 흐름:
#   (1) memory_update_agent  →  대화 기록을 state["memory"]에 누적 저장
#   (2) memory_llm_agent     →  저장된 memory를 기반으로 LLM이 답변 생성
#
#  RAG 파이프라인에서 검색 후 단계(Post-processing)에 들어가거나, 과거 질의 참조(MEMORY_QUERY) 시에만 실행되는 보조 agent




from typing import Dict, Any, List
from model_utils import call_llm_text  # 이미 쓰고 있는 LLM 호출 함수라고 가정

AgentState = Dict[str, Any]


def memory_update_agent(state: AgentState) -> AgentState:
    """검색/답변이 끝난 후, 이번 턴 정보를 memory에 쌓는 agent."""
    history: List[dict] = state.get("memory", []) or []

    entry = {
        "query_text": state.get("query_text"),
        "answer": state.get("answer"),
        "top_papers": state.get("top_papers", []),
    }
    history.append(entry)
    state["memory"] = history
    return state


def memory_llm_agent(state: AgentState) -> AgentState:
    """
    MEMORY_QUERY인 경우:
    - Elasticsearch / 임베딩 검색은 건너뛰고
    - 지금까지의 memory만 가지고 LLM이 답하도록 하는 agent.
    """
    query = state.get("query_text", "")
    history: List[dict] = state.get("memory", []) or []

    if not history:
        state["answer"] = "아직 이전 질문/추천 기록이 없어서 기억해 줄 내용이 없어."
        return state

    # 히스토리를 LLM이 보기 좋게 텍스트로 펼치기
    lines = []
    for i, h in enumerate(history, start=1):
        q = h.get("query_text") or ""
        a = h.get("answer") or ""
        lines.append(f"[{i}] 질문: {q}\n    답변: {a[:200]}...")

    history_text = "\n\n".join(lines)

    system_prompt = """
너는 논문 검색 에이전트의 메모리 관리자다.

- 아래는 지금까지의 대화에서 사용자가 했던 질문과 그에 대한 답변 기록이다.
- 사용자의 현재 질문이 "내 첫번째 질문이 뭐였지?", "방금 추천해 준 첫번째 논문 다시 알려줘" 처럼
  과거 대화를 참조하는 경우, 검색을 새로 하지 말고 아래 history만 보고 답해라.

- 가능한 한:
  - "첫 번째 질문"이면 history에서 가장 처음 질문을 찾아서 알려주고,
  - "마지막 질문"이면 가장 최근 질문을 알려주고,
  - "이전에 추천해 준 논문들"이면 그때의 질문과 맥락을 같이 설명해줘라.
"""

    user_prompt = f"""
[사용자의 현재 질문]
{query}

[지금까지의 질문/답변 기록]
{history_text}

위 기록만을 사용해서, 사용자의 현재 질문에 자연스럽게 한국어로 답변해줘.
새로운 논문 검색 결과를 지어내지 말고, history 안에 있는 내용만 가지고 답해.
"""

    answer = call_llm_text(system_prompt, user_prompt)
    state["answer"] = answer
    return state
