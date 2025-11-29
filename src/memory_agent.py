# memory_agent.py
# 검색 결과 대화 메모리 관리/누적 

from state import AgentState

def memory_update_agent(state: AgentState) -> AgentState:
    memory = state.get("memory", []) or []
    query = state.get("query_text", "")
    answer = state.get("answer", "")
    if query and answer:
        memory.append({"query": query, "answer": answer})
    return {"memory": memory}
