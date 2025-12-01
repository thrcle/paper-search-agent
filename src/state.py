# 검색 상태를 구조적으로 관리하는 타입 정의 

from typing import TypedDict, Literal, List, Dict, Any
from typing_extensions import NotRequired

UtterType = Literal["KEYWORD_TOPIC", "NL_TOPIC", "SPECIFIC_PAPER"]
# UtterType = Literal["KEYWORD_TOPIC", "NL_TOPIC", "SPECIFIC_PAPER", "MEMORY_QUERY", "FOLLOWUP_RECOMMEND"]

SearchStrategy = Literal["sparse", "dense", "hybrid"]

class AgentState(TypedDict, total=False):
    query_text: str
    utterance_type: UtterType
    search_strategy: SearchStrategy
    keyword_hits: List[Dict[str, Any]]
    semantic_hits: List[Dict[str, Any]]
    top_papers: List[Dict[str, Any]]
    answer: str
    memory: List[Dict[str, Any]]
    relevance_score: float
    reasoning_trace: NotRequired[List[str]]

def append_trace(state: AgentState, msg: str) -> None:
    """
    노드별 판단 과정을 남기기 위한 trace helper.

    - state["reasoning_trace"] 리스트에 msg를 순서대로 쌓는다.
    - 없으면 리스트를 새로 만들어서 초기화.
    """
    trace = state.get("reasoning_trace") or []
    # 방어적으로 복사해두면 참조 꼬이는 일 줄어듦
    trace = list(trace)
    trace.append(msg)
    state["reasoning_trace"] = trace

