# 검색 상태를 구조적으로 관리하는 타입 정의 

from typing import TypedDict, Literal, List, Dict, Any

UtterType = Literal["KEYWORD_TOPIC", "NL_TOPIC", "SPECIFIC_PAPER"]
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
