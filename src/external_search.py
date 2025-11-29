# 외부 API 보완 모듈
# 평가 점수가 낮으면 외부 검색 API를 사용하여 추가 정보를 가져옴 


# external_search.py
import requests

def fetch_external_answer(query: str) -> str:
    """
    외부 검색 API(DuckDuckGo 등)를 이용해 보완 정보 가져오기.
    향후 arXiv / CrossRef / Semantic Scholar로 교체 가능.
    """
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        r = requests.get(url, timeout=8)
        data = r.json()
        abstract = data.get("AbstractText") or ""
        related = data.get("RelatedTopics", [])

        if abstract:
            return f"[보완정보]\n{abstract}"

        if related:
            extra = related[0].get("Text") or related[0].get("Result") or ""
            return f" [보완정보]\n{extra}"

        return "[보완정보 없음]"
    except Exception as e:
        print(f"[fetch_external_answer] Error: {e}")
        return " [보완정보 조회 실패]"
