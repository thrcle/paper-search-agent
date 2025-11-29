# # 논문 -> 임베딩 -> es 인덱싱 

# # ingest_papers.py
# import json
# from pathlib import Path
# from typing import List, Dict, Any

# from elasticsearch import helpers

# from settings import (
#     client, es, ES_INDEX, EMBED_FIELD,
#     TITLE_FIELD, ABSTRACT_FIELD, CONTENT_FIELD,
#     YEAR_FIELD, CITATION_FIELD, VENUE_FIELD, URL_FIELD,
# )

# # 인덱스 매핑이 없을 경우 생성
# from es_utils import create_papers_index

# DATA_PATH = Path("data/papers.jsonl")

# # 각 논문(제목 + 초록) 텍스트를 OpenAI text-embedding-3-large 모델로 임베딩 생성
# def get_embeddings(texts: List[str]) -> List[List[float]]:
#     resp = client.embeddings.create(
#         model="text-embedding-3-large",
#         input=texts,
#     )
#     return [d.embedding for d in resp.data]

# # data/papers.jsonl 파일을 한 줄씩 읽어서 JSON 객체 리스트로 변환
# def load_papers(path: Path) -> List[Dict[str, Any]]:
#     papers: List[Dict[str, Any]] = []
#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             papers.append(json.loads(line))
#     return papers

# # title + abstract 합쳐서 검색용 통합 텍스트 만듦
# def build_content(title: str, abstract: str) -> str:
#     return f"{title}\n\n{abstract}"


# # 논문 데이터를 batch 단위로 ES 인덱스에 등록
# def bulk_index_papers(papers: List[Dict[str, Any]], batch_size: int = 64) -> None:
#     create_papers_index()  # 없으면 생성

#     for i in range(0, len(papers), batch_size):
#         batch = papers[i : i + batch_size]

#         texts = [build_content(p.get("title", ""), p.get("abstract", "")) for p in batch]
#         embeddings = get_embeddings(texts)

#         actions = []
#         for p, emb, content in zip(batch, embeddings, texts):
#             doc = {
#                 TITLE_FIELD: p.get("title", ""),
#                 ABSTRACT_FIELD: p.get("abstract", ""),
#                 CONTENT_FIELD: content,
#                 YEAR_FIELD: p.get("year"),
#                 CITATION_FIELD: p.get("citations", 0),
#                 VENUE_FIELD: p.get("venue"),
#                 URL_FIELD: p.get("url"),
#                 EMBED_FIELD: emb,
#             }

#             actions.append({
#                 "_index": ES_INDEX,
#                 "_id": p.get("id"),
#                 "_source": doc,
#             })

#         helpers.bulk(es, actions)
#         print(f"[bulk_index_papers] indexed {i + len(batch)} / {len(papers)}")

# # 전체 파이프라인 실행: JSONL 로드 → 임베딩 생성 → ES 업로드
# def main():
#     papers = load_papers(DATA_PATH)
#     print(f"[main] loaded {len(papers)} papers")
#     bulk_index_papers(papers)

# if __name__ == "__main__":
#     main()
# ingest_papers.py
import json
from pathlib import Path
from typing import List, Dict, Any

from elasticsearch import helpers

from settings import (
    es, ES_INDEX, EMBED_FIELD,
    TITLE_FIELD, ABSTRACT_FIELD, CONTENT_FIELD,
    YEAR_FIELD, CITATION_FIELD, VENUE_FIELD, URL_FIELD,
)

# 인덱스 매핑이 없을 경우 생성
from es_utils import create_papers_index

# 네가 보여준 JSONL이 있는 경로
DATA_PATH = Path("data/arxiv_embeddings.jsonl")



# data/papers.jsonl 파일을 한 줄씩 읽어서 JSON 객체 리스트로 변환
def load_papers(path: Path) -> List[Dict[str, Any]]:
    papers: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            papers.append(json.loads(line))
    return papers


# title + chunk text 합쳐서 검색용 통합 텍스트 만듦
# (예: "Mamba: ... \n\n though this change prevents ...")
def build_content(title: str, text: str) -> str:
    return f"{title}\n\n{text}"


def extract_year(published: Any) -> Any:
    """
    published: "2023-12-01" 같은 문자열이 온다고 가정
    연도만 int로 뽑아서 YEAR_FIELD에 넣어줌.
    문제가 생기면 그냥 None으로 리턴.
    """
    if not published:
        return None
    try:
        s = str(published)
        return int(s[:4])
    except Exception:
        return None


# 논문 데이터를 batch 단위로 ES 인덱스에 등록
def bulk_index_papers(papers: List[Dict[str, Any]], batch_size: int = 64) -> None:
    create_papers_index()  # 없으면 생성

    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]

        actions = []
        for p in batch:
            title = p.get("title", "")
            text = p.get("text", "")  # JSONL의 chunk 텍스트
            content = build_content(title, text)

            # JSONL에 이미 들어있는 embedding 사용
            emb = p.get("embedding", [])

            # 연도 추출 (published: "2023-12-01" → 2023)
            year = extract_year(p.get("published"))

            doc = {
                TITLE_FIELD: title,
                # 기존에 ABSTRACT_FIELD를 쓰고 있으면 text를 같이 넣거나 빈 문자열로 둘 수 있음
                # chunk text
                ABSTRACT_FIELD: text,
                CONTENT_FIELD: content,
                YEAR_FIELD: year,
                CITATION_FIELD: p.get("citation_count", 0),
                VENUE_FIELD: p.get("venue"),               # JSONL에 없으면 None으로 들어감
                URL_FIELD: p.get("paper_id") or p.get("url"),
                EMBED_FIELD: emb,
            }

            actions.append({
                "_index": ES_INDEX,
                "_id": p.get("id"),  # "2312.00752v2_0" 이런 chunk 단위 id
                "_source": doc,
            })

        helpers.bulk(es, actions)
        print(f"[bulk_index_papers] indexed {i + len(batch)} / {len(papers)}")


# 전체 파이프라인 실행: JSONL 로드 → ES 업로드 (임베딩은 JSONL에서 그대로 사용)
def main():
    papers = load_papers(DATA_PATH)
    print(f"[main] loaded {len(papers)} papers")
    bulk_index_papers(papers)


if __name__ == "__main__":
    main()
