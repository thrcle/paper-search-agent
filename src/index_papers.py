# 논문 -> 임베딩 -> es 인덱싱 

# ingest_papers.py
import json
from pathlib import Path
from typing import List, Dict, Any

from elasticsearch import helpers

from settings import (
    client, es, ES_INDEX, EMBED_FIELD,
    TITLE_FIELD, ABSTRACT_FIELD, CONTENT_FIELD,
    YEAR_FIELD, CITATION_FIELD, VENUE_FIELD, URL_FIELD,
)

# 인덱스 매핑이 없을 경우 생성
from es_utils import create_papers_index

DATA_PATH = Path("data/papers.jsonl")

# 각 논문(제목 + 초록) 텍스트를 OpenAI text-embedding-3-large 모델로 임베딩 생성
def get_embeddings(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts,
    )
    return [d.embedding for d in resp.data]

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

# title + abstract 합쳐서 검색용 통합 텍스트 만듦
def build_content(title: str, abstract: str) -> str:
    return f"{title}\n\n{abstract}"


# 논문 데이터를 batch 단위로 ES 인덱스에 등록
def bulk_index_papers(papers: List[Dict[str, Any]], batch_size: int = 64) -> None:
    create_papers_index()  # 없으면 생성

    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]

        texts = [build_content(p.get("title", ""), p.get("abstract", "")) for p in batch]
        embeddings = get_embeddings(texts)

        actions = []
        for p, emb, content in zip(batch, embeddings, texts):
            doc = {
                TITLE_FIELD: p.get("title", ""),
                ABSTRACT_FIELD: p.get("abstract", ""),
                CONTENT_FIELD: content,
                YEAR_FIELD: p.get("year"),
                CITATION_FIELD: p.get("citations", 0),
                VENUE_FIELD: p.get("venue"),
                URL_FIELD: p.get("url"),
                EMBED_FIELD: emb,
            }

            actions.append({
                "_index": ES_INDEX,
                "_id": p.get("id"),
                "_source": doc,
            })

        helpers.bulk(es, actions)
        print(f"[bulk_index_papers] indexed {i + len(batch)} / {len(papers)}")

# 전체 파이프라인 실행: JSONL 로드 → 임베딩 생성 → ES 업로드
def main():
    papers = load_papers(DATA_PATH)
    print(f"[main] loaded {len(papers)} papers")
    bulk_index_papers(papers)

if __name__ == "__main__":
    main()
