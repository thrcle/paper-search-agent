# # es_utils.py
# from typing import Dict, Any
# from settings import es, ES_INDEX, EMBED_FIELD

# def create_papers_index(dimension: int = 3072) -> None:
#     """
#     논문 인덱스 생성 (dense_vector 포함)
#     dimension은 사용하는 OpenAI embedding dimension 맞춰주면 됨.
#     """
#     if es.indices.exists(index=ES_INDEX):
#         print(f"[create_papers_index] index '{ES_INDEX}' already exists")
#         return

#     mapping: Dict[str, Any] = {
#         "mappings": {
#             "properties": {
#                 "title": {"type": "text"},
#                 "abstract": {"type": "text"},
#                 "content": {"type": "text"},
#                 "year": {"type": "integer"},
#                 "citations": {"type": "integer"},
#                 "venue": {"type": "keyword"},
#                 "url": {"type": "keyword"},
#                 EMBED_FIELD: {
#                     "type": "dense_vector",
#                     "dims": dimension,
#                     "index": True,
#                     "similarity": "cosine",
#                 },
#             }
#         }
#     }

#     es.indices.create(index=ES_INDEX, body=mapping)
#     print(f"[create_papers_index] created index '{ES_INDEX}'")

# def delete_papers_index() -> None:
#     if es.indices.exists(index=ES_INDEX):
#         es.indices.delete(index=ES_INDEX)
#         print(f"[delete_papers_index] deleted index '{ES_INDEX}'")
#     else:
#         print(f"[delete_papers_index] index '{ES_INDEX}' does not exist")

# es_utils.py
from typing import Dict, Any
from settings import es, ES_INDEX, EMBED_FIELD


def create_papers_index(dimension: int | None = None) -> None:

    if es.indices.exists(index=ES_INDEX):
        print(f"[create_papers_index] index '{ES_INDEX}' already exists")
        return

    # 현재 JSONL 기준 기본 임베딩 차원 (예: bge-m3 384차원)
    if dimension is None:
        dimension = 384

    mapping: Dict[str, Any] = {
        "mappings": {
            "properties": {
                # ingest_papers.py에서 TITLE_FIELD, ABSTRACT_FIELD, CONTENT_FIELD 로 들어가는 필드들
                "title": {"type": "text"},
                "abstract": {"type": "text"},   # JSONL의 "text"를 여기로 매핑
                "content": {"type": "text"},    # title + abstract 합친 통합 필드

                # 연도, 인용수 등
                "year": {"type": "integer"},
                "citations": {"type": "integer"},

                # venue는 지금 None 넣고 있지만, 스키마는 keyword로 미리 열어둠
                "venue": {"type": "keyword"},

                # paper_id → URL 로 매핑해서 저장하는 필드
                "url": {"type": "keyword"},

                # 임베딩 필드
                EMBED_FIELD: {
                    "type": "dense_vector",
                    "dims": dimension,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        }
    }

    es.indices.create(index=ES_INDEX, body=mapping)
    print(f"[create_papers_index] created index '{ES_INDEX}' (dims={dimension})")


def delete_papers_index() -> None:
    if es.indices.exists(index=ES_INDEX):
        es.indices.delete(index=ES_INDEX)
        print(f"[delete_papers_index] deleted index '{ES_INDEX}'")
    else:
        print(f"[delete_papers_index] index '{ES_INDEX}' does not exist")
