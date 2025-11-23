# es_utils.py
from typing import Dict, Any
from config import es, ES_INDEX, EMBED_FIELD

def create_papers_index(dimension: int = 3072) -> None:
    """
    논문 인덱스 생성 (dense_vector 포함)
    dimension은 사용하는 OpenAI embedding dimension 맞춰주면 됨.
    """
    if es.indices.exists(index=ES_INDEX):
        print(f"[create_papers_index] index '{ES_INDEX}' already exists")
        return

    mapping: Dict[str, Any] = {
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "abstract": {"type": "text"},
                "content": {"type": "text"},
                "year": {"type": "integer"},
                "citations": {"type": "integer"},
                "venue": {"type": "keyword"},
                "url": {"type": "keyword"},
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
    print(f"[create_papers_index] created index '{ES_INDEX}'")

def delete_papers_index() -> None:
    if es.indices.exists(index=ES_INDEX):
        es.indices.delete(index=ES_INDEX)
        print(f"[delete_papers_index] deleted index '{ES_INDEX}'")
    else:
        print(f"[delete_papers_index] index '{ES_INDEX}' does not exist")
