# 📚 Paper Search Agent  
Hybrid IR + RAG + LangGraph 기반 논문 검색 에이전트

이 프로젝트는 **LLM + 검색엔진(ElasticSearch + Qdrant)** 을 결합해  
사용자가 입력한 키워드/질문에 대해 논문을 검색하고 요약/정리해주는  
**Agentic RAG 기반 Paper Search System**입니다.

LangGraph를 사용해 검색 → 재검색 → 정교화 → 답변 생성과 같은  
에이전트 흐름을 구성하였으며, 로컬 LLM(Qwen/Llama)과  
임베딩 모델(bge-m3) 기반 하이브리드 검색을 제공합니다.

---

## 🚀 Features (기능 요약)

### 🔍 1. 논문 검색 (Hybrid IR)
- **Sparse 검색**: BM25(ElasticSearch)
- **Dense 검색**: Qdrant + bge-m3 embedding
- 두 점수를 혼합해 최적의 검색 결과 제공

### 🧠 2. Agentic LangGraph Workflow
- 쿼리 생성 → 문서 검색 → 재검색(Refine) → RAG 응답
- 상태 흐름을 명확하게 관리하며 추론 과정 안정화

### 🧩 3. RAG 기반 응답 생성
- 검색된 문서 기반으로 LLM이 요약 및 근거 포함 답변 생성
- hallucination 방지 (검색 기반 Evidence 우선)

### 📝 4. 노트북 기반 실험 환경 제공
`notebook/qwen3_es_qdrant_langgraph_rag.ipynb` 에 전체 파이프라인 포함  
LLM 호출, Retriever, RAG Flow 실험 가능

---

## 🏗 Architecture

```mermaid
flowchart LR
    A[User Query] --> B[LLM (Qwen/Llama)]
    B --> C[Query Rewriting / Expansion]
    C --> D[Retriever]

    D --> E[(ElasticSearch)]
    D --> F[(Qdrant Vector DB)]

    E --> G[Hybrid Reranker]
    F --> G[Hybrid Reranker]

    G --> H[RAG Context Builder]
    H --> I[LLM Answer Generator]

    I --> J[Final Response]
```

---

## 📦 Project Structure

```
paper-search-agent/
 ├── notebook/
 │    └── qwen3_es_qdrant_langgraph_rag.ipynb
 ├── docker-compose.yml
 ├── .gitignore
 └── README.md
```

---

## 🔧 Installation

### 1) 가상환경 생성 & 패키지 설치

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🤖 LLM / Embedding 모델 다운로드 (중요)

이 레포는 모델 파일을 포함하지 않습니다.  
로컬에서 직접 다운받아 아래 구조로 저장하세요.

```
models/
 ├── llm/
 │    └── llama-3.1-8b-instruct.Q4_K_M.gguf   (예시)
 └── embed/
      └── bge-m3-q4_k_m.gguf
```

### 예시 다운로드 링크
- **Llama 3.1 8B Instruct GGUF**: HuggingFace / TheBloke / bartowski 중 택1  
- **bge-m3 embedding GGUF**: BAAI/bge-m3

---

## 🐳 Run with Docker (ElasticSearch + Qdrant)

```bash
docker-compose up -d
```

실행 후:
- ES: `localhost:9200`
- Qdrant: `localhost:6333`

---

## 🧪 Run Notebook

```bash
jupyter lab
```

- `notebook/qwen3_es_qdrant_langgraph_rag.ipynb` 열어서 실행  
- 검색 → RAG → 응답 생성 전체 파이프라인 테스트 가능

---

## 📌 TODO (Roadmap)

- [ ] 검색 재랭킹 개선 (RRF → Learned Reranker)
- [ ] LangGraph Memory 적용
- [ ] Fine-grained Query Rewriting
- [ ] Multi-step Deep Search
- [ ] Agent tools 자동 확장


---

## ⭐ License

MIT License  
모델 파일은 각 제공처의 라이선스를 따릅니다.
