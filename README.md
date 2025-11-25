
# 🧠 Paper Search Agent (LangGraph + OpenAI + Elasticsearch)

대규모 논문 검색을 위한 **Agentic RAG 시스템** 프로젝트입니다.  
OpenAI API와 Elasticsearch를 기반으로 LangGraph를 사용해 다음 단계를 자동화합니다:

> **질문 분류 → 검색 전략 결정 → Sparse/Dense 검색 → 결과 융합 → 요약 생성**

---

## 🚀 프로젝트 개요
이 시스템은 LangGraph를 활용해 논문 검색 과정을 완전 자동화한 Agentic RAG 파이프라인입니다.  
사용자의 질문을 이해하고, 키워드 기반 / 의미 기반 검색을 병렬 수행한 뒤  
RRF 융합을 통해 결과를 랭킹하고 OpenAI 모델로 요약합니다.

---

## 📂 폴더 구조
```

PAPER_SEARCH_AGENT_PROJECT/
├─ data/
│   └─ papers.jsonl              # 샘플 논문 데이터
│
├─ src/
│   ├─ settings.py               # OpenAI & ES 설정 및 초기화
│   ├─ es_utils.py               # ES 인덱스 생성/삭제 유틸
│   ├─ index_papers.py          # 논문 데이터 인덱싱 스크립트
│   └─ paper_agent.ipynb         # LangGraph 기반 검색 Agent 노트북
│
├─ docker-compose.yml            # Elasticsearch 실행용
├─ requirements.txt              # 의존성 목록
├─ .gitignore                    # 환경파일, 모델, venv 제외
└─ README.md

````
---

## ⚙️ 환경 설정
1. **.env 파일 생성 (프로젝트 루트에 위치)**
   ```bash
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ES_HOST=localhost
   ES_PORT=9200
   ES_USER=elastic
   ES_PASSWORD=changeme
   ES_INDEX=papers_index
```

2. **필수 패키지 설치**

   ```bash
   pip install -r requirements.txt
   ```

3. **Elasticsearch 실행**

   ```bash
   docker compose up -d
   ```

---

## 💾 데이터셋 설정

### 2023~2025년도, 각 년도별 인용수 기준 각 2000 개씩 전체 6개 수집
- 각 년도별 AI 관련 논문 1만개 수집 후, 인용수 기준 2000개씩 선정
- 빠르고 높은 성능의 'all-MiniLM-L6-v2' 모델로 임베딩 적용
- jsonl data size : 20,228 , vector size : 384, chunk size : 500
  

**Sample Dataset **

```json
[샘플 데이터 구조 (첫 번째 항목)] { "id": "2312.00752v2_0", "paper_id": "[http://arxiv.org/abs/2312.00752v2](http://arxiv.org/abs/2312.00752v2)", "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", "published": "2023-12-01", "citation_count": 4579, "text": "Foundation models, now powering most of the exciting applications in deep learning,... but they have not performed as well as attention on important modalities such as language", "embedding": [ -0.04355737194418907, -0.10571599006652832, ... -0.04330563545227051, -0.006554255727678537 ], "metadata": { "source": "arxiv", "categories": [ "cs.LG", "cs.AI" ] } }
```

---

## 🧩 실행 순서

### ① 데이터 인덱싱

```bash
python src/ingest_papers.py
```

→ `data/papers.jsonl`의 논문 데이터를 Elasticsearch에 업로드하고 인덱스 생성

### ② Agent 실행 (Jupyter Notebook)

`src/paper_agent.ipynb`를 열고 아래 셀을 실행:

```python
run_query("RAG retriever 관련 최신 논문 알려줘")
```

결과:

* LangGraph가 질문 유형을 분류하고
* Sparse + Dense 검색 수행 후 RRF로 결과 통합
* OpenAI 모델이 관련 논문을 요약 및 설명 출력

---

## 🧱 주요 구성 요소

| 모듈                    | 역할                                 |
| --------------------- | ---------------------------------- |
| **settings.py**       | `.env` 로드 및 OpenAI/ES 클라이언트 초기화    |
| **es_utils.py**       | 인덱스 생성, 삭제, 매핑 관리                  |
| **ingest_papers.py**  | `papers.jsonl` 데이터를 임베딩 후 인덱싱      |
| **paper_agent.ipynb** | LangGraph 기반 Agent 흐름 실행 (검색 + 요약) |

---

## 🧠 기술 스택

* **LangGraph** – Agentic Workflow 구성
* **Elasticsearch** – BM25 + dense vector hybrid 검색
* **OpenAI API** – `gpt-4o` / `text-embedding-3-large`
* **Python-dotenv** – 환경 변수 관리

---

## 🔒 주의사항

* `.env`, `venv_paper_agent/`, `*.gguf` 등은 Git에 포함되지 않습니다.
* API Key는 절대 커밋하지 말고 `.env`로만 관리하세요.
* `data/papers.jsonl`은 샘플 데이터로, 실제 대용량 데이터는 `.gitignore`에 추가하세요.

---

## 🧩 향후 개선 방향

* LangGraph Memory 추가로 질의 이력 반영
* Hybrid retrieval 가중치 자동 최적화
* Paper Summarizer Agent 분리 및 API화

---

## ✨ 예시 명령어 요약

```bash
# 환경 구성
pip install -r requirements.txt

# Elasticsearch 실행
docker compose up -d

# 논문 인덱싱
python src/ingest_papers.py

# Agent 실행 (노트북)
jupyter lab src/paper_agent.ipynb
```

---

```
---
```
