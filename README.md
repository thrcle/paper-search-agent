# 🩺 Dr.Paper – AI 논문 검색 · 진단 에이전트

Elasticsearch  + LangGraph + LLM 을 이용해서
연구 주제에 맞는 논문을 검색/요약/추천해 주는 에이전트 프로젝트.

* BM25(키워드) + Dense Embedding(의미 기반) 하이브리드 검색
* LangGraph 로 상태/워크플로우 관리
* 메모리 기반 Multi-turn 질의 지원
* 외부 논문 API(Tavily/OpenAlex 등)로 보완 검색

---

## 📂 프로젝트 구조

```
paper-search-agent/
├── data/
│   └── arxiv_embeddings.jsonl        # 논문 + chunk + embedding 데이터
├── src/
│   ├── app.py                        # Streamlit 웹 UI
│   ├── es_utils.py                   # Elasticsearch 인덱스 생성/매핑 유틸
│   ├── evaluation.py                 # 답변 관련도 평가 로직
│   ├── external_search.py            # Semantic Scholar / OpenAlex 외부 검색 에이전트
│   ├── index_papers.py               # JSONL → Elasticsearch 인덱싱 스크립트
│   ├── main_agent.py                 # LangGraph 기반 메인 오케스트레이터
│   ├── memory_agent.py               # 대화/검색 히스토리 메모리 관리
│   ├── model_utils.py                # OpenAI 클라이언트, 임베딩/LLM 호출 유틸
│   ├── search_agents.py              # BM25 / Dense / RRF / 요약 에이전트
│   ├── settings.py                   # 환경변수, ES 설정, 필드명 상수
│   ├── state.py                      # AgentState 타입 정의
│   └── visualize_graph.ipynb         # LangGraph 시각화용 노트북
└── README.md
```

---

## ⚙️ 환경 설정

### 1️⃣ Python 가상환경 설정

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ 환경 변수 (.env)

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
OPENALEX_BASE_URL=https://api.openalex.org
ES_HOST=http://localhost:9200
ES_INDEX=paper_index
```

### 3️⃣ Elasticsearch 실행

```bash
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.1
```

---

## 🧱 인덱싱 파이프라인

```bash
cd src
python index_papers.py
```

* `arxiv_embeddings.jsonl` 데이터를 로드하고, 각 논문을 Elasticsearch 인덱스로 업로드.
* 인덱스 필드:

  * `title`, `abstract`, `content`, `year`, `citations`, `venue`, `url`, `embedding`

---

## 🧠 LangGraph 기반 실행 순서
<img width="549" height="877" alt="image" src="https://github.com/user-attachments/assets/6dafb822-76e0-4346-b9b7-da3be7cbe853" />

### 전체 노드 흐름

```
__start__ → classify_utterance_agent → strategy_agent
    ├─ sparse → keyword_search_agent
    ├─ dense  → semantic_search_agent
    └─ hybrid → merge_and_select_agent
→ evaluate_answer_agent
    ├─ True  → external_search_agent
    └─ False → memory_update_agent
→ __end__
```

### 1️⃣ 질의 유형 분류(classify_utterance_agent )
* 사용자의 발화를 4가지 타입으로 구분 
*  타입	설명	예시
NL_TOPIC	자연어형 주제 질의	“RAG 관련 논문 추천해줘”
KEYWORD_TOPIC	키워드 기반	“ICLR 2024 RAG”
SPECIFIC_PAPER	특정 논문	“MAIN-RAG 논문 원문 보여줘”
MEMORY_QUERY	대화 히스토리 질의	“내가 전에 뭐 물어봤지?”

### 2️⃣ 검색 전략 결정 (strategy_agent)
* 입력: utterance_type
* 출력: search_strategy = sparse / dense / hybrid
* 결정 기준
키워드 중심 → sparse
문맥형 질문 → dense
포괄적 질의 → hybrid
LLM 예시 출력:
```
🤖 [LLM Decision] strategy=hybrid,
reason=“키워드와 의미 정보를 모두 활용하는 것이 적합하다고 판단함”
```

### 3️⃣ 검색 전략  
* 인용수 가중치 포함 BM25 기반 키워드 검색 (keyword_search_agent)
* 벡터 기반 유사도 검색 (semantic_search_agent)
* RRF 융합 (Reciprocal Rank Fusion) 상위 N개 논문 추출 

### 4️⃣ 답변품질 평가(evaluate_answer_agent)
* 문장 단위 유사도 + 키워드 정밀도 혼합 점수 계산
* 점수 < 0.6 → 외부 검색으로 분기

### 5️⃣ external_search_agent
* OpenAlex / Semantic Scholar API 호출
* 최신 논문/리뷰를 보완해 답변 통합

### 6️⃣ memory_update_agent
* 메모리에 질의, 답변, 전략, 점수 기록 → 멀티턴 질의 대응

---

## 🧩 실행 방법

```bash
# 1. Elasticsearch 실행
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.11.1

# 2. 인덱싱 수행
cd src
python index_papers.py

# 3. LangGraph 에이전트 실행 테스트
python main_agent.py

# 4. Streamlit 웹 인터페이스 실행
streamlit run app.py
```

---

## 🔍 실행 플로우 요약

1. 사용자의 질의 입력
2. 인텐트 분류 및 검색 전략 결정
3. BM25 / Dense 검색 수행
4. 결과 융합 및 요약 생성
5. 답변 관련도 평가 → 기준치 미달 시 외부 검색 수행
6. 최종 답변 및 메모리 저장

---

## 💡 예시 출력

```
💬 Utterance Type  : NL_TOPIC
🧭 Search Strategy : hybrid
📊 Relevance Score : 0.73
🤖 LLM Decision    : [LLM Decision] strategy=hybrid, reason=키워드와 의미 정보를 모두 활용함
🌐 External Search : hybrid
─────────────────────────────────────────────
[답변]
RAG의 효율적 Retrieval 개선에 관한 주요 연구는 MAIN-RAG(2024), Self-RAG(2023) 등이 있습니다.
```

---
## 💻 Streamlit UI 구성
<img width="849" height="551" alt="image" src="https://github.com/user-attachments/assets/cbe02373-57f4-4fb7-9c9b-6e90c3dd034c" />

주요 기능
* 검색창:	자연어 또는 키워드 질의 입력 (예: “LLM 관련 최신 논문”)
* AI 진단:	질의 유형, 전략, 관련도 점수, 검색 경로 표시
* 논문 카드	각 논문 제목, 인용수, 연도, 요약 표시 및 논문 URL 연결
* 진료 기록:	이전 질의 내역 및 메모리 기반 대화 히스토리 확인

---

## 🚀 향후 개선 로드맵

* 필터링(연도/컨퍼런스/인용수) 기능 강화
* Citation Graph 기반 논문 클러스터링
* Multi-Agent 협업 Reasoning 개선
