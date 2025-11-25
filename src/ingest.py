import os
import requests
import xml.etree.ElementTree as ET
import json
import pandas as pd
import time
from datetime import datetime

# 임베딩 라이브러리 확인
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[ERROR] 'sentence_transformers' 라이브러리가 필요합니다.")
    print("pip install sentence-transformers 명령어로 설치해주세요.")
    exit()

# -------------------
# 설정
# -------------------
TARGET_CATEGORIES = [
    "cs.AI",   # Artificial Intelligence (인공지능)
    "cs.LG",   # Machine Learning (머신러닝 - CS)
    "stat.ML", # Machine Learning (머신러닝 - 통계)
    "cs.CL",   # Computation and Language (자연어 처리/NLP)
    "cs.CV",   # Computer Vision (컴퓨터 비전)
    "cs.RO",   # Robotics (로봇공학)
    "cs.NE"    # Neural and Evolutionary Computing (신경망 및 진화 연산)
]

# 연도 설정 (3개년)
TARGET_YEARS = [2023, 2024, 2025]

# 추출 설정
TARGET_COUNT_PER_YEAR = 2000  # 연도별 최종 저장할 상위 논문 수 (3개년 총 6,000개)
FETCH_CANDIDATES = 10000      # 연도별 arXiv에서 수집할 논문 수
                              # 2000개씩 5번 요청

OUTPUT_XLSX = "arxiv_ai_top_cited_6k.xlsx"
OUTPUT_JSONL = "arxiv_embeddings_6k.jsonl"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# -------------------
# 1) arXiv API 요청 함수 (배치 처리 적용)
# -------------------
def fetch_arxiv_papers(categories, start_date, end_date, max_results):
    print(f"[arXiv] {start_date} ~ {end_date} 기간, 총 {max_results}개 후보 수집 시작...")
    
    # 날짜 포맷 변환
    start_fmt = start_date.replace("-", "") + "0000"
    end_fmt = end_date.replace("-", "") + "2359"

    # 쿼리 생성
    cat_query_parts = [f"cat:{cat}" for cat in categories]
    cat_query_str = "+OR+".join(cat_query_parts)
    cat_group = f"%28{cat_query_str}%29"
    date_query = f"submittedDate:[{start_fmt}+TO+{end_fmt}]"
    final_query = f"{cat_group}+AND+{date_query}"

    # --- 배치 처리 변수 ---
    all_papers = []
    start_index = 0
    BATCH_SIZE = 2000  # 한 번에 요청할 최대 개수 (arXiv 권장)

    while len(all_papers) < max_results:
        # 이번 턴에 요청할 개수 계산
        current_max = min(BATCH_SIZE, max_results - len(all_papers))
        
        print(f"   >> 요청 중: start={start_index}, count={current_max} ...")

        url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query={final_query}&"
            f"start={start_index}&"            # 시작 위치 변경
            f"max_results={current_max}&"      # 요청 개수
            f"sortBy=submittedDate&sortOrder=descending"
        )

        try:
            response = requests.get(url)
            response.raise_for_status()
        except Exception as e:
            print(f"[ERROR] 요청 실패 (start={start_index}): {e}")
            break

        # XML 파싱
        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        batch_papers = []

        for entry in root.findall("atom:entry", ns):
            authors_list = [author.find("atom:name", ns).text for author in entry.findall("atom:author", ns)]
            categories_list = [c.attrib["term"] for c in entry.findall("atom:category", ns)]

            title = entry.find("atom:title", ns).text
            title = title.replace("\n", " ").strip() if title else ""
            
            summary = entry.find("atom:summary", ns).text
            summary = summary.replace("\n", " ").strip() if summary else ""

            arxiv_url = entry.find("atom:id", ns).text
            arxiv_id = arxiv_url.split("/abs/")[-1]

            paper = {
                "id": arxiv_url,
                "arxiv_id": arxiv_id,
                "title": title,
                "summary": summary,
                "authors": list(authors_list),
                "categories": list(categories_list),
                "published": entry.find("atom:published", ns).text[:10],
                "citation_count": 0 
            }
            batch_papers.append(paper)
        
        # 결과가 없으면 종료 
        if not batch_papers:
            print("   >> 더 이상 가져올 데이터가 없습니다.")
            break

        all_papers.extend(batch_papers)
        start_index += len(batch_papers) # 다음 시작 인덱스 갱신

        print(f"   >> 현재 확보된 논문: {len(all_papers)}개")
        
        # arXiv 서버 부하 방지를 위한 대기 (필수)
        if len(all_papers) < max_results:
            time.sleep(3)

    print(f"[arXiv] 수집 완료. 총 {len(all_papers)}개 후보 확보.")
    return all_papers

# -------------------
# 2) Semantic Scholar API로 인용수 조회 및 상위 추출
# -------------------
def update_citations_and_filter(papers, target_count):
    if not papers:
        return []

    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    params = {"fields": "citationCount,title"}
    batch_size = 100
    
    print(f"[S2] Semantic Scholar 인용수 조회 중... (총 {len(papers)}건)")

    for i in range(0, len(papers), batch_size):
        batch_papers = papers[i:i + batch_size]
        paper_ids = [f"ARXIV:{p['arxiv_id'].split('v')[0]}" for p in batch_papers]
        
        try:
            r = requests.post(url, json={"ids": paper_ids}, params=params)
            if r.status_code == 200:
                data = r.json()
                for j, item in enumerate(data):
                    if item and 'citationCount' in item:
                        batch_papers[j]['citation_count'] = item['citationCount']
            
            # 진행 상황 표시 (선택사항)
            if i % 1000 == 0 and i > 0:
                print(f"   >> {i}건 조회 완료...")
                
            time.sleep(1) # API 부하 방지
        except Exception as e:
            print(f"[ERROR] 인용수 조회 중 에러: {e}")

    # 1. 인용수 기준 내림차순 정렬
    papers.sort(key=lambda x: x['citation_count'], reverse=True)
    
    # 2. 상위 N개만 자르기 (Top-K)
    top_papers = papers[:target_count]
    
    if top_papers:
        min_cit = top_papers[-1]['citation_count']
        max_cit = top_papers[0]['citation_count']
        print(f"[Filtering] 상위 {len(top_papers)}개 선정 완료 (인용수 범위: {max_cit} ~ {min_cit}회)")
    
    return top_papers

# -------------------
# 3) 텍스트 청킹
# -------------------
def chunk_text(text, size=500, overlap=50):
    if not text: return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + size, text_len)
        chunks.append(text[start:end])
        if end == text_len: break
        start += (size - overlap)
    return chunks

# -------------------
# 4) 임베딩 및 저장
# -------------------
def process_embeddings_and_save(papers, output_path):
    print(f"[Embedding] 모델 로딩 중... (all-MiniLM-L6-v2)")
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    print(f"[Embedding] 총 {len(papers)}개 논문에 대한 임베딩 생성 시작...")
    
    total_chunks = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, paper in enumerate(papers):
            if idx % 100 == 0: print(f" - 진행률: {idx}/{len(papers)}")
            
            chunks = chunk_text(paper['summary'], size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            if not chunks: continue

            embeddings = model.encode(chunks)

            for i, (chunk_text_str, vector) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{paper['arxiv_id']}_{i}"
                record = {
                    "id": chunk_id,
                    "paper_id": paper['id'],
                    "title": paper['title'],
                    "published": paper['published'],
                    "citation_count": paper['citation_count'],
                    "text": chunk_text_str,
                    "embedding": vector.tolist(),
                    "metadata": {
                        "source": "arxiv",
                        "categories": paper['categories']
                    }
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"[Save] JSONL 저장 완료 ({total_chunks} 청크).")

# -------------------
# 5) 엑셀 저장
# -------------------
def save_excel(papers, output_excel):
    simple_data = []
    for p in papers:
        simple_data.append({
            "citation_count": p["citation_count"],
            "published": p["published"],
            "title": p["title"],
            "id": p["id"],
            "summary": p["summary"],
            "authors": ", ".join(p["authors"]),
            "categories": ", ".join(p["categories"]),
        })

    df = pd.DataFrame(simple_data)
    df.to_excel(output_excel, index=False)
    print(f"[Save] Excel 저장 완료.")

# -------------------
# 메인 실행
# -------------------
if __name__ == "__main__":
    final_papers = []

    # --- 연도별 순차 수집 ---
    for year in TARGET_YEARS:
        print(f"\n=== {year}년도 데이터 처리 시작 ===")
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        # 1. 해당 연도 후보군 수집 (배치 처리 적용됨)
        # FETCH_CANDIDATES 만큼 가져오기 위해 2000개씩 끊어서 호출함
        candidates = fetch_arxiv_papers(TARGET_CATEGORIES, start_date, end_date, FETCH_CANDIDATES)
        
        # 2. 인용수 조회 및 Top 2000 선정
        if candidates:
            top_cited = update_citations_and_filter(candidates, TARGET_COUNT_PER_YEAR)
            final_papers.extend(top_cited)
        else:
            print(f"[Warn] {year}년도 데이터가 없습니다.")

    print(f"\n=== 최종 결과 집계: 총 {len(final_papers)}개 논문 ===")

    # --- 저장 ---
    if final_papers:
        process_embeddings_and_save(final_papers, OUTPUT_JSONL)
        save_excel(final_papers, OUTPUT_XLSX)
    else:
        print("[Error] 수집된 논문이 없습니다.")