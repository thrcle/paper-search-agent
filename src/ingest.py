import os
import requests
import xml.etree.ElementTree as ET
import json
import pandas as pd
from datetime import datetime

# 임베딩을 위한 라이브러리 (설치 필요: pip install sentence-transformers)
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

# 날짜 설정 (시작일 < 종료일)
START_DATE = "20220101"
END_DATE = "20251231"

MAX_RESULTS = 5000
OUTPUT_XLSX = "arxiv_ai_papers.xlsx"
OUTPUT_JSONL = "arxiv_embeddings.jsonl"

# 청킹 설정
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# -------------------
# 1) arXiv API 요청 함수
# -------------------
def fetch_arxiv_papers(categories, start_date, end_date, max_results):
    print(f"[INFO] 기간: {start_date} ~ {end_date}")

    # 쿼리 생성
    cat_query_parts = [f"cat:{cat}" for cat in categories]
    cat_query_str = "+OR+".join(cat_query_parts)
    cat_group = f"%28{cat_query_str}%29" # 괄호 인코딩
    date_query = f"submittedDate:[{start_date}+TO+{end_date}]"
    final_query = f"{cat_group}+AND+{date_query}"

    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query={final_query}&"
        f"start=0&max_results={max_results}&"
        f"sortBy=submittedDate&sortOrder=descending"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        print(f"[ERROR] 요청 실패: {e}")
        return []

    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []

    for entry in root.findall("atom:entry", ns):
        authors_list = [author.find("atom:name", ns).text for author in entry.findall("atom:author", ns)]
        categories_list = [c.attrib["term"] for c in entry.findall("atom:category", ns)]

        title = entry.find("atom:title", ns).text
        title = title.replace("\n", " ").strip() if title else ""
        
        summary = entry.find("atom:summary", ns).text
        summary = summary.replace("\n", " ").strip() if summary else ""

        paper = {
            "id": entry.find("atom:id", ns).text,
            "title": title,
            "summary": summary,
            "authors": list(authors_list), # 리스트 유지
            "categories": list(categories_list), # 리스트 유지
            "published": entry.find("atom:published", ns).text[:10],
            "source_url": entry.find("atom:id", ns).text,
        }
        papers.append(paper)

    print(f"[INFO] arXiv 수집 완료! 총 {len(papers)}개 논문.")
    return papers

# -------------------
# 2) 텍스트 청킹 함수
# -------------------
def chunk_text(text, size=500, overlap=50):
    """
    주어진 텍스트를 size만큼 자르고, overlap만큼 겹치게 하여 리스트로 반환
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        
        # 더 이상 자를 것이 없으면 종료
        if end == text_len:
            break
            
        # 다음 시작점 계산 (현재 시작점 + 사이즈 - 오버랩)
        # 즉, 500자 읽고 450자 지점부터 다시 읽음
        start += (size - overlap)
        
    return chunks

# -------------------
# 3) 임베딩 생성 및 JSONL 저장
# -------------------
def process_embeddings_and_save(papers, output_path):
    print(f"[INFO] 임베딩 모델 로딩 중... (all-MiniLM-L6-v2)")
    # 로컬에서 동작하는 가벼운 SOTA 모델
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    print(f"[INFO] 청킹 및 임베딩 생성 시작...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        total_chunks = 0
        
        for paper in papers:
            # 1. Summary 청킹
            chunks = chunk_text(paper['summary'], size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            
            if not chunks:
                continue

            # 2. 청크들에 대해 임베딩 생성 (일괄 처리)
            # embeddings는 numpy array 리스트로 반환됨
            embeddings = model.encode(chunks)

            # 3. 각 청크별 데이터 구성 및 저장
            for i, (chunk_text_str, vector) in enumerate(zip(chunks, embeddings)):
                
                # 고유 ID 생성 (논문ID_청크인덱스)
                chunk_id = f"{paper['id'].split('/')[-1]}_{i}"
                
                record = {
                    "id": chunk_id,
                    "paper_id": paper['id'],
                    "title": paper['title'],
                    "published": paper['published'],
                    "text": chunk_text_str,            # 실제 청킹된 텍스트
                    "embedding": vector.tolist(),      # numpy array -> list 변환 필수
                    "metadata": {
                        "source": "arxiv",
                        "categories": paper['categories'],
                        "chunk_index": i
                    }
                }
                
                # JSONL 한 줄 쓰기
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"[INFO] 저장 완료! 총 {total_chunks}개의 청크가 생성되었습니다.")
    print(f"[INFO] 파일 경로: {output_path}")

# -------------------
# 4) 엑셀 저장
# -------------------
def save_excel(papers, output_excel):
    # 엑셀 저장을 위해 리스트를 문자열로 변환한 단순화된 데이터 생성
    simple_data = []
    for p in papers:
        simple_data.append({
            "id": p["id"],
            "title": p["title"],
            "summary": p["summary"],
            "authors": ", ".join(p["authors"]),
            "categories": ", ".join(p["categories"]),
            "published": p["published"],
            "source_url": p["source_url"]
        })

    df = pd.DataFrame(simple_data)
    df.to_excel(output_excel, index=False)
    print(f"[INFO] Excel 저장 완료 → {output_excel}")

# -------------------
# 메인 실행
# -------------------
if __name__ == "__main__":
    # 1. 데이터 수집
    papers = fetch_arxiv_papers(TARGET_CATEGORIES, START_DATE, END_DATE, MAX_RESULTS)

    if papers:
        # 2. 임베딩 생성 및 JSONL 저장
        process_embeddings_and_save(papers, OUTPUT_JSONL)
        
        # 3. 원본 데이터 엑셀 저장
        save_excel(papers, OUTPUT_XLSX)
    else:
        print("[INFO] 데이터가 없어 종료합니다.")