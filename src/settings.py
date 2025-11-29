# # config.py
# import os
# from elasticsearch import Elasticsearch
# from openai import OpenAI   
# # --- 환경변수 ---
# # export OPENAI_API_KEY="sk-..."
# from dotenv import load_dotenv
# from pathlib import Path
# import os

# BASE_DIR = Path(__file__).resolve().parent.parent
# load_dotenv(BASE_DIR / ".env")

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# client = OpenAI()

# # --- Elasticsearch 설정 ---
# ES_HOST = os.getenv("ES_HOST", "localhost")
# ES_PORT = int(os.getenv("ES_PORT", "9200"))
# ES_USER = os.getenv("ES_USER", "elastic")
# ES_PASSWORD = os.getenv("ES_PASSWORD", "changeme")

# es = Elasticsearch(
#     hosts=[{"host": ES_HOST, "port": ES_PORT, "scheme": "http"}],
#     basic_auth=(ES_USER, ES_PASSWORD),
# )

# # --- 인덱스 / 필드 이름 ---
# ES_INDEX = "papers_index_v2"

# TITLE_FIELD = "title"
# ABSTRACT_FIELD = "abstract"
# CONTENT_FIELD = "content"  # title+abstract 합친 필드 써도 됨
# YEAR_FIELD = "year"
# CITATION_FIELD = "citations"
# VENUE_FIELD = "venue"
# URL_FIELD = "url"
# EMBED_FIELD = "embedding"  # dense_vector 필드

# # --- 검색 설정 ---
# TOP_K_SPARSE = 30
# TOP_K_DENSE = 30
# TOP_K_FINAL = 10

import os
from pathlib import Path
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import OpenAI

# --- .env 경로를 project_root 기준으로 지정 ---
BASE_DIR = Path(__file__).resolve().parent.parent  # src의 상위 폴더
load_dotenv(BASE_DIR / ".env")

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(" OPENAI_API_KEY not found. Check your .env path or content.")

client = OpenAI(api_key=OPENAI_API_KEY)

# # settings.py
# import os
# from elasticsearch import Elasticsearch
# from openai import OpenAI
# from dotenv import load_dotenv
# from pathlib import Path

# BASE_DIR = Path(__file__).resolve().parent
# load_dotenv(BASE_DIR / ".env")

# # --- OpenAI ---
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=OPENAI_API_KEY)

# --- Elasticsearch ---
ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = int(os.getenv("ES_PORT", "9200"))
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "changeme")

es = Elasticsearch(
    hosts=[{"host": ES_HOST, "port": ES_PORT, "scheme": "http"}],
    basic_auth=(ES_USER, ES_PASSWORD),
)

# --- 인덱스 / 필드 이름 ---
ES_INDEX = "papers_index_v2"
TITLE_FIELD = "title"
ABSTRACT_FIELD = "abstract"
CONTENT_FIELD = "content"
YEAR_FIELD = "year"
CITATION_FIELD = "citations"
VENUE_FIELD = "venue"
URL_FIELD = "url"
EMBED_FIELD = "embedding"

# --- 검색 설정 ---
TOP_K_SPARSE = 30
TOP_K_DENSE = 30
TOP_K_FINAL = 10

