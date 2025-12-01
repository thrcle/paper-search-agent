# app.py
import streamlit as st
import pandas as pd
from main_agent import run_query

# ─────────────────────────────────────
# 페이지 & 세션 설정
# ─────────────────────────────────────
st.set_page_config(
    page_title="Dr.Paper - AI 논문 진단 에이전트",
    page_icon="🩺",
    layout="wide",
)

# 세션 메모리 초기화
if "memory" not in st.session_state:
    st.session_state["memory"] = []  # run_query에 넘길 메모리

# 마지막 판단 로그 저장용
if "last_trace" not in st.session_state:
    st.session_state["last_trace"] = []    

# ─────────────────────────────────────
# 상단 헤더
# ─────────────────────────────────────
st.markdown("<h1 style='text-align: center;'>🩺 Dr.Paper</h1>", unsafe_allow_html=True)
st.markdown(
    """
<p style="text-align:center; color:gray; font-size:16px;">
AI 기반 연구 논문 검색·추천 에이전트<br>
당신의 연구 주제에 맞는 논문을 진단하고, 처방해 드립니다.
</p>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────
# 입력 영역
# ─────────────────────────────────────
query = st.text_area(
    "🔎 관심 연구 주제나 질문을 입력하세요",
    placeholder="예: RAG",
    height=80,
)
search_btn = st.button("논문 진단하기", use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────
# 중간 레이아웃: 왼쪽 AI 진단(+진료기록) / 오른쪽 논문 카드
# ─────────────────────────────────────
left_col, right_col = st.columns([1, 3])

state = None
top_papers = []
answer = ""
utter_type = "NL_TOPIC"
strategy = "hybrid"
rel_str = "N/A"

# ─────────────────────────────────────
# 검색 버튼 클릭 시
# ─────────────────────────────────────
if search_btn and query.strip():
    with st.spinner("AI가 논문을 분석 중입니다..."):
        state = run_query(query.strip(), memory=st.session_state["memory"])

    # run_query에서 memory_update_agent가 채운 메모리를 다시 세션에 저장
    updated_memory = state.get("memory", st.session_state["memory"]) if state else []
    st.session_state["memory"] = updated_memory

    answer = state.get("answer", "") if state else ""
    top_papers = state.get("top_papers", []) if state else []
    utter_type = state.get("utterance_type", "NL_TOPIC") if state else "NL_TOPIC"
    strategy = state.get("search_strategy", "hybrid") if state else "hybrid"
    relevance = state.get("relevance_score", None) if state else None
    rel_str = f"{relevance:.2f}" if isinstance(relevance, (int, float)) else "N/A"

    # 판단 로그 가져오기
    trace = state.get("reasoning_trace", []) if state else []
    st.session_state["last_trace"] = trace


    # ───────── 왼쪽: AI 진단 + 진료기록 토글 ─────────
    with left_col:
        st.markdown("### 🩺 AI 진단")

        st.markdown(
            f"""
**질의 유형**  
- {utter_type}

**선택된 검색 전략**  
- {strategy}

"""
        )

        # ✅ main_agent.py의 LLM 판단 정보 표시
        if state:
            llm_decision = state.get("llm_decision")
            external_strategy = state.get("external_strategy")

            if llm_decision or external_strategy:
                
                st.markdown("**LLM 판단 요약**")
                if llm_decision:
                    st.markdown(
                        f"<div style='color:#444; font-size:13px; padding-left:8px;'>🧠 {llm_decision}</div>",
                        unsafe_allow_html=True,
                    )
                if external_strategy:
                    st.markdown(
                        f"<div style='color:#666; font-size:13px; padding-left:8px;'>🌐 External Search: <b>{external_strategy}</b></div>",
                        unsafe_allow_html=True,
                    )
            # 🔹 여기: 판단 로그 토글 추가
            trace = state.get("reasoning_trace", []) or []
            with st.expander("🧠 판단 로그 보기", expanded=False):
                if trace:
                    for i, line in enumerate(trace, start=1):
                        st.markdown(f"**{i}.** {line}")
                else:
                    st.info("이번 턴 판단 로그가 없습니다.")

        st.markdown("---")
        st.markdown("**진단 단계**")
        st.markdown(
            """
1. 문장 의미 벡터화  
2. 초록·키워드 기반 후보 논문 탐색  
3. 키워드 BM25 + 임베딩 검색 결합  
4. LLM 기반 요약 및 응답 생성  
5. 품질 평가 및 필요한 경우 외부 논문 API로 보완
"""
        )

        st.markdown("---")


        # 📝 진료기록: 왼쪽 패널에서 토글(expander)로 표시
        memory = st.session_state["memory"]
        with st.expander("📝 진료 기록 보기", expanded=False):
            if isinstance(memory, list) and memory:
                histories = []
                for m in memory:
                    if isinstance(m, dict):
                        q_text = m.get("query_text") or m.get("query")
                    else:
                        q_text = str(m)
                    if q_text:
                        histories.append(q_text)

                if histories:
                    df_hist = pd.DataFrame(
                        [{"이전 질문": q} for q in histories[::-1]]
                    )
                    st.dataframe(df_hist, use_container_width=True, hide_index=True)
                else:
                    st.info("아직 저장된 진료 기록이 없습니다.")
            else:
                st.info("아직 저장된 진료 기록이 없습니다.")

    # ───────── 오른쪽: 처방된 논문 카드 ─────────
    with right_col:
        st.markdown(f"### 💊 처방된 논문")

        if not top_papers:
            st.warning("🔍 관련 논문을 찾지 못했습니다.")
        else:
            for i, p in enumerate(top_papers, start=1):
                title = p.get("title", "제목 없음")
                year = p.get("year", "")
                citations = p.get("citations", 0)
                url = p.get("url", "")
                score = p.get("score", 0.0)
                preview = (p.get("content") or "").replace("\n", " ")[:200]
                summary = p.get("summary") or preview
                score_pct = int(min(max(score, 0.0), 1.0) * 100)
                if url and not url.startswith("http"):
                    url = "https://" + url

                st.markdown(
                    f"""
<div style="padding:14px 18px; margin-bottom:10px; border-radius:12px; border:1px solid #e5e5e5; background-color:white;">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div style="font-size:15px; color:#666;"> 논문 #{i}</div>
    <div style="font-size:14px; color:#4a6cf7; font-weight:bold;">{score_pct}% <span style="color:#999; font-weight:normal;">의미 기반 매칭</span></div>
  </div>
  <div style="margin-top:6px; font-size:18px; font-weight:600;">
    {f'<a href="{url}" target="_blank" style="text-decoration:none; color:#4a6cf7;">🔗 {title}</a>' if url else title}
  </div>
  <div style="margin-top:2px; font-size:13px; color:#777;">
    {year} · 인용 {citations}
  </div>
  <div style="margin-top:8px; padding:8px 10px; background-color:#f7f7ff; border-radius:8px; font-size:13px; color:#555;">
    {summary}
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────
# 검색 전 기본 화면
# ─────────────────────────────────────
else:
    with left_col:
        st.markdown(f"### 🧑‍⚕️ AI 진단")
        st.write("질문을 입력하고 **논문 진단하기** 버튼을 눌러주세요.")
        st.markdown(
            """
1. 문장 의미 벡터화  
2. 초록과 의미 비교  
3. 키워드로 정밀도 확보  
"""
        )
    # 🔹 검색 전 상태용 판단 로그 토글
        with st.expander("🧠 판단 로그 보기", expanded=False):
            st.info("아직 판단 로그가 없습니다. 질문을 입력하고 진단을 실행해 주세요.")
        # 기본 상태에서도 진료기록 토글은 보이게
        memory = st.session_state["memory"]
        with st.expander("📝 진료 기록 보기", expanded=False):
            if isinstance(memory, list) and memory:
                histories = []
                for m in memory:
                    if isinstance(m, dict):
                        q_text = m.get("query_text") or m.get("query")
                    else:
                        q_text = str(m)
                    if q_text:
                        histories.append(q_text)

                if histories:
                    df_hist = pd.DataFrame(
                        [{"이전 질문": q} for q in histories[::-1]]
                    )
                    st.dataframe(df_hist, use_container_width=True, hide_index=True)
                else:
                    st.info("아직 저장된 진료 기록이 없습니다.")
            else:
                st.info("아직 저장된 진료 기록이 없습니다.")

    # with right_col:
    #     st.markdown("### 💊 처방된 논문")
    #     st.info("아직 처방된 논문이 없습니다. 질문을 입력해 주세요.")
with right_col:
    # st.markdown("### 💊 처방된 논문")

    # 메모리 기반 질의인 경우
    if utter_type == "MEMORY_QUERY":
        if answer:
            st.markdown(
                f"""
<div style='padding:18px; border-radius:12px; background-color:#f0f7ff; border:1px solid #cfe3ff;'>
  <b>🧠 이전 진단 기록 기반 응답</b><br>
  {answer}
</div>
""",
                unsafe_allow_html=True,
            )
            st.info("이 질문은 이전 진단 기록(메모리)을 기반으로 답변했어요. 새로운 논문 검색은 수행하지 않았습니다.")
        else:
            st.warning("아직 저장된 진료 기록이 없어, 첫 번째 질문을 찾지 못했어요. 먼저 일반 논문 질의를 실행해 주세요.")

    # 일반 검색 질의
    else:
        if not top_papers:
            st.warning("🔍 관련 논문을 찾지 못했습니다.")
        else:
            for i, p in enumerate(top_papers, start=1):
                title = p.get("title", "제목 없음")
                year = p.get("year", "")
                citations = p.get("citations", 0)
                url = p.get("url", "")
                score = p.get("score", 0.0)
                preview = (p.get("content") or "").replace("\n", " ")[:200]
                summary = p.get("summary") or preview
                score_pct = int(min(max(score, 0.0), 1.0) * 100)
                if url and not url.startswith("http"):
                    url = "https://" + url

                st.markdown(
                    f"""
<div style="padding:14px 18px; margin-bottom:10px; border-radius:12px; border:1px solid #e5e5e5; background-color:white;">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div style="font-size:15px; color:#666;"> 논문 #{i}</div>
    <div style="font-size:14px; color:#4a6cf7; font-weight:bold;">{score_pct}% <span style="color:#999; font-weight:normal;">의미 기반 매칭</span></div>
  </div>
  <div style="margin-top:6px; font-size:18px; font-weight:600;">
    {f'<a href="{url}" target="_blank" style="text-decoration:none; color:#4a6cf7;">🔗 {title}</a>' if url else title}
  </div>
  <div style="margin-top:2px; font-size:13px; color:#777;">
    {year} · 인용 {citations}
  </div>
  <div style="margin-top:8px; padding:8px 10px; background-color:#f7f7ff; border-radius:8px; font-size:13px; color:#555;">
    {summary}
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )


