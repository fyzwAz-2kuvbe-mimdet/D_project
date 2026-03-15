"""
D_project.py — 우리 아이 일기 감성 분석기 (Streamlit Cloud 배포용)
===================================================================
역할  : 데스크탑에서 학습·저장된 model_bundle.pkl 을 로드하여
        일기 텍스트를 입력받고 창의성·사회성·자존감을 분석합니다.
        TensorFlow 는 사용하지 않으므로 Streamlit Cloud 에서 바로 실행됩니다.

필요 파일:
  model_bundle.pkl   ← train_model.py 로 생성한 모델 번들 (같은 폴더에 위치)

실행  : streamlit run D_project.py
"""

import os
import pickle
import random
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# =============================================================================
# [상수]
# =============================================================================
GEMINI_MODEL   = "gemini-2.5-flash"
BUNDLE_PATH    = "model_bundle.pkl"   # train_model.py 가 생성하는 파일

THEME = {
    "bg":         "#f0f8ff",
    "card":       "#ffffff",
    "card_bg":    "#e3f2fd",
    "border":     "#90caf9",
    "primary":    "#2196f3",
    "primary_dk": "#1565c0",
    "text":       "#1a237e",
    "text_muted": "#90caf9",
    "danger":     "#ef5350",
    "fire":       "#ff7043",
}

CATS      = ["창의성", "사회성", "자존감"]
CAT_EMOJI = ["🎨",     "🤝",     "👑"]
CAT_COLOR = ["#FF6B6B", "#4ECDC4", "#FFD93D"]
CAT_BG    = ["#FFF0F0", "#F0FFFE", "#FFFBEA"]
CAT_DARK  = ["#CC3333", "#2A9D8F", "#C8960C"]


# =============================================================================
# [1] 모델 번들 로드
#     @st.cache_resource → 앱 재실행마다 pkl 을 다시 읽지 않습니다
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_bundle(path: str) -> dict | None:
    """
    model_bundle.pkl 을 읽어 딕셔너리로 반환합니다.
    파일이 없으면 None 을 반환합니다.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)

        # 가중치로 모델 복원 (TensorFlow 필요 없이 numpy 연산으로 추론)
        # → 모델 구조 JSON + 가중치로 Keras 모델 재구성
        import tensorflow as tf
        model = tf.keras.models.model_from_json(bundle["model_cfg"])
        model.set_weights(bundle["weights"])
        bundle["model"] = model
        return bundle
    except Exception as e:
        st.error(f"모델 로드 오류: {e}")
        return None


# =============================================================================
# [2] 추론 함수 (번들 내 tokenizer + model 사용)
# =============================================================================
def predict(bundle: dict, text: str) -> np.ndarray:
    """
    입력 텍스트의 카테고리별 확률을 반환합니다.
    bundle["tokenizer"] 와 bundle["model"] 을 사용합니다.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tok    = bundle["tokenizer"]
    model  = bundle["model"]
    maxlen = bundle["maxlen"]

    seq    = tok.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen, padding="post")
    return model.predict(padded, verbose=0)[0]


# =============================================================================
# [3] Gemini API — 따뜻한 한마디 생성
# =============================================================================
def gemini_warm_message(diary: str, best_cat: str, probs: np.ndarray) -> str:
    """
    Gemini API 로 일기 내용 기반 맞춤형 따뜻한 한마디를 생성합니다.
    API 키 미설정 또는 오류 시 템플릿 메시지로 폴백합니다.
    """
    try:
        from google import genai

        client    = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        prob_text = "  /  ".join(f"{CATS[i]} {probs[i]*100:.1f}%" for i in range(3))

        prompt = (
            "당신은 아이의 성장을 따뜻하게 응원하는 부모 또는 선생님입니다.\n\n"
            "아이가 쓴 일기를 읽고, 분석 결과를 바탕으로 아이에게 직접 전하는 "
            "따뜻하고 구체적인 한마디를 한국어로 작성하세요.\n\n"
            "[규칙]\n"
            "- 아이에게 직접 말하는 2인칭(\"너는\", \"네가\") 형식\n"
            "- 일기 내용을 구체적으로 언급할 것\n"
            f"- 가장 높은 지표({best_cat})를 중심으로 칭찬\n"
            "- 2~3문장, 이모지 1~2개 포함\n"
            "- 과도한 칭찬보다 진심 어린 공감과 격려\n\n"
            f"[일기 내용]\n{diary}\n\n"
            f"[분석 결과]\n{prob_text}\n가장 돋보이는 지표: {best_cat}\n\n"
            "아이에게 따뜻한 한마디를 전해주세요."
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text.strip()

    except KeyError:
        return _fallback(int(np.argmax(probs)))
    except Exception as e:
        st.warning(f"Gemini 연결 오류 ({e}) → 기본 메시지로 대체합니다.")
        return _fallback(int(np.argmax(probs)))


def _fallback(best_idx: int) -> str:
    """GEMINI_API_KEY 없을 때 카테고리별 템플릿 메시지."""
    pool = {
        0: [
            "네가 직접 생각해서 만들어낸 것들이 정말 특별해! 🎨 세상 어디에도 없는 너만의 아이디어가 너의 가장 큰 보물이란다.",
            "오늘 네가 보여준 창의력이 정말 대단했어! ✨ 생각을 현실로 만드는 그 능력, 앞으로도 꼭 키워나가렴.",
            "남들과 다르게 생각하는 것이 너의 특기야! 🌟 그 호기심이 언젠가 세상을 바꿀 거라고 믿어.",
        ],
        1: [
            "친구와 함께하는 시간을 소중히 여기는 네 마음이 정말 따뜻해! 🤝 그 마음 덕분에 네 곁에 좋은 친구들이 모이는 거야.",
            "다른 사람의 마음을 헤아릴 줄 아는 네가 정말 자랑스러워! 💛 오늘 네가 보여준 배려가 친구들에게 큰 힘이 됐을 거야.",
            "함께해서 더 즐거운 하루를 만들어 가는 너! 🌈 그 따뜻한 마음씨가 너를 정말 특별한 사람으로 만들어 준단다.",
        ],
        2: [
            "오늘 네가 해낸 일을 보면서 정말 뿌듯했어! 👑 스스로를 믿고 끝까지 해내는 너의 모습, 정말 멋있어.",
            "네가 스스로 해냈다는 사실이 가장 중요해! ⭐ 그 용기와 끈기가 앞으로 더 큰 성공을 만들어 줄 거야.",
            "오늘 하루 당당하게 살아간 너가 정말 자랑스러워! 🏆 자신을 사랑하는 그 마음, 언제나 잃지 말고 간직해 줘.",
        ],
    }
    return random.choice(pool.get(best_idx, pool[2]))


# =============================================================================
# [4] Plotly 차트
# =============================================================================
def chart_bar(probs: np.ndarray):
    best    = int(np.argmax(probs))
    ylabels = [f"{CAT_EMOJI[i]} {CATS[i]}" for i in range(3)]
    fig = go.Figure(go.Bar(
        x=[p * 100 for p in probs],
        y=ylabels,
        orientation="h",
        marker=dict(
            color=CAT_COLOR,
            opacity=[1.0 if i == best else 0.35 for i in range(3)],
            line=dict(width=0),
        ),
        text=[f"<b>{p*100:.1f}%</b>" for p in probs],
        textposition="outside",
        textfont=dict(size=14),
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 125], showticklabels=False,
                   showgrid=False, zeroline=False),
        yaxis=dict(tickfont=dict(size=15)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=55, t=10, b=10),
        height=200,
    )
    return fig


def chart_radar(probs: np.ndarray):
    theta = CATS + [CATS[0]]
    r     = [p * 100 for p in probs] + [probs[0] * 100]
    fig = go.Figure(go.Scatterpolar(
        r=r, theta=theta,
        fill="toself",
        fillcolor="rgba(33,150,243,0.15)",
        line=dict(color=THEME["primary"], width=2.5),
        marker=dict(size=8, color=CAT_COLOR + [CAT_COLOR[0]]),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(size=9), ticksuffix="%",
                            gridcolor="#DDE"),
            angularaxis=dict(tickfont=dict(size=14), gridcolor="#DDE"),
            bgcolor="rgba(0,0,0,0)",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=45, r=45, t=25, b=25),
        height=270,
    )
    return fig


# =============================================================================
# [5] 전역 CSS
# =============================================================================
def inject_css():
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700;800&display=swap');
html, body, [class*="css"] {{
    font-family: 'Nanum Gothic', sans-serif;
    background-color: {THEME["bg"]};
    color: {THEME["text"]};
}}
.block-container {{ max-width: 820px; padding-top: 1.8rem; }}

.hero {{
    text-align: center;
    padding: 1.5rem 1rem 1.1rem;
    background: linear-gradient(135deg, {THEME["card_bg"]}, #ffffff);
    border-radius: 20px;
    border: 1px solid {THEME["border"]};
    margin-bottom: 1.6rem;
}}
.hero-title {{
    font-size: 2.3rem; font-weight: 900; line-height: 1.2;
    background: linear-gradient(135deg, {THEME["primary"]}, {THEME["fire"]});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    display: inline-block;
}}
.hero-sub {{ color: {THEME["text_muted"]}; font-size: .88rem; margin-top: .35rem; }}

.sec-label {{
    font-size: .7rem; font-weight: 800; letter-spacing: 1.5px;
    color: {THEME["text_muted"]}; text-transform: uppercase; margin-bottom: .5rem;
}}

.model-ok   {{ color: #4ECDC4; font-size: .82rem; font-weight: 700; margin-top:.3rem; }}
.model-none {{ color: {THEME["danger"]}; font-size: .82rem; font-weight: 700; margin-top:.3rem; }}

.metric-card {{
    border-radius: 14px; padding: .9rem; text-align: center; margin-bottom: .4rem;
}}

.warm-box {{
    background: linear-gradient(135deg, #FFFBEA, #EFF9FF);
    border: 1.5px solid {THEME["border"]};
    border-radius: 18px; padding: 1.4rem 1.8rem; margin-top: .6rem;
    overflow: hidden;
}}
.warm-ql {{
    font-size: 3.0rem; color: {THEME["primary"]}; opacity: .28;
    font-family: Georgia, serif; line-height: .8;
    display: inline-block; margin-bottom: -.4rem;
}}
.warm-qr {{
    font-size: 3.0rem; color: {THEME["primary"]}; opacity: .28;
    font-family: Georgia, serif; line-height: .8;
    display: block; text-align: right; margin-top: -.4rem;
}}
.warm-text {{
    font-size: 1.04rem; line-height: 1.9; color: #444;
    font-style: italic; padding: .3rem .6rem;
}}
.best-badge {{
    display: inline-block; border-radius: 12px;
    padding: .5rem 1.1rem; font-size: .88rem; font-weight: 800; margin-top: .7rem;
}}

.info-box {{
    background: {THEME["card_bg"]};
    border: 1px solid {THEME["border"]};
    border-radius: 14px; padding: 1rem 1.4rem; margin-top: .5rem;
    font-size: .88rem; line-height: 1.8; color: {THEME["text"]};
}}
.info-box code {{
    background: #fff; border-radius: 6px;
    padding: .15rem .45rem; font-size: .82rem; color: {THEME["primary_dk"]};
}}
.step-num {{
    display: inline-block; background: {THEME["primary"]};
    color: #fff; border-radius: 50%; width: 1.5rem; height: 1.5rem;
    text-align: center; line-height: 1.5rem; font-size: .78rem;
    font-weight: 800; margin-right: .4rem;
}}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# [6] 메인 앱
# =============================================================================
def main():
    st.set_page_config(
        page_title="우리 아이 일기 분석기",
        page_icon="📔",
        layout="centered",
    )
    inject_css()

    # ── 히어로 헤더 ──────────────────────────────────────────────────────────
    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">📔 우리 아이 일기 분석기</div>'
        '<div class="hero-sub">'
        'TensorFlow 딥러닝 &nbsp;·&nbsp; Google Gemini AI'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # ── 모델 번들 로드 ───────────────────────────────────────────────────────
    bundle = load_bundle(BUNDLE_PATH)

    # ── 모델 없음 → 안내 화면 표시 ──────────────────────────────────────────
    if bundle is None:
        st.markdown(
            '<p class="model-none">● 모델 파일(model_bundle.pkl)이 없습니다.</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="info-box">'
            '<b>📋 사용 방법 — 데스크탑에서 모델을 먼저 학습하세요</b><br><br>'
            '<span class="step-num">1</span>'
            '데스크탑에서 <code>train_model.py</code> 실행<br>'
            '<span class="step-num">2</span>'
            '생성된 <code>model_bundle.pkl</code> 파일을 GitHub 저장소에 커밋<br>'
            '<span class="step-num">3</span>'
            'Streamlit Cloud 에서 앱을 재시작하면 자동으로 모델을 로드합니다<br><br>'
            '학습 데이터를 추가하려면 <code>training_data.json</code> 에 문장을 추가한 뒤<br>'
            '<code>train_model.py</code> 를 다시 실행하고 <code>model_bundle.pkl</code> 을 다시 업로드하세요.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ── 모델 정보 표시 ───────────────────────────────────────────────────────
    cats = bundle.get("categories", CATS)
    st.markdown(
        f'<p class="model-ok">● 모델 로드 완료'
        f' &nbsp;│&nbsp; 카테고리: {" · ".join(cats)}'
        f' &nbsp;│&nbsp; 번들 버전: {bundle.get("version","–")}</p>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── 일기 입력 ────────────────────────────────────────────────────────────
    st.markdown('<p class="sec-label">✏️ 오늘의 일기를 입력하세요</p>',
                unsafe_allow_html=True)

    diary = st.text_area(
        label="일기",
        placeholder=(
            "오늘 있었던 일을 자유롭게 써보세요!\n"
            "예) 오늘 새로운 게임 맵을 혼자서 만들어봤는데 "
            "생각보다 너무 잘 나와서 기분이 최고였다!"
        ),
        height=130,
        label_visibility="collapsed",
        key="diary_input",
    )

    _, col_btn = st.columns([3, 1])
    with col_btn:
        run_btn = st.button("🔍 분석하기", use_container_width=True, type="primary")

    if not run_btn:
        return

    if not diary.strip():
        st.warning("일기를 먼저 입력해 주세요!")
        return

    # ── 딥러닝 추론 ──────────────────────────────────────────────────────────
    with st.spinner("딥러닝 분석 중…"):
        probs    = predict(bundle, diary)
        best_idx = int(np.argmax(probs))

    st.divider()

    # ── 그래프 2종 ───────────────────────────────────────────────────────────
    st.markdown('<p class="sec-label">📊 분석 결과</p>', unsafe_allow_html=True)
    col_bar, col_radar = st.columns([1.1, 1])
    with col_bar:
        st.caption("막대 차트")
        st.plotly_chart(chart_bar(probs), use_container_width=True,
                        config={"displayModeBar": False})
    with col_radar:
        st.caption("레이더 차트")
        st.plotly_chart(chart_radar(probs), use_container_width=True,
                        config={"displayModeBar": False})

    # ── 수치 카드 3개 ────────────────────────────────────────────────────────
    cols = st.columns(3)
    for i, col in enumerate(cols):
        with col:
            is_best = (i == best_idx)
            bg      = CAT_BG[i]    if is_best else "#F8F8F8"
            border  = CAT_COLOR[i] if is_best else "#DDD"
            fw      = "900"        if is_best else "600"
            top_tag = (
                f"<div style='font-size:.62rem;color:{CAT_DARK[i]};"
                f"font-weight:800;margin-top:3px'>★ TOP</div>"
            ) if is_best else ""
            st.markdown(
                f'<div class="metric-card" '
                f'style="background:{bg};border:2px solid {border};">'
                f'<div style="font-size:1.9rem">{CAT_EMOJI[i]}</div>'
                f'<div style="font-weight:{fw};font-size:.88rem;color:#333">{CATS[i]}</div>'
                f'<div style="font-size:1.45rem;font-weight:900;color:{CAT_COLOR[i]}">'
                f'{probs[i]*100:.1f}%</div>'
                f'{top_tag}</div>',
                unsafe_allow_html=True,
            )

    # ── 최고 지표 뱃지 ───────────────────────────────────────────────────────
    st.markdown(
        f'<div class="best-badge" '
        f'style="background:{CAT_BG[best_idx]};'
        f'color:{CAT_DARK[best_idx]};'
        f'border:1.5px solid {CAT_COLOR[best_idx]};">'
        f'{CAT_EMOJI[best_idx]}&nbsp;'
        f'오늘 일기에서는 <b>{CATS[best_idx]}</b> 지표가 가장 돋보여요!'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Gemini 따뜻한 한마디 ─────────────────────────────────────────────────
    st.markdown('<p class="sec-label">💌 따뜻한 한마디 (Gemini AI)</p>',
                unsafe_allow_html=True)
    with st.spinner("💬 Gemini가 따뜻한 말을 만들고 있어요…"):
        msg = gemini_warm_message(diary, CATS[best_idx], probs)

    st.markdown(
        f'<div class="warm-box">'
        f'<span class="warm-ql">&ldquo;</span>'
        f'<p class="warm-text">{msg}</p>'
        f'<span class="warm-qr">&rdquo;</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 다른 일기 분석하기"):
        st.rerun()

    # ── 푸터 ─────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("학습 데이터 추가 → training_data.json 수정 → train_model.py 재실행 → model_bundle.pkl 커밋")


if __name__ == "__main__":
    main()


# =============================================================================
# ⚠️  [Firestore 보안 규칙 설정 안내]
# =============================================================================
# 이 버전은 Firestore 를 사용하지 않습니다.
# 모델 데이터는 model_bundle.pkl 파일로 관리됩니다.
# Firebase 를 추가하려면 이전 버전(탭3 포함)을 참고하세요.
# =============================================================================
