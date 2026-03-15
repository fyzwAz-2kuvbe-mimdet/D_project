"""
D_project.py — 우리 아이 일기 감성 분석기 (Streamlit Cloud 배포용)
===================================================================
TensorFlow 완전 불필요 — numpy 만으로 추론합니다.
데스크탑에서 train_model.py 로 학습 후 저장된 model_bundle.pkl 을 로드합니다.
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
GEMINI_MODEL = "gemini-2.5-flash"
BUNDLE_PATH  = "model_bundle.pkl"

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
# [1] numpy 전용 추론 엔진
#     TensorFlow 없이 가중치 행렬 연산만으로 Bidirectional LSTM 추론
# =============================================================================

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def _tanh(x):
    return np.tanh(np.clip(x, -500, 500))

def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def _relu(x):
    return np.maximum(0, x)

def _lstm_step(x, h_prev, c_prev, W, U, b):
    """
    LSTM 단일 스텝 연산 (numpy)
    gates = sigmoid/tanh(W·x + U·h + b)
    """
    z    = W @ x + U @ h_prev + b
    d    = len(h_prev)
    i    = _sigmoid(z[0*d : 1*d])   # input gate
    f    = _sigmoid(z[1*d : 2*d])   # forget gate
    g    = _tanh(   z[2*d : 3*d])   # cell gate
    o    = _sigmoid(z[3*d : 4*d])   # output gate
    c    = f * c_prev + i * g
    h    = o * _tanh(c)
    return h, c

def _run_lstm(seq, weights_dict, units):
    """단방향 LSTM 전체 시퀀스 실행"""
    W = weights_dict["W"]   # (4*units, emb_dim)
    U = weights_dict["U"]   # (4*units, units)
    b = weights_dict["b"]   # (4*units,)
    h = np.zeros(units)
    c = np.zeros(units)
    for x in seq:
        h, c = _lstm_step(x, h, c, W, U, b)
    return h

def numpy_predict(bundle: dict, text: str) -> np.ndarray:
    """
    TensorFlow 없이 numpy 만으로 추론합니다.
    bundle 에서 가중치와 tokenizer 를 꺼내 직접 행렬 연산합니다.

    모델 구조: Embedding → BiLSTM → Dense(relu) → Dropout(추론시 비활성) → Dense(softmax)
    """
    weights  = bundle["weights"]
    tok      = bundle["tokenizer"]
    maxlen   = bundle["maxlen"]

    # ── 토크나이징 + 패딩 ────────────────────────────────────────────────────
    word_index = tok.word_index
    oov_idx    = tok.word_index.get("<OOV>", 1)
    tokens     = text.replace("\n", " ").split()
    seq        = [word_index.get(w, oov_idx) for w in tokens][:maxlen]
    seq        = seq + [0] * (maxlen - len(seq))   # post-padding

    # ── 가중치 언패킹 ────────────────────────────────────────────────────────
    # Keras Sequential: [Embedding, BiLSTM(forward), BiLSTM(backward),
    #                    Dense(relu), Dense(softmax)]
    # weights 순서: [emb, W_f, U_f, b_f, W_b, U_b, b_b, W_d1, b_d1, W_d2, b_d2]
    emb_matrix = weights[0]                          # (vocab, emb_dim)
    W_f, U_f, b_f = weights[1], weights[2], weights[3]   # forward LSTM
    W_b, U_b, b_b = weights[4], weights[5], weights[6]   # backward LSTM
    W_d1, b_d1    = weights[7], weights[8]           # Dense relu
    W_d2, b_d2    = weights[9], weights[10]          # Dense softmax

    units = b_d1.shape[0]   # Dense 유닛 수

    # ── Embedding lookup ─────────────────────────────────────────────────────
    embedded = np.array([
        emb_matrix[idx] if idx < len(emb_matrix) else np.zeros(emb_matrix.shape[1])
        for idx in seq
    ])  # (maxlen, emb_dim)

    # ── Bidirectional LSTM ───────────────────────────────────────────────────
    h_fwd = _run_lstm(embedded,        {"W": W_f, "U": U_f, "b": b_f}, W_f.shape[0]//4)
    h_bwd = _run_lstm(embedded[::-1],  {"W": W_b, "U": U_b, "b": b_b}, W_b.shape[0]//4)
    h_bi  = np.concatenate([h_fwd, h_bwd])   # (2*lstm_units,)

    # ── Dense(relu) ──────────────────────────────────────────────────────────
    d1 = _relu(W_d1.T @ h_bi + b_d1)

    # ── Dense(softmax) ───────────────────────────────────────────────────────
    logits = W_d2.T @ d1 + b_d2
    probs  = _softmax(logits)

    return probs


# =============================================================================
# [2] 모델 번들 로드  (@st.cache_resource → 최초 1회만 실행)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_bundle(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return bundle
    except Exception as e:
        st.error(f"모델 로드 오류: {e}")
        return None


# =============================================================================
# [3] Gemini API — 따뜻한 한마디
# =============================================================================
def gemini_warm_message(diary: str, best_cat: str, probs: np.ndarray) -> str:
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
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return response.text.strip()
    except KeyError:
        return _fallback(int(np.argmax(probs)))
    except Exception as e:
        st.warning(f"Gemini 연결 오류 ({e}) → 기본 메시지로 대체합니다.")
        return _fallback(int(np.argmax(probs)))


def _fallback(best_idx: int) -> str:
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
def chart_bar(probs):
    best    = int(np.argmax(probs))
    ylabels = [f"{CAT_EMOJI[i]} {CATS[i]}" for i in range(3)]
    fig = go.Figure(go.Bar(
        x=[p * 100 for p in probs], y=ylabels, orientation="h",
        marker=dict(color=CAT_COLOR,
                    opacity=[1.0 if i == best else 0.35 for i in range(3)],
                    line=dict(width=0)),
        text=[f"<b>{p*100:.1f}%</b>" for p in probs],
        textposition="outside", textfont=dict(size=14),
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 125], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(tickfont=dict(size=15)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=55, t=10, b=10), height=200,
    )
    return fig


def chart_radar(probs):
    theta = CATS + [CATS[0]]
    r     = [p * 100 for p in probs] + [probs[0] * 100]
    fig = go.Figure(go.Scatterpolar(
        r=r, theta=theta, fill="toself",
        fillcolor="rgba(33,150,243,0.15)",
        line=dict(color=THEME["primary"], width=2.5),
        marker=dict(size=8, color=CAT_COLOR + [CAT_COLOR[0]]),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(size=9), ticksuffix="%", gridcolor="#DDE"),
            angularaxis=dict(tickfont=dict(size=14), gridcolor="#DDE"),
            bgcolor="rgba(0,0,0,0)",
        ),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=45, r=45, t=25, b=25), height=270,
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
    background-color: {THEME["bg"]}; color: {THEME["text"]};
}}
.block-container {{ max-width: 820px; padding-top: 1.8rem; }}
.hero {{
    text-align: center; padding: 1.5rem 1rem 1.1rem;
    background: linear-gradient(135deg, {THEME["card_bg"]}, #ffffff);
    border-radius: 20px; border: 1px solid {THEME["border"]}; margin-bottom: 1.6rem;
}}
.hero-title {{
    font-size: 2.3rem; font-weight: 900; line-height: 1.2;
    background: linear-gradient(135deg, {THEME["primary"]}, {THEME["fire"]});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: inline-block;
}}
.hero-sub {{ color: {THEME["text_muted"]}; font-size: .88rem; margin-top: .35rem; }}
.sec-label {{
    font-size: .7rem; font-weight: 800; letter-spacing: 1.5px;
    color: {THEME["text_muted"]}; text-transform: uppercase; margin-bottom: .5rem;
}}
.model-ok   {{ color: #4ECDC4; font-size: .82rem; font-weight: 700; margin-top:.3rem; }}
.model-none {{ color: {THEME["danger"]}; font-size: .82rem; font-weight: 700; margin-top:.3rem; }}
.metric-card {{ border-radius: 14px; padding: .9rem; text-align: center; margin-bottom: .4rem; }}
.warm-box {{
    background: linear-gradient(135deg, #FFFBEA, #EFF9FF);
    border: 1.5px solid {THEME["border"]}; border-radius: 18px;
    padding: 1.4rem 1.8rem; margin-top: .6rem; overflow: hidden;
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
    background: {THEME["card_bg"]}; border: 1px solid {THEME["border"]};
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
    st.set_page_config(page_title="우리 아이 일기 분석기", page_icon="📔", layout="centered")
    inject_css()

    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">📔 우리 아이 일기 분석기</div>'
        '<div class="hero-sub">TensorFlow 딥러닝 &nbsp;·&nbsp; Google Gemini AI</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # 모델 번들 로드
    bundle = load_bundle(BUNDLE_PATH)

    # 모델 없음 → 안내 화면
    if bundle is None:
        st.markdown('<p class="model-none">● 모델 파일(model_bundle.pkl)이 없습니다.</p>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box"><b>📋 사용 방법</b><br><br>'
            '<span class="step-num">1</span>데스크탑에서 <code>train_model.py</code> 실행<br>'
            '<span class="step-num">2</span>생성된 <code>model_bundle.pkl</code>을 GitHub에 커밋<br>'
            '<span class="step-num">3</span>Streamlit Cloud 앱 재시작 → 자동 로드</div>',
            unsafe_allow_html=True,
        )
        return

    # 모델 정보 표시
    cats = bundle.get("categories", CATS)
    st.markdown(
        f'<p class="model-ok">● 모델 로드 완료'
        f' &nbsp;│&nbsp; 카테고리: {" · ".join(cats)}'
        f' &nbsp;│&nbsp; 버전: {bundle.get("version", "–")}</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    # 일기 입력
    st.markdown('<p class="sec-label">✏️ 오늘의 일기를 입력하세요</p>', unsafe_allow_html=True)
    diary = st.text_area(
        label="일기", label_visibility="collapsed",
        placeholder=(
            "오늘 있었던 일을 자유롭게 써보세요!\n"
            "예) 오늘 새로운 게임 맵을 혼자서 만들어봤는데 생각보다 너무 잘 나와서 기분이 최고였다!"
        ),
        height=130, key="diary_input",
    )
    _, col_btn = st.columns([3, 1])
    with col_btn:
        run_btn = st.button("🔍 분석하기", use_container_width=True, type="primary")

    if not run_btn:
        return
    if not diary.strip():
        st.warning("일기를 먼저 입력해 주세요!")
        return

    # 추론
    with st.spinner("딥러닝 분석 중…"):
        probs    = numpy_predict(bundle, diary)
        best_idx = int(np.argmax(probs))

    st.divider()

    # 그래프 2종
    st.markdown('<p class="sec-label">📊 분석 결과</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.caption("막대 차트")
        st.plotly_chart(chart_bar(probs), use_container_width=True,
                        config={"displayModeBar": False})
    with c2:
        st.caption("레이더 차트")
        st.plotly_chart(chart_radar(probs), use_container_width=True,
                        config={"displayModeBar": False})

    # 수치 카드 3개
    cols = st.columns(3)
    for i, col in enumerate(cols):
        with col:
            is_best = (i == best_idx)
            bg     = CAT_BG[i]    if is_best else "#F8F8F8"
            border = CAT_COLOR[i] if is_best else "#DDD"
            fw     = "900"        if is_best else "600"
            top    = (f"<div style='font-size:.62rem;color:{CAT_DARK[i]};"
                      f"font-weight:800;margin-top:3px'>★ TOP</div>") if is_best else ""
            st.markdown(
                f'<div class="metric-card" style="background:{bg};border:2px solid {border};">'
                f'<div style="font-size:1.9rem">{CAT_EMOJI[i]}</div>'
                f'<div style="font-weight:{fw};font-size:.88rem;color:#333">{CATS[i]}</div>'
                f'<div style="font-size:1.45rem;font-weight:900;color:{CAT_COLOR[i]}">'
                f'{probs[i]*100:.1f}%</div>{top}</div>',
                unsafe_allow_html=True,
            )

    # 최고 지표 뱃지
    st.markdown(
        f'<div class="best-badge" style="background:{CAT_BG[best_idx]};'
        f'color:{CAT_DARK[best_idx]};border:1.5px solid {CAT_COLOR[best_idx]};">'
        f'{CAT_EMOJI[best_idx]}&nbsp;오늘 일기에서는 <b>{CATS[best_idx]}</b> 지표가 가장 돋보여요!'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Gemini 따뜻한 한마디
    st.markdown('<p class="sec-label">💌 따뜻한 한마디 (Gemini AI)</p>', unsafe_allow_html=True)
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

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("학습 데이터 추가 → training_data.json 수정 → train_model.py 재실행 → model_bundle.pkl 커밋")


if __name__ == "__main__":
    main()
