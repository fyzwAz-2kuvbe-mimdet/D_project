"""
D_project.py — 우리 아이 일기 감성 분석기
==========================================
스택  : Streamlit · TensorFlow (Bidirectional LSTM) · Firebase Firestore · Google Gemini API
구조  :
  📄 탭1 — 일기 분석   : 일기 입력 → 딥러닝 분석 → 그래프 → Gemini 따뜻한 한마디
  📚 탭2 — 학습 데이터 : Firestore 데이터 조회 / 추가 / 삭제
  🧠 탭3 — 모델 학습   : Epoch·데이터 소스 선택 후 재학습 (언제든 가능)

실행  : streamlit run D_project.py
"""

# ── 표준 라이브러리 ────────────────────────────────────────────────────────────
import time
import random
import datetime

# ── 외부 라이브러리 ────────────────────────────────────────────────────────────
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# =============================================================================
# [상수] 모델 ID — 버전 교체 시 여기만 수정
# =============================================================================
GEMINI_MODEL = "gemini-2.5-flash"

# =============================================================================
# [상수] UI 테마 — 색상을 바꾸면 전체 UI에 반영됩니다
# =============================================================================
THEME = {
    "bg":         "#f0f8ff",   # 앱 전체 배경 (하늘빛 흰색)
    "card":       "#ffffff",   # 카드/대화창 배경
    "card_bg":    "#e3f2fd",   # 연한 카드 배경
    "border":     "#90caf9",   # 테두리 색
    "primary":    "#2196f3",   # 메인 강조색 (버튼, 제목)
    "primary_dk": "#1565c0",   # 진한 강조색
    "text":       "#1a237e",   # 본문 텍스트
    "text_muted": "#90caf9",   # 흐린 텍스트
    "danger":     "#ef5350",   # 경고색
    "fire":       "#ff7043",   # 불꽃 강조색
}

# =============================================================================
# [상수] 카테고리 정보
# =============================================================================
CATS      = ["창의성", "사회성", "자존감"]
CAT_EMOJI = ["🎨",     "🤝",     "👑"]
CAT_COLOR = ["#FF6B6B", "#4ECDC4", "#FFD93D"]
CAT_BG    = ["#FFF0F0", "#F0FFFE", "#FFFBEA"]
CAT_DARK  = ["#CC3333", "#2A9D8F", "#C8960C"]

# =============================================================================
# [상수] 기본 학습 데이터 — Firestore가 비어있거나 연결 실패 시 사용
# =============================================================================
DEFAULT_TRAINING_DATA = [
    {"text": "오늘 찰흙으로 아무도 생각 못한 새로운 모양의 스퀴시를 만들었다.", "label": 0},
    {"text": "캡컷으로 나만의 재미있는 짧은 영상을 기발하게 편집해봤다.",       "label": 0},
    {"text": "블록을 조립해서 설명서에 없는 나만의 우주선을 완성했다.",          "label": 0},
    {"text": "색종이로 설명서 없이 내가 생각한 동물을 접어봤다.",               "label": 0},
    {"text": "그림을 그리다가 새로운 색 조합을 발견해서 신기했다.",              "label": 0},
    {"text": "레고로 책에 없는 나만의 집을 설계해서 만들었다.",                 "label": 0},
    {"text": "로블록스에서 친구들이랑 같이 협동해서 어려운 맵을 탈출했다.",       "label": 1},
    {"text": "친구한테 내 간식을 나눠줬더니 기분이 좋았다.",                    "label": 1},
    {"text": "친구랑 의견이 달랐지만 대화로 풀고 재미있게 놀았다.",              "label": 1},
    {"text": "같이 게임하다가 팀원이 실수했을 때 괜찮다고 말해줬다.",            "label": 1},
    {"text": "모둠 활동에서 내가 먼저 아이디어를 내서 친구들이 좋아했다.",        "label": 1},
    {"text": "전학 온 친구한테 먼저 말을 걸어서 같이 점심을 먹었다.",            "label": 1},
    {"text": "내가 만든 작품을 보고 내 자신이 너무 자랑스러웠다.",              "label": 2},
    {"text": "오늘 하루도 내가 하고 싶은 대로 신나고 당당하게 보냈다.",          "label": 2},
    {"text": "어려운 미션이었지만 포기하지 않고 끝까지 해낸 내가 멋지다.",        "label": 2},
    {"text": "수학 문제를 혼자 다 풀었을 때 뿌듯함이 넘쳤다.",                  "label": 2},
    {"text": "체육 시간에 달리기에서 기록을 갱신해서 기분이 최고였다.",           "label": 2},
    {"text": "오늘 내가 계획한 대로 숙제와 운동을 모두 해냈다.",               "label": 2},
]

# =============================================================================
# [1] Firebase 초기화
#     @st.cache_resource → Streamlit 재실행마다 중복 초기화 방지 (ValueError 예방)
# =============================================================================
@st.cache_resource
def init_firebase():
    """
    Firebase Admin SDK를 초기화하고 Firestore 클라이언트를 반환합니다.
    실패 시 None을 반환 → 앱은 기본 데이터로 계속 동작합니다.
    """
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore as fb_firestore

        # secrets.toml 의 [firebase] 섹션을 dict 로 변환
        fb_cfg = dict(st.secrets["firebase"])

        # ⚠️ private_key 의 \n 이 문자열 '\\n' 으로 깨지는 현상 방지
        fb_cfg["private_key"] = fb_cfg["private_key"].replace("\\n", "\n")

        # 이미 초기화된 앱이 있으면 재사용, 없으면 신규 생성
        if not firebase_admin._apps:
            cred = credentials.Certificate(fb_cfg)
            firebase_admin.initialize_app(cred)

        return fb_firestore.client()

    except KeyError:
        # secrets.toml 에 [firebase] 섹션이 없는 경우 (로컬 테스트 등)
        return None
    except Exception as e:
        st.warning(f"⚠️ Firebase 연결 실패: {e}\n기본 내장 데이터로 동작합니다.")
        return None


# =============================================================================
# [2] Firestore CRUD 헬퍼 함수
# =============================================================================

def _get_firestore_timestamp():
    """Firestore 서버 타임스탬프를 안전하게 반환합니다."""
    try:
        from google.cloud import firestore as gfs
        return gfs.SERVER_TIMESTAMP
    except Exception:
        return datetime.datetime.utcnow().isoformat()


def load_from_firestore(db) -> list:
    """
    Firestore 'training_data' 컬렉션의 모든 문서를 읽어옵니다.
    각 항목에 doc_id 를 포함시켜 삭제 시 활용합니다.
    비어있거나 실패하면 DEFAULT_TRAINING_DATA 를 반환합니다.
    """
    if db is None:
        return DEFAULT_TRAINING_DATA.copy()
    try:
        docs = db.collection("training_data").order_by("label").stream()
        result = []
        for doc in docs:
            row = doc.to_dict()
            row["doc_id"] = doc.id
            result.append(row)
        return result if result else DEFAULT_TRAINING_DATA.copy()
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return DEFAULT_TRAINING_DATA.copy()


def add_to_firestore(db, text: str, label: int) -> bool:
    """새 학습 문장을 Firestore에 추가합니다."""
    if db is None:
        return False
    try:
        from firebase_admin import firestore as fb_fs
        db.collection("training_data").add({
            "text":       text,
            "label":      label,
            "created_at": fb_fs.SERVER_TIMESTAMP,
        })
        return True
    except Exception as e:
        st.error(f"추가 오류: {e}")
        return False


def delete_from_firestore(db, doc_id: str) -> bool:
    """Firestore에서 특정 문서를 삭제합니다."""
    if db is None:
        return False
    try:
        db.collection("training_data").document(doc_id).delete()
        return True
    except Exception as e:
        st.error(f"삭제 오류: {e}")
        return False


def seed_firestore_if_empty(db):
    """Firestore가 비어있을 때 기본 데이터를 자동으로 시드합니다."""
    if db is None:
        return
    try:
        existing = list(db.collection("training_data").limit(1).stream())
        if not existing:
            for item in DEFAULT_TRAINING_DATA:
                db.collection("training_data").add({
                    "text":  item["text"],
                    "label": item["label"],
                })
    except Exception:
        pass  # 시드 실패는 무시하고 진행


# =============================================================================
# [3] 학습 데이터 캐시 로더
#     Firestore Client 는 hashable 하지 않으므로 _db 접두사로 캐시 키 우회
#     ttl=60 → 60초마다 자동 갱신 / .clear() 로 즉시 갱신
# =============================================================================
@st.cache_data(ttl=60, show_spinner=False)
def cached_load(_db_id: str, _db) -> list:
    """
    _db_id  : str (캐시 키용 — db 객체는 해싱 불가라 별도 ID 사용)
    _db     : Firestore Client
    """
    return load_from_firestore(_db)


def get_training_data(db) -> list:
    """캐시를 통해 학습 데이터를 가져옵니다."""
    db_id = "firestore" if db is not None else "local"
    return cached_load(db_id, db)


# =============================================================================
# [4] TensorFlow 딥러닝 모델 학습
# =============================================================================
def build_and_train_model(training_data: list, epochs: int = 100):
    """
    Bidirectional LSTM 모델을 학습합니다.

    반환값 : (model, tokenizer, maxlen, log_lines, epoch_data, final_acc)
    """
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    texts  = [d["text"]  for d in training_data]
    labels = [d["label"] for d in training_data]
    maxlen = 20

    # ── 텍스트 토크나이징 ────────────────────────────────────────────────────
    tok = Tokenizer(num_words=1000, oov_token="<OOV>")
    tok.fit_on_texts(texts)
    seqs   = tok.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=maxlen, padding="post")

    # ── 모델 설계: Embedding → Bidirectional LSTM → Dense ────────────────────
    model = tf.keras.Sequential([
        # 단어를 32차원 벡터로 임베딩
        tf.keras.layers.Embedding(input_dim=1000, output_dim=32, input_length=maxlen),
        # 양방향 LSTM — 문장의 앞뒤 맥락을 동시에 파악
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),          # 과적합 방지
        tf.keras.layers.Dense(3, activation="softmax"),  # 3개 카테고리 확률
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    # ── 학습 로그 수집 콜백 ──────────────────────────────────────────────────
    log_lines  = []
    epoch_data = {"loss": [], "acc": []}

    class _LogCB(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            epoch_data["loss"].append(logs["loss"])
            epoch_data["acc"].append(logs["accuracy"])
            if (epoch + 1) % 10 == 0:
                log_lines.append(
                    f"Epoch {epoch+1:>4}/{epochs}  "
                    f"loss={logs['loss']:.4f}  "
                    f"acc={logs['accuracy']:.4f}"
                )

    model.fit(
        padded, np.array(labels),
        epochs=epochs, verbose=0,
        callbacks=[_LogCB()],
    )

    final_acc = epoch_data["acc"][-1] if epoch_data["acc"] else 0.0
    log_lines.append(f"✅ 학습 완료!  최종 정확도: {final_acc * 100:.1f}%")

    return model, tok, maxlen, log_lines, epoch_data, final_acc


def run_predict(model, tok, maxlen: int, text: str) -> np.ndarray:
    """입력 텍스트의 카테고리별 확률(numpy array)을 반환합니다."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq    = tok.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen, padding="post")
    return model.predict(padded, verbose=0)[0]


# =============================================================================
# [5] Google Gemini API — 따뜻한 한마디 생성
# =============================================================================
def generate_warm_message(diary: str, best_cat: str, probs: np.ndarray) -> str:
    """
    Gemini API 로 아이 맞춤 따뜻한 한마디를 생성합니다.
    API 키가 없거나 오류 시 템플릿 메시지로 폴백합니다.
    """
    try:
        from google import genai  # google-genai 라이브러리 (구형 google-generativeai 사용 금지)

        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

        prob_text = "  /  ".join(
            f"{CATS[i]} {probs[i]*100:.1f}%" for i in range(3)
        )
        prompt = (
            "당신은 아이의 성장을 따뜻하게 응원하는 부모 또는 선생님입니다.\n\n"
            "아이가 쓴 일기를 읽고, 분석 결과를 바탕으로 아이에게 직접 전하는 "
            "따뜻하고 구체적인 한마디를 한국어로 작성하세요.\n\n"
            "[규칙]\n"
            "- 아이에게 직접 말하는 2인칭(\"너는\", \"네가\") 형식\n"
            "- 일기 내용을 구체적으로 언급할 것\n"
            f"- 가장 높은 지표({best_cat})를 중심으로 칭찬\n"
            "- 2~3문장, 이모지 1~2개 포함\n"
            "- 진심 어린 공감과 격려 (과도한 칭찬 지양)\n\n"
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
        # secrets.toml 에 GEMINI_API_KEY 가 없는 경우
        return _fallback_message(int(np.argmax(probs)))
    except Exception as e:
        st.warning(f"Gemini 연결 오류 ({e}) → 기본 메시지로 대체합니다.")
        return _fallback_message(int(np.argmax(probs)))


def _fallback_message(best_idx: int) -> str:
    """Gemini를 사용할 수 없을 때 카테고리별 템플릿 메시지를 무작위 반환합니다."""
    templates = {
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
    pool = templates.get(best_idx, templates[2])
    return random.choice(pool)


# =============================================================================
# [6] Plotly 차트
# =============================================================================

def chart_bar(probs: np.ndarray):
    """수평 막대 차트 — 카테고리별 확률."""
    best   = int(np.argmax(probs))
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
        text=[f"<b>{p * 100:.1f}%</b>" for p in probs],
        textposition="outside",
        textfont=dict(size=14),
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 125], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(tickfont=dict(size=15)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=55, t=10, b=10),
        height=200,
    )
    return fig


def chart_radar(probs: np.ndarray):
    """레이더(방사형) 차트 — 균형감 시각화."""
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


def chart_training_curve(epoch_data: dict):
    """학습 곡선 — Loss(좌축) + Accuracy(우축)."""
    xs = list(range(1, len(epoch_data["loss"]) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=epoch_data["loss"],
        name="Loss",
        line=dict(color="#FF6B6B", width=2.2),
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=epoch_data["acc"],
        name="Accuracy",
        line=dict(color="#4ECDC4", width=2.2),
        yaxis="y2",
    ))
    fig.update_layout(
        xaxis=dict(title="Epoch", gridcolor="#EEE"),
        yaxis=dict(title="Loss",     gridcolor="#EEE",
                   titlefont=dict(color="#FF6B6B")),
        yaxis2=dict(title="Accuracy", overlaying="y", side="right",
                    range=[0, 1.05], titlefont=dict(color="#4ECDC4")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.5, y=-0.18, orientation="h", xanchor="center"),
        margin=dict(l=10, r=55, t=10, b=50),
        height=280,
    )
    return fig


# =============================================================================
# [7] 전역 CSS 주입
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

/* ── 히어로 헤더 ── */
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

/* ── 섹션 레이블 ── */
.sec-label {{
    font-size: .7rem; font-weight: 800; letter-spacing: 1.5px;
    color: {THEME["text_muted"]}; text-transform: uppercase; margin-bottom: .5rem;
}}

/* ── 수치 카드 (inline style 로 배경·테두리 지정) ── */
.metric-card {{
    border-radius: 14px; padding: .9rem; text-align: center;
    margin-bottom: .4rem;
}}

/* ── 따뜻한 한마디 박스 ── */
.warm-box {{
    background: linear-gradient(135deg, #FFFBEA, #EFF9FF);
    border: 1.5px solid {THEME["border"]};
    border-radius: 18px;
    padding: 1.4rem 1.8rem;
    margin-top: .6rem;
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

/* ── TOP 뱃지 ── */
.best-badge {{
    display: inline-block; border-radius: 12px;
    padding: .5rem 1.1rem; font-size: .88rem; font-weight: 800;
    margin-top: .7rem;
}}

/* ── 모델 상태 표시 ── */
.model-ready {{ color: #4ECDC4; font-size: .82rem; font-weight: 700; margin-top: .3rem; }}
.model-none  {{ color: {THEME["danger"]}; font-size: .82rem; font-weight: 700; margin-top: .3rem; }}

/* ── 학습 로그 콘솔 ── */
.log-box {{
    background: #1A1A2E; color: #4ECDC4;
    border-radius: 10px; padding: .9rem 1.1rem;
    font-family: monospace; font-size: .78rem;
    line-height: 1.8; max-height: 210px; overflow-y: auto;
}}

/* ── 데이터 행 ── */
.data-row {{
    background: {THEME["card"]};
    border: 1px solid {THEME["border"]};
    border-radius: 10px;
    padding: .55rem 1rem;
    margin-bottom: .4rem;
    display: flex;
    align-items: center;
    gap: .8rem;
}}
.data-row-label {{
    margin-left: auto;
    font-size: .72rem;
    font-weight: 700;
    white-space: nowrap;
}}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# [8] 탭 1 — 일기 분석
# =============================================================================
def tab_analyze(db):
    st.markdown('<p class="sec-label">✏️ 오늘의 일기를 입력하세요</p>',
                unsafe_allow_html=True)

    diary = st.text_area(
        label="일기",
        placeholder=(
            "오늘 있었던 일을 자유롭게 써보세요!\n"
            "예) 오늘 새로운 게임 맵을 혼자서 만들어봤는데 생각보다 너무 잘 나와서 기분이 최고였다!"
        ),
        height=130,
        label_visibility="collapsed",
        key="diary_input",
    )

    _, col_btn = st.columns([3, 1])
    with col_btn:
        analyze_btn = st.button("🔍 분석하기", use_container_width=True, type="primary")

    # 모델 상태 표시
    if st.session_state.get("model") is None:
        st.markdown(
            '<p class="model-none">● 모델 미학습 — [🧠 모델 학습] 탭에서 먼저 학습하세요</p>',
            unsafe_allow_html=True,
        )
    else:
        n   = st.session_state.get("train_n", 0)
        acc = st.session_state.get("final_acc", 0.0)
        st.markdown(
            f'<p class="model-ready">● 모델 준비 완료'
            f' &nbsp;│&nbsp; Bidirectional LSTM'
            f' &nbsp;│&nbsp; {n}개 문장'
            f' &nbsp;│&nbsp; 정확도 {acc * 100:.1f}%</p>',
            unsafe_allow_html=True,
        )

    if not analyze_btn:
        return

    # ── 입력 검증 ────────────────────────────────────────────────────────────
    if not diary.strip():
        st.warning("일기를 먼저 입력해 주세요!")
        return
    if st.session_state.get("model") is None:
        st.error("먼저 [🧠 모델 학습] 탭에서 학습을 진행해 주세요!")
        return

    # ── 딥러닝 예측 ──────────────────────────────────────────────────────────
    with st.spinner("딥러닝 분석 중…"):
        probs    = run_predict(
            st.session_state["model"],
            st.session_state["tokenizer"],
            st.session_state["maxlen"],
            diary,
        )
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
            top_html = (
                f"<div style='font-size:.62rem;color:{CAT_DARK[i]};"
                f"font-weight:800;margin-top:3px'>★ TOP</div>"
            ) if is_best else ""

            st.markdown(
                f'<div class="metric-card" '
                f'style="background:{bg};border:2px solid {border};">'
                f'<div style="font-size:1.9rem">{CAT_EMOJI[i]}</div>'
                f'<div style="font-weight:{fw};font-size:.88rem;color:#333">{CATS[i]}</div>'
                f'<div style="font-size:1.45rem;font-weight:900;color:{CAT_COLOR[i]}">'
                f'{probs[i] * 100:.1f}%</div>'
                f'{top_html}</div>',
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
        msg = generate_warm_message(diary, CATS[best_idx], probs)

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


# =============================================================================
# [9] 탭 2 — 학습 데이터 관리
# =============================================================================
def tab_data(db):
    st.markdown('<p class="sec-label">📚 학습 데이터 관리</p>',
                unsafe_allow_html=True)

    if db is None:
        st.warning("Firebase가 연결되지 않았습니다. secrets.toml 의 [firebase] 섹션을 확인하세요.")
        st.info("현재는 기본 내장 데이터(읽기 전용)를 표시합니다.")
        _render_data_rows(DEFAULT_TRAINING_DATA, readonly=True)
        return

    # ── 새 문장 추가 ─────────────────────────────────────────────────────────
    with st.expander("➕ 새 학습 문장 추가", expanded=False):
        new_text = st.text_input(
            "문장",
            key="new_text_input",
            placeholder="예) 오늘 친구와 함께 그림을 그렸다.",
        )
        new_label = st.selectbox(
            "카테고리",
            options=[0, 1, 2],
            format_func=lambda x: f"{CAT_EMOJI[x]} {CATS[x]}",
            key="new_label_select",
        )
        if st.button("추가하기", type="primary", key="btn_add"):
            if not new_text.strip():
                st.warning("문장을 입력해 주세요.")
            else:
                ok = add_to_firestore(db, new_text.strip(), new_label)
                if ok:
                    st.success("✅ 추가 완료! [🧠 모델 학습] 탭에서 재학습을 진행하세요.")
                    cached_load.clear()
                    st.rerun()

    st.divider()

    # ── 데이터 목록 ──────────────────────────────────────────────────────────
    data = get_training_data(db)
    st.caption(f"총 {len(data)}개 문장 저장됨")
    _render_data_rows(data, db=db, readonly=False)


def _render_data_rows(data: list, db=None, readonly: bool = True):
    """데이터 목록 행을 렌더링합니다. readonly=False 이면 삭제 버튼 표시."""
    for item in data:
        idx    = item.get("label", 0)
        text   = item.get("text", "")
        doc_id = item.get("doc_id", "")

        col_txt, col_del = st.columns([5, 1])
        with col_txt:
            st.markdown(
                f'<div class="data-row">'
                f'<span style="font-size:1.25rem">{CAT_EMOJI[idx]}</span>'
                f'<span style="font-size:.86rem;color:#333;flex:1">{text}</span>'
                f'<span class="data-row-label" style="color:{CAT_COLOR[idx]}">'
                f'{CATS[idx]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_del:
            if not readonly and doc_id:
                if st.button("🗑️", key=f"del_{doc_id}", help="이 문장 삭제"):
                    ok = delete_from_firestore(db, doc_id)
                    if ok:
                        st.toast("삭제 완료!")
                        cached_load.clear()
                        st.rerun()


# =============================================================================
# [10] 탭 3 — 딥러닝 모델 학습
# =============================================================================
def tab_train(db):
    st.markdown('<p class="sec-label">🧠 딥러닝 모델 학습</p>',
                unsafe_allow_html=True)

    # ── 현재 모델 상태 안내 ──────────────────────────────────────────────────
    if st.session_state.get("model") is not None:
        n   = st.session_state.get("train_n", 0)
        acc = st.session_state.get("final_acc", 0.0)
        st.info(
            f"현재 학습된 모델: **{n}개 문장 · 정확도 {acc * 100:.1f}%**\n\n"
            "재학습 버튼을 누르면 선택한 데이터로 새로 학습합니다."
        )
    else:
        st.warning("⚠️ 아직 학습된 모델이 없습니다. 아래 버튼으로 학습을 시작하세요!")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 학습 옵션 ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.slider("학습 Epoch 수", min_value=30, max_value=300,
                           value=100, step=10, key="slider_epochs")
    with col2:
        use_firestore = st.radio(
            "학습 데이터 출처",
            options=["Firestore (최신 데이터)", "기본 내장 데이터"],
            key="radio_data_src",
        ) == "Firestore (최신 데이터)"

    train_btn = st.button("🚀 학습 시작", type="primary",
                           use_container_width=True, key="btn_train")

    if not train_btn:
        return

    # ── 학습 데이터 준비 ─────────────────────────────────────────────────────
    if use_firestore and db is not None:
        with st.spinner("Firestore에서 최신 데이터를 불러오는 중…"):
            training_data = load_from_firestore(db)   # 캐시 우회하여 최신 데이터 강제 로드
    else:
        training_data = DEFAULT_TRAINING_DATA.copy()

    # label 이 없는 행(doc_id 전용 row 등) 필터링
    training_data = [d for d in training_data if "text" in d and "label" in d]

    if len(training_data) < 6:
        st.error(f"학습 데이터가 {len(training_data)}개뿐입니다. 최소 6개 이상 필요합니다.")
        return

    st.markdown(f"📦 학습 데이터: **{len(training_data)}개** 문장")
    st.divider()

    # ── 학습 실행 ────────────────────────────────────────────────────────────
    progress = st.progress(0, text="학습 준비 중…")
    log_slot = st.empty()

    with st.spinner(f"🧠 Bidirectional LSTM 학습 중… ({epochs} epochs)"):
        t0 = time.time()
        model, tok, maxlen, log_lines, epoch_data, final_acc = \
            build_and_train_model(training_data, epochs=epochs)
        elapsed = time.time() - t0

    progress.progress(1.0, text=f"✅ 학습 완료! ({elapsed:.1f}초 소요)")

    # 로그 콘솔 출력
    log_slot.markdown(
        '<div class="log-box">' + "<br>".join(log_lines) + "</div>",
        unsafe_allow_html=True,
    )

    # ── 학습 곡선 차트 ───────────────────────────────────────────────────────
    st.markdown('<p class="sec-label">📈 학습 곡선</p>', unsafe_allow_html=True)
    st.plotly_chart(chart_training_curve(epoch_data), use_container_width=True,
                    config={"displayModeBar": False})

    # ── Session State 에 모델 저장 ───────────────────────────────────────────
    st.session_state["model"]     = model
    st.session_state["tokenizer"] = tok
    st.session_state["maxlen"]    = maxlen
    st.session_state["train_n"]   = len(training_data)
    st.session_state["final_acc"] = final_acc

    st.success(
        f"🎉 학습 완료! 정확도 **{final_acc * 100:.1f}%** "
        f"→ [📄 일기 분석] 탭에서 바로 사용할 수 있습니다."
    )


# =============================================================================
# [11] 메인 진입점
# =============================================================================
def main():
    # ── 페이지 기본 설정 (반드시 가장 먼저 호출) ─────────────────────────────
    st.set_page_config(
        page_title="우리 아이 일기 분석기",
        page_icon="📔",
        layout="centered",
    )

    # 전역 CSS 주입
    inject_css()

    # ── 히어로 헤더 ──────────────────────────────────────────────────────────
    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">📔 우리 아이 일기 분석기</div>'
        '<div class="hero-sub">'
        'TensorFlow 딥러닝 &nbsp;·&nbsp; Firebase Firestore &nbsp;·&nbsp; Google Gemini AI'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # ── Firebase 초기화 (캐시됨 — 중복 초기화 없음) ──────────────────────────
    db = init_firebase()
    if db is not None:
        seed_firestore_if_empty(db)  # 최초 실행 시 기본 데이터 자동 시드

    # ── Session State 초기화 ─────────────────────────────────────────────────
    for key in ("model", "tokenizer", "maxlen", "train_n", "final_acc"):
        if key not in st.session_state:
            st.session_state[key] = None

    # ── 3개 탭 ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📄 일기 분석", "📚 학습 데이터", "🧠 모델 학습"])
    with tab1:
        tab_analyze(db)
    with tab2:
        tab_data(db)
    with tab3:
        tab_train(db)

    # ── 푸터 ─────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        "학습 데이터 형식 (Firestore): "
        "`{ text: '문장', label: 0 }`  "
        "· label → 0=창의성  1=사회성  2=자존감"
    )


if __name__ == "__main__":
    main()


# =============================================================================
# ⚠️  [Firestore 보안 규칙 설정 안내]
# =============================================================================
# Firebase Console > Firestore Database > 규칙(Rules) 탭에서
# 프로덕션 배포 전 반드시 아래와 같이 보안 규칙을 변경하세요.
# 테스트 모드(allow read, write: if true)는 모든 접근을 허용하므로 매우 위험합니다.
#
# 권장 규칙 예시:
# ─────────────────────────────────────────────────────
# rules_version = '2';
# service cloud.firestore {
#   match /databases/{database}/documents {
#     match /training_data/{doc} {
#       allow read:  if true;                     // 읽기는 모두 허용
#       allow write: if request.auth != null;     // 쓰기는 인증 사용자만
#     }
#   }
# }
# ─────────────────────────────────────────────────────
