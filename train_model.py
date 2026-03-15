"""
train_model.py — 데스크탑 전용 딥러닝 학습 스크립트
====================================================
변경사항: model_bundle.pkl 에 keras/tensorflow 객체를 일절 포함하지 않습니다.
          가중치(numpy 배열)와 tokenizer 단어사전(dict)만 저장합니다.
          → Streamlit Cloud 에서 keras 없이 numpy 만으로 추론 가능

실행: python train_model.py
필요: pip install -r requirements_desktop.txt
"""

import os
import json
import pickle
import platform
import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =============================================================================
# [설정값]
# =============================================================================
TRAINING_DATA_FILE = "training_data.json"
OUTPUT_BUNDLE      = "model_bundle.pkl"
EPOCHS             = 150
MAX_WORDS          = 1000
MAX_LEN            = 20
EMBEDDING_DIM      = 32
LSTM_UNITS         = 32
CATEGORIES         = ["창의성", "사회성", "자존감"]

# =============================================================================
# [기본 학습 데이터] — training_data.json 없을 때 자동 생성
# =============================================================================
DEFAULT_DATA = {
    "categories": CATEGORIES,
    "training_data": [
        {"text": "오늘 찰흙으로 아무도 생각 못한 새로운 모양의 스퀴시를 만들었다.", "label": 0},
        {"text": "캡컷으로 나만의 재미있는 짧은 영상을 기발하게 편집해봤다.",       "label": 0},
        {"text": "블록을 조립해서 설명서에 없는 나만의 우주선을 완성했다.",          "label": 0},
        {"text": "색종이로 설명서 없이 내가 생각한 동물을 접어봤다.",               "label": 0},
        {"text": "그림을 그리다가 새로운 색 조합을 발견해서 신기했다.",              "label": 0},
        {"text": "레고로 책에 없는 나만의 집을 설계해서 만들었다.",                 "label": 0},
        {"text": "스케치북에 상상 속 나라를 처음부터 끝까지 그려봤다.",              "label": 0},
        {"text": "폐상자로 로봇 장난감을 직접 만들어서 친구한테 선물했다.",           "label": 0},
        {"text": "로블록스에서 친구들이랑 같이 협동해서 어려운 맵을 탈출했다.",       "label": 1},
        {"text": "친구한테 내 간식을 나눠줬더니 기분이 좋았다.",                    "label": 1},
        {"text": "친구랑 의견이 달랐지만 대화로 풀고 재미있게 놀았다.",              "label": 1},
        {"text": "같이 게임하다가 팀원이 실수했을 때 괜찮다고 말해줬다.",            "label": 1},
        {"text": "모둠 활동에서 내가 먼저 아이디어를 내서 친구들이 좋아했다.",        "label": 1},
        {"text": "전학 온 친구한테 먼저 말을 걸어서 같이 점심을 먹었다.",            "label": 1},
        {"text": "슬퍼 보이는 친구 옆에 앉아서 말을 들어줬다.",                    "label": 1},
        {"text": "모둠 발표 준비를 친구들과 역할 나눠서 잘 마쳤다.",               "label": 1},
        {"text": "내가 만든 작품을 보고 내 자신이 너무 자랑스러웠다.",              "label": 2},
        {"text": "오늘 하루도 내가 하고 싶은 대로 신나고 당당하게 보냈다.",          "label": 2},
        {"text": "어려운 미션이었지만 포기하지 않고 끝까지 해낸 내가 멋지다.",        "label": 2},
        {"text": "수학 문제를 혼자 다 풀었을 때 뿌듯함이 넘쳤다.",                  "label": 2},
        {"text": "체육 시간에 달리기에서 기록을 갱신해서 기분이 최고였다.",           "label": 2},
        {"text": "오늘 내가 계획한 대로 숙제와 운동을 모두 해냈다.",               "label": 2},
        {"text": "발표하기 무서웠지만 손을 들고 끝까지 말했다.",                    "label": 2},
        {"text": "실수했지만 다시 도전해서 결국 성공했다.",                         "label": 2},
    ],
}


# =============================================================================
# [1] 데이터 로드
# =============================================================================
def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"'{filepath}' 없음 → 기본 데이터로 자동 생성합니다.")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_DATA, f, ensure_ascii=False, indent=2)
        print(f"'{filepath}' 생성 완료\n")

    with open(filepath, encoding="utf-8") as f:
        raw = json.load(f)

    data = raw.get("training_data", raw)
    print(f"학습 데이터: {len(data)}개 문장")
    return data


# =============================================================================
# [2] 전처리
# =============================================================================
def preprocess(data):
    texts  = [d["text"]  for d in data]
    labels = [d["label"] for d in data]

    tok = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tok.fit_on_texts(texts)

    seqs   = tok.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post")

    return padded, np.array(labels), tok


# =============================================================================
# [3] 모델 설계 및 학습
# =============================================================================
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_UNITS)),
        tf.keras.layers.Dense(LSTM_UNITS, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(CATEGORIES), activation="softmax"),
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def train(model, X, y):
    print(f"\n학습 시작 — {EPOCHS} epochs")
    print("─" * 50)

    log = {"loss": [], "acc": []}

    class CB(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            log["loss"].append(logs["loss"])
            log["acc"].append(logs["accuracy"])
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:>4}/{EPOCHS}"
                      f"  loss={logs['loss']:.4f}"
                      f"  acc={logs['accuracy']:.4f}")

    model.fit(X, y, epochs=EPOCHS, verbose=0, callbacks=[CB()])
    print("─" * 50)
    print(f"학습 완료!  최종 정확도: {log['acc'][-1]*100:.1f}%\n")
    return log


# =============================================================================
# [4] 학습 곡선 저장
# =============================================================================
def save_plot(log):
    os.makedirs("training_result", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 4))
    xs = range(1, len(log["loss"]) + 1)

    ax1.plot(xs, log["loss"],  color="#FF6B6B", lw=2, label="Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="#FF6B6B")
    ax1.tick_params(axis="y", labelcolor="#FF6B6B")

    ax2 = ax1.twinx()
    ax2.plot(xs, log["acc"], color="#4ECDC4", lw=2, label="Accuracy")
    ax2.set_ylabel("Accuracy", color="#4ECDC4")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="#4ECDC4")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")

    plt.title("학습 곡선 (Loss & Accuracy)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("training_result/training_curve.png", dpi=150)
    plt.show()
    print("학습 곡선 저장: training_result/training_curve.png")


# =============================================================================
# [5] 모델 번들 저장 — keras 객체 완전 제거
#
# 저장 내용:
#   weights    : numpy 배열 리스트 (model.get_weights())
#   word_index : 단어→인덱스 dict (tok.word_index)
#   oov_token  : "<OOV>" 문자열
#   maxlen     : 정수
#   categories : 문자열 리스트
#   version    : "2.0"
#
# keras Tokenizer 객체 자체는 저장하지 않으므로
# Streamlit Cloud 에서 keras 없이 로드 가능합니다.
# =============================================================================
def save_bundle(model, tok):
    bundle = {
        "weights":    model.get_weights(),   # numpy 배열만 저장
        "word_index": tok.word_index,         # 순수 dict
        "oov_token":  "<OOV>",
        "maxlen":     MAX_LEN,
        "categories": CATEGORIES,
        "version":    "2.0",
    }
    with open(OUTPUT_BUNDLE, "wb") as f:
        pickle.dump(bundle, f)

    size_kb = os.path.getsize(OUTPUT_BUNDLE) / 1024
    print(f"\n모델 번들 저장 완료: {OUTPUT_BUNDLE}  ({size_kb:.1f} KB)")
    print("\n다음 단계:")
    print(f"  1. '{OUTPUT_BUNDLE}' 을 GitHub 에 커밋·푸시")
    print("  2. Streamlit Cloud 앱이 자동으로 이 파일을 로드합니다\n")


# =============================================================================
# [6] 빠른 예측 테스트
# =============================================================================
def quick_test(model, tok):
    tests = [
        "오늘 새로운 게임 맵을 혼자서 만들어봤는데 생각보다 너무 잘 나와서 기분이 최고였다!",
        "친구가 넘어졌을 때 도와줘서 고맙다는 말을 들었다.",
        "어려운 수학 문제를 끝까지 풀어서 내 자신이 자랑스러웠다.",
    ]
    print("\n빠른 예측 테스트:")
    print("─" * 60)
    for sent in tests:
        seq    = tok.texts_to_sequences([sent])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
        probs  = model.predict(padded, verbose=0)[0]
        best   = int(np.argmax(probs))
        result = "  ".join(f"{CATEGORIES[i]}:{probs[i]*100:.1f}%" for i in range(3))
        print(f"  입력: {sent[:35]}…")
        print(f"  결과: {result}  →  ★ {CATEGORIES[best]}\n")


# =============================================================================
# [메인]
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  우리 아이 일기 분석기 — 딥러닝 학습")
    print("=" * 60 + "\n")

    data          = load_data(TRAINING_DATA_FILE)
    X, y, tok     = preprocess(data)
    model         = build_model()
    model.summary()
    log           = train(model, X, y)
    save_plot(log)
    quick_test(model, tok)
    save_bundle(model, tok)
