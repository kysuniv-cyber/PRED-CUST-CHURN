import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# =============================
# 공통 스타일
# =============================
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.2rem;
}

.sub-title {
    font-size: 1.05rem;
    color: #64748b;
    margin-bottom: 1.5rem;
}

.card {
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    min-height: 120px;
}

.card-title {
    font-size: 1rem;
    color: #475569;
    margin-bottom: 0.6rem;
}

.card-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0f172a;
}

.section-card {
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 18px 20px;
    margin-top: 12px;
}

.section-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.25rem;
}

.small-muted {
    font-size: 0.95rem;
    color: #64748b;
}

div[data-testid="stSidebarNav"] {
    padding-top: 1rem;
}

div[data-testid="stSidebarNav"]::before {
    content: "보험 이탈 예측\\A고객 관리 시스템";
    white-space: pre-line;
    display: block;
    font-size: 2rem;
    line-height: 1.5;
    font-weight: 800;
    color: #2563eb;
    margin-bottom: 1.2rem;
    padding-left: 0.2rem;
}

div[data-testid="stSidebarNav"] ul {
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

def render_model_dashboard(model, X_test, y_test, y_probs):
    # 제목
    st.title("Model Analysis")
    st.markdown('<div class="main-title">모델 분석</div>', unsafe_allow_html=True)

    # 1. 모델 성능 지표 (Metrics)
    # y_probs가 2차원 배열일 경우(클래스별 확률), 이탈(1) 확률만 추출
    if len(y_probs.shape) > 1:
        y_probs_churn = y_probs[:, 1]
    else:
        y_probs_churn = y_probs

    y_pred = (y_probs_churn >= 0.5).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Recall (재현율)", f"{report['1']['recall']:.2%}")
    col2.metric("Precision (정밀도)", f"{report['1']['precision']:.2%}")
    col3.metric("F1-Score", f"{report['1']['f1-score']:.2%}")

    st.divider()

    # 2. 혼동 행렬 & 변수 중요도 (2열 배치)
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("📍 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax_cm,
                    xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig_cm)

    with right_col:
        st.subheader("🔑 Feature Importance")
        # Decision Tree는 feature_importances_를 기본 제공합니다.
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=X_test.columns).sort_values(ascending=True)

        # 상위 10개 출력
        fig_fi, ax_fi = plt.subplots(figsize=(5, 4))
        feat_imp.tail(10).plot(kind='barh', color='#2ecc71', ax=ax_fi)
        plt.title("Key Predictors of Churn")
        plt.xlabel("Importance Score")
        st.pyplot(fig_fi)

    st.divider()

    # 3. 임계값 시뮬레이터 (Figma 스타일 핵심 기능)
    st.subheader("⚙️ Threshold Tuning Simulation")
    st.write("예측 기준을 변경하여 비즈니스 손실(이탈자 방치)을 최소화하세요.")

    threshold = st.slider("이탈 판단 기준점 설정", 0.05, 0.95, 0.5, 0.05)

    y_pred_adj = (y_probs_churn >= threshold).astype(int)
    adj_report = classification_report(y_test, y_pred_adj, output_dict=True)

    # 시뮬레이션 결과 요약
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"현재 재현율(Recall): **{adj_report['1']['recall']:.2% Rose}**")
        st.progress(adj_report['1']['recall'])
    with c2:
        st.write(f"현재 정밀도(Precision): **{adj_report['1']['precision']:.2% Blue}**")
        st.progress(adj_report['1']['precision'])

    if adj_report['1']['recall'] < 0.5:
        st.error("❗ 주의: 현재 모델은 이탈자의 절반도 못 잡고 있습니다. 임계값을 낮추세요.")


import streamlit as st
import joblib
import pandas as pd

# 위에서 만든 함수를 같은 파일 혹은 import해서 준비하세요.

# 1. 페이지 설정 (Figma 느낌 나도록 와이드 모드)
st.set_page_config(layout="wide", page_title="Insurance Churn Dashboard")

#-----------------------------------------------------
# 2. 모델 및 데이터 로드 (파일 경로는 본인 설정에 맞게 수정)
@st.cache_resource
def load_data_and_model():
    # 학습 때 저장한 pkl 파일 불러오기
    model = joblib.load('./model/churn_model_jyhong.pkl')
    # 테스트 데이터 (X_test, y_test 데이터프레임/시리즈 형태)
    data = joblib.load('./data/test_data.pkl')
    X_test = data['X_test']
    y_test = data['y_test']
    return model, X_test, y_test


try:
    model, X_test, y_test = load_data_and_model()

    # load_data_and_model()
    # 3. 모델 예측 확률 계산 (대시보드 핵심 재료)
    # Decision Tree의 경우 클래스 1(이탈)에 대한 확률을 가져옵니다.
    y_probs = model.predict_proba(X_test)[:, 1]

    # 4. 화면 탭 구성 (Figma의 상단 탭 메뉴 느낌)
    tab1, tab2 = st.tabs(["📈 Model Performance", "🔍 Customer Prediction"])

    with tab1:
        # 아까 만든 함수 호출!
        render_model_dashboard(model, X_test, y_test, y_probs)

    with tab2:
        st.header("Individual Customer Analysis")
        st.write("개별 고객 데이터를 입력하여 이탈 가능성을 확인하세요.")
        # 여기에 고객 한 명씩 예측하는 UI를 추가할 수 있습니다.

except FileNotFoundError as e :
    st.error("🚨 모델 파일(.pkl)이나 데이터 파일(.csv)을 찾을 수 없습니다. 경로를 확인해주세요!")
    st.expander("에러 상세 정보 보기").write(f"상세 메시지: {e}")