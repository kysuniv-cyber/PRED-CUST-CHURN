import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# =============================
# 1. 페이지 설정 및 공통 스타일 (사용자 제공 코드 유지)
# =============================
st.set_page_config(
    page_title="보험 이탈 예측",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* 컨테이너 여백 설정 */
.block-container { 
    padding-top: 1.2rem; 
    padding-bottom: 2rem; 
    padding-left: 2rem; 
    padding-right: 2rem; 
}

/* 메인 타이틀 */
.main-title { 
    font-size: 2.2rem; 
    font-weight: 800; 
    color: #0f172a; 
    margin-bottom: 0.2rem; 
}

/* 서브 타이틀 */
.sub-title { 
    font-size: 1.05rem; 
    color: #64748b; 
    margin-bottom: 1.5rem; 
}

/* 모델 정보 그라데이션 헤더 */
.model-info-header {
    background: linear-gradient(90deg, #4f46e5 0%, #7e22ce 100%);
    border-radius: 18px;
    padding: 25px 30px;
    color: white;
    margin-bottom: 20px;
}

/* 헤더 내부 그리드 레이아웃 */
.model-info-grid { 
    display: flex; 
    justify-content: space-between; 
    align-items: center; 
}

/* 요약 지표 카드 */
.card {
    background-color: white; 
    border: 1px solid #e5e7eb; 
    border-radius: 18px;
    padding: 20px 22px; 
    box-shadow: 0 1px 2px rgba(0,0,0,0.04); 
    min-height: 120px;
}

/* 카드 내 제목 */
.card-title { 
    font-size: 1rem; 
    color: #475569; 
    margin-bottom: 0.6rem; 
}

/* 카드 내 수치 값 */
.card-value { 
    font-size: 2.2rem; 
    font-weight: 800; 
    color: #0f172a; 
}

/* 섹션 구분 카드 (차트 영역 등) */
.section-card {
    background-color: white; 
    border: 1px solid #e5e7eb; 
    border-radius: 18px;
    padding: 18px 20px; 
    margin-top: 12px;
}

/* 섹션 제목 */
.section-title { 
    font-size: 1.6rem; 
    font-weight: 800; 
    color: #0f172a; 
    margin-bottom: 0.25rem; 
}

/* 사이드바 로고 및 텍스트 영역 */
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
</style>
""", unsafe_allow_html=True)


# =============================
# 2. 모델 및 데이터 로드 (실제 경로 연동)
# =============================
@st.cache_resource
def load_ml_resources():
    # 파일 경로를 본인 환경에 맞게 수정하세요
    model_path = Path("./model/churn_model_jyhong.pkl")
    data_path = Path("./data/test_data.pkl")

    if model_path.exists() and data_path.exists():
        model = joblib.load(model_path)
        data = joblib.load(data_path)
        return model, data['X_test'], data['y_test']
    return None, None, None


model, X_test, y_test = load_ml_resources()

# =============================
# 3. 모델 성능 화면 렌더링
# =============================
if model is not None:
    # --- 데이터 계산 ---
    y_probs = model.predict_proba(X_test)[:, 1]

    # 최적의 임계치는 수동으로 지정(?)
    threshold = 0.35
    y_pred = (y_probs >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    report = classification_report(y_test, y_pred, output_dict=True)

    # --- UI 렌더링 ---
    st.markdown('<div class="main-title">모델 성능</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">예측 모델 성능 평가 및 모니터링</div>', unsafe_allow_html=True)

    # 모델 정보 그라데이션 카드 (이미지 2 스타일)
    st.markdown(f"""
    <div class="model-info-header">
        <div class="model-info-grid">
            <div>
                <div style="font-size: 1.5rem; font-weight: 700;">모델명 하드코딩</div>
                <div style="opacity: 0.9; font-size: 0.95rem; margin-top: 5px;">
                    마지막 업데이트: 업데이트일 하드코딩 | 학습 데이터: {len(X_test):,}건
                </div>
            </div>
            <div style="text-align: right;">
                <div style="opacity: 0.8; font-size: 0.8rem;">모델 버전 / 예측 수</div>
                <div style="font-size: 1.6rem; font-weight: 800;">v2.3(hard코딩) / {len(y_pred):,}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 성능 지표 3개 (이미지 2 하단)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="card"><div class="card-title">Accuracy</div><div class="card-value">{report["accuracy"]:.2%}</div></div>',
            unsafe_allow_html=True)
    with col2:
        st.markdown(
            f'<div class="card"><div class="card-title">F1 Score</div><div class="card-value">{report["1"]["f1-score"]:.2%}</div></div>',
            unsafe_allow_html=True)
    with col3:
        st.markdown(
            f'<div class="card"><div class="card-title">ROC AUC</div><div class="card-value">{roc_auc:.2%}</div></div>',
            unsafe_allow_html=True)

    # 상세 분석 영역 (이미지 4, 6 스타일)
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">ROC Curve</div>', unsafe_allow_html=True)
        fig_roc = px.line(x=fpr, y=tpr, labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        fig_roc.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350)
        st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True,
                           x=['Negative', 'Positive'], y=['Negative', 'Positive'],
                           color_continuous_scale='Blues')
        fig_cm.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350)
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 특성 중요도 (이미지 6)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">주요 예측 변수 (Feature Importance)</div>', unsafe_allow_html=True)

    # 실제 모델에서 중요도 추출
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X_test.columns).sort_values(ascending=True).tail(10)

    fig_imp = px.bar(x=feat_imp.values, y=feat_imp.index, orientation='h',
                     color_discrete_sequence=['#6366f1'])
    fig_imp.update_layout(xaxis_title="Importance", yaxis_title="Features", height=400)
    st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("⚠️ 모델 또는 테스트 데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")