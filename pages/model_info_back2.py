import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# 1. 페이지 설정 및 디자인 CSS
st.set_page_config(layout="wide", page_title="Insurance Churn Dashboard")

st.markdown("""
    <style>
    /* 타이틀 및 서브타이틀 스타일 */
    .main-title-div { font-size: 32px; font-weight: 800; color: #1E1E1E; margin-bottom: 5px; }
    .sub-title-div { font-size: 16px; color: #666666; margin-bottom: 30px; }

    /* 이미지 2의 상단 그라데이션 카드 */
    .model-header-card {
        background: linear-gradient(90deg, #4361ee 0%, #7209b7 100%);
        padding: 25px; border-radius: 15px; color: white; margin-bottom: 25px;
    }
    .model-header-top { display: flex; justify-content: space-between; align-items: center; }
    .model-header-name {font-size: 24px; font-weight: 700; }
    .model-header-ver { text-align: right; }

    /* 메트릭 카드 스타일 */
    .metric-container {
        background: white; border: 1px solid #E0E0E0; padding: 20px; border-radius: 12px;
        text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    </style>
    """, unsafe_allow_html=True)


# 2. 데이터 및 모델 로드 함수 (기존 로직 유지)
@st.cache_resource
def load_data_and_model():
    try:
        # 실제 파일 경로에 맞춰 수정하세요
        model = joblib.load('./model/churn_model_jyhong.pkl')
        data = joblib.load('./data/test_data.pkl')
        return model, data['X_test'], data['y_test']
    except:
        # 테스트용 더미 데이터 (파일이 없을 경우를 대비)
        return None, None, None


model, X_test, y_test = load_data_and_model()

# 3. 사이드바 메뉴 (이미지 1 디자인)
with st.sidebar:
    st.markdown(
        "<div style='padding: 20px 0px;'><h2 style='color: #4361ee; margin:0;'>보험 이탈 예측</h2><p style='color: #888;'>고객 관리 시스템</p></div>",
        unsafe_allow_html=True)
    menu = st.sidebar.radio("Menu", ["홈", "고객 이탈 예측", "고객 분석", "위험 고객 관리", "모델 성능", "설정"], index=4)

if menu == "모델 성능":
    if model is not None:
        # 모델 예측 결과 계산
        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= 0.5).astype(int)

        # 지표 계산
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        report = classification_report(y_test, y_pred, output_dict=True)
        acc = report['accuracy']
        f1 = report['1']['f1-score']
        prec = report['1']['precision']
        rec = report['1']['recall']

        # --- 화면 그리기 ---

        # 타이틀 (div 태그)
        st.markdown('<div class="main-title-div">모델 성능</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title-div">예측 모델 성능 평가 및 모니터링</div>', unsafe_allow_html=True)

        # 상단 모델 정보 카드 (이미지 2)
        st.markdown(f"""
            <div class="model-header-card">
                <div class="model-header-top">
                    <div>
                        <div class="model-header-name">Gradient Boosting Classifier</div>
                        <div style="opacity:0.8; font-size:14px;">마지막 업데이트: 2026년 3월 8일 | 학습 데이터: {len(X_test):,}건</div>
                    </div>
                    <div class="model-header-ver">
                        <span style="opacity:0.8; font-size:12px;">모델 버전 / 예측 수</span><br>
                        <span style="font-size:22px; font-weight:700;">v2.3 / {len(y_pred):,}</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # 지표 카드 3개 (이미지 2 하단)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Accuracy", f"{acc:.2%}")
        with m2:
            st.metric("F1 Score", f"{f1:.2%}")
        with m3:
            st.metric("ROC AUC", f"{roc_auc:.2%}")

        st.divider()

        # 하단 1열: 추가 지표 & 모델 설정 (이미지 3)
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("추가 성능 지표")
            st.write(f"Precision (정밀도): {prec:.2%}")
            st.progress(prec)
            st.write(f"Recall (재현율): {rec:.2%}")
            st.progress(rec)

        with col_right:
            st.subheader("모델 설정")
            # 실제 모델 파라미터 추출 가능 (예: model.n_estimators)
            params = {"알고리즘": "Gradient Boosting", "트리 개수": "100", "최대 깊이": "5", "학습률": "0.1"}
            st.table(pd.DataFrame([params]).T.rename(columns={0: '설정값'}))

        # 하단 2열: ROC Curve & Confusion Matrix (이미지 4)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("ROC Curve")
            fig_roc = px.line(x=fpr, y=tpr, labels={'x': 'FPR', 'y': 'TPR'})
            fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig_roc, use_container_width=True)

        with c2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, x=['Negative', 'Positive'], y=['Negative', 'Positive'],
                               color_continuous_scale='Blues')
            st.plotly_chart(fig_cm, use_container_width=True)

        # 하단 3열: 특성 중요도 (이미지 6)
        st.subheader("🔑 특성 중요도 (Feature Importance)")
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=X_test.columns).sort_values(ascending=True).tail(6)
        fig_imp = px.bar(x=feat_imp.values, y=feat_imp.index, orientation='h', color_discrete_sequence=['#7209b7'])
        st.plotly_chart(fig_imp, use_container_width=True)

    else:
        st.error("🚨 모델 파일을 로드할 수 없습니다. 경로를 확인해주세요.")