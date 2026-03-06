import streamlit as st
import pandas as pd
import joblib

import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model_path = os.path.join(BASE_DIR, "churn_model.pkl")
threshold_path = os.path.join(BASE_DIR, "threshold.pkl")

model = joblib.load(model_path)
threshold = joblib.load(threshold_path)

# ----------------------------
# 기본 설정
# ----------------------------
st.set_page_config(
    page_title="실시간 이탈 위험 예측",
    page_icon="🔎",
    layout="wide"
)

# ----------------------------
# 스타일
# ----------------------------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 10px;
}
.sub-box {
    border: 1px solid #2f3542;
    border-radius: 14px;
    padding: 24px;
    margin-top: 10px;
    background-color: #0f172a;
}
.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 20px;
}
.result-low {
    padding: 18px;
    border-radius: 12px;
    background-color: #163d2a;
    color: #9df3c4;
    font-size: 22px;
    font-weight: 700;
}
.result-mid {
    padding: 18px;
    border-radius: 12px;
    background-color: #4a3b12;
    color: #ffe08a;
    font-size: 22px;
    font-weight: 700;
}
.result-high {
    padding: 18px;
    border-radius: 12px;
    background-color: #4b1f1f;
    color: #ff9b9b;
    font-size: 22px;
    font-weight: 700;
}
.small-text {
    font-size: 15px;
    color: #cbd5e1;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------
# 제목
# ----------------------------
st.markdown('<div class="main-title">🔎 실시간 이탈 위험 예측</div>', unsafe_allow_html=True)
st.markdown('<div class="small-text">가설 기반 변수와 학습된 이탈 예측 모델을 활용해 고객의 이탈 위험도를 실시간으로 예측합니다.</div>', unsafe_allow_html=True)

st.markdown('<div class="sub-box">', unsafe_allow_html=True)
st.markdown('<div class="section-title">고객 정보 입력</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("나이", min_value=18, max_value=100, value=35, step=1)
    customer_tenure_months = st.number_input("가입 기간(개월)", min_value=0, max_value=600, value=24, step=1)
    premium_change_pct = st.number_input("보험료 변화율", value=0.05, step=0.01, format="%.2f")
    quote_requested_flag = st.selectbox("견적 요청 여부", [0, 1], format_func=lambda x: "예" if x == 1 else "아니오")

with col2:
    autopay_enabled = st.selectbox("자동이체 여부", [0, 1], format_func=lambda x: "사용" if x == 1 else "미사용")
    late_payment_count_12m = st.number_input("최근 1년 연체 횟수", min_value=0, max_value=20, value=0, step=1)
    missed_payment_flag = st.selectbox("연체 심각 여부", [0, 1], format_func=lambda x: "예" if x == 1 else "아니오")
    coverage_downgrade_flag = st.selectbox("보장 축소 여부", [0, 1], format_func=lambda x: "예" if x == 1 else "아니오")

predict_btn = st.button("이탈 위험 예측하기", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# 예측
# ----------------------------
if predict_btn:
    input_df = pd.DataFrame([{
        'age': age,
        'customer_tenure_months': customer_tenure_months,
        'premium_change_pct': premium_change_pct,
        'quote_requested_flag': quote_requested_flag,
        'autopay_enabled': autopay_enabled,
        'late_payment_count_12m': late_payment_count_12m,
        'missed_payment_flag': missed_payment_flag,
        'coverage_downgrade_flag': coverage_downgrade_flag
    }])

    prob = model.predict_proba(input_df)[:, 1][0]
    pred = int(prob >= threshold)

    st.markdown("### 예측 결과")

    risk_percent = prob * 100

    if risk_percent < 30:
        st.markdown(
            f'<div class="result-low">✅ 낮은 위험군 | 이탈 확률: {risk_percent:.2f}%</div>',
            unsafe_allow_html=True
        )
    elif risk_percent < 70:
        st.markdown(
            f'<div class="result-mid">⚠️ 중간 위험군 | 이탈 확률: {risk_percent:.2f}%</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-high">🚨 높은 위험군 | 이탈 확률: {risk_percent:.2f}%</div>',
            unsafe_allow_html=True
        )

    st.write(f"**최종 분류 기준 threshold:** {threshold}")
    st.write(f"**예측 클래스:** {'이탈 위험 고객' if pred == 1 else '유지 가능 고객'}")

    # 입력값 요약
    st.markdown("### 입력값 요약")
    st.dataframe(input_df, use_container_width=True)

    # 간단 해석
    st.markdown("### 해석 포인트")
    reasons = []

    if premium_change_pct > 0.1:
        reasons.append("- 보험료 상승률이 높아 이탈 가능성이 커질 수 있습니다.")
    if late_payment_count_12m >= 2:
        reasons.append("- 최근 연체 횟수가 많아 유지 리스크가 증가할 수 있습니다.")
    if quote_requested_flag == 1:
        reasons.append("- 견적 요청 이력이 있어 다른 보험 상품을 비교 중일 가능성이 있습니다.")
    if coverage_downgrade_flag == 1:
        reasons.append("- 보장 축소 이력이 있어 비용 부담 기반 이탈 신호일 수 있습니다.")
    if customer_tenure_months < 12:
        reasons.append("- 가입 기간이 짧아 고객 충성도가 낮을 가능성이 있습니다.")
    if missed_payment_flag == 1:
        reasons.append("- 연체 심각 여부가 활성화되어 있어 강한 위험 신호로 볼 수 있습니다.")

    if reasons:
        for r in reasons:
            st.write(r)
    else:
        st.write("- 입력된 정보 기준 뚜렷한 고위험 신호는 상대적으로 적습니다.")