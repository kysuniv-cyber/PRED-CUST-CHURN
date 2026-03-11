import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="What-If 시뮬레이션", layout="wide")

st.title("📈 What-If 비즈니스 시뮬레이션")
st.markdown("""
**"만약 우리의 정책을 바꾼다면, 이탈률은 어떻게 변할까?"** 현재 고객 데이터를 바탕으로 비즈니스 액션에 따른 이탈률 변화를 시뮬레이션합니다.
""")

# 임시 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_csv("././data/insurance_policyholder_churn_synthetic.csv")
    df['churn_prob'] = df['churn_probability_true'] # 실제로는 모델 예측값을 매핑
    return df

df = load_data()

# --- 시뮬레이션 설정 UI ---
st.sidebar.header("🛠️ 시뮬레이션 시나리오 설정")

st.sidebar.subheader("시나리오 1: 보상 정책 개선")
reject_reduction = st.sidebar.slider(
    "청구 거절 건수 감소 목표 (%)",
    min_value=0, max_value=50, value=10, step=5,
    help="재심사 캠페인 등을 통해 청구 거절율을 낮췄을 때의 효과를 시뮬레이션합니다."
)

st.sidebar.subheader("시나리오 2: 보험료 방어")
premium_control = st.sidebar.slider(
    "보험료 인상률 최대치 제한 (%)",
    min_value=0, max_value=20, value=5, step=1,
    help="급격한 보험료 인상을 막아 고객 이탈을 방지합니다."
)

st.sidebar.markdown("---")
run_sim = st.sidebar.button("🚀 시뮬레이션 실행", type="primary")

# --- 메인 화면 로직 ---
if run_sim:
    st.markdown("### 📊 시뮬레이션 결과")

    # ---------------------------------------------------------
    # [핵심 로직] 실제 환경에서는 아래 과정을 거칩니다:
    # 1. current_probs = model.predict_proba(df)[:, 1]
    # 2. sim_df = df.copy()
    # 3. sim_df['num_rejected_claims_12m'] = sim_df['num_rejected_claims_12m'] * (1 - reject_reduction/100)
    # 4. sim_df['premium_change_pct'] = np.where(sim_df['premium_change_pct'] > premium_control/100, premium_control/100, sim_df['premium_change_pct'])
    # 5. sim_probs = model.predict_proba(sim_df)[:, 1]
    # ---------------------------------------------------------

    # 임시 Mock 데이터로 결과 산출 (실제 모델 연산으로 대체해야 함)
    current_churn_rate = 5.2
    # 거절율 감소와 보험료 통제에 비례하여 이탈률이 떨어진다고 가정 (Mock 계산)
    expected_churn_rate = current_churn_rate - (reject_reduction * 0.08) - ((10 - premium_control) * 0.05)
    expected_churn_rate = max(1.5, round(expected_churn_rate, 2))  # 최소 방어선 설정

    saved_customers = int((current_churn_rate - expected_churn_rate) / 100 * len(df))

    # 결과 요약 Metric 출력
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="현재 예상 이탈률", value=f"{current_churn_rate}%")
    with col2:
        st.metric(
            label="시뮬레이션 후 예상 이탈률",
            value=f"{expected_churn_rate}%",
            delta=f"-{round(current_churn_rate - expected_churn_rate, 2)}%p",
            delta_color="inverse"
        )
    with col3:
        st.metric(
            label="이탈 방어 예상 고객 수",
            value=f"{saved_customers}명",
            delta="Retention 증가!"
        )

    st.markdown("---")

    # 시각화 비교
    st.subheader("💡 기대 효과 분석")
    fig, ax = plt.subplots(figsize=(8, 4))

    categories = ['현재 상태 (AS-IS)', '정책 개선 후 (TO-BE)']
    rates = [current_churn_rate, expected_churn_rate]

    sns.barplot(x=categories, y=rates, palette=['#ff9999', '#66b3ff'], ax=ax)

    for i, v in enumerate(rates):
        ax.text(i, v + 0.1, f"{v}%", ha='center', fontweight='bold')

    ax.set_ylabel("예상 이탈률 (%)")
    ax.set_ylim(0, max(rates) + 2)

    st.pyplot(fig)

    st.info(
        f"**Action Item**: 청구 거절 건수를 **{reject_reduction}%** 줄이고, 보험료 인상률을 **{premium_control}%** 이내로 통제할 경우, 전체 예상 이탈률이 **{current_churn_rate}%에서 {expected_churn_rate}%로 감소**할 것으로 예상됩니다.")

else:
    st.info("👈 좌측 사이드바에서 시나리오 조건을 설정한 후 **'시뮬레이션 실행'** 버튼을 눌러주세요.")