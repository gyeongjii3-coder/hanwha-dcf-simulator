"""
한화솔루션 DCF 가치평가 & 몬테카를로 시뮬레이터 (Streamlit 화면).
실행: streamlit run app.py
계산 로직은 dcf.py에 있다.
"""

import numpy as np
import plotly.express as px
import streamlit as st

from dcf import Assumptions, implied_margin, monte_carlo, sensitivity, valuate

st.set_page_config(page_title="한화솔루션 DCF 시뮬레이터", page_icon="☀️", layout="wide")

D = Assumptions()  # 기본 가정값

# ---------------- 사이드바: 가정 입력 ----------------
st.sidebar.header("가정 입력")
st.sidebar.caption("기본값의 출처와 근거는 README를 참고하세요. 모든 값은 바꿀 수 있습니다.")

with st.sidebar.expander("매출 성장률", expanded=True):
    growth = tuple(
        st.number_input(f"{D.base_year + i + 1}년 (%)", value=g * 100, step=1.0,
                        key=f"g{i}") / 100
        for i, g in enumerate(D.growth)
    )

with st.sidebar.expander("수익성·투자", expanded=True):
    margin_start = st.number_input("1년차 영업이익률 (%)", value=D.margin_start * 100, step=0.5) / 100
    margin_end = st.number_input("5년차 영업이익률 (%)", value=D.margin_end * 100, step=0.5) / 100
    tax_rate = st.number_input("법인세율 (%)", value=D.tax_rate * 100, step=0.5) / 100
    da_pct = st.number_input("감가상각비 / 매출 (%)", value=D.da_pct * 100, step=0.5) / 100
    capex_pct = st.number_input("CAPEX / 매출 (%)", value=D.capex_pct * 100, step=0.5) / 100
    nwc_pct = st.number_input("운전자본 증가 / 매출 증가 (%)", value=D.nwc_pct * 100, step=1.0) / 100

with st.sidebar.expander("할인율·자본구조", expanded=True):
    wacc = st.number_input("WACC (%)", value=D.wacc * 100, step=0.25) / 100
    terminal_growth = st.number_input("영구성장률 (%)", value=D.terminal_growth * 100, step=0.25) / 100
    net_debt = st.number_input("순차입금 (조원)", value=D.net_debt, step=0.1)
    shares = st.number_input("발행주식수 (백만주)", value=D.shares, step=0.1)
    price = st.number_input("비교할 현재 주가 (원)", value=32_900, step=100)

a = Assumptions(
    base_year=D.base_year, base_revenue=D.base_revenue, growth=growth,
    margin_start=margin_start, margin_end=margin_end, tax_rate=tax_rate,
    da_pct=da_pct, capex_pct=capex_pct, nwc_pct=nwc_pct, wacc=wacc,
    terminal_growth=terminal_growth, net_debt=net_debt, shares=shares,
)

# ---------------- 본문 ----------------
st.title("한화솔루션 DCF 가치평가 시뮬레이터")
st.write(
    "큐셀(신재생에너지) 부문이 매출의 절반 이상을 차지하는 한화솔루션을 "
    "연결 기준 FCFF 방식으로 평가합니다. 왼쪽에서 가정을 바꾸면 모든 결과가 즉시 다시 계산됩니다."
)

try:
    r = valuate(a)
except ValueError as e:
    st.error(f"{e} 왼쪽에서 WACC를 올리거나 영구성장률을 낮춰 주세요.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("기업가치 (EV)", f"{r['enterprise_value']:.2f}조원")
c2.metric("주주가치", f"{r['equity_value']:.2f}조원")
c3.metric("주당가치", f"{r['per_share']:,.0f}원",
          delta=f"{r['per_share'] / price - 1:+.1%} vs 현재가")
c4.metric("EV 중 터미널가치 비중", f"{r['tv_share']:.0%}")

if r["equity_value"] < 0:
    st.warning("현재 가정에서는 기업가치가 순차입금보다 작아 주주가치가 음수입니다. "
               "차입금이 많은 기업은 가정 변화에 주당가치가 크게 흔들린다는 점을 보여줍니다.")

tab1, tab2, tab3, tab4 = st.tabs(["현금흐름 추정", "민감도 분석", "몬테카를로", "역DCF"])

with tab1:
    st.subheader("연도별 FCF 계산 과정")
    st.caption("단위: 조원. 이 표를 엑셀에서 똑같이 계산해 결과를 교차검증할 수 있습니다.")
    t = r["table"].copy()
    fmt = {c: "{:.3f}" for c in t.columns if c not in ("연도", "영업이익률")}
    fmt["영업이익률"] = "{:.1%}"
    st.dataframe(t.style.format(fmt), hide_index=True, width="stretch")
    st.write(
        f"예측기간 FCF 현재가치 합계 **{r['pv_fcf_sum']:.3f}조원** + "
        f"터미널가치 현재가치 **{r['pv_terminal']:.3f}조원** "
        f"(할인 전 {r['terminal_value']:.3f}조원) = 기업가치 **{r['enterprise_value']:.3f}조원**"
    )

with tab2:
    st.subheader("WACC × 영구성장률 민감도 (주당가치, 원)")
    waccs = [wacc + d for d in (-0.01, -0.005, 0, 0.005, 0.01)]
    gs = [terminal_growth + d for d in (-0.01, -0.005, 0, 0.005, 0.01)]
    sens = sensitivity(a, waccs, gs)
    st.dataframe(
        sens.style.format("{:,.0f}", na_rep="-").background_gradient(cmap="RdYlGn", axis=None),
        width="stretch",
    )
    st.caption("가운데 칸이 현재 가정입니다. 할인율 1%p 차이가 주당가치를 얼마나 바꾸는지 확인해 보세요.")

with tab3:
    st.subheader("몬테카를로 시뮬레이션")
    st.caption("WACC, 영구성장률, 5년차 영업이익률, 연도별 성장률을 정규분포로 흔들어 DCF를 반복 계산합니다.")
    m1, m2, m3 = st.columns(3)
    n_sims = m1.select_slider("시뮬레이션 횟수", [1_000, 5_000, 10_000, 50_000], value=10_000)
    margin_sd = m2.number_input("영업이익률 표준편차 (%p)", value=2.0, step=0.5) / 100
    wacc_sd = m3.number_input("WACC 표준편차 (%p)", value=0.75, step=0.25) / 100

    sims = monte_carlo(a, n_sims=n_sims, margin_sd=margin_sd, wacc_sd=wacc_sd)
    p5, p50, p95 = np.percentile(sims, [5, 50, 95])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("하위 5%", f"{p5:,.0f}원")
    k2.metric("중앙값", f"{p50:,.0f}원")
    k3.metric("상위 5%", f"{p95:,.0f}원")
    k4.metric("현재가를 넘을 확률", f"{(sims > price).mean():.0%}")

    clipped = np.clip(sims, np.percentile(sims, 1), np.percentile(sims, 99))
    fig = px.histogram(x=clipped, nbins=80, labels={"x": "주당가치 (원)"})
    fig.add_vline(x=price, line_dash="dash", annotation_text="현재 주가")
    fig.add_vline(x=p50, annotation_text="중앙값")
    fig.update_layout(yaxis_title="빈도", showlegend=False, bargap=0.02)
    st.plotly_chart(fig, width="stretch")
    st.caption("그래프는 극단값(상하위 1%)을 잘라 표시합니다. 통계치는 전체 결과 기준입니다.")

with tab4:
    st.subheader("역DCF: 시장은 무엇을 기대하고 있나")
    m = implied_margin(a, price)
    if np.isnan(m):
        st.info("다른 가정을 유지한 채 영업이익률만으로는 현재 주가를 설명할 수 없습니다.")
    else:
        st.write(
            f"다른 가정이 그대로라면, 현재 주가 **{price:,.0f}원**을 정당화하려면 "
            f"5년차 영업이익률이 **{m:.1%}** 수준이어야 합니다. "
            f"(현재 가정: {margin_end:.1%})"
        )
        st.caption("주가가 가정하는 수익성이 현실적인지 판단하는 것이 역DCF의 목적입니다.")

with st.expander("방법론과 한계"):
    st.markdown(
        """
- **FCFF 방식**: FCF = 세후영업이익 + 감가상각비 − CAPEX − 순운전자본 증가. 연말 할인 가정.
- **터미널가치**: 고든 성장 모형, TV = FCF₅ × (1+g) / (WACC − g).
- **단순화한 부분**: 부문별(신재생·케미칼·첨단소재) 분리 없이 연결 기준으로 추정했고,
  비지배지분·지분법 투자자산·미국 세액공제(AMPC) 정책 변화는 별도로 반영하지 않았습니다.
- 이 앱은 학습·포트폴리오 목적이며 투자 권유가 아닙니다.
"""
    )
