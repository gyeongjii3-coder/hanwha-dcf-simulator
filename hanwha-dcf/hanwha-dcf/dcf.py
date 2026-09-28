"""
DCF(현금흐름할인법) 가치평가 계산 모듈.

화면(app.py)과 계산 로직을 분리해 두었기 때문에,
이 파일의 함수들은 test_dcf.py로 따로 검증할 수 있다.

단위: 금액은 모두 '조원', 주식 수는 '백만주', 주당가치는 '원'.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass
class Assumptions:
    base_year: int = 2025            # 기준연도(실적 확정 연도)
    base_revenue: float = 13.3544    # 기준연도 매출 (조원)
    growth: tuple = (0.20, 0.06, 0.05, 0.04, 0.03)  # 예측기간 연도별 매출성장률
    margin_start: float = 0.01       # 예측 1년차 영업이익률
    margin_end: float = 0.10         # 예측 마지막 해 영업이익률 (선형으로 증가)
    tax_rate: float = 0.24           # 법인세율
    da_pct: float = 0.055            # 감가상각비 / 매출
    capex_pct: float = 0.065         # 설비투자(CAPEX) / 매출
    nwc_pct: float = 0.10            # 순운전자본 증가 / 매출 증가분
    wacc: float = 0.085              # 가중평균자본비용 (할인율)
    terminal_growth: float = 0.015   # 영구성장률
    net_debt: float = 11.55          # 순차입금 (조원)
    shares: float = 225.9            # 발행주식수 (백만주)


def project_fcf(a: Assumptions) -> pd.DataFrame:
    """연도별 매출 → 영업이익 → FCF → 현재가치까지 계산한 표를 반환."""
    n = len(a.growth)
    years = [a.base_year + i + 1 for i in range(n)]

    revenue = []
    prev = a.base_revenue
    for g in a.growth:
        prev = prev * (1 + g)
        revenue.append(prev)
    revenue = np.array(revenue)

    margins = np.linspace(a.margin_start, a.margin_end, n)
    ebit = revenue * margins
    nopat = ebit * (1 - a.tax_rate)
    da = revenue * a.da_pct
    capex = revenue * a.capex_pct
    rev_change = np.diff(np.concatenate([[a.base_revenue], revenue]))
    delta_nwc = rev_change * a.nwc_pct
    fcf = nopat + da - capex - delta_nwc

    # 연말 할인 가정: t년차 현금흐름을 (1+WACC)^t로 나눈다
    t = np.arange(1, n + 1)
    discount = 1 / (1 + a.wacc) ** t
    pv = fcf * discount

    return pd.DataFrame({
        "연도": years,
        "매출": revenue,
        "영업이익률": margins,
        "영업이익(EBIT)": ebit,
        "세후영업이익(NOPAT)": nopat,
        "감가상각비(+)": da,
        "CAPEX(-)": capex,
        "운전자본증가(-)": delta_nwc,
        "FCF": fcf,
        "할인계수": discount,
        "FCF 현재가치": pv,
    })


def valuate(a: Assumptions) -> dict:
    """DCF 결과(기업가치, 주주가치, 주당가치 등)를 계산."""
    if a.terminal_growth >= a.wacc:
        raise ValueError("영구성장률은 WACC보다 작아야 합니다.")

    table = project_fcf(a)
    last_fcf = table["FCF"].iloc[-1]
    n = len(table)

    # 고든 성장 모형: TV = FCF_(n+1) / (WACC - g)
    terminal_value = last_fcf * (1 + a.terminal_growth) / (a.wacc - a.terminal_growth)
    pv_terminal = terminal_value / (1 + a.wacc) ** n
    pv_fcf_sum = table["FCF 현재가치"].sum()

    enterprise_value = pv_fcf_sum + pv_terminal
    equity_value = enterprise_value - a.net_debt
    # 조원 → 원: ×1e12, 백만주 → 주: ×1e6
    per_share = equity_value * 1e12 / (a.shares * 1e6)

    return {
        "table": table,
        "pv_fcf_sum": pv_fcf_sum,
        "terminal_value": terminal_value,
        "pv_terminal": pv_terminal,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "per_share": per_share,
        "tv_share": pv_terminal / enterprise_value if enterprise_value > 0 else np.nan,
    }


def sensitivity(a: Assumptions, waccs, growths) -> pd.DataFrame:
    """WACC × 영구성장률 조합별 주당가치 표."""
    rows = {}
    for w in waccs:
        row = {}
        for g in growths:
            if g >= w:
                row[f"{g:.1%}"] = np.nan
            else:
                row[f"{g:.1%}"] = valuate(replace(a, wacc=w, terminal_growth=g))["per_share"]
        rows[f"{w:.1%}"] = row
    df = pd.DataFrame(rows).T
    df.index.name = "WACC \\ 영구성장률"
    return df


def monte_carlo(a: Assumptions, n_sims=10_000, wacc_sd=0.0075, g_sd=0.005,
                margin_sd=0.02, growth_sd=0.03, seed=42) -> np.ndarray:
    """
    핵심 가정을 확률분포(정규분포)로 바꿔 n_sims번 DCF를 돌린다.
    - WACC, 영구성장률, 목표 영업이익률, 연도별 성장률에 불확실성 부여
    - 영구성장률이 WACC에 너무 가까우면 값이 폭발하므로 WACC-1%p로 상한 설정
    반환값: 시뮬레이션별 주당가치 배열 (원)
    """
    rng = np.random.default_rng(seed)
    n = len(a.growth)

    wacc = rng.normal(a.wacc, wacc_sd, n_sims)
    g = np.minimum(rng.normal(a.terminal_growth, g_sd, n_sims), wacc - 0.01)
    margin_end = rng.normal(a.margin_end, margin_sd, n_sims)
    growth = np.array(a.growth) + rng.normal(0, growth_sd, (n_sims, n))

    # 벡터 연산으로 1만 번을 한꺼번에 계산 (반복문보다 훨씬 빠름)
    revenue = a.base_revenue * np.cumprod(1 + growth, axis=1)
    frac = np.linspace(0, 1, n)
    margins = a.margin_start + (margin_end[:, None] - a.margin_start) * frac
    ebit = revenue * margins
    nopat = ebit * (1 - a.tax_rate)
    prev_rev = np.column_stack([np.full(n_sims, a.base_revenue), revenue[:, :-1]])
    fcf = (nopat + revenue * a.da_pct - revenue * a.capex_pct
           - (revenue - prev_rev) * a.nwc_pct)

    t = np.arange(1, n + 1)
    discount = 1 / (1 + wacc[:, None]) ** t
    pv_sum = (fcf * discount).sum(axis=1)
    tv = fcf[:, -1] * (1 + g) / (wacc - g)
    pv_tv = tv / (1 + wacc) ** n

    equity = pv_sum + pv_tv - a.net_debt
    return equity * 1e12 / (a.shares * 1e6)


def implied_margin(a: Assumptions, target_price: float,
                   lo=-0.10, hi=0.50, tol=1e-6) -> float:
    """
    역DCF: 현재 주가를 정당화하려면 마지막 해 영업이익률이 몇 %여야 하는지
    이분법(bisection)으로 찾는다. 범위 안에 해가 없으면 NaN.
    """
    f = lambda m: valuate(replace(a, margin_end=m))["per_share"] - target_price
    if f(lo) * f(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2
