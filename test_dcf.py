"""
계산 로직 검증 테스트. 실행: pytest -q

AI가 작성한 코드를 그대로 믿지 않고, 손으로 계산할 수 있는
단순한 경우를 만들어 결과가 일치하는지 확인한다.
"""

from dataclasses import replace

import numpy as np
import pytest

from dcf import Assumptions, implied_margin, monte_carlo, valuate

# 손계산이 가능한 단순 케이스:
# 매출 100 고정, 영업이익률 20%, 세율 25% → NOPAT 15
# 감가상각 = CAPEX, 매출 변화 없음 → FCF = 15 (매년 동일)
# WACC 10%, 영구성장률 0% → 기업가치 = 15 / 0.10 = 150
SIMPLE = Assumptions(
    base_year=2025, base_revenue=100, growth=(0.0, 0.0),
    margin_start=0.20, margin_end=0.20, tax_rate=0.25,
    da_pct=0.05, capex_pct=0.05, nwc_pct=0.10,
    wacc=0.10, terminal_growth=0.0, net_debt=30, shares=1.0,
)


def test_fcf_hand_calculation():
    table = valuate(SIMPLE)["table"]
    assert np.allclose(table["FCF"], [15, 15])


def test_enterprise_value_equals_perpetuity():
    r = valuate(SIMPLE)
    # 1년차 15/1.1 + 2년차 15/1.21 + TV(150)/1.21 = 150
    assert r["pv_fcf_sum"] == pytest.approx(15 / 1.1 + 15 / 1.21)
    assert r["enterprise_value"] == pytest.approx(150)


def test_equity_and_per_share():
    r = valuate(SIMPLE)
    assert r["equity_value"] == pytest.approx(120)          # 150 - 30
    assert r["per_share"] == pytest.approx(120e12 / 1e6)    # 조원 / 백만주 → 원


def test_growth_must_be_below_wacc():
    with pytest.raises(ValueError):
        valuate(replace(SIMPLE, terminal_growth=0.10))


def test_monte_carlo_without_noise_matches_dcf():
    base = Assumptions()
    sims = monte_carlo(base, n_sims=100, wacc_sd=0, g_sd=0, margin_sd=0, growth_sd=0)
    assert np.allclose(sims, valuate(base)["per_share"])


def test_implied_margin_round_trip():
    base = Assumptions()
    price = valuate(replace(base, margin_end=0.15))["per_share"]
    assert implied_margin(base, price) == pytest.approx(0.15, abs=1e-4)
