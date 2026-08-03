"""
VaR 모델 단위 테스트 (joint system + fixed claim)
=================================================
ES와 동일한 처리: g_VaR(y) >= y 이므로 Y0 != F0.
VaR feasibility는 quantile-hedging cost로 판정한다.
"""

import numpy as np
import pytest
from scipy.stats import norm

from ldi import var_model as VaR, params as P


BASE = dict(F0=1.0, alpha=0.10)


class TestJointSystem:

    @pytest.mark.parametrize("F0,alpha", [(1.0, 0.10), (1.0, 0.05), (1.0, 0.20),
                                          (0.9, 0.10), (1.2, 0.10)])
    def test_budget_residual(self, F0, alpha):
        """|Psi_VaR(0,Y0) - F0| < 1e-10."""
        s = VaR.solve_var(F0, alpha)
        if not s['binding']:
            pytest.skip("non-binding")
        assert abs(float(VaR.psi(s['Y0'], s['k_alpha'], P.T)) - F0) < 1e-10

    def test_threshold_proportional_to_Y0(self):
        """k_alpha = lambda(alpha)·Y0 (P-measure quantile)."""
        s = VaR.solve_var(**BASE)
        assert s['k_alpha'] == pytest.approx(VaR.lambda_ratio(0.10) * s['Y0'],
                                            rel=1e-12)

    def test_binding_probability_is_alpha(self):
        """P(Y_T < k_alpha) = alpha by construction."""
        s = VaR.solve_var(**BASE)
        z = (np.log(s['k_alpha'] / s['Y0']) - P.m_P * P.T) / (P.sigma_Y * np.sqrt(P.T))
        assert float(norm.cdf(z)) == pytest.approx(0.10, abs=1e-12)

    def test_Y0_below_F0_when_binding(self):
        s = VaR.solve_var(**BASE)
        assert s['binding'] and s['Y0'] < s['F0']

    def test_baseline_regression(self):
        """새 baseline 기준값 (2026-08 재계산)."""
        s = VaR.solve_var(**BASE)
        assert s['Y0'] == pytest.approx(0.916206, abs=1e-5)
        assert s['k_alpha'] == pytest.approx(0.714926, abs=1e-5)

    def test_well_funded_non_binding(self):
        s = VaR.solve_var(F0=2.0, alpha=0.10)
        assert not s['binding'] and s['Y0'] == 2.0


class TestFeasibility:

    def test_quantile_hedge_cost_baseline(self):
        """C_VaR(0.10) ≈ 0.7664 < F0 = 1 — VaR is comfortably feasible."""
        assert VaR.quantile_hedge_cost(0.10) == pytest.approx(0.766378, abs=1e-6)
        assert VaR.is_feasible(1.0, 0.10)

    def test_cost_decreasing_in_alpha(self):
        """더 엄격한 alpha일수록 비용이 크다."""
        cs = [VaR.quantile_hedge_cost(a) for a in [0.02, 0.05, 0.10, 0.25, 0.5]]
        assert np.all(np.diff(cs) < 0)

    def test_cost_limit_is_full_defeasance(self):
        """alpha -> 0 에서 비용 -> k·e^{-r̃T} (= ES floor + F0)."""
        assert VaR.quantile_hedge_cost(1e-10) == pytest.approx(
            P.k * np.exp(-P.r_tilde * P.T), rel=1e-6)

    def test_alpha_min_consistent(self):
        a = VaR.alpha_min(1.0)
        assert VaR.quantile_hedge_cost(a) == pytest.approx(1.0, abs=1e-9)
        assert a == pytest.approx(0.015975, abs=1e-5)

    def test_infeasible_below_alpha_min(self):
        a = VaR.alpha_min(1.0)
        with pytest.raises(VaR.InfeasibleError):
            VaR.solve_var(1.0, a * 0.5)

    def test_var_floor_below_es_floor(self):
        """논문 포인트: 같은 F0에서 VaR는 feasible, ES는 훨씬 빠듯하다."""
        assert VaR.quantile_hedge_cost(0.10) < 1.0
        assert P.eps_min(1.0) > 0.0


class TestGamblingIncentive:

    def test_A_exceeds_1_somewhere(self):
        """VaR의 핵심 성질: digital 때문에 A > 1 구간이 존재."""
        s = VaR.solve_var(**BASE)
        y = np.linspace(0.2, 2.0, 1000)
        A = VaR.adjustment_factor(y, s['k_alpha'], P.T)
        assert np.max(A) > 1.0

    def test_gambling_peak_near_threshold(self):
        s = VaR.solve_var(**BASE)
        y = np.linspace(0.2, 2.0, 2000)
        A = VaR.adjustment_factor(y, s['k_alpha'], P.T)
        y_peak = y[int(np.argmax(A))]
        assert abs(y_peak - s['k_alpha']) < 0.25

    def test_gambling_intensifies_near_maturity(self):
        s = VaR.solve_var(**BASE)
        y = np.linspace(0.3, 1.5, 2000)
        peaks = [np.max(VaR.adjustment_factor(y, s['k_alpha'], P.T - t))
                 for t in (0.0, 5.0, 8.0)]
        assert peaks[0] < peaks[1] < peaks[2]

    def test_claim_dominates_reference(self):
        s = VaR.solve_var(**BASE)
        y = np.linspace(0.01, 5.0, 1000)
        assert np.all(VaR.claim(y, s['k_alpha']) >= y - 1e-15)

    def test_psi_converges_to_claim(self):
        """Digital 불연속 때문에 k_alpha 근방은 제외하고 비교."""
        s = VaR.solve_var(**BASE)
        y = np.linspace(0.2, 2.0, 400)
        y = y[np.abs(y - s['k_alpha']) > 0.05]
        near = VaR.psi(y, s['k_alpha'], 1e-8)
        assert np.max(np.abs(near - VaR.claim(y, s['k_alpha']))) < 1e-3
