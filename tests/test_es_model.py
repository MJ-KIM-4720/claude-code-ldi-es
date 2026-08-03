"""
ES 모델 단위 테스트 (joint system + fixed claim)
================================================
2026-08 이론 수정 반영:
  - Y0는 F0가 아니다. (Y0, k_eps)는 budget + binding ES 연립해.
  - eps <= eps_min 이면 해가 존재하지 않는다 (feasibility floor).
  - A_ES(t,y)는 t=0에 고정된 claim의 delta. 경로 위에서 k_eps 재계산 금지.
"""

import numpy as np
import pytest

from ldi import es_model as ES, params as P
from ldi.bs_utils import bs_put


BASE = dict(F0=1.0, eps=0.10)


# ═══════════════════════════════════════════════════════════
# Feasibility band
# ═══════════════════════════════════════════════════════════

class TestFeasibility:

    def test_eps_min_formula(self):
        """eps_min = max(k·e^{-r̃T} - F0, 0) ≈ 0.087629 at baseline."""
        assert P.eps_min(1.0) == pytest.approx(
            P.k * np.exp(-P.r_tilde * P.T) - 1.0, abs=1e-14)
        assert P.eps_min(1.0) == pytest.approx(0.0876289, abs=1e-6)

    def test_eps_merton(self):
        assert P.eps_merton(1.0) == pytest.approx(0.1526135, abs=1e-6)

    def test_band_ordering(self):
        """eps_min <= eps_M always (put dominates its discounted intrinsic)."""
        for F0 in [0.5, 0.8, 1.0, 1.3, 2.0]:
            lo, hi = P.eps_band(F0)
            assert lo <= hi + 1e-15

    def test_old_baseline_is_infeasible(self):
        """이전 baseline eps=0.05 는 floor 아래 — 해가 없어야 한다."""
        with pytest.raises(ES.InfeasibleError):
            ES.solve_es(F0=1.0, eps=0.05)

    def test_infeasible_below_floor(self):
        lo = P.eps_min(1.0)
        with pytest.raises(ES.InfeasibleError):
            ES.solve_es(F0=1.0, eps=lo * 0.999)
        with pytest.raises(ES.InfeasibleError):
            ES.solve_es(F0=1.0, eps=lo)          # boundary itself unattainable

    def test_non_strict_returns_flag(self):
        s = ES.solve_es(F0=1.0, eps=0.05, strict=False)
        assert s['feasible'] is False
        assert np.isnan(s['Y0'])

    def test_slack_above_eps_M(self):
        s = ES.solve_es(F0=1.0, eps=P.eps_merton(1.0) * 1.01)
        assert s['feasible'] and not s['binding']
        assert s['Y0'] == 1.0 and s['k_eps'] == P.k and s['c'] == 1.0


# ═══════════════════════════════════════════════════════════
# The joint system is actually solved
# ═══════════════════════════════════════════════════════════

class TestJointSystem:

    @pytest.mark.parametrize("F0,eps", [(1.0, 0.10), (1.0, 0.09), (1.0, 0.15),
                                        (1.2, 0.05), (1.5, 0.02), (0.99, 0.10)])
    def test_budget_residual(self, F0, eps):
        """|Psi_ES(0,Y0) - F0| < 1e-10."""
        s = ES.solve_es(F0, eps)
        if not s['binding']:
            pytest.skip("non-binding")
        res = float(ES.psi(s['Y0'], s['k_eps'], s['c'], P.T)) - F0
        assert abs(res) < 1e-10

    @pytest.mark.parametrize("F0,eps", [(1.0, 0.10), (1.0, 0.09), (1.0, 0.15),
                                        (1.2, 0.05), (1.5, 0.02), (0.99, 0.10)])
    def test_constraint_residual(self, F0, eps):
        """|c·Put(0,Y0,k_eps) - eps| < 1e-10."""
        s = ES.solve_es(F0, eps)
        if not s['binding']:
            pytest.skip("non-binding")
        lhs = s['c'] * bs_put(s['Y0'], s['k_eps'], P.r_tilde, P.sigma_Y, P.T)
        assert abs(lhs - eps) < 1e-10

    def test_Y0_below_F0_when_binding(self):
        """보호 claim은 reference process보다 비싸므로 Y0 < F0."""
        s = ES.solve_es(**BASE)
        assert s['Y0'] < s['F0']
        assert s['c'] > 1.0 and s['k_eps'] < P.k

    def test_baseline_regression(self):
        """새 baseline 기준값 (2026-08 재계산)."""
        s = ES.solve_es(**BASE)
        assert s['Y0'] == pytest.approx(0.803420, abs=1e-5)
        assert s['k_eps'] == pytest.approx(0.727553, abs=1e-5)
        assert s['c'] == pytest.approx(1.374471, abs=1e-5)

    def test_limit_eps_to_eps_min(self):
        """eps -> eps_min+ 에서 Y0 -> 0 (단조 감소).

        수렴은 매우 느리다: 초과분 d = eps - eps_min에 대해
        d = k·e^{-r̃T}(N(-d2)-1) + Y0·N(d1) 이고 두 항 모두
        exp(-(ln Y0)²/(2σ²T)) 속도로 사라지므로 Y0는 log 스케일로만
        0에 접근한다 (d=1e-14 에서도 Y0 ≈ 0.19).
        """
        lo = P.eps_min(1.0)
        prev = np.inf
        for d in [1e-2, 1e-3, 1e-4, 1e-6, 1e-9, 1e-12, 1e-14]:
            Y0 = ES.solve_es(1.0, lo + d)['Y0']
            assert Y0 < prev
            prev = Y0
        assert prev < 0.20
        assert ES.solve_es(1.0, lo + 1e-9)['Y0'] < 0.30

    def test_limit_eps_to_eps_M(self):
        """eps -> eps_M- 에서 Y0 -> F0, k_eps -> k, c -> 1."""
        hi = P.eps_merton(1.0)
        s = ES.solve_es(1.0, hi * (1 - 1e-9))
        assert s['Y0'] == pytest.approx(1.0, abs=1e-7)
        assert s['k_eps'] == pytest.approx(P.k, abs=1e-6)
        assert s['c'] == pytest.approx(1.0, abs=1e-6)

    def test_Y0_monotone_in_eps(self):
        lo, hi = P.eps_band(1.0)
        grid = np.linspace(lo + 1e-4, hi - 1e-4, 25)
        Y0s = [ES.solve_es(1.0, e)['Y0'] for e in grid]
        assert np.all(np.diff(Y0s) > 0)


# ═══════════════════════════════════════════════════════════
# Wedge identity & derivatives
# ═══════════════════════════════════════════════════════════

class TestWedgeIdentity:

    @pytest.mark.parametrize("t", [0.0, 2.5, 5.0, 7.5, 9.5])
    @pytest.mark.parametrize("y", [0.3, 0.6, 0.8, 1.0, 1.4, 2.0])
    def test_wedge_matches_finite_difference(self, t, y):
        """Psi - y·Psi_y = k·e^{-r̃τ}[N(-d2(k)) - N(-d2(k_eps))], vs FD < 1e-6."""
        s = ES.solve_es(**BASE)
        tau = P.T - t
        ke, c = s['k_eps'], s['c']

        h = 1e-6 * y
        psi_p = float(ES.psi(y + h, ke, c, tau))
        psi_m = float(ES.psi(y - h, ke, c, tau))
        dpsi_fd = (psi_p - psi_m) / (2 * h)

        assert float(ES.dpsi_dy(y, ke, c, tau)) == pytest.approx(dpsi_fd, abs=1e-6)

        wedge_fd = float(ES.psi(y, ke, c, tau)) - y * dpsi_fd
        assert float(ES.wedge(y, ke, c, tau)) == pytest.approx(wedge_fd, abs=1e-6)

    def test_wedge_non_negative_everywhere(self):
        """수학적으로는 wedge > 0. 다만 y가 아주 작으면 N(-d2(k))와
        N(-d2(k_eps))가 둘 다 double precision에서 정확히 1.0이 되어
        차이가 0으로 underflow한다 (baseline에서 y <~ 0.10). 그래서
        전 구간에서는 >= 0, 분해 가능한 구간에서는 > 0 을 검증한다."""
        s = ES.solve_es(**BASE)
        y = np.linspace(0.02, 5.0, 500)
        for t in (0.0, 3.0, 6.0, 9.0):
            assert np.all(ES.wedge(y, s['k_eps'], s['c'], P.T - t) >= 0)

    def test_wedge_strictly_positive_in_relevant_band(self):
        """underflow가 일어나지 않는 경제적 관심 구간에서는 엄격히 양수.

        underflow 경계는 sigma_Y·sqrt(tau)에 따라 움직이므로 (만기에
        가까울수록 더 넓은 구간이 saturate) 고정 구간 [0.5, 2.0]로 검증.
        """
        s = ES.solve_es(**BASE)
        y = np.linspace(0.5, 2.0, 400)
        for t in (0.0, 3.0, 6.0, 9.0):
            assert np.all(ES.wedge(y, s['k_eps'], s['c'], P.T - t) > 0)


# ═══════════════════════════════════════════════════════════
# No gambling: 0 < A < 1, U-shape
# ═══════════════════════════════════════════════════════════

class TestNoGambling:

    def test_A_between_0_and_1(self):
        """0 < A <= 1 (machine precision): 도박 인센티브 없음.

        A는 wedge form (1 - wedge/Psi)으로 계산하므로 clamping 없이
        상한이 정확히 성립한다.
        """
        s = ES.solve_es(**BASE)
        y = np.linspace(0.02, 8.0, 800)
        for t in (0.0, 2.5, 5.0, 7.5, 9.9):
            A = ES.adjustment_factor(y, s['k_eps'], s['c'], P.T - t)
            assert np.all(A > 0), f"A <= 0 at t={t}"
            assert np.all(A <= 1.0), f"A > 1 at t={t}, max={A.max()}"

    def test_A_strictly_below_1_wherever_wedge_resolves(self):
        """상대 wedge가 machine epsilon 위인 모든 (t,y)에서 A < 1 이 엄격히 성립.

        wedge/Psi <= eps_machine 이면 1 - wedge/Psi 가 정확히 1.0으로
        반올림된다 (부동소수점 한계이지 모형의 성질이 아니다).
        """
        s = ES.solve_es(**BASE)
        y = np.linspace(0.02, 8.0, 800)
        checked = 0
        for t in (0.0, 2.5, 5.0, 7.5, 9.9):
            tau = P.T - t
            w = np.asarray(ES.wedge(y, s['k_eps'], s['c'], tau))
            ps = np.asarray(ES.psi(y, s['k_eps'], s['c'], tau))
            A = np.asarray(ES.adjustment_factor(y, s['k_eps'], s['c'], tau))
            m = (w / ps) > 1e-15
            assert np.all(A[m] < 1.0), f"A >= 1 at t={t}, max={A[m].max()}"
            checked += int(m.sum())
        assert checked > 2000            # 테스트가 공허하지 않은지 확인

    def test_A_matches_direct_quotient(self):
        """wedge form과 직접 계산 y·Psi_y/Psi 가 일치 (대수적 동치)."""
        s = ES.solve_es(**BASE)
        y = np.linspace(0.3, 3.0, 200)
        for t in (0.0, 5.0):
            tau = P.T - t
            direct = y * ES.dpsi_dy(y, s['k_eps'], s['c'], tau) / \
                ES.psi(y, s['k_eps'], s['c'], tau)
            A = ES.adjustment_factor(y, s['k_eps'], s['c'], tau)
            assert np.allclose(A, direct, atol=1e-12)

    def test_A_tends_to_1_at_both_ends(self):
        """U-shape: y->0 (claim ≈ c·y, linear) 와 y->inf (unconstrained) 양쪽 →1."""
        s = ES.solve_es(**BASE)
        for tau in (P.T, P.T / 2):
            A_lo = float(ES.adjustment_factor(1e-4, s['k_eps'], s['c'], tau))
            A_hi = float(ES.adjustment_factor(1e3, s['k_eps'], s['c'], tau))
            assert A_lo == pytest.approx(1.0, abs=1e-3)
            assert A_hi == pytest.approx(1.0, abs=1e-3)

    def test_A_has_interior_dip(self):
        s = ES.solve_es(**BASE)
        y = np.linspace(0.05, 5.0, 2000)
        A = ES.adjustment_factor(y, s['k_eps'], s['c'], P.T)
        i = int(np.argmin(A))
        assert 0 < i < len(y) - 1                  # interior minimum
        assert A[i] < 0.7
        assert 0.5 < y[i] < 1.5

    def test_cross_sectional_A_leq_1(self):
        for F0 in [0.99, 1.0, 1.05, 1.2, 1.5, 2.0]:
            A = ES.cross_sectional_A(F0, 0.10, strict=False)
            assert np.isnan(A) or A <= 1.0 + 1e-12


# ═══════════════════════════════════════════════════════════
# Claim function
# ═══════════════════════════════════════════════════════════

class TestClaim:

    def test_claim_shape(self):
        s = ES.solve_es(**BASE)
        ke, c = s['k_eps'], s['c']
        assert float(ES.claim(0.5 * ke, ke, c)) == pytest.approx(c * 0.5 * ke)
        assert float(ES.claim(0.5 * (ke + P.k), ke, c)) == pytest.approx(P.k)
        assert float(ES.claim(1.5, ke, c)) == pytest.approx(1.5)

    def test_claim_dominates_reference(self):
        """g_ES(y) >= y everywhere — the reason A <= 1 and the floor exists."""
        s = ES.solve_es(**BASE)
        y = np.linspace(0.01, 5.0, 1000)
        assert np.all(ES.claim(y, s['k_eps'], s['c']) >= y - 1e-15)

    def test_psi_converges_to_claim(self):
        s = ES.solve_es(**BASE)
        y = np.linspace(0.2, 2.0, 50)
        near = ES.psi(y, s['k_eps'], s['c'], 1e-8)
        assert np.max(np.abs(near - ES.claim(y, s['k_eps'], s['c']))) < 1e-3
