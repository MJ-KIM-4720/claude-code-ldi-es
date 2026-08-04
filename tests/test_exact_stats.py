"""
Exact (model-implied) statistics 테스트
=======================================
closed form 값이 (a) 독립적인 수치적분, (b) MC, (c) 오더에 명시된
기대값과 일치하는지 검증한다.
"""

import numpy as np
import pytest
from scipy.stats import norm
from scipy.integrate import quad

from ldi import params as P, es_model as ES, var_model as VaR
from ldi import exact_stats as X, simulate as SIM


BASE_ES = ES.solve_es()
BASE_VAR = VaR.solve_var()


def _g(y, c, k_low, k=1.0):
    """Piecewise claim, scalar."""
    if y < k_low:
        return c * y
    if y < k:
        return k
    return y


def _numeric_moment(a, Y0, c, k_low, k=1.0):
    """E[g(Y_T)^a] by quadrature over the lognormal density — независимая check."""
    m, s = X.log_moments(Y0)
    f = lambda z: _g(np.exp(m + s * z), c, k_low, k) ** a * norm.pdf(z)
    val, _ = quad(f, -12, 12, limit=400)
    return val


# ═══════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════

class TestBuildingBlocks:

    def test_lambda_plus_upsilon_is_full_moment(self):
        m, s = X.log_moments(0.9)
        for a in (1, 2, -2):
            full = np.exp(a * m + a ** 2 * s ** 2 / 2)
            assert X.Lambda(a, 1.1, m, s) + X.Upsilon(a, 1.1, m, s) == \
                pytest.approx(full, rel=1e-12)

    def test_lambda_matches_quadrature(self):
        m, s = X.log_moments(0.85)
        for a, K in [(1, 0.8), (2, 1.0), (-2, 1.2)]:
            z_star = (np.log(K) - m) / s          # split at the discontinuity
            f = lambda z: np.exp(a * (m + s * z)) * norm.pdf(z)
            num, _ = quad(f, -12, z_star, limit=400)
            assert X.Lambda(a, K, m, s) == pytest.approx(num, rel=1e-8)

    def test_prob_below_matches_quantile(self):
        m, s = X.log_moments(1.0)
        for q in (0.01, 0.05, 0.25, 0.5, 0.9):
            assert X.prob_below(X.y_quantile(q, m, s), m, s) == \
                pytest.approx(q, abs=1e-12)


# ═══════════════════════════════════════════════════════════
# Assembly vs independent quadrature
# ═══════════════════════════════════════════════════════════

class TestAgainstQuadrature:

    @pytest.mark.parametrize("name", ['merton', 'es', 'var'])
    def test_mean_and_second_moment(self, name):
        if name == 'merton':
            Y0, c, kl = P.F0, 1.0, P.k
        elif name == 'es':
            Y0, c, kl = BASE_ES['Y0'], BASE_ES['c'], BASE_ES['k_eps']
        else:
            Y0, c, kl = BASE_VAR['Y0'], 1.0, BASE_VAR['k_alpha']
        st = X.claim_stats(Y0, c=c, k_low=kl)
        assert st['mean'] == pytest.approx(_numeric_moment(1, Y0, c, kl), rel=1e-8)
        second = st['std'] ** 2 + st['mean'] ** 2
        assert second == pytest.approx(_numeric_moment(2, Y0, c, kl), rel=1e-8)

    def test_certainty_equivalent(self):
        Y0, c, kl = BASE_ES['Y0'], BASE_ES['c'], BASE_ES['k_eps']
        st = X.claim_stats(Y0, c=c, k_low=kl)
        omg = 1 - P.GAMMA
        assert st['ce'] ** omg == pytest.approx(
            _numeric_moment(omg, Y0, c, kl), rel=1e-8)

    def test_merton_closed_forms(self):
        """Merton reduces to plain lognormal formulas."""
        m, s = X.log_moments(P.F0)
        st = X.merton_stats()
        assert st['mean'] == pytest.approx(np.exp(m + s ** 2 / 2), rel=1e-12)
        assert st['ce'] == pytest.approx(
            np.exp(m + (1 - P.GAMMA) * s ** 2 / 2), rel=1e-12)
        assert st['prob_shortfall'] == pytest.approx(
            norm.cdf((np.log(P.k) - m) / s), rel=1e-12)


# ═══════════════════════════════════════════════════════════
# Tail statistics
# ═══════════════════════════════════════════════════════════

class TestTailStats:

    @pytest.mark.parametrize("q", [0.01, 0.05, 0.10, 0.30, 0.50])
    def test_quantile_and_bottom_mean_consistent(self, q):
        """bottom-q mean <= Q_q, and both reproduce a fine numerical sample."""
        Y0, c, kl = BASE_ES['Y0'], BASE_ES['c'], BASE_ES['k_eps']
        st = X.claim_stats(Y0, c=c, k_low=kl, tail_q=q)
        m, s = X.log_moments(Y0)
        # deterministic grid of the lognormal (inverse-CDF sampling, no RNG)
        u = (np.arange(400_000) + 0.5) / 400_000
        F = np.array([_g(y, c, kl) for y in np.exp(m + s * norm.ppf(u))])
        F_sorted = np.sort(F)
        n_tail = int(round(q * len(F_sorted)))
        assert st['q5'] == pytest.approx(F_sorted[n_tail - 1], rel=2e-3)
        assert st['bottom5_mean'] == pytest.approx(
            F_sorted[:n_tail].mean(), rel=2e-3)
        assert st['bottom5_mean'] <= st['q5'] + 1e-12

    def test_bottom_mean_branches(self):
        """q가 세 구간을 넘나들 때도 단조 증가해야 한다."""
        Y0, c, kl = BASE_ES['Y0'], BASE_ES['c'], BASE_ES['k_eps']
        vals = [X.claim_stats(Y0, c=c, k_low=kl, tail_q=q)['bottom5_mean']
                for q in (0.02, 0.05, 0.2, 0.24, 0.3, 0.45, 0.6, 0.9)]
        assert np.all(np.diff(vals) > 0)

    def test_var_quantile_below_alpha(self):
        """q=5% < alpha=10% 이면 VaR의 Q5는 보호 없는 구간에 있다."""
        st = X.var_stats(sol=BASE_VAR)
        m, s = X.log_moments(BASE_VAR['Y0'])
        assert st['q5'] == pytest.approx(X.y_quantile(0.05, m, s), rel=1e-12)


# ═══════════════════════════════════════════════════════════
# Constraint recovery
# ═══════════════════════════════════════════════════════════

class TestConstraintRecovery:

    def test_var_prob_shortfall_equals_alpha(self):
        for a in (0.05, 0.08118, 0.10, 0.15):
            st = X.var_stats(alpha=a)
            assert st['prob_shortfall'] == pytest.approx(a, abs=1e-12)

    def test_es_prob_shortfall_is_prob_below_k_eps(self):
        st = X.es_stats(sol=BASE_ES)
        m, s = X.log_moments(BASE_ES['Y0'])
        assert st['prob_shortfall'] == pytest.approx(
            norm.cdf((np.log(BASE_ES['k_eps']) - m) / s), rel=1e-12)

    def test_cond_shortfall_identity(self):
        for st in (X.es_stats(sol=BASE_ES), X.var_stats(sol=BASE_VAR)):
            assert st['cond_shortfall'] * st['prob_shortfall'] == \
                pytest.approx(st['exp_shortfall'], rel=1e-12)


# ═══════════════════════════════════════════════════════════
# Calibration
# ═══════════════════════════════════════════════════════════

class TestCalibration:

    def test_equal_ce_alpha(self):
        eq = X.match_alpha_equal_ce()
        assert eq['bracketed']
        assert eq['alpha'] == pytest.approx(0.08118, abs=1e-4)
        assert eq['ce_achieved'] == pytest.approx(eq['ce_target'], rel=1e-9)
        assert abs(eq['alpha'] - 0.081178) < 1e-5

    def test_equal_ce_is_deterministic(self):
        """MC와 달리 seed 의존성이 없어야 한다."""
        a1 = X.match_alpha_equal_ce()['alpha']
        a2 = X.match_alpha_equal_ce()['alpha']
        assert a1 == a2

    def test_threshold_matched_alpha(self):
        thr = X.match_alpha_threshold()
        assert thr['alpha'] == pytest.approx(0.106663, abs=1e-5)
        assert thr['k_alpha'] == pytest.approx(BASE_ES['k_eps'], rel=1e-12)
        # budget must hold at the implied Y0
        assert float(VaR.psi(thr['Y0'], thr['k_alpha'], P.T)) == \
            pytest.approx(P.F0, abs=1e-10)

    def test_threshold_matched_agrees_with_solver(self):
        thr = X.match_alpha_threshold()
        s = VaR.solve_var(alpha=thr['alpha'])
        assert s['k_alpha'] == pytest.approx(thr['k_alpha'], rel=1e-8)
        assert s['Y0'] == pytest.approx(thr['Y0'], rel=1e-8)


# ═══════════════════════════════════════════════════════════
# VaR feasibility bound
# ═══════════════════════════════════════════════════════════

class TestAlphaMin:

    def test_closed_form_matches_numeric(self):
        assert VaR.alpha_min() == pytest.approx(VaR.alpha_min_numeric(), abs=1e-12)
        assert VaR.alpha_min() == pytest.approx(0.01597, abs=1e-4)

    def test_cost_at_alpha_min_equals_F0(self):
        assert VaR.quantile_hedge_cost(VaR.alpha_min()) == \
            pytest.approx(P.F0, abs=1e-10)

    def test_gamma_free(self):
        """lambda = gamma·sigma_Y = sqrt(theta_sq): alpha_min must not move."""
        base = VaR.alpha_min()
        for g in (2.0, 3.0, 5.0, 8.0, 12.0):
            with P.override_params(GAMMA=g):
                assert VaR.market_price_of_risk() == pytest.approx(
                    np.sqrt(P.theta_sq), rel=1e-12)
                assert VaR.alpha_min() == pytest.approx(base, abs=1e-12)

    def test_zero_when_target_defeasible(self):
        """r̃ > 0 이면 target을 무위험으로 완전히 커버할 수 있어 bound가 0."""
        with P.override_params(MU_I=0.010):
            assert P.r_tilde > 0
            assert VaR.alpha_min() == 0.0

    def test_positive_and_binding_when_underfunded(self):
        with P.override_params(MU_I=0.030):
            assert VaR.alpha_min() == pytest.approx(0.02148, abs=1e-4)

    def test_all_calibrations_feasible(self):
        a_min = VaR.alpha_min()
        for a in (P.alpha, X.match_alpha_equal_ce()['alpha'],
                  X.match_alpha_threshold()['alpha']):
            assert a > a_min

    def test_quantile_hedge_cost_two_forms_agree(self):
        """closed form vs the d2-based form in the solver."""
        from ldi.bs_utils import bs_d2
        for a in (0.02, 0.05, 0.10, 0.25):
            lam = VaR.lambda_ratio(a)
            d2form = float(P.k * np.exp(-P.r_tilde * P.T)
                           * norm.cdf(bs_d2(1.0, lam, P.r_tilde, P.sigma_Y, P.T)))
            assert VaR.quantile_hedge_cost(a) == pytest.approx(d2form, rel=1e-12)


# ═══════════════════════════════════════════════════════════
# delta_L channel isolation
# ═══════════════════════════════════════════════════════════

class TestDeltaL:

    def test_baseline_value(self):
        assert P.delta_L() == pytest.approx(0.0484, abs=1e-12)

    def test_asset_side_invariant(self):
        """delta_L만 움직이고 자산 파라미터는 전부 고정되어야 한다."""
        sig, tot, th = P.sigma_Y, P.Pi_star.copy(), P.theta_sq
        for d in (0.040, 0.043, 0.046, 0.052):
            with P.override_delta_L(d):
                assert P.sigma_Y == pytest.approx(sig, rel=1e-14)
                assert P.theta_sq == pytest.approx(th, rel=1e-14)
                assert np.allclose(P.Pi_star, tot, rtol=1e-14)
                assert P.r_tilde == pytest.approx(P.r - d, abs=1e-14)

    @pytest.mark.parametrize("d,expected", [(0.040, 0.0), (0.043, 0.030455),
                                            (0.046, 0.061837), (0.0484, 0.087629),
                                            (0.052, 0.127497)])
    def test_eps_min_grid(self, d, expected):
        with P.override_delta_L(d):
            assert P.eps_min(1.0) == pytest.approx(expected, abs=1e-5)

    def test_status_transitions(self):
        """slack -> binding -> infeasible 순서."""
        status = []
        for d in (0.040, 0.043, 0.046, 0.0484, 0.052):
            with P.override_delta_L(d):
                s = ES.solve_es(strict=False)
                status.append('infeasible' if not s['feasible']
                              else ('slack' if not s['binding'] else 'binding'))
        assert status == ['slack', 'binding', 'binding', 'binding', 'infeasible']

    def test_restores_baseline(self):
        before = (P.BETA_0, P.r_tilde)
        with P.override_delta_L(0.052):
            pass
        assert (P.BETA_0, P.r_tilde) == before


# ═══════════════════════════════════════════════════════════
# Exact vs Monte Carlo
# ═══════════════════════════════════════════════════════════

class TestAgainstMonteCarlo:

    def test_within_three_se(self):
        """exact 통계가 MC ± 3 SE 안에 있어야 한다 (Merton P(F<k) 포함)."""
        mc = SIM.run(n_paths=10_000, n_boot=200)
        rows = {'merton': X.merton_stats(), 'es': X.es_stats(sol=BASE_ES),
                'var': X.var_stats(sol=BASE_VAR)}
        pairs = [('mean', 'mean'), ('std', 'std'),
                 ('prob_shortfall', 'shortfall_prob'),
                 ('exp_shortfall', 'exp_shortfall'),
                 ('q5', 'q05'), ('bottom5_mean', 'cvar05'), ('ce', 'CE')]
        for name, st in rows.items():
            for ex_key, mc_key in pairs:
                se = mc[name]['se'][mc_key]
                z = (st[ex_key] - mc[name]['stats'][mc_key]) / se
                assert abs(z) < 3.0, f'{name}.{ex_key} is {z:+.2f} SE from MC'
