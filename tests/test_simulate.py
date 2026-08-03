"""
Fixed-claim Monte Carlo 테스트
==============================
핵심: 경로 위에서 threshold를 재계산하지 않는다. 그리고 delta-hedging
전략이 실제로 claim을 복제한다 (스텝을 늘리면 오차가 감소).
"""

import numpy as np
import pytest

from ldi import params as P, es_model as ES, var_model as VaR, simulate as SIM


N = 1500
NS = 60


class TestReferencePaths:

    def test_exact_scheme_moments(self):
        """ln Y_T ~ N(ln Y0 + m_P·T, sigma_Y²·T)."""
        _, Y, _ = SIM.reference_paths(1.0, n_paths=40000, n_steps=NS, seed=1)
        lg = np.log(Y[:, -1])
        se = P.sigma_Y * np.sqrt(P.T) / np.sqrt(40000)
        assert abs(np.mean(lg) - P.m_P * P.T) < 4 * se
        assert np.std(lg) == pytest.approx(P.sigma_Y * np.sqrt(P.T), rel=0.02)

    def test_step_count_does_not_change_terminal_law(self):
        """Exact scheme: 스텝 수는 terminal 분포를 바꾸지 않는다."""
        _, Y1, _ = SIM.reference_paths(1.0, n_paths=20000, n_steps=60, seed=3)
        _, Y2, _ = SIM.reference_paths(1.0, n_paths=20000, n_steps=240, seed=3)
        assert np.mean(np.log(Y1[:, -1])) == pytest.approx(
            np.mean(np.log(Y2[:, -1])), abs=0.01)


class TestFixedClaim:

    def test_claim_is_fixed_not_resolved(self):
        """F_T는 t=0의 (k_eps,c)로 만든 payoff와 정확히 일치해야 한다."""
        s = ES.solve_es()
        t, Y, _ = SIM.reference_paths(s['Y0'], N, NS, seed=5)
        out = SIM.evaluate('es', s, t, Y)
        assert np.allclose(out['F'][:, -1], ES.claim(Y[:, -1], s['k_eps'], s['c']))

    def test_initial_value_equals_F0(self):
        """Psi(0,Y0) = F0 for every model (budget constraint holds)."""
        res = SIM.run(n_paths=N, n_steps=NS, with_se=False)
        for name in ('merton', 'es', 'var'):
            assert res[name]['F'][:, 0] == pytest.approx(P.F0, abs=1e-10)

    def test_es_terminal_dominates_var_in_deep_tail(self):
        """ES는 tail에서 부분 보호(c·y), VaR는 포기 → 최악 구간에서 ES가 낫다."""
        res = SIM.run(n_paths=6000, n_steps=NS, with_se=False)
        assert res['es']['stats']['cvar05'] > res['var']['stats']['cvar05']


class TestReplication:

    def test_replication_error_shrinks_with_steps(self):
        """자기금융 이산 복제 오차가 스텝 2배마다 ~1/sqrt(2)로 감소."""
        errs = []
        for ns in (60, 240):
            r = SIM.run(n_paths=N, n_steps=ns, with_se=False, models=('es',))
            errs.append(r['es']['repl_err_mean'])
        assert errs[1] < errs[0] * 0.75
        assert errs[1] < 5e-3

    def test_merton_replication_is_exact(self):
        """A=1이면 복제 포트폴리오가 곧 reference process."""
        r = SIM.run(n_paths=500, n_steps=NS, with_se=False, models=('merton',))
        assert r['merton']['repl_err_max'] < 1e-10


class TestConstraintRecovery:

    def test_realized_shortfall_probability_matches_alpha(self):
        """VaR 전략의 실현 P(F_T<k)가 alpha에 수렴."""
        res = SIM.run(n_paths=40000, n_steps=NS, with_se=False,
                      models=('var',))
        se = np.sqrt(0.1 * 0.9 / 40000)
        assert abs(res['var']['stats']['shortfall_prob'] - P.alpha) < 4 * se

    def test_realized_Q_shortfall_matches_epsilon(self):
        """ES 전략의 실현 Q-shortfall이 eps 근처이고 eps_min 이상."""
        res = SIM.run(n_paths=40000, n_steps=NS, with_se=False, models=('es',))
        q, se = res['es']['q_shortfall'], res['es']['q_shortfall_se']
        assert abs(q - P.epsilon) < 4 * se
        assert q >= P.eps_min(P.F0) - 4 * se

    def test_merton_Q_shortfall_matches_eps_M(self):
        res = SIM.run(n_paths=40000, n_steps=NS, with_se=False,
                      models=('merton',))
        q = res['merton']['q_shortfall']
        assert abs(q - P.eps_merton(P.F0)) < 4 * res['merton']['q_shortfall_se']


class TestMatching:

    def test_equal_ce_matching(self):
        m = SIM.match_alpha_equal_ce(n_paths=N, n_steps=NS)
        assert m['bracketed']
        assert abs(m['achieved_ce_loss'] - m['target_ce_loss']) < 1e-6
        assert 0 < m['alpha'] < 0.5

    def test_threshold_matching(self):
        m = SIM.match_alpha_threshold()
        assert m['bracketed']
        assert m['k_alpha'] == pytest.approx(m['k_eps'], abs=1e-10)


class TestStats:

    def test_cvar_is_tail_mean(self):
        x = np.linspace(0.0, 1.0, 1001)
        assert SIM.cvar(x, 0.05) == pytest.approx(0.025, abs=1e-3)

    def test_certainty_equivalent_of_constant(self):
        assert SIM.certainty_equivalent(np.full(100, 1.3)) == pytest.approx(1.3)

    def test_bootstrap_se_positive(self):
        rng = np.random.default_rng(0)
        se = SIM.bootstrap_se(rng.lognormal(0, 0.2, 800), n_boot=50)
        assert all(v >= 0 for v in se.values())
        assert se['mean'] > 0
