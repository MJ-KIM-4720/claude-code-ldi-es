"""
ES 모델 단위 테스트
====================
Known Results (CLAUDE.md)에 기반한 regression test.
"""

import pytest
from ldi import es_model as ES, params as P


class TestESThreshold:
    """ES threshold solver 검증."""

    def test_well_funded_non_binding(self):
        """y0=1.5 → constraint 안 걸림."""
        k_eps, c, binding = ES.solve_threshold(y0=1.5)
        assert not binding
        assert k_eps == P.k
        assert c == 1.0

    def test_underfunded_binding(self):
        """y0=0.8 → constraint 걸림, k_eps < k."""
        k_eps, c, binding = ES.solve_threshold(y0=0.8)
        assert binding
        assert k_eps < P.k
        assert c > 1.0

    def test_binding_condition_satisfied(self):
        """c * Put(y0, k_eps) = epsilon 확인."""
        from ldi.bs_utils import bs_put

        y0 = 1.0
        k_eps, c, binding = ES.solve_threshold(y0)
        assert binding
        lhs = c * bs_put(y0, k_eps, P.r_tilde, P.sigma_Y, P.T)
        assert lhs == pytest.approx(P.epsilon, abs=1e-8)


class TestESAdjustmentFactor:
    """A(y0) 값이 Known Results와 일치하는지 확인."""

    def test_A_at_y0_0_1(self):
        A = ES.cross_sectional_A(y0=0.1)
        assert A == pytest.approx(0.29, abs=0.01)

    def test_A_at_y0_0_9(self):
        A = ES.cross_sectional_A(y0=0.9)
        assert A == pytest.approx(0.50, abs=0.01)

    def test_A_at_y0_1_0(self):
        A = ES.cross_sectional_A(y0=1.0)
        assert A == pytest.approx(0.65, abs=0.01)

    def test_A_at_y0_1_5(self):
        A = ES.cross_sectional_A(y0=1.5)
        assert A == pytest.approx(1.00, abs=0.01)

    def test_A_always_leq_1(self):
        """ES의 핵심 성질: A <= 1 (도박 인센티브 없음)."""
        for y0 in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.5]:
            A = ES.cross_sectional_A(y0)
            assert A <= 1.0 + 1e-10, f"A={A} > 1 at y0={y0}"
