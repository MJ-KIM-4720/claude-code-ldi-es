"""
VaR 모델 단위 테스트
=====================
Known Results (CLAUDE.md)에 기반한 regression test.
"""

import pytest
from ldi import var_model as VaR, params as P


class TestVaRThreshold:
    """VaR threshold solver 검증."""

    def test_well_funded_non_binding(self):
        """y0=1.5 → constraint 안 걸림."""
        k_alpha, binding = VaR.solve_threshold(y0=1.5)
        assert not binding

    def test_underfunded_binding(self):
        """y0=0.8 → constraint 걸림."""
        k_alpha, binding = VaR.solve_threshold(y0=0.8)
        assert binding
        assert k_alpha < P.k


class TestVaRAdjustmentFactor:
    """A(y0) 값이 Known Results와 일치하는지 확인."""

    def test_A_at_y0_0_1(self):
        A = VaR.cross_sectional_A(y0=0.1)
        assert A == pytest.approx(1.80, abs=0.01)

    def test_A_at_y0_0_9(self):
        A = VaR.cross_sectional_A(y0=0.9)
        assert A == pytest.approx(0.89, abs=0.01)

    def test_A_at_y0_1_0(self):
        A = VaR.cross_sectional_A(y0=1.0)
        assert A == pytest.approx(0.92, abs=0.01)

    def test_A_at_y0_1_5(self):
        A = VaR.cross_sectional_A(y0=1.5)
        assert A == pytest.approx(1.00, abs=0.01)

    def test_underfunded_can_exceed_1(self):
        """VaR의 핵심 성질: underfunded에서 A > 1 (도박 인센티브)."""
        A = VaR.cross_sectional_A(y0=0.1)
        assert A > 1.0, f"Expected gambling incentive at y0=0.1, got A={A}"
