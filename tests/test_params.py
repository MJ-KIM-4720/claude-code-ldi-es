"""
파라미터 검증 테스트
====================
Merton ≈ 84%, r_tilde ≈ -0.023, sigma_Y ≈ 0.0711
"""

import pytest
from ldi import params as P


class TestBaseParameters:
    """기본 파라미터가 올바르게 설정되었는지 확인."""

    def test_R_and_r_not_swapped(self):
        """R=0.02(real), r=0.04(nominal) — 스왑되면 Merton이 650%로 폭발."""
        assert P.R == 0.02
        assert P.r == 0.04

    def test_constraint_defaults(self):
        assert P.alpha == 0.05
        assert P.epsilon == 0.10
        assert P.T == 5.0
        assert P.k == 1.0
        assert P.GAMMA == 3.0


class TestDerivedQuantities:
    """파생 값이 CLAUDE.md의 기준과 일치하는지 확인."""

    def test_r_tilde(self):
        assert P.r_tilde == pytest.approx(-0.023, abs=1e-4)

    def test_sigma_Y(self):
        assert P.sigma_Y == pytest.approx(0.0711, abs=1e-3)

    def test_merton_total(self):
        """Merton total ≈ 84%. 650%이면 R/r 스왑된 것."""
        total = P.Pi_star.sum()
        assert total == pytest.approx(0.84, abs=0.01)

    def test_merton_components_positive(self):
        assert P.Pi_star[0] > 0  # pi*_S > 0
        assert P.Pi_star[1] > 0  # pi*_I > 0
