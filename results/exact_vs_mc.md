# Exact vs Monte Carlo reconciliation

MC: N=10000, steps=120, seed=20260803, bootstrap SE.

| statistic | exact | MC | SE | (exact-MC)/SE | within 3 SE |
|---|---|---|---|---|---|
| Merton mean | 1.10562 | 1.10972 | 0.00296 | -1.39 | yes |
| Merton std | 0.27838 | 0.28070 | 0.00267 | -0.87 | yes |
| Merton prob_shortfall | 0.38935 | 0.37560 | 0.00487 | +2.83 | yes |
| Merton exp_shortfall | 0.05941 | 0.05859 | 0.00101 | +0.81 | yes |
| Merton cond_shortfall | 0.15258 | 0.15599 | 0.00180 | -1.89 | yes |
| Merton q5 | 0.71310 | 0.70834 | 0.00377 | +1.26 | yes |
| Merton bottom5_mean | 0.64554 | 0.64228 | 0.00376 | +0.87 | yes |
| Merton ce | 1.00824 | 1.01053 | 0.00275 | -0.83 | yes |
| ES (eps=0.1) mean | 1.01516 | 1.01595 | 0.00157 | -0.50 | yes |
| ES (eps=0.1) std | 0.14701 | 0.15003 | 0.00230 | -1.31 | yes |
| ES (eps=0.1) prob_shortfall | 0.24791 | 0.24090 | 0.00432 | +1.62 | yes |
| ES (eps=0.1) exp_shortfall | 0.03249 | 0.03254 | 0.00076 | -0.06 | yes |
| ES (eps=0.1) cond_shortfall | 0.13106 | 0.13506 | 0.00206 | -1.94 | yes |
| ES (eps=0.1) q5 | 0.78746 | 0.78220 | 0.00417 | +1.26 | yes |
| ES (eps=0.1) bottom5_mean | 0.71285 | 0.70925 | 0.00415 | +0.87 | yes |
| ES (eps=0.1) ce | 0.98530 | 0.98504 | 0.00155 | +0.17 | yes |
| VaR (alpha=0.1) mean | 1.06981 | 1.07132 | 0.00240 | -0.63 | yes |
| VaR (alpha=0.1) std | 0.22387 | 0.22700 | 0.00263 | -1.19 | yes |
| VaR (alpha=0.1) prob_shortfall | 0.10000 | 0.10040 | 0.00305 | -0.13 | yes |
| VaR (alpha=0.1) exp_shortfall | 0.03611 | 0.03653 | 0.00112 | -0.38 | yes |
| VaR (alpha=0.1) cond_shortfall | 0.36109 | 0.36385 | 0.00190 | -1.45 | yes |
| VaR (alpha=0.1) q5 | 0.65335 | 0.64899 | 0.00346 | +1.26 | yes |
| VaR (alpha=0.1) bottom5_mean | 0.59144 | 0.58846 | 0.00344 | +0.87 | yes |
| VaR (alpha=0.1) ce | 0.99213 | 0.99168 | 0.00288 | +0.16 | yes |
