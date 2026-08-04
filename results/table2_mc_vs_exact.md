# Table 2 (MC) vs closed form

N = 1,000,000, seed = 20260803, exact lognormal terminal draws (no path discretisation), common random numbers.
Tolerance |MC - exact| < 0.001 on every checked column.

```
row                                     statistic                 MC       exact     |diff|  status
-----------------------------------------------------------------------------------------------
Merton                                  mean                1.106154    1.105618   5.37e-04  ok
Merton                                  std                 0.278438    0.278380   5.81e-05  ok
Merton                                  prob_shortfall      0.388900    0.389351   4.51e-04  ok
Merton                                  exp_shortfall       0.059264    0.059408   1.44e-04  ok
Merton                                  cond_shortfall      0.152390    0.152582   1.93e-04  ok
Merton                                  q5                  0.713728    0.713099   6.29e-04  ok
Merton                                  bottom5_mean        0.646239    0.645537   7.03e-04  ok
Merton                                  ce                  1.008790    1.008236   5.54e-04  ok
Merton                                  atom_mass           0.000000    0.000000   0.00e+00  ok
ES (eps=0.1)                            mean                1.015477    1.015161   3.16e-04  ok
ES (eps=0.1)                            std                 0.146944    0.147011   6.67e-05  ok
ES (eps=0.1)                            prob_shortfall      0.247346    0.247907   5.61e-04  ok
ES (eps=0.1)                            exp_shortfall       0.032363    0.032490   1.27e-04  ok
ES (eps=0.1)                            cond_shortfall      0.130843    0.131058   2.15e-04  ok
ES (eps=0.1)                            q5                  0.788154    0.787460   6.94e-04  ok
ES (eps=0.1)                            bottom5_mean        0.713628    0.712852   7.76e-04  ok
ES (eps=0.1)                            ce                  0.985675    0.985302   3.73e-04  ok
ES (eps=0.1)                            atom_mass           0.478075    0.478446   3.71e-04  ok
VaR (alpha=0.1)                         mean                1.070300    1.069808   4.92e-04  ok
VaR (alpha=0.1)                         std                 0.223796    0.223871   7.43e-05  ok
VaR (alpha=0.1)                         prob_shortfall      0.099564    0.100000   4.36e-04  ok
VaR (alpha=0.1)                         exp_shortfall       0.035924    0.036109   1.85e-04  ok
VaR (alpha=0.1)                         cond_shortfall      0.360813    0.361089   2.75e-04  ok
VaR (alpha=0.1)                         q5                  0.653922    0.653346   5.76e-04  ok
VaR (alpha=0.1)                         bottom5_mean        0.592088    0.591445   6.44e-04  ok
VaR (alpha=0.1)                         ce                  0.992773    0.992132   6.41e-04  ok
VaR (alpha=0.1)                         atom_mass           0.428345    0.428688   3.43e-04  ok
VaR equal-CE (alpha=0.08118)            mean                1.057647    1.057157   4.90e-04  ok
VaR equal-CE (alpha=0.08118)            std                 0.207208    0.207346   1.38e-04  ok
VaR equal-CE (alpha=0.08118)            prob_shortfall      0.080685    0.081178   4.93e-04  ok
VaR equal-CE (alpha=0.08118)            exp_shortfall       0.031873    0.032084   2.11e-04  ok
VaR equal-CE (alpha=0.08118)            cond_shortfall      0.395035    0.395236   2.02e-04  ok
VaR equal-CE (alpha=0.08118)            q5                  0.634191    0.633632   5.59e-04  ok
VaR equal-CE (alpha=0.08118)            bottom5_mean        0.574223    0.573598   6.24e-04  ok
VaR equal-CE (alpha=0.08118)            ce                  0.985996    0.985302   6.94e-04  ok
VaR equal-CE (alpha=0.08118)            atom_mass           0.496474    0.496341   1.33e-04  ok
VaR threshold-matched (alpha=0.10666)   mean                1.073845    1.073379   4.66e-04  ok
VaR threshold-matched (alpha=0.10666)   std                 0.228552    0.228585   3.29e-05  ok
VaR threshold-matched (alpha=0.10666)   prob_shortfall      0.106344    0.106663   3.19e-04  ok
VaR threshold-matched (alpha=0.10666)   exp_shortfall       0.037262    0.037413   1.50e-04  ok
VaR threshold-matched (alpha=0.10666)   cond_shortfall      0.350395    0.350757   3.62e-04  ok
VaR threshold-matched (alpha=0.10666)   q5                  0.659382    0.658801   5.81e-04  ok
VaR threshold-matched (alpha=0.10666)   bottom5_mean        0.597032    0.596382   6.49e-04  ok
VaR threshold-matched (alpha=0.10666)   ce                  0.994554    0.993970   5.84e-04  ok
VaR threshold-matched (alpha=0.10666)   atom_mass           0.408258    0.408668   4.10e-04  ok
```
