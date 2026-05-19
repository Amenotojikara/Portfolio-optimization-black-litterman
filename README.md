# Sector rotation under Mean-Variance and Black-Litterman

Coursework project, MSc Financial Mathematics.

This repository contains an empirical study of the Black-Litterman (BL) model
against plain mean-variance optimisation (MVO) on the eleven SPDR sector ETFs
that together partition the S&P 500. The strategies are estimated on a 60-month
rolling window from January 2015 through April 2025, rebalanced monthly, and
compared by their out-of-sample Sharpe ratio. SPY, the equal-risk-contribution
("risk parity") portfolio and the global minimum-variance portfolio are
included as benchmarks.

The main result is what the simulation literature (Best & Grauer 1991, Chopra
& Ziemba 1993) predicts: anchoring expected returns to a market-implied
equilibrium prior and updating Bayesian-style with views produces a meaningful
improvement in out-of-sample Sharpe relative to plain MVO. Over the 62-month
test window I find Sharpe ratios of 0.60 (BL with momentum views), 0.52 (BL
with three fixed analyst views), 0.48 (SPY), 0.45 (minimum variance), 0.43
(risk parity), and 0.25 (naive Markowitz). The ranking is robust to linear
transaction costs of up to 20 bp per dollar traded.

## Repository contents

```
portfolio_optimization.ipynb   the main analysis
fetch_prices.py                one-off utility to refresh data/prices.csv from Stooq
data/prices.csv                price panel (Jan 2015 - Apr 2025)
figures/                       output figures referenced by the notebook
requirements.txt               minimal pip dependencies
```

## How to run

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python fetch_prices.py            # optional: refresh data/prices.csv
jupyter notebook portfolio_optimization.ipynb
```

The notebook does no network I/O. It reads `data/prices.csv`, which ships
with the repository as a snapshot. `fetch_prices.py` rebuilds that file from
Stooq (https://stooq.com), a free source that requires no API key. Note that
Stooq does not provide dividend-adjusted prices; the absolute Sharpe ratios
are therefore slightly biased downward by the sector dividend yield, but the
relative ranking of strategies is unaffected.

## Notes and caveats

The three "fixed analyst views" are chosen with hindsight on the 2015-2025
window. I am explicit about this in §5 of the notebook. A more honest exercise
would lock the views at the start of the sample using ex-ante reasoning, or
run leave-one-out cross-validation over a wider set of candidate views.

XLC (Communication Services) was created by the 2018 GICS reclassification
and has less than 60 months of history until mid-2023. The pandas covariance
estimator uses pairwise-available data; the alternative is to backfill with
the predecessor Telecom Services index, which I left as a possible extension.

Sharpe-ratio differences over ten years of monthly data have wide confidence
intervals. With a stationary bootstrap over annual blocks I would expect the
two BL strategies to be statistically separable from MVO but not from each
other, and the three model-free benchmarks (MinVar, RiskParity, SPY) to be
indistinguishable.

## References

Black, F. and Litterman, R. (1992). "Global portfolio optimization."
*Financial Analysts Journal* 48(5), 28-43.

He, G. and Litterman, R. (1999). "The intuition behind Black-Litterman model
portfolios." Goldman Sachs Investment Management Research.

Markowitz, H. (1952). "Portfolio selection." *Journal of Finance* 7(1), 77-91.

Meucci, A. (2010). "The Black-Litterman approach: original model and
extensions." In *Encyclopedia of Quantitative Finance*. Wiley.

Michaud, R. O. (1989). "The Markowitz optimization enigma: is 'optimized'
optimal?" *Financial Analysts Journal* 45(1), 31-42.

A full reference list is given at the end of the notebook.
