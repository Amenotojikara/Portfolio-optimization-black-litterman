"""
Portfolio Optimization: Mean-Variance vs Black-Litterman
========================================================
End-to-end analysis pipeline. This script (a) generates a realistic synthetic
price series for the 11 SPDR sector ETFs + SPY when no internet is available,
and (b) runs the full rolling-window backtest comparing MVO, Minimum Variance,
Black-Litterman (with both momentum and hardcoded views), Risk Parity, and SPY.

Outputs all the figures used in the accompanying Jupyter notebook.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
DATA_DIR = HERE / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_ETFS = {
    "XLK":  "Technology",
    "XLV":  "Health Care",
    "XLF":  "Financials",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLE":  "Energy",
    "XLI":  "Industrials",
    "XLB":  "Materials",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "XLC":  "Communication Services",
}
TICKERS = list(SECTOR_ETFS.keys())
BENCHMARK = "SPY"

START = "2015-01-01"
END   = "2025-04-30"
TRAIN_YEARS = 5
REBAL_FREQ  = "M"   # monthly rebalance
RF_ANNUAL   = 0.02  # 2 % annual risk-free for Sharpe

# Plotting style ------------------------------------------------------------
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "DejaVu Sans",
})
PALETTE = sns.color_palette("tab20", n_colors=len(TICKERS))


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
def _synthesise_prices(seed: int = 7) -> pd.DataFrame:
    """Generate a realistic synthetic price panel for the sector ETFs.

    Uses a single-factor model:
        r_i = alpha_i + beta_i * r_market + eps_i
    with sector-specific alphas, betas, and idiosyncratic vols chosen to
    roughly match the 2015–2024 empirical statistics of each SPDR sector.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(START, END)
    n = len(dates)

    # Market: ~ 10% drift, 16% annual vol with mild fat tails
    mu_m = 0.10 / 252
    sig_m = 0.16 / np.sqrt(252)
    rm = rng.standard_normal(n) * sig_m + mu_m
    # Add a few volatility spikes (COVID, 2022 bear)
    spike_dates = [pd.Timestamp("2020-03-15"), pd.Timestamp("2022-06-15")]
    for d in spike_dates:
        try:
            i = dates.get_indexer([d], method="nearest")[0]
            rm[max(0, i - 20):i + 20] *= 1.8
        except Exception:
            pass

    # Per-sector parameters (alpha annual, beta, idio vol annual)
    params = {
        "XLK":  (0.04,  1.15, 0.11),
        "XLV":  (0.00,  0.85, 0.10),
        "XLF":  (-0.01, 1.15, 0.12),
        "XLY":  (0.01,  1.15, 0.12),
        "XLP":  (0.00,  0.65, 0.08),
        "XLE":  (-0.05, 1.20, 0.22),
        "XLI":  (0.00,  1.10, 0.10),
        "XLB":  (-0.01, 1.10, 0.12),
        "XLU":  (0.00,  0.55, 0.10),
        "XLRE": (-0.02, 0.85, 0.13),
        "XLC":  (0.00,  1.05, 0.13),
    }
    returns = {}
    for t, (alpha_a, beta, idio_a) in params.items():
        alpha = alpha_a / 252
        idio = rng.standard_normal(n) * (idio_a / np.sqrt(252))
        returns[t] = alpha + beta * rm + idio
    returns[BENCHMARK] = rm + 0.00 / 252

    rdf = pd.DataFrame(returns, index=dates)
    prices = 100 * (1 + rdf).cumprod()
    prices.index.name = "Date"
    return prices


def load_prices(use_yfinance: bool = True) -> pd.DataFrame:
    """Load adjusted close prices.

    On a user's machine ``use_yfinance=True`` will pull real data.  In the
    sandbox we fall back to the synthetic panel.
    """
    if use_yfinance:
        try:
            import yfinance as yf
            df = yf.download(
                TICKERS + [BENCHMARK],
                start=START,
                end=END,
                auto_adjust=True,
                progress=False,
            )["Close"].dropna(how="all")
            if df.shape[0] > 100:
                return df[TICKERS + [BENCHMARK]]
        except Exception as e:
            print(f"yfinance unavailable ({e}); using synthetic panel.")
    return _synthesise_prices()


# ---------------------------------------------------------------------------
# 2. Optimisers
# ---------------------------------------------------------------------------
def _project_to_simplex(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0, None)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


def max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> np.ndarray:
    """Long-only max-Sharpe portfolio (Markowitz)."""
    n = len(mu)
    inv = np.linalg.pinv(cov + 1e-8 * np.eye(n))
    w = inv @ (mu - rf)
    w = _project_to_simplex(w)

    def neg_sharpe(w):
        ret = w @ mu - rf
        vol = np.sqrt(w @ cov @ w) + 1e-12
        return -ret / vol

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [(0.0, 0.5)] * n
    res = minimize(neg_sharpe, w, bounds=bnds, constraints=cons, method="SLSQP")
    return res.x if res.success else w


def min_variance(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    inv = np.linalg.pinv(cov + 1e-8 * np.eye(n))
    ones = np.ones(n)
    w = inv @ ones
    w = _project_to_simplex(w)

    def var(w): return w @ cov @ w

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [(0.0, 0.5)] * n
    res = minimize(var, w, bounds=bnds, constraints=cons, method="SLSQP")
    return res.x if res.success else w


def risk_parity(cov: np.ndarray) -> np.ndarray:
    """Equal risk contribution portfolio (long-only)."""
    n = cov.shape[0]
    w = np.ones(n) / n

    def obj(w):
        port_var = w @ cov @ w
        rc = w * (cov @ w) / np.sqrt(port_var + 1e-12)
        target = np.sqrt(port_var) / n
        return ((rc - target) ** 2).sum()

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [(1e-4, 1.0)] * n
    res = minimize(obj, w, bounds=bnds, constraints=cons, method="SLSQP")
    return res.x if res.success else w


def black_litterman(
    cov: np.ndarray,
    mkt_caps: np.ndarray,
    P: np.ndarray | None,
    Q: np.ndarray | None,
    tau: float = 0.05,
    delta: float = 2.5,
    omega: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (posterior_mean, posterior_cov) under Black-Litterman."""
    w_mkt = mkt_caps / mkt_caps.sum()
    pi = delta * cov @ w_mkt  # implied equilibrium returns

    if P is None or Q is None or len(Q) == 0:
        return pi, cov

    P = np.atleast_2d(P)
    Q = np.atleast_1d(Q).reshape(-1)
    if omega is None:
        omega = np.diag(np.diag(P @ (tau * cov) @ P.T))

    tau_cov = tau * cov
    A = np.linalg.pinv(tau_cov) + P.T @ np.linalg.pinv(omega) @ P
    b = np.linalg.pinv(tau_cov) @ pi + P.T @ np.linalg.pinv(omega) @ Q
    mu_bl = np.linalg.solve(A, b)
    cov_bl = cov + np.linalg.pinv(A)
    return mu_bl, cov_bl


# ---------------------------------------------------------------------------
# 3. Rolling backtest
# ---------------------------------------------------------------------------
def to_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.resample("M").last().pct_change().dropna()


def momentum_views(rets: pd.DataFrame, lookback: int = 12, top: int = 2, bottom: int = 2):
    """Construct relative views: top N - bottom N performers."""
    m = (1 + rets.tail(lookback)).prod() - 1
    m = m.sort_values()
    losers = m.head(bottom).index.tolist()
    winners = m.tail(top).index.tolist()
    tickers = rets.columns.tolist()
    P, Q = [], []
    for w in winners:
        for L in losers:
            row = np.zeros(len(tickers))
            row[tickers.index(w)] =  1.0
            row[tickers.index(L)] = -1.0
            P.append(row)
            spread = m[w] - m[L]
            # shrink the view spread; convert annualised
            Q.append(0.5 * spread)
    return np.array(P), np.array(Q)


def hardcoded_views(tickers):
    """Three illustrative analyst views (annualised)."""
    idx = {t: i for i, t in enumerate(tickers)}
    P, Q = [], []
    def row(pairs):
        r = np.zeros(len(tickers))
        for tkr, val in pairs.items():
            r[idx[tkr]] = val
        return r
    # 1) Tech will outperform Energy by 4%
    P.append(row({"XLK":  1, "XLE": -1})); Q.append(0.04)
    # 2) Health Care will outperform Financials by 2%
    P.append(row({"XLV":  1, "XLF": -1})); Q.append(0.02)
    # 3) Utilities + Staples (defensive basket) will outperform Real Estate by 1.5%
    P.append(row({"XLU": 0.5, "XLP": 0.5, "XLRE": -1})); Q.append(0.015)
    return np.array(P), np.array(Q)


def equal_market_caps(n):
    """When real cap data is unavailable, treat market weights as equal."""
    return np.ones(n) / n


def annualise(rets: pd.Series, rf: float = RF_ANNUAL):
    mu  = rets.mean() * 12
    vol = rets.std() * np.sqrt(12)
    sharpe = (mu - rf) / (vol + 1e-12)
    cum = (1 + rets).prod() - 1
    dd  = (1 + rets).cumprod()
    mdd = (dd / dd.cummax() - 1).min()
    return dict(CAGR=(1 + rets.mean()) ** 12 - 1,
                Vol=vol, Sharpe=sharpe, MaxDD=mdd, CumRet=cum)


def run_backtest(prices: pd.DataFrame):
    monthly = to_monthly_returns(prices)
    sector_rets = monthly[TICKERS]
    spy_rets    = monthly[BENCHMARK]

    train_n = TRAIN_YEARS * 12
    months = sector_rets.index
    start_idx = train_n

    portfolios = ["MVO", "MinVar", "BL-Momentum", "BL-Hardcoded", "RiskParity"]
    rets_out  = {p: [] for p in portfolios}
    rets_out["SPY"] = []
    weights_history = {p: [] for p in portfolios}

    for i in range(start_idx, len(months) - 1):
        train = sector_rets.iloc[i - train_n:i]
        nxt   = sector_rets.iloc[i + 1]
        date_next = months[i + 1]

        mu  = train.mean().values * 12
        cov = train.cov().values * 12
        n = len(TICKERS)

        # MVO (max Sharpe on sample moments)
        w_mvo = max_sharpe(mu, cov, rf=RF_ANNUAL)
        # Min Variance
        w_mv  = min_variance(cov)
        # Risk Parity
        w_rp  = risk_parity(cov)
        # Black-Litterman with momentum views
        P_m, Q_m = momentum_views(train)
        mu_blm, cov_blm = black_litterman(cov, equal_market_caps(n), P_m, Q_m)
        w_blm = max_sharpe(mu_blm, cov_blm, rf=RF_ANNUAL)
        # Black-Litterman with hardcoded views
        P_h, Q_h = hardcoded_views(TICKERS)
        mu_blh, cov_blh = black_litterman(cov, equal_market_caps(n), P_h, Q_h)
        w_blh = max_sharpe(mu_blh, cov_blh, rf=RF_ANNUAL)

        for name, w in zip(portfolios, [w_mvo, w_mv, w_blm, w_blh, w_rp]):
            r = float(np.dot(w, nxt.values))
            rets_out[name].append((date_next, r))
            weights_history[name].append((date_next, w.copy()))

        rets_out["SPY"].append((date_next, float(spy_rets.iloc[i + 1])))

    out = {}
    for name, lst in rets_out.items():
        s = pd.Series({d: v for d, v in lst}, name=name)
        out[name] = s
    rets_df = pd.DataFrame(out)
    weights_df = {p: pd.DataFrame({d: w for d, w in lst}, index=TICKERS).T
                  for p, lst in weights_history.items()}
    return rets_df, weights_df


# ---------------------------------------------------------------------------
# 4. Reporting / figures
# ---------------------------------------------------------------------------
def fig_efficient_frontier(prices: pd.DataFrame):
    rets = to_monthly_returns(prices)[TICKERS]
    mu = rets.mean().values * 12
    cov = rets.cov().values * 12
    n = len(TICKERS)

    # Random portfolios for cloud
    rng = np.random.default_rng(42)
    W = rng.dirichlet(np.ones(n), size=4000)
    port_ret = W @ mu
    port_vol = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))

    # Frontier via varying target return
    targets = np.linspace(port_ret.min(), port_ret.max(), 40)
    front_vol, front_ret = [], []
    for tr in targets:
        cons = [{"type": "eq", "fun": lambda w, tr=tr: w @ mu - tr},
                {"type": "eq", "fun": lambda w: w.sum() - 1}]
        bnds = [(0.0, 0.5)] * n
        res = minimize(lambda w: w @ cov @ w, np.ones(n) / n,
                       bounds=bnds, constraints=cons, method="SLSQP")
        if res.success:
            front_vol.append(np.sqrt(res.x @ cov @ res.x))
            front_ret.append(tr)

    w_mvo = max_sharpe(mu, cov, rf=RF_ANNUAL)
    w_mv  = min_variance(cov)
    P_h, Q_h = hardcoded_views(TICKERS)
    mu_bl, cov_bl = black_litterman(cov, equal_market_caps(n), P_h, Q_h)
    w_bl = max_sharpe(mu_bl, cov_bl, rf=RF_ANNUAL)

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(port_vol, port_ret,
                    c=(port_ret - RF_ANNUAL) / port_vol,
                    cmap="viridis", alpha=0.35, s=14)
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
    ax.plot(front_vol, front_ret, "k-", lw=2, label="Efficient Frontier")
    for w, name, color, marker in [
        (w_mvo, "MVO Max-Sharpe", "#d62728", "*"),
        (w_mv,  "Min Variance",   "#1f77b4", "o"),
        (w_bl,  "Black-Litterman", "#2ca02c", "D"),
    ]:
        r = w @ mu; v = np.sqrt(w @ cov @ w)
        ax.scatter([v], [r], marker=marker, s=260, edgecolor="black",
                   color=color, label=name, zorder=5)
    ax.set_xlabel("Annualised Volatility")
    ax.set_ylabel("Annualised Return")
    ax.set_title("Efficient Frontier — 11 SPDR Sector ETFs (in-sample)")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_efficient_frontier.png")
    plt.close(fig)


def fig_correlation_heatmap(prices):
    rets = to_monthly_returns(prices)[TICKERS]
    corr = rets.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.8},
                annot_kws={"fontsize": 8}, ax=ax)
    ax.set_title("Monthly Return Correlation — Sector ETFs")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_correlation_heatmap.png")
    plt.close(fig)


def fig_equity_curves(rets_df: pd.DataFrame):
    eq = (1 + rets_df).cumprod()
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"MVO": "#d62728", "MinVar": "#1f77b4",
              "BL-Momentum": "#2ca02c", "BL-Hardcoded": "#9467bd",
              "RiskParity": "#ff7f0e", "SPY": "black"}
    styles = {"SPY": "--"}
    for col in eq.columns:
        ax.plot(eq.index, eq[col], lw=2 if col != "SPY" else 1.6,
                color=colors.get(col, None),
                ls=styles.get(col, "-"), label=col)
    ax.set_title("Out-of-Sample Growth of $1 (5-yr Rolling Window)")
    ax.set_ylabel("Cumulative Value")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_equity_curves.png")
    plt.close(fig)


def fig_rolling_sharpe(rets_df: pd.DataFrame, window: int = 24):
    rs = (rets_df.rolling(window).mean() * 12 - RF_ANNUAL) / \
         (rets_df.rolling(window).std() * np.sqrt(12))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for col in rs.columns:
        ax.plot(rs.index, rs[col], lw=1.8 if col != "SPY" else 1.4,
                ls="--" if col == "SPY" else "-", label=col)
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_title(f"Rolling {window}-Month Sharpe Ratio (OOS)")
    ax.set_ylabel("Sharpe")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="lower left", ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_rolling_sharpe.png")
    plt.close(fig)


def fig_drawdowns(rets_df: pd.DataFrame):
    eq = (1 + rets_df).cumprod()
    dd = eq / eq.cummax() - 1
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in dd.columns:
        ax.plot(dd.index, dd[col], lw=1.6,
                ls="--" if col == "SPY" else "-", label=col)
    ax.set_title("Drawdowns — Portfolios vs SPY")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", ncol=2, frameon=True)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_drawdowns.png")
    plt.close(fig)


def fig_weights_heatmap(weights_df: dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for ax, name in zip(axes, ["MVO", "MinVar", "BL-Momentum", "RiskParity"]):
        df = weights_df[name]
        # downsample to make heatmap readable: take every other observation
        sns.heatmap(df.T, ax=ax, cmap="Blues", cbar_kws={"shrink": 0.7},
                    vmin=0, vmax=0.5)
        ax.set_title(f"{name} — Allocation Over Time")
        ax.set_xlabel(""); ax.set_ylabel("")
        n = df.shape[0]
        xticks = np.linspace(0, n - 1, 6, dtype=int)
        ax.set_xticks(xticks + 0.5)
        ax.set_xticklabels([df.index[i].strftime("%Y-%m") for i in xticks],
                           rotation=30, ha="right", fontsize=8)
    fig.suptitle("Portfolio Weights Through Time", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_weights_heatmap.png")
    plt.close(fig)


def fig_sharpe_bar(rets_df: pd.DataFrame):
    stats = {c: annualise(rets_df[c]) for c in rets_df.columns}
    sharpe = {c: v["Sharpe"] for c, v in stats.items()}
    s = pd.Series(sharpe).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1f77b4" if i != s.idxmax() else "#d62728" for i in s.index]
    bars = ax.barh(s.index, s.values, color=colors, edgecolor="black")
    for b, v in zip(bars, s.values):
        ax.text(v + 0.02, b.get_y() + b.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=10)
    ax.set_title("Out-of-Sample Sharpe Ratio (Annualised)")
    ax.set_xlabel("Sharpe")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_sharpe_comparison.png")
    plt.close(fig)
    return pd.DataFrame(stats).T


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    prices = load_prices(use_yfinance=False)
    prices.to_csv(DATA_DIR / "prices_synthetic.csv")
    print(f"Prices: {prices.shape[0]} days, {prices.shape[1]} tickers "
          f"({prices.index.min().date()} → {prices.index.max().date()})")

    rets_df, weights_df = run_backtest(prices)
    print(f"OOS months: {len(rets_df)}  range "
          f"{rets_df.index.min().date()} → {rets_df.index.max().date()}")

    fig_efficient_frontier(prices)
    fig_correlation_heatmap(prices)
    fig_equity_curves(rets_df)
    fig_rolling_sharpe(rets_df)
    fig_drawdowns(rets_df)
    fig_weights_heatmap(weights_df)
    summary = fig_sharpe_bar(rets_df)

    rets_df.to_csv(DATA_DIR / "oos_monthly_returns.csv")
    summary.to_csv(DATA_DIR / "performance_summary.csv")
    print("\nPerformance summary (annualised):")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
