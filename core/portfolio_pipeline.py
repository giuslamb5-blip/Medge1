# core/portfolio_pipeline.py

import numpy as np
import pandas as pd

from .portfolio_data import DataLoader, portfolio_value_from_prices, pct_returns
from .portfolio_metrics import (
    drawdown_series,
    cagr_from_returns,
    vol_ann,
    ulcer_index_from_equity,
    burke_ratio,
    sterling_ratio,
    kappa_ratio,
    sharpe_ratio,
    sortino_ratio,
    rachev_ratio,
    es_cvar,
    omega_ratio,
    pain_ratio,
)


def compute_pipeline(
    tickers,
    weights=None,
    start: str = "2020-01-01",
    end: str | None = None,
    rf_annual: float = 0.00,
    initial: float = 100_000.0,
    rachev_alpha: float = 0.05,
):
    """
    Pipeline principale:
      - scarica i prezzi
      - costruisce l'equity
      - calcola rendimenti asset/portafoglio
      - calcola le metriche
    """
    prices, valid_tickers, missing = DataLoader(tickers, start, end).fetch()
    if len(valid_tickers) == 0:
        raise RuntimeError(
            "Tutti i ticker forniti sono risultati non validi."
        )

    n = len(valid_tickers)
    if (weights is None) or (len(weights) != n):
        w = np.repeat(1.0 / n, n)
    else:
        w = np.array(weights, dtype=float)
        s = float(w.sum())
        w = np.repeat(1.0 / n, n) if s <= 0 else w / s

    prices = prices[valid_tickers]
    equity = portfolio_value_from_prices(prices, w, initial=initial)

    rets_assets = pct_returns(prices)
    rets_port = (
        equity.pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # =============== Metriche ===============
    cagr = cagr_from_returns(rets_port)
    sigma_ann = vol_ann(rets_port)
    mdd = drawdown_series(equity).min()
    ulcer = ulcer_index_from_equity(equity)
    burke = burke_ratio(equity, cagr)
    sterling = sterling_ratio(equity, cagr)
    calmar = cagr / abs(mdd) if (mdd is not None and mdd != 0) else np.nan
    mar = calmar
    pain = pain_ratio(equity, cagr)
    kappa3 = kappa_ratio(rets_port, mar_annual=rf_annual, order=3)
    sharpe = sharpe_ratio(rets_port, rf_annual=rf_annual)
    sortino = sortino_ratio(rets_port, rf_annual=rf_annual)
    rachev = rachev_ratio(rets_port, alpha=rachev_alpha)
    es95 = es_cvar(rets_port, 0.95)
    es99 = es_cvar(rets_port, 0.99)
    omega0 = omega_ratio(rets_port, threshold=0.0)

    metrics = pd.Series(
        {
            "CAGR": cagr,
            "Volatilità ann.": sigma_ann,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max Drawdown": mdd,
            "Ulcer Index": ulcer,
            "Burke": burke,
            "Kappa(3)": kappa3,
            "Sterling": sterling,
            "Calmar": calmar,
            "MAR": mar,
            "Pain Ratio": pain,
            f"Rachev (α={rachev_alpha:.2f})": rachev,
            "ES/CVaR 95% (giorn.)": es95,
            "ES/CVaR 99% (giorn.)": es99,
            "Omega(τ=0)": omega0,
        }
    )

    return {
        "prices": prices,
        "equity": equity,
        "returns_portfolio": rets_port,
        "returns_assets": rets_assets,
        "weights": pd.Series(w, index=valid_tickers),
        "metrics": metrics,
        "correlation": rets_assets.corr(),
        "missing": missing,
    }
