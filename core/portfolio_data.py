# core/portfolio_data.py

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252


# =============================
# 🧩 Helper dati prezzi
# =============================
def ensure_dataframe(prices):
    """Se è una Series la converte in DataFrame, altrimenti la restituisce com'è."""
    if isinstance(prices, pd.Series):
        return prices.to_frame(name="TICKER")
    return prices


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Pulisce e normalizza i prezzi:
      - ordina per data
      - ffill/bfill
      - rimuove colonne costanti
      - normalizza indice a date giornaliere senza tz
      - rimuove duplicati.
    """
    prices = prices.sort_index().ffill().bfill().dropna(how="any")
    nunique = prices.nunique()
    prices = prices.loc[:, nunique > 1]

    idx = pd.to_datetime(prices.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    try:
        idx = idx.tz_localize(None)
    except Exception:
        pass
    prices.index = idx.normalize()
    prices = prices[~prices.index.duplicated(keep="last")]
    return prices


def extract_close(df: pd.DataFrame, tickers_requested):
    """
    Estrae la colonna/colonne 'Close' da un DataFrame yfinance
    gestendo il caso MultiIndex.
    """
    if df is None or len(df) == 0:
        raise RuntimeError("Nessun dato scaricato.")

    if isinstance(df.columns, pd.MultiIndex):
        # yfinance può dare livelli [campo, ticker] o [ticker, campo]
        for lvl in [0, 1]:
            try:
                if "Close" in df.columns.get_level_values(lvl):
                    close = df.xs("Close", level=lvl, axis=1).copy()
                    break
            except Exception:
                pass
        else:
            raise KeyError("Colonna 'Close' mancante nei dati.")
    else:
        if "Close" in df.columns:
            close = df[["Close"]].copy()
            colname = tickers_requested[0] if len(tickers_requested) == 1 else "CLOSE"
            close.columns = [colname]
        else:
            raise KeyError("Colonna 'Close' mancante nei dati.")

    close = ensure_dataframe(close)
    return clean_prices(close)


def pct_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Rendimenti percentuali semplici a partire dai prezzi."""
    return (
        prices.pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="all")
    )


def log_returns(close: pd.Series) -> pd.Series:
    """Rendimenti logaritmici da una serie di prezzi di chiusura."""
    s = (
        pd.Series(close)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    return np.log(s).diff().replace([np.inf, -np.inf], np.nan).dropna()


def portfolio_value_from_prices(
    prices: pd.DataFrame,
    weights: np.ndarray,
    initial: float = 100_000.0,
) -> pd.Series:
    """
    Costruisce l'equity line del portafoglio:
    - prezzi normalizzati a 1 al primo giorno
    - moltiplicati per i pesi
    - scalati per il capitale iniziale.
    """
    return (prices / prices.iloc[0]).dot(weights) * float(initial)


# =============================
# 🚚 Data Loader multiplo
# =============================
class DataLoader:
    """
    Wrapper semplice intorno a yfinance:
    scarica i prezzi per una lista di ticker e restituisce solo i Close puliti.
    """

    def __init__(self, tickers, start, end):
        self.tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
        self.start = start
        self.end = end

    def fetch(self):
        if len(self.tickers) == 0:
            raise ValueError("Nessun ticker valido.")

        data = yf.download(
            self.tickers,
            start=self.start,
            end=self.end,
            auto_adjust=True,
            progress=False,
            group_by="column",
        )

        close = extract_close(data, self.tickers)
        valid = list(close.columns)
        missing = sorted(set(self.tickers) - set(valid))
        return close, valid, missing


# =============================
# 📈 OHLCV singolo ticker
# =============================
def load_ohlcv_from_yf(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Carica OHLCV per un singolo ticker e restituisce un DataFrame con colonne:
      ['open','high','low','close','volume'] (lowercase), indice giornaliero pulito.

    Gestisce anche il caso MultiIndex di yfinance appiattendo le colonne.
    """
    t = str(ticker).strip()
    if not t:
        return pd.DataFrame()

    df = yf.download(t, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # Appiattisci eventuale MultiIndex -> stringhe lowercase
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for tup in df.columns:
            parts = [
                str(x).strip().lower().replace(" ", "_")
                for x in tup
                if x is not None and str(x).strip() != ""
            ]
            new_cols.append("_".join(parts))
        df.columns = new_cols
    else:
        df.columns = [
            str(c).strip().lower().replace(" ", "_") for c in df.columns
        ]

    # Trova le colonne richieste (fallback su adj_close per close)
    def _pick(name: str) -> str | None:
        for c in df.columns:
            if c == name:
                return c
        for c in df.columns:
            if (
                c.startswith(name + "_")
                or c.endswith("_" + name)
                or f"_{name}_" in c
                or name in c
            ):
                return c
        return None

    open_col = _pick("open")
    high_col = _pick("high")
    low_col = _pick("low")
    close_col = _pick("close") or _pick("adj_close")
    volume_col = _pick("volume")

    needed = dict(
        open=open_col,
        high=high_col,
        low=low_col,
        close=close_col,
        volume=volume_col,
    )
    if any(v is None for v in needed.values()):
        return pd.DataFrame()

    out = df[
        [
            needed["open"],
            needed["high"],
            needed["low"],
            needed["close"],
            needed["volume"],
        ]
    ].copy()
    out.columns = ["open", "high", "low", "close", "volume"]
    out = out.astype(float).sort_index()

    # pulizia indice
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    try:
        idx = idx.tz_localize(None)
    except Exception:
        pass
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")]

    return out.dropna(how="any")
