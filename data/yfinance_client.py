"""
yfinance-based option chain builder for SPY.

Replaces the per-contract Massive/Polygon aggregate calls with a single
yfinance call per expiry date — fetches a full chain in seconds instead of hours.

Price proxy: lastPrice (last trade price), same disclosed simplification as
the Massive approach (close price). Contracts with lastPrice == 0 or NaN
are dropped (no qualifying trades).
"""
from datetime import date, timedelta

import pandas as pd
import yfinance as yf


def build_chain_yfinance(
    underlying: str,
    expiry_window_days: int = 90,
    strike_band: float = 0.25,
    verbose: bool = True,
) -> list[dict]:
    """Fetch the full option chain via yfinance.

    Parameters
    ----------
    underlying         : equity ticker, e.g. 'SPY'
    expiry_window_days : include expiries up to this many days from today
    strike_band        : ±fraction of spot to filter strikes
    verbose            : print progress

    Returns
    -------
    List of dicts with keys: ticker, underlying, strike, expiration,
    option_type, close_price
    """
    ticker = yf.Ticker(underlying)

    spot = ticker.fast_info["last_price"]
    lo = spot * (1 - strike_band)
    hi = spot * (1 + strike_band)
    cutoff = date.today() + timedelta(days=expiry_window_days)

    if verbose:
        print(f"{underlying} spot: ${spot:.2f}")
        print(f"Strike band: [{lo:.2f} – {hi:.2f}]")

    available_expiries = [
        d for d in ticker.options
        if date.fromisoformat(d) <= cutoff
    ]

    if verbose:
        print(f"Expiries within {expiry_window_days} days: {available_expiries}\n")

    records = []
    for expiry in available_expiries:
        if verbose:
            print(f"Fetching {expiry}...", end=" ")

        chain = ticker.option_chain(expiry)

        for option_type, df in [("call", chain.calls), ("put", chain.puts)]:
            filtered = df[
                (df["strike"] >= lo) &
                (df["strike"] <= hi) &
                (df["lastPrice"] > 0) &
                (df["lastPrice"].notna())
            ].copy()

            for _, row in filtered.iterrows():
                records.append({
                    "ticker": row["contractSymbol"],
                    "underlying": underlying,
                    "strike": float(row["strike"]),
                    "expiration": expiry,
                    "option_type": option_type,
                    "close_price": float(row["lastPrice"]),
                })

        if verbose:
            print(f"{len(records)} records so far")

    if verbose:
        print(f"\nDone. {len(records)} contracts with qualifying trades.")

    return records
