"""
Builds the SPY option chain snapshot from the Massive/Polygon API.

Flow:
  1. Pull all reference contracts within the expiry window and strike band.
  2. For each contract, pull the EOD aggregate.
  3. Drop contracts with no qualifying trade (empty aggregate).
  4. Return a list of clean records ready for SQLite insertion.
"""
import sys
from data.massive_client import MassiveClient


def build_chain(
    client: MassiveClient,
    underlying: str,
    spot: float,
    expiry_gte: str,
    expiry_lte: str,
    strike_band: float = 0.25,
    verbose: bool = True,
) -> list[dict]:
    """Fetch and assemble the option chain.

    Parameters
    ----------
    client       : authenticated MassiveClient
    underlying   : equity ticker, e.g. 'SPY'
    spot         : current spot price used to compute the strike band
    expiry_gte   : earliest expiry date to include (YYYY-MM-DD)
    expiry_lte   : latest expiry date to include (YYYY-MM-DD)
    strike_band  : fraction of spot defining the ±band around spot
    verbose      : print progress to stdout

    Returns
    -------
    List of dicts with keys: ticker, underlying, strike, expiration,
    option_type, close_price
    """
    lo = round(spot * (1 - strike_band), 2)
    hi = round(spot * (1 + strike_band), 2)

    if verbose:
        print(f"Fetching reference contracts: {underlying}, expiry [{expiry_gte} – {expiry_lte}], strike [{lo} – {hi}]")

    contracts = client.get_option_contracts(
        underlying=underlying,
        expiration_date_gte=expiry_gte,
        expiration_date_lte=expiry_lte,
        strike_price_gte=lo,
        strike_price_lte=hi,
    )

    if verbose:
        print(f"  {len(contracts)} contracts found. Fetching EOD aggregates...")

    records = []
    for i, contract in enumerate(contracts, 1):
        ticker = contract["ticker"]
        agg = client.get_prev_agg(ticker)

        if agg is None:
            if verbose:
                print(f"  [{i}/{len(contracts)}] {ticker} — no trades, skipping")
            continue

        records.append({
            "ticker": ticker,
            "underlying": underlying,
            "strike": float(contract["strike_price"]),
            "expiration": contract["expiration_date"],
            "option_type": contract["contract_type"],  # 'call' or 'put'
            "close_price": float(agg["c"]),
        })

        if verbose:
            print(f"  [{i}/{len(contracts)}] {ticker}  close={agg['c']}")

    if verbose:
        print(f"\nDone. {len(records)} contracts with qualifying trades.")

    return records
