"""
One-time script to fetch the SPY option chain snapshot and write it to SQLite.

Usage:
    python scripts/fetch_snapshot.py --source yfinance   # fast (~seconds)
    python scripts/fetch_snapshot.py --source massive    # slow (~15-20 min, rate-limited)

The dashboard reads only from the resulting SQLite file — it never calls the API live.

Price proxy: daily close / last-trade price (not bid-ask mid).
This is a disclosed simplification — Massive free tier has no quotes endpoint
and yfinance lastPrice is the last recorded trade, not a true mid.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage.db import init_db, insert_chain, load_chain

DB_PATH = Path("data/spy_chain.db")
UNDERLYING = "SPY"
EXPIRY_WINDOW_DAYS = 90
STRIKE_BAND = 0.25


def fetch_massive() -> list[dict]:
    from datetime import date, timedelta
    from data.massive_client import MassiveClient
    from data.chain_builder import build_chain

    client = MassiveClient()

    print("Fetching SPY spot price...")
    spot = client.get_spot(UNDERLYING)
    print(f"  SPY spot: ${spot:.2f}")

    today = date.today()
    expiry_gte = today.isoformat()
    expiry_lte = (today + timedelta(days=EXPIRY_WINDOW_DAYS)).isoformat()

    print(f"\nBuilding option chain via Massive API (~15-20 min due to rate limiting)...")
    return build_chain(
        client=client,
        underlying=UNDERLYING,
        spot=spot,
        expiry_gte=expiry_gte,
        expiry_lte=expiry_lte,
        strike_band=STRIKE_BAND,
        verbose=True,
    )


def fetch_yfinance() -> list[dict]:
    from data.yfinance_client import build_chain_yfinance

    print("Building option chain via yfinance...")
    return build_chain_yfinance(
        underlying=UNDERLYING,
        expiry_window_days=EXPIRY_WINDOW_DAYS,
        strike_band=STRIKE_BAND,
        verbose=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Fetch SPY option chain snapshot.")
    parser.add_argument(
        "--source",
        choices=["massive", "yfinance"],
        default="yfinance",
        help="Data source: 'yfinance' (fast, no key) or 'massive' (rate-limited, needs API key)",
    )
    args = parser.parse_args()

    if args.source == "massive":
        records = fetch_massive()
    else:
        records = fetch_yfinance()

    if not records:
        print("No records returned. Check date (markets may have been closed) or connectivity.")
        sys.exit(1)

    print(f"\nWriting {len(records)} records to {DB_PATH}...")
    init_db(DB_PATH)
    insert_chain(DB_PATH, records)

    df = load_chain(DB_PATH)
    print(f"Snapshot saved. {len(df)} rows, {df['expiration'].nunique()} expiries, "
          f"{df['strike'].nunique()} unique strikes.")


if __name__ == "__main__":
    main()
