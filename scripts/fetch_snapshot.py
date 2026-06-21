"""
One-time script to fetch the SPY option chain snapshot and write it to SQLite.

Run once from the project root:
    python scripts/fetch_snapshot.py

This takes ~15-20 minutes due to the 5 calls/minute free-tier rate limit.
The dashboard reads only from the resulting SQLite file — it never calls the API live.
"""
import sys
from pathlib import Path
from datetime import date, timedelta

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.massive_client import MassiveClient
from data.chain_builder import build_chain
from storage.db import init_db, insert_chain

DB_PATH = Path("data/spy_chain.db")
UNDERLYING = "SPY"
EXPIRY_WINDOW_DAYS = 90   # look ahead window for expiries
STRIKE_BAND = 0.25        # ±25% around spot


def main():
    client = MassiveClient()

    print("Fetching SPY spot price...")
    spot = client.get_spot(UNDERLYING)
    print(f"  SPY spot: ${spot:.2f}")

    today = date.today()
    expiry_gte = today.isoformat()
    expiry_lte = (today + timedelta(days=EXPIRY_WINDOW_DAYS)).isoformat()

    print(f"\nBuilding option chain (this will take ~15-20 minutes)...")
    records = build_chain(
        client=client,
        underlying=UNDERLYING,
        spot=spot,
        expiry_gte=expiry_gte,
        expiry_lte=expiry_lte,
        strike_band=STRIKE_BAND,
        verbose=True,
    )

    if not records:
        print("No records returned. Check the date (markets may have been closed).")
        sys.exit(1)

    print(f"\nWriting {len(records)} records to {DB_PATH}...")
    init_db(DB_PATH)
    insert_chain(DB_PATH, records)
    print("Done.")


if __name__ == "__main__":
    main()
