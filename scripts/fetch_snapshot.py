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
import math
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage.db import init_db, insert_chain, load_chain, save_calibration_cache

DB_PATH = Path("data/spy_chain.db")
CACHE_PATH = Path("data/calibration_cache.json")
UNDERLYING = "SPY"
EXPIRY_WINDOW_DAYS = 90
STRIKE_BAND = 0.25
DEFAULT_R = 0.05


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


def _get_spot() -> float:
    import yfinance as yf
    return float(yf.Ticker(UNDERLYING).fast_info["last_price"])


def _make_serializable(obj):
    """Recursively convert numpy types and NaN to JSON-safe equivalents."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_make_serializable(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) else f
    if isinstance(obj, bool):
        return obj
    return obj


def build_calibration_cache(spot: float, r: float = DEFAULT_R) -> None:
    from vol_surface.iv_inversion import implied_vol
    from vol_surface.svi import fit_slice, svi_total_var
    from vol_surface.ssvi import fit_ssvi
    from vol_surface.heston_calibration import calibrate_heston
    from vol_surface.sanos import fit_sanos_surface

    print(f"\nBuilding calibration cache (spot={spot:.2f}, r={r})...")
    df = load_chain(DB_PATH)

    ivs_by_expiry: dict[str, list] = {}
    svi_fits: dict[str, dict] = {}

    for expiry in sorted(df["expiration"].unique()):
        T = max((date.fromisoformat(expiry) - date.today()).days / 365.0, 1 / 365)
        slice_df = df[df["expiration"] == expiry].copy()

        ivs = []
        for _, row in slice_df.iterrows():
            try:
                iv = implied_vol(spot, row["strike"], T, r, row["mid_price"], row["option_type"])
                ivs.append(iv)
            except ValueError:
                ivs.append(float("nan"))

        slice_df["iv"] = ivs
        slice_df["T"] = T
        slice_df["log_moneyness"] = np.log(slice_df["strike"] / (spot * math.exp(r * T)))
        slice_df["total_var"] = slice_df["iv"] ** 2 * T
        slice_df = slice_df.dropna(subset=["iv"])
        slice_df = slice_df[slice_df["iv"].between(0.01, 1.5)]

        if len(slice_df) < 5:
            print(f"  {expiry}: skipped (fewer than 5 valid IVs)")
            continue

        ivs_by_expiry[expiry] = slice_df.to_dict(orient="records")

        try:
            params = fit_slice(slice_df["log_moneyness"].values, slice_df["total_var"].values)
            svi_fits[expiry] = params
            print(f"  SVI  {expiry}: OK")
        except Exception as e:
            print(f"  SVI  {expiry}: failed -- {e}")

    # SSVI global fit
    ssvi_result = None
    if len(svi_fits) >= 2:
        slices, thetas = [], {}
        for expiry, params in svi_fits.items():
            rows = ivs_by_expiry[expiry]
            theta = float(svi_total_var(np.array([0.0]), params)[0])
            thetas[expiry] = theta
            slices.append({
                "log_moneyness": np.array([row["log_moneyness"] for row in rows]),
                "total_var":     np.array([row["total_var"]     for row in rows]),
                "theta":         theta,
            })
        try:
            ssvi_result = fit_ssvi(slices)
            ssvi_result["thetas"] = thetas
            print(f"  SSVI global: OK (RMSE={ssvi_result['rmse']:.6f})")
        except Exception as e:
            print(f"  SSVI global: failed -- {e}")

    # Heston calibration
    heston_result = None
    heston_smiles: list[dict] = []
    if ivs_by_expiry:
        all_rows = [row for rows in ivs_by_expiry.values() for row in rows]
        full_df = pd.DataFrame(all_rows)
        try:
            print("  Heston calibration (~30-40s)...")
            heston_result = calibrate_heston(full_df, S=spot, r=r)
            print(f"  Heston: OK (RMSE={heston_result['rmse']:.6f})")
        except Exception as e:
            print(f"  Heston: failed -- {e}")

    # Pre-compute Heston smile curves per expiry
    if heston_result:
        from vol_surface.iv_inversion import implied_vol as _iv
        from pricing.heston_cf import heston_price as _heston_price
        print("  Pre-computing Heston smiles...")
        for expiry, rows in ivs_by_expiry.items():
            T = max((date.fromisoformat(expiry) - date.today()).days / 365.0, 1 / 365)
            F = spot * math.exp(r * T)
            exp_df = pd.DataFrame(rows)
            k_lo = float(exp_df["log_moneyness"].min())
            k_hi = float(exp_df["log_moneyness"].max())
            k_grid = np.linspace(k_lo, k_hi, 200)
            strikes_grid = F * np.exp(k_grid)
            smile_strikes, smile_ivs = [], []
            for K in strikes_grid:
                try:
                    price = _heston_price(
                        spot, K, T, r,
                        heston_result["v0"], heston_result["kappa"],
                        heston_result["theta"], heston_result["xi"],
                        heston_result["rho"], option_type="call",
                    )
                    iv = _iv(spot, K, T, r, price, "call")
                    if 0.01 <= iv <= 1.5:
                        smile_strikes.append(float(K))
                        smile_ivs.append(float(iv))
                except Exception:
                    pass
            if smile_strikes:
                heston_smiles.append({
                    "expiry":  expiry,
                    "strikes": smile_strikes,
                    "ivs":     smile_ivs,
                })
                print(f"  Heston smile {expiry}: {len(smile_strikes)} points")

    # SANOS per-expiry fit
    sanos_results: list[dict] = []
    if ivs_by_expiry:
        sanos_slices = []
        for expiry, rows in ivs_by_expiry.items():
            sl_df = pd.DataFrame(rows)
            T = max((date.fromisoformat(expiry) - date.today()).days / 365.0, 1 / 365)
            sanos_slices.append({
                "expiry":       expiry,
                "T":            T,
                "strikes":      sl_df["strike"].values,
                "mids":         sl_df["mid_price"].values,
                "option_types": sl_df["option_type"].values,
                "bids":         sl_df["bid"].values if "bid" in sl_df.columns else None,
                "asks":         sl_df["ask"].values if "ask" in sl_df.columns else None,
            })
        try:
            print("  SANOS calibration...")
            raw = fit_sanos_surface(sanos_slices, S=spot, r=r, spread_weighted=False)
            for res in raw:
                sanos_results.append({
                    "expiry":       res["expiry"],
                    "T":            res["T"],
                    "pure_strikes": res["pure_strikes"].tolist(),
                    "fitted_vols":  [
                        v if np.isfinite(v) else None
                        for v in res["fitted_vols"].tolist()
                    ],
                    "market_vols":  [
                        v if np.isfinite(v) else None
                        for v in res["market_vols"].tolist()
                    ],
                    "atm_vol":      res["atm_vol"],
                    "atm_call":     res["atm_call"],
                    "rmse":         res["rmse"] if np.isfinite(res["rmse"]) else None,
                    "status":       res["status"],
                })
                print(f"  SANOS {res['expiry']}: OK (RMSE={res['rmse']:.4f})")
        except Exception as e:
            print(f"  SANOS: failed -- {e}")

    cache = _make_serializable({
        "generated_at":  datetime.utcnow().isoformat(),
        "spot":          spot,
        "r":             r,
        "svi_fits":      svi_fits,
        "ssvi":          ssvi_result,
        "heston":        heston_result,
        "heston_smiles": heston_smiles,
        "sanos":         sanos_results,
        "ivs":           ivs_by_expiry,
    })
    save_calibration_cache(CACHE_PATH, cache)
    print(f"Calibration cache saved -> {CACHE_PATH}")


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

    spot = _get_spot()
    build_calibration_cache(spot)


if __name__ == "__main__":
    main()
