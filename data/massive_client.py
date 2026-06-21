"""Thin wrapper around the Polygon/Massive REST API with rate-limit throttling."""
import time
import os
import requests
from dotenv import load_dotenv

load_dotenv()

_BASE_URL = "https://api.polygon.io"
# Free tier: 5 calls/minute → minimum 12 s between calls
_MIN_INTERVAL = 12.0


class MassiveClient:
    def __init__(self, api_key: str | None = None, calls_per_minute: int = 5):
        self._api_key = api_key or os.environ["MASSIVE_API_KEY"]
        self._min_interval = 60.0 / calls_per_minute
        self._last_call: float = 0.0

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_spot(self, ticker: str) -> float:
        """Return previous-day close price for an equity ticker."""
        data = self._get(f"/v2/aggs/ticker/{ticker}/prev")
        return float(data["results"][0]["c"])

    def get_option_contracts(
        self,
        underlying: str,
        expiration_date_gte: str,
        expiration_date_lte: str,
        strike_price_gte: float,
        strike_price_lte: float,
        limit: int = 250,
    ) -> list[dict]:
        """Paginate through reference contracts and return all matching records."""
        contracts = []
        params = {
            "underlying_ticker": underlying,
            "expiration_date.gte": expiration_date_gte,
            "expiration_date.lte": expiration_date_lte,
            "strike_price.gte": strike_price_gte,
            "strike_price.lte": strike_price_lte,
            "limit": limit,
            "sort": "ticker",
            "order": "asc",
        }

        while True:
            data = self._get("/v3/reference/options/contracts", params)
            contracts.extend(data.get("results", []))
            cursor = data.get("next_url")
            if not cursor:
                break
            # next_url contains the full URL; extract cursor param for next call
            params = {"cursor": cursor.split("cursor=")[-1]}

        return contracts

    def get_prev_agg(self, options_ticker: str) -> dict | None:
        """Return previous-day OHLCV for one option contract, or None if no trades."""
        data = self._get(f"/v2/aggs/ticker/{options_ticker}/prev")
        results = data.get("results", [])
        if not results:
            return None
        return results[0]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> dict:
        self._throttle()
        merged = {"apiKey": self._api_key, **(params or {})}
        response = requests.get(_BASE_URL + path, params=merged, timeout=30)
        response.raise_for_status()
        return response.json()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()
