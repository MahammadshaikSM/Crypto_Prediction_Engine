"""CoinMarketCap client shared by prediction_tracker.py and site/build_forecast.py.

API key: environment variable CMC_API_KEY (GitHub secret in the daily Action),
or a git-ignored .cmc_key file next to this script for local runs.

Coins are returned in the shape the tracker already uses:
  id (CMC slug), cmc_id, symbol (lower case), name, image, current_price,
  price_change_percentage_24h, price_change_percentage_7d_in_currency,
  total_volume, market_cap, tags, daily_closes (oldest -> newest, today last)

Credit use per run (CMC "1 credit per 100 records"):
  listings/latest limit=250            ~3 credits
  quotes/historical, 16 days x ~250    ~40 credits
"""
import os, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests

BASE = "https://pro-api.coinmarketcap.com"
KEY_FILE = Path(__file__).resolve().parent / ".cmc_key"
LOGO = "https://s2.coinmarketcap.com/static/img/coins/64x64/{}.png"


def api_key():
    key = os.environ.get("CMC_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
    if not key:
        raise RuntimeError("No CoinMarketCap key: set CMC_API_KEY or create .cmc_key")
    return key


def _get(path, params, retries=4):
    headers = {"X-CMC_PRO_API_KEY": api_key(), "Accept": "application/json"}
    for attempt in range(retries):
        try:
            r = requests.get(BASE + path, params=params, headers=headers, timeout=60)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 60)))
                continue
            body = r.json()
            status = body.get("status", {})
            if r.status_code != 200 or status.get("error_code"):
                raise RuntimeError(f"CMC {path} {r.status_code}: {status.get('error_message')}")
            return body["data"]
        except (requests.RequestException, ValueError) as e:
            if attempt == retries - 1:
                raise RuntimeError(f"CMC {path} failed: {e}") from e
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"CMC {path} rate-limited {retries} times")


def listings(limit=250):
    """Latest market data for the top `limit` coins by market cap."""
    out = []
    for c in _get("/v1/cryptocurrency/listings/latest", {"limit": limit, "convert": "USD"}):
        q = c["quote"]["USD"]
        out.append({
            "id": c["slug"], "cmc_id": c["id"], "symbol": c["symbol"].lower(), "name": c["name"],
            "image": LOGO.format(c["id"]), "current_price": q.get("price"),
            "price_change_percentage_24h": q.get("percent_change_24h"),
            "price_change_percentage_7d_in_currency": q.get("percent_change_7d"),
            "total_volume": q.get("volume_24h"), "market_cap": q.get("market_cap"),
            "tags": c.get("tags") or [],
        })
    return out


def daily_closes(cmc_ids, days=16, batch=100):
    """{cmc_id: [daily close, ...]} for the last `days` days (UTC midnight snapshots)."""
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    out = {}
    ids = [str(i) for i in cmc_ids]
    for i in range(0, len(ids), batch):
        data = _get("/v2/cryptocurrency/quotes/historical", {
            "id": ",".join(ids[i:i + batch]), "interval": "daily", "convert": "USD",
            "time_start": start.isoformat(), "time_end": end.isoformat()})
        for cid, obj in (data or {}).items():
            obj = obj[0] if isinstance(obj, list) and obj else obj
            quotes = (obj or {}).get("quotes") or []
            prices = [q["quote"]["USD"]["price"] for q in sorted(quotes, key=lambda q: q["timestamp"])
                      if q.get("quote", {}).get("USD", {}).get("price")]
            out[int(cid)] = prices
        time.sleep(2.1)  # stay under the per-minute rate limit
    return out


def prices_by_symbol(wanted):
    """Current USD price for coins outside the top listing, matched by symbol then name.

    wanted: list of (symbol, name). Returns {symbol_lower: price}. Costs ~1 credit.
    When several coins share a symbol, the one whose name matches wins, else the largest.
    """
    if not wanted:
        return {}
    data = _get("/v2/cryptocurrency/quotes/latest",
                {"symbol": ",".join(sorted({s.upper() for s, _ in wanted})), "convert": "USD", "skip_invalid": "true"})
    out = {}
    for sym, name in wanted:
        cands = data.get(sym.upper()) or []
        cands = cands if isinstance(cands, list) else [cands]
        cands = [c for c in cands if (c.get("quote") or {}).get("USD", {}).get("price")]
        if not cands:
            continue
        named = [c for c in cands if c.get("name", "").lower() == (name or "").lower()]
        best = (named or sorted(cands, key=lambda c: -(c["quote"]["USD"].get("market_cap") or 0)))[0]
        out[sym.lower()] = best["quote"]["USD"]["price"]
    return out


def fetch_market(limit=250, days=16):
    """Listings plus daily closes; today's live price is appended as the latest close."""
    coins = listings(limit)
    hist = daily_closes([c["cmc_id"] for c in coins], days=days)
    for c in coins:
        closes = list(hist.get(c["cmc_id"], []))
        if c["current_price"]:
            closes.append(c["current_price"])
        c["daily_closes"] = closes
    return coins
