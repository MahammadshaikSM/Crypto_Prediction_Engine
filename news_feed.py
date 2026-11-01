"""News, sentiment and catalysts for the stock watchlist (free sources, no API key).

Per stock, once a day:
  headlines   Google News RSS for '"<company>" stock' over the last 7 days (title, source, link, time)
  tags        keyword tags per headline: Deal / M&A, Earnings, Analyst, Legal / regulatory, Product / partner,
              Capital (buyback, offering, split, dividend, IPO)
  tone        a rough headline sentiment from a small finance word list: +1 per positive word, -1 per
              negative word; the stock's 7-day tone is the average over its headlines
  analysts    Yahoo Finance consensus (strong buy .. sell), number of analysts, mean price target and the
              upside to it, rating changes in the last 30 days (upgrades vs downgrades)
  earnings    next earnings date and days until it
  short       short interest as % of float

The tone is a keyword count, not a model: it misses sarcasm, context and headlines about other companies.
None of this feeds the weekly call; it has no backtest. It is context for reading the watchlist.
"""
import re, time, xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import requests

QUERY = {  # what to search for each ticker (company names find more relevant news than tickers)
    "SNDK": "Sandisk", "GOOGL": "Alphabet", "MU": "Micron", "INTC": "Intel", "PLTR": "Palantir",
    "NVDA": "Nvidia", "NFLX": "Netflix", "MRVL": "Marvell", "SKHY": "SK hynix", "HOOD": "Robinhood",
    "CBRS": "Cerebras", "AMZN": "Amazon", "TSM": "TSMC", "QCOM": "Qualcomm", "IBM": "IBM", "DELL": "Dell",
    "MSTR": "MSTR", "MSFT": "Microsoft", "SPCX": "SpaceX", "AMD": "AMD", "TSLA": "Tesla",
    "META": "Meta Platforms", "DRAM": "memory chip stocks",
}
TAGS = {
    "Deal / M&A": r"\b(acquir\w*|acquisition|merger|merges?|merged|buyout|takeover|agree\w* to buy|stake in|spin[- ]?off)\b",
    "Earnings": r"\b(earnings|quarter\w*|q[1-4]\b|results|revenue|eps|guidance|outlook|forecast)\b",
    "Analyst": r"\b(upgrade\w*|downgrade\w*|price target|rating|outperform|underperform|overweight|underweight|analyst\w*)\b",
    "Legal / regulatory": r"\b(probe|lawsuit|sue[sd]?|antitrust|sec\b|investigation|ban\w*|export control\w*|fine[sd]?|regulat\w*|tariff\w*)\b",
    "Product / partner": r"\b(launch\w*|unveil\w*|partner\w*|contract|deal with|agreement|orders?|supply deal)\b",
    "Capital": r"\b(buyback|repurchase|offering|split|dividend|ipo|debt|bond\w*|raise\w* \$)\b",
}
POS = set("""beat beats beating surge surges surged soar soars soared jump jumps jumped rally rallies rallied gain gains gained
rise rises rising rose climb climbs climbed record upgrade upgraded upgrades raise raises raised boost boosts boosted strong stronger
win wins won bullish outperform tops topped rebound rebounds rebounded expands expansion approval approved partnership buyback
growth grows grew higher high best breakthrough profit profitable optimistic upbeat""".split())
NEG = set("""miss misses missed fall falls fell drop drops dropped plunge plunges plunged slump slumps slumped sink sinks sank
tumble tumbles tumbled slide slides slid decline declines declined downgrade downgraded downgrades cut cuts lawsuit probe
investigation recall layoffs layoff weak weaker warning warns bearish underperform lower low loss losses fraud fine fined ban
banned delay delays delayed concern concerns risk risks selloff sell-off crash crashes worst halt halted short""".split())


def tone(text):
    words = re.findall(r"[a-z\-]+", text.lower())
    return sum(w in POS for w in words) - sum(w in NEG for w in words)


def headlines(ticker, days=7, limit=12):
    q = f'"{QUERY.get(ticker, ticker)}" stock when:{days}d' if ticker != "DRAM" else f"{QUERY['DRAM']} when:{days}d"
    url = f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"
    r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0 (Ichnora research)"})
    r.raise_for_status()
    out, seen = [], set()
    for it in ET.fromstring(r.content).findall(".//item"):
        title = (it.findtext("title") or "").strip()
        source = (it.findtext("source") or "").strip()
        if source and title.endswith(" - " + source):
            title = title[: -len(" - " + source)]
        link = it.findtext("link") or ""
        key = re.sub(r"\W+", " ", title.lower())[:80]
        if not title or key in seen or not link.startswith("https://"):
            continue
        seen.add(key)
        try:
            when = parsedate_to_datetime(it.findtext("pubDate")).astimezone(timezone.utc)
        except (TypeError, ValueError):
            continue
        out.append(dict(title=title, source=source, link=link, published=when.strftime("%Y-%m-%dT%H:%MZ"),
                        tone=tone(title), tags=[k for k, rx in TAGS.items() if re.search(rx, title, re.I)]))
    out.sort(key=lambda h: h["published"], reverse=True)
    return out[:limit]


def analysts(ticker):
    import yfinance as yf
    t = yf.Ticker(ticker)
    out = {}
    try:
        info = t.info or {}
        price, target = info.get("currentPrice") or info.get("regularMarketPrice"), info.get("targetMeanPrice")
        out.update(consensus=info.get("recommendationKey"), analysts=info.get("numberOfAnalystOpinions"),
                   target=target, upside=(target / price - 1) if price and target else None,
                   short_float=info.get("shortPercentOfFloat"))
    except Exception:
        pass
    try:
        ud = t.upgrades_downgrades
        if ud is not None and len(ud):
            since = datetime.now(timezone.utc) - timedelta(days=30)
            idx = ud.index.tz_localize("UTC") if ud.index.tz is None else ud.index
            recent = ud[idx >= since]
            act = recent["Action"].str.lower() if "Action" in recent else []
            out.update(upgrades_30d=int((act == "up").sum()), downgrades_30d=int((act == "down").sum()),
                       rating_changes=[dict(date=str(i.date()), firm=r.get("Firm"), action=r.get("Action"),
                                            to=r.get("ToGrade"), target=r.get("currentPriceTarget"))
                                       for i, r in recent.head(4).iterrows()])
    except Exception:
        pass
    try:
        cal = t.calendar or {}
        dates = cal.get("Earnings Date") or []
        nxt = min((d for d in dates if d >= datetime.now(timezone.utc).date()), default=None)
        if nxt:
            out.update(earnings=str(nxt), earnings_in=(nxt - datetime.now(timezone.utc).date()).days)
    except Exception:
        pass
    return out


def collect(tickers, pause=1.0):
    """{ticker: {headlines, tone, n, tags, analysts...}} with failures recorded per stock, never raised."""
    out = {}
    for t in tickers:
        row = {}
        try:
            h = headlines(t)
            row.update(headlines=h, n=len(h), tone=(sum(x["tone"] for x in h) / len(h)) if h else None,
                       tags=sorted({tag for x in h for tag in x["tags"]}))
        except Exception as e:
            row["news_error"] = f"{type(e).__name__}: {e}"[:160]
        if t != "DRAM":
            row.update(analysts(t))
        out[t] = row
        time.sleep(pause)
    return out
