"""Download resolved Manifold markets and the market price at fixed points in each market's life.

Two files are written to data/ and can be re-used by analyze.py:
  markets.jsonl    one line per market (question, outcome, times, trader count, sample)
  snapshots.jsonl  one line per (market, horizon): the last traded probability before that moment

The script is resumable: re-running it skips anything already in the files.

Usage:  python fetch.py
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

API = "https://api.manifold.markets/v0"
DATA = Path(__file__).parent / "data"
MARKETS_FILE = DATA / "markets.jsonl"
SNAPSHOTS_FILE = DATA / "snapshots.jsonl"

DAY_MS = 24 * 60 * 60 * 1000
MIN_TRADERS = 20            # ignore markets almost nobody traded
SMALL_SAMPLE_SIZE = 2000    # how many smaller, recent markets to add to the popular ones
REQUESTS_PER_SECOND = 6     # Manifold allows 500 requests per minute per IP

# Moments at which we read the market price.
#   o7   = 7 days after the market opened. The only horizon a trader could know in advance,
#          so it is the one used for backtests.
#   mid  = halfway between creation and resolution
#   d30 / d7 / d1 = that many days before resolution
# mid and dN depend on when the market resolved, which a trader cannot know at the time and
# which depends on the outcome (YES markets often resolve early). See the report.
HORIZONS = ["o7", "mid", "d30", "d7", "d1"]

WORKERS = 16                # requests in flight at once; each takes a few seconds to answer

session = requests.Session()
session.headers["User-Agent"] = "manifold-calibration-study"
_rate_lock = threading.Lock()
_next_slot = 0.0


def _wait_for_slot():
    """Space requests out so we never exceed REQUESTS_PER_SECOND, even across threads."""
    global _next_slot
    with _rate_lock:
        now = time.time()
        slot = max(now, _next_slot)
        _next_slot = slot + 1 / REQUESTS_PER_SECOND
    time.sleep(max(0.0, slot - now))


def get(path, **params):
    """GET from the Manifold API with rate limiting and retries."""
    for attempt in range(6):
        _wait_for_slot()
        try:
            r = session.get(f"{API}/{path}", params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"{r.status_code} {r.text[:200]}")
    raise RuntimeError(f"gave up on {path} {params}")


def keep(m):
    """True for a resolved YES/NO binary market that enough people traded."""
    return (
        m.get("outcomeType") == "BINARY"
        and m.get("mechanism") == "cpmm-1"
        and m.get("isResolved")
        and m.get("resolution") in ("YES", "NO")
        and m.get("uniqueBettorCount", 0) >= MIN_TRADERS
        and m.get("resolutionTime", 0) > m.get("createdTime", 0) + DAY_MS
    )


def slim(m, sample):
    return {
        "id": m["id"],
        "question": m["question"],
        "url": m["url"],
        "token": m.get("token", "MANA"),
        "resolution": m["resolution"],
        "created": m["createdTime"],
        "close": m.get("closeTime"),
        "resolved": m["resolutionTime"],
        "traders": m["uniqueBettorCount"],
        "volume": m.get("volume"),
        "sample": sample,
    }


def fetch_markets():
    """Sample A: the 2,000 most popular resolved markets. Sample B: smaller recent ones."""
    markets = {}

    # Sample A. The search endpoint only pages up to offset 1000, so this is the top 2,000.
    for offset in (0, 1000):
        page = get("search-markets", term="", filter="resolved", contractType="BINARY",
                   sort="most-popular", limit=1000, offset=offset)
        for m in page:
            if keep(m):
                markets[m["id"]] = slim(m, "popular")
    popular_floor = min(m["traders"] for m in markets.values())
    print(f"popular sample: {len(markets)} markets, fewest traders = {popular_floor}")

    # Sample B. Page backwards through the newest resolved markets and keep the smaller ones.
    before = None
    added = 0
    while added < SMALL_SAMPLE_SIZE:
        params = dict(term="", filter="resolved", contractType="BINARY", sort="newest", limit=1000)
        if before:
            params["beforeTime"] = before
        page = get("search-markets", **params)
        if not page:
            break
        for m in page:
            if keep(m) and m["uniqueBettorCount"] < popular_floor and m["id"] not in markets:
                markets[m["id"]] = slim(m, "smaller")
                added += 1
                if added >= SMALL_SAMPLE_SIZE:
                    break
        before = min(m["createdTime"] for m in page)
        print(f"smaller sample: {added} markets so far")

    with MARKETS_FILE.open("w", encoding="utf-8") as f:
        for m in markets.values():
            f.write(json.dumps(m) + "\n")
    return list(markets.values())


def horizon_time(m, horizon):
    if horizon == "mid":
        return m["created"] + (m["resolved"] - m["created"]) // 2
    if horizon.startswith("o"):
        t = m["created"] + int(horizon[1:]) * DAY_MS
        # Only defined if the market was still unresolved at that moment.
        return t if t < m["resolved"] else m["created"]
    days = int(horizon[1:])
    return m["resolved"] - days * DAY_MS


def fetch_snapshots(markets):
    done = set()
    if SNAPSHOTS_FILE.exists():
        for line in SNAPSHOTS_FILE.open(encoding="utf-8"):
            s = json.loads(line)
            done.add((s["id"], s["horizon"]))

    todo = [(m, h) for m in markets for h in HORIZONS if (m["id"], h) not in done]
    print(f"snapshots to fetch: {len(todo)} ({len(done)} already done)")

    with ThreadPoolExecutor(WORKERS) as pool, SNAPSHOTS_FILE.open("a", encoding="utf-8") as out:
        rows = pool.map(lambda job: snapshot(*job), todo)
        bad = 0
        for i, row in enumerate(rows, 1):
            bad += row.pop("bad")
            out.write(json.dumps(row) + "\n")
            if i % 400 == 0:
                out.flush()
                print(f"snapshots: {i}/{len(todo)}, bad = {bad}")
    if bad:
        raise RuntimeError(f"{bad} snapshots came back at or after their requested time")


def snapshot(m, h):
    """The last traded probability strictly before horizon h of market m."""
    t = horizon_time(m, h)
    row = {"id": m["id"], "horizon": h, "time": t, "prob": None, "bet_time": None, "bad": 0}
    if t > m["created"]:
        bets = get("bets", contractId=m["id"], beforeTime=t, limit=1)
        if bets:
            bet = bets[0]
            # Guard against the API ignoring beforeTime: that would silently replace
            # every snapshot with the final price.
            if bet["createdTime"] >= t:
                row["bad"] = 1
            else:
                row["prob"] = bet["probAfter"]
                row["bet_time"] = bet["createdTime"]
    return row


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    if MARKETS_FILE.exists():
        markets = [json.loads(line) for line in MARKETS_FILE.open(encoding="utf-8")]
        print(f"loaded {len(markets)} markets from {MARKETS_FILE.name}")
    else:
        markets = fetch_markets()
    fetch_snapshots(markets)
    print("done")
