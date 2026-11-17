"""Write docs/data.json: a compact copy of the dataset for the interactive website.

Usage:  python export_site_data.py   (after fetch.py and analyze.py)
"""

import json
from pathlib import Path

from analyze import DATA, DAY_MS, HEADLINE, HORIZONS, read_jsonl, resolved_while_open

ROOT = Path(__file__).parent


def main():
    markets = read_jsonl(DATA / "markets.jsonl").reset_index(drop=True)
    snaps = read_jsonl(DATA / "snapshots.jsonl").drop_duplicates(["id", "horizon"], keep="last")
    snaps = snaps.dropna(subset=["prob"])

    index = {mid: i for i, mid in enumerate(markets["id"])}
    early = resolved_while_open(markets).astype(int)

    # One row per market: [question, url, outcome (1 = YES), traders, resolved day, resolved while open, sample]
    market_rows = [
        [m.question, m.url, int(m.resolution == "YES"), int(m.traders),
         int(m.resolved // DAY_MS), int(early[i]), 0 if m.sample == "popular" else 1]
        for i, m in enumerate(markets.itertuples())
    ]
    # One row per snapshot: [market index, horizon index, price in tenths of a percent, day the price was read]
    snapshot_rows = [
        [index[s.id], HORIZONS.index(s.horizon), int(round(s.prob * 1000)), int(s.time // DAY_MS)]
        for s in snaps.itertuples() if s.id in index and s.horizon in HORIZONS
    ]

    results = json.loads((ROOT / "results.json").read_text())
    out = {
        "horizons": HORIZONS,
        "headline": HEADLINE,
        "markets": market_rows,
        "snapshots": snapshot_rows,
        "summary": {
            "markets": results["markets"],
            "created_range": results["created_range"],
            "early_resolution": results["early_resolution"],
            "horizons": {h: {k: results["horizons"][h][k] for k in
                             ("n", "brier", "brier_base_rate", "skill_vs_base_rate", "log_loss", "slope", "intercept")}
                         for h in HORIZONS},
            "slope_ci_headline": results["horizons"][HEADLINE].get("slope_ci"),
        },
    }
    path = ROOT / "docs" / "data.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB, {len(market_rows)} markets, {len(snapshot_rows)} snapshots)")


if __name__ == "__main__":
    main()
