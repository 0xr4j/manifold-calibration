# How well do prediction-market prices predict outcomes?

A calibration study of 3,915 resolved binary markets on [Manifold Markets](https://manifold.markets).
An event contract's price is meant to be the probability of the event. This project checks that claim against the outcomes: when a market said 70%, did the event happen about 70% of the time? How does accuracy change as resolution nears, and with the number of traders?

**Report:** [`report/report.pdf`](report/report.pdf) (two pages, with all figures and numbers).

<!-- RESULTS -->

## Method

1. **Markets.** Two samples of resolved YES/NO markets from the public Manifold API, each with at least 20 traders and at least one day between creation and resolution:
   - the 2,000 most popular resolved markets (the API's search only pages this deep), and
   - 2,000 smaller recent markets with fewer traders than the least popular market in the first sample.

   Markets resolved to a percentage (MKT) or cancelled are excluded.
2. **Prices.** For each market, the price is the probability right after the last trade strictly before four moments: halfway through its life, and 30, 7 and 1 days before it resolved. Each price comes from one `GET /v0/bets?contractId=…&beforeTime=T&limit=1` call. The fetcher checks every returned trade is earlier than `T`, so a silently ignored filter cannot leak final prices into the data. A market with no trade before a moment is left out of that horizon, and the count is reported.
3. **Scores.**
   - Brier score and log loss, each compared with always quoting the sample's base rate.
   - Calibration tables with 95% Wilson intervals, with finer bins at the tails.
   - A calibration slope and intercept from a logistic regression of the outcome on logit(price), with 1,000 bootstrap resamples of markets. A perfectly calibrated market has slope 1 and intercept 0.
   - The Murphy decomposition of Brier into reliability, resolution and uncertainty.
   - A favourite-longshot table that folds each market onto its likelier side.

**Headline horizon.** The headline is halfway through each market's life, not one day before resolution. Many "will X happen by date" markets resolve YES as soon as X happens, while NO markets wait for the deadline, so a price taken just before resolution often already knows the answer. The report measures this effect directly.

## Run it

```bash
python -m pip install -r requirements.txt
python fetch.py      # about 40 minutes; resumable; writes data/markets.jsonl and data/snapshots.jsonl
python analyze.py    # writes results.json and figures/
python report.py     # writes report/report.html from results.json (print it to PDF from a browser)
python -m pytest -q  # checks the statistics on synthetic data where the answer is known
```

The fetched data is committed in `data/`, so `analyze.py` and `report.py` run without the network.

## Files

| File | What it does |
|---|---|
| `fetch.py` | Downloads markets and price snapshots, rate-limited to 6 requests per second |
| `metrics.py` | Brier, log loss, Wilson intervals, calibration fit, Murphy decomposition, favourite table |
| `analyze.py` | Scores every horizon and liquidity group, draws the figures, writes `results.json` |
| `report.py` | Builds the report from `results.json`, so no number in it is typed by hand |
| `tests/` | Synthetic checks: a calibrated market gives slope ≈ 1 and reliability ≈ 0 |

## Limits

- **Play money.** Manifold trades in mana, which has no cash value, so incentives are weaker than on Kalshi or Polymarket. Every market in this sample uses mana. Kalshi and Polymarket were not reachable from the network this study was run on.
- **Sample.** The popular sample is biased toward heavily traded markets, and the smaller sample is biased toward recent ones. The liquidity comparison mixes both, so trader count and time period are partly confounded.
- **Resolution.** Manifold markets are resolved by their creators, not by an exchange.
- **Price.** The snapshot is the last traded price, not a mid-quote, so a stale price in a thinly traded market counts as that market's forecast.
