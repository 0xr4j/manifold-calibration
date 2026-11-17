"""Measure how well Manifold prices predict outcomes. Writes results.json and figures/.

Usage:  python analyze.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import metrics  # noqa: E402

ROOT = Path(__file__).parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
DAY_MS = 24 * 60 * 60 * 1000
HORIZONS = ["o7", "mid", "d30", "d7", "d1"]
# o7 is the headline: it is the only moment a trader could pick in advance. The others are
# counted from the resolution time, which a trader does not know and which depends on the outcome.
HEADLINE = "o7"
HORIZON_LABELS = {"o7": "7 days after opening", "mid": "Halfway through life", "d30": "30 days before resolution",
                  "d7": "7 days before resolution", "d1": "1 day before resolution"}
INK, ACCENT, MUTED = "#1f2a44", "#c0392b", "#8a94a6"


def read_jsonl(path):
    """Read a JSON-lines file without pandas guessing at dates."""
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


def load():
    markets = read_jsonl(DATA / "markets.jsonl")
    snaps = read_jsonl(DATA / "snapshots.jsonl")
    snaps = snaps.drop_duplicates(["id", "horizon"], keep="last")
    df = snaps.merge(markets, on="id", how="inner")
    df["y"] = (df["resolution"] == "YES").astype(float)
    df["early"] = resolved_while_open(df)
    return markets, df


def resolved_while_open(m):
    """True for markets resolved while trading was still open.

    When a creator resolves an open market, Manifold moves its close time to the moment of
    resolution. So a close time within a minute of the resolution time marks a market that was
    still open when it resolved: mostly early resolutions ("X happened before the deadline"),
    plus some resolved on their deadline day before trading closed. A market that ran to its
    deadline closes first and is resolved later.
    """
    return (m["resolved"] - m["close"]).abs() < 60_000


def score(sub):
    p, y = sub["prob"].to_numpy(), sub["y"].to_numpy()
    base = y.mean()
    out = {
        "n": int(len(sub)),
        "yes_share": float(base),
        "brier": metrics.brier(p, y),
        "brier_base_rate": metrics.brier(np.full_like(p, base), y),
        "log_loss": metrics.log_loss(p, y),
        "log_loss_base_rate": metrics.log_loss(np.full_like(p, base), y),
        **metrics.calibration_fit(p, y),
        **metrics.murphy_decomposition(p, y),
    }
    out["skill_vs_base_rate"] = 1 - out["brier"] / out["brier_base_rate"]
    return out


def walk_forward(head, min_edge=0.03):
    """Learn the calibration correction from markets already resolved on a split date, then trade later prices.

    Split date S = median time at which the headline prices were read. Train on markets resolved
    before S; test on prices read on or after S. Buy YES when the corrected chance beats the price
    by min_edge, NO when it falls short by min_edge. One contract per trade, no fees.
    Mirrors the fitted-correction rule on the website.
    """
    rows = head.sort_values("time")
    split = rows["time"].iloc[len(rows) // 2]
    train = rows[rows["resolved"] < split]
    test = rows[rows["time"] >= split]
    fit = metrics.calibration_fit(train["prob"].to_numpy(), train["y"].to_numpy())
    p, y = test["prob"].to_numpy(), test["y"].to_numpy()
    fair = 1 / (1 + np.exp(-(fit["intercept"] + fit["slope"] * metrics.logit(p))))
    buy_yes, buy_no = fair - p >= min_edge, p - fair >= min_edge
    profit = np.concatenate([(y - p)[buy_yes], (p - y)[buy_no]])
    cost = np.concatenate([p[buy_yes], (1 - p)[buy_no]])
    n = len(profit)
    half = 1.96 * profit.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return {
        "split": pd.to_datetime(split, unit="ms").strftime("%b %Y"),
        "train_markets": int(len(train)), "test_markets": int(len(test)), "fit": fit,
        # Fitted after the fact on the test period, only to show whether the pattern moved.
        "test_period_fit": metrics.calibration_fit(p, y),
        "trades": int(n), "buy_yes": int(buy_yes.sum()), "buy_no": int(buy_no.sum()), "min_edge": min_edge,
        "avg_profit": float(profit.mean()), "avg_profit_ci": [float(profit.mean() - half), float(profit.mean() + half)],
        "return_on_stake": float(profit.sum() / cost.sum()),
    }


def calibration_plot(sub, path, title):
    p, y = sub["prob"].to_numpy(), sub["y"].to_numpy()
    rows = metrics.calibration_table(p, y)
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6.4, 5.4), height_ratios=[4, 1], sharex=True)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Perfect calibration")
    x = [r["mean_price"] for r in rows]
    yr = [r["yes_rate"] for r in rows]
    # Clip at zero: when every market in a bin resolves the same way, rounding can make the
    # Wilson bound land a hair inside the point estimate.
    err = [[max(0.0, r["yes_rate"] - r["ci_low"]) for r in rows],
           [max(0.0, r["ci_high"] - r["yes_rate"]) for r in rows]]
    ax.errorbar(x, yr, yerr=err, fmt="o", color=INK, ecolor=INK, elinewidth=1, capsize=3,
                ms=5, label="Observed YES rate (95% CI)")
    ax.set_ylabel("Share that resolved YES")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=11, loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)
    axh.hist(p, bins=50, range=(0, 1), color=MUTED)
    axh.set_xlabel("Market price (probability of YES)")
    axh.set_ylabel("Markets")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return rows


def bar_plot(labels, values, path, ylabel, title, baseline=None, baseline_label=None):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(labels, values, color=INK, width=0.6)
    if baseline is not None:
        ax.axhline(baseline, color=ACCENT, lw=1, ls="--", label=baseline_label)
        ax.legend(frameon=False, fontsize=9)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def favourite_plot(sub, path):
    rows = metrics.favourite_table(sub["prob"].to_numpy(), sub["y"].to_numpy())
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(rows))
    gap = [r["yes_rate"] - r["mean_price"] for r in rows]
    lo = [r["ci_low"] - r["mean_price"] for r in rows]
    hi = [r["ci_high"] - r["mean_price"] for r in rows]
    ax.axhline(0, color=MUTED, lw=1)
    yerr = [np.maximum(0, np.subtract(gap, lo)), np.maximum(0, np.subtract(hi, gap))]
    ax.errorbar(x, gap, yerr=yerr, fmt="o", color=INK,
                capsize=3, ms=5)
    ax.set_xticks(x, [f"{r['bin_low']:.0%}–{r['bin_high']:.0%}\n(n={r['n']})" for r in rows], fontsize=8)
    ax.set_xlabel("Price of the favourite (the likelier side)")
    ax.set_ylabel("Win rate minus price")
    ax.set_title("Do favourites win as often as their price says?", fontsize=11, loc="left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return rows


def main():
    FIG.mkdir(exist_ok=True)
    markets, df = load()
    results = {"markets": int(len(markets)),
               "markets_by_sample": markets["sample"].value_counts().to_dict(),
               "tokens": markets["token"].value_counts().to_dict(),
               "yes_share_all": float((markets["resolution"] == "YES").mean()),
               "created_range": [pd.to_datetime(markets["created"].min(), unit="ms").strftime("%b %Y"),
                                 pd.to_datetime(markets["created"].max(), unit="ms").strftime("%b %Y")],
               "horizons": {}}

    # Early resolution: how often each outcome resolved while the market was still open.
    m = markets.dropna(subset=["close"]).copy()
    m["early"] = resolved_while_open(m)
    results["early_resolution"] = {
        "yes": float(m.loc[m["resolution"] == "YES", "early"].mean()),
        "no": float(m.loc[m["resolution"] == "NO", "early"].mean()),
    }

    for h in HORIZONS:
        sub = df[df["horizon"] == h]
        have = sub.dropna(subset=["prob"])
        r = score(have)
        # o7 has no price when the market resolved within 7 days; the others when nobody had traded yet.
        r["dropped_no_price"] = int(len(sub) - len(have))
        if h in (HEADLINE, "mid"):
            r.update(metrics.bootstrap_fit(have["prob"].to_numpy(), have["y"].to_numpy()))
        results["horizons"][h] = r

    head = df[df["horizon"] == HEADLINE].dropna(subset=["prob"]).copy()
    results["calibration_headline"] = calibration_plot(
        head, FIG / "calibration_o7.png", "Calibration 7 days after each market opened")
    results["calibration_mid"] = calibration_plot(
        df[df["horizon"] == "mid"].dropna(subset=["prob"]), FIG / "calibration_mid.png",
        "Calibration halfway through each market's life")
    results["favourites_headline"] = favourite_plot(head, FIG / "favourites_o7.png")

    # Accuracy by when the price is read.
    hs = [h for h in HORIZONS if results["horizons"][h]["n"] > 0]
    bar_plot([HORIZON_LABELS[h].replace(" before resolution", "\nbefore").replace(" through life", "\nthrough life")
              .replace(" after opening", "\nafter opening") for h in hs],
             [results["horizons"][h]["brier"] for h in hs], FIG / "brier_by_horizon.png",
             "Brier score (lower is better)", "Accuracy by when the price is read",
             baseline=results["horizons"][HEADLINE]["brier_base_rate"],
             baseline_label="Always quoting the base rate (7 days after opening)")

    # Liquidity: split markets into thirds by number of traders.
    head["traders_third"] = pd.qcut(head["traders"], 3, labels=["fewest", "middle", "most"])
    liq = {}
    for label, sub in head.groupby("traders_third", observed=True):
        r = score(sub)
        r["traders_range"] = [int(sub["traders"].min()), int(sub["traders"].max())]
        r.update(metrics.bootstrap_fit(sub["prob"].to_numpy(), sub["y"].to_numpy(), n_boot=500))
        liq[str(label)] = r
    results["liquidity_headline"] = liq
    bar_plot([f"{k} traders\n({v['traders_range'][0]}–{v['traders_range'][1]})" for k, v in liq.items()],
             [v["brier"] for v in liq.values()], FIG / "brier_by_liquidity.png",
             "Brier score (lower is better)", "Accuracy 7 days after opening, by number of traders")

    results["walk_forward"] = walk_forward(head)

    # A day before resolution: markets resolved while still open vs those that ran to their deadline.
    late = df[df["horizon"] == "d1"].dropna(subset=["prob"])
    results["d1_by_early"] = {str(k): score(g) for k, g in late.groupby("early") if len(g) > 50}
    ran = late[~late["early"]]
    results["d1_ran_to_close_after_close"] = float((ran["time"] > ran["close"]).mean())

    (ROOT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k in ("markets", "markets_by_sample", "tokens")}, indent=2))
    for h in HORIZONS:
        r = results["horizons"][h]
        print(f"{h:>4}: n={r['n']:5d}  brier={r['brier']:.4f}  base={r['brier_base_rate']:.4f}  "
              f"slope={r['slope']:.2f}  intercept={r['intercept']:.2f}  dropped={r['dropped_no_price']}")


if __name__ == "__main__":
    main()
