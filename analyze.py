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
HORIZONS = ["mid", "d30", "d7", "d1"]
HORIZON_LABELS = {"mid": "Halfway through life", "d30": "30 days before resolution",
                  "d7": "7 days before resolution", "d1": "1 day before resolution"}
INK, ACCENT, MUTED = "#1f2a44", "#c0392b", "#8a94a6"


def load():
    markets = pd.read_json(DATA / "markets.jsonl", lines=True)
    snaps = pd.read_json(DATA / "snapshots.jsonl", lines=True)
    snaps = snaps.drop_duplicates(["id", "horizon"], keep="last")
    df = snaps.merge(markets, on="id", how="inner")
    df["y"] = (df["resolution"] == "YES").astype(float)
    # A market "resolved early" if it resolved more than a day before its scheduled close.
    df["early"] = df["resolved"] < df["close"] - DAY_MS
    return markets, df


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


def calibration_plot(sub, path, title):
    p, y = sub["prob"].to_numpy(), sub["y"].to_numpy()
    rows = metrics.calibration_table(p, y)
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6.4, 6.4), height_ratios=[4, 1], sharex=True)
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

    # Early resolution: how often each outcome resolved well before its scheduled close.
    m = markets.dropna(subset=["close"]).copy()
    m["early"] = m["resolved"] < m["close"] - DAY_MS
    results["early_resolution"] = {
        "yes": float(m.loc[m["resolution"] == "YES", "early"].mean()),
        "no": float(m.loc[m["resolution"] == "NO", "early"].mean()),
    }

    for h in HORIZONS:
        sub = df[df["horizon"] == h]
        have = sub.dropna(subset=["prob"])
        r = score(have)
        r["dropped_no_price"] = int(len(sub) - len(have))
        if h == "mid":
            r.update(metrics.bootstrap_fit(have["prob"].to_numpy(), have["y"].to_numpy()))
        results["horizons"][h] = r

    mid = df[(df["horizon"] == "mid")].dropna(subset=["prob"]).copy()
    results["calibration_mid"] = calibration_plot(
        mid, FIG / "calibration_mid.png", "Calibration halfway through each market's life")
    results["calibration_d1"] = calibration_plot(
        df[df["horizon"] == "d1"].dropna(subset=["prob"]), FIG / "calibration_d1.png",
        "Calibration 1 day before resolution")
    results["favourites_mid"] = favourite_plot(mid, FIG / "favourites_mid.png")

    # Accuracy against time left.
    hs = [h for h in HORIZONS if results["horizons"][h]["n"] > 0][::-1]
    order = ["mid", "d30", "d7", "d1"]
    hs = [h for h in order if h in hs]
    bar_plot([HORIZON_LABELS[h].replace(" before resolution", "\nbefore").replace(" through life", "\nthrough life")
              for h in hs],
             [results["horizons"][h]["brier"] for h in hs], FIG / "brier_by_horizon.png",
             "Brier score (lower is better)", "Accuracy improves as resolution nears",
             baseline=results["horizons"]["mid"]["brier_base_rate"],
             baseline_label="Always quoting the base rate (halfway)")

    # Liquidity: split markets into thirds by number of traders.
    mid["traders_third"] = pd.qcut(mid["traders"], 3, labels=["fewest", "middle", "most"])
    liq = {}
    for label, sub in mid.groupby("traders_third", observed=True):
        r = score(sub)
        r["traders_range"] = [int(sub["traders"].min()), int(sub["traders"].max())]
        r.update(metrics.bootstrap_fit(sub["prob"].to_numpy(), sub["y"].to_numpy(), n_boot=500))
        liq[str(label)] = r
    results["liquidity_mid"] = liq
    bar_plot([f"{k} traders\n({v['traders_range'][0]}–{v['traders_range'][1]})" for k, v in liq.items()],
             [v["brier"] for v in liq.values()], FIG / "brier_by_liquidity.png",
             "Brier score (lower is better)", "Halfway accuracy by number of traders")

    # Early resolution as a source of look-ahead in late snapshots.
    late = df[df["horizon"] == "d1"].dropna(subset=["prob"])
    results["d1_by_early"] = {str(k): score(g) for k, g in late.groupby("early") if len(g) > 50}

    (ROOT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k in ("markets", "markets_by_sample", "tokens")}, indent=2))
    for h in HORIZONS:
        r = results["horizons"][h]
        print(f"{h:>4}: n={r['n']:5d}  brier={r['brier']:.4f}  base={r['brier_base_rate']:.4f}  "
              f"slope={r['slope']:.2f}  intercept={r['intercept']:.2f}  dropped={r['dropped_no_price']}")


if __name__ == "__main__":
    main()
