"""Build report/report.html from results.json and figures/. Every number comes from results.json.

Usage:  python report.py      then open report/report.html and print it to PDF.
"""

import base64
import json
from pathlib import Path

ROOT = Path(__file__).parent
R = json.loads((ROOT / "results.json").read_text())
H = R["horizons"]


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def img(name):
    data = base64.b64encode((ROOT / "figures" / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{name}">'


def slope_reading(r):
    lo, hi = r["slope_ci"]
    if lo <= 1 <= hi:
        return "consistent with 1, so prices were neither too extreme nor too timid overall"
    if lo > 1:
        return ("above 1: outcomes were more extreme than prices suggested, so prices were too timid "
                "and a price of 20% should have been lower and 80% higher")
    return ("below 1: prices were too extreme, so markets were overconfident and long shots won "
            "more often than their prices implied")


def intercept_reading(r):
    lo, hi = r["intercept_ci"]
    if lo <= 0 <= hi:
        return "no significant tilt toward YES or NO"
    return ("YES resolved more often than priced" if lo > 0 else "YES resolved less often than priced")


def table(rows, cols):
    head = "".join(f"<th>{c}</th>" for c, _ in cols)
    body = "".join("<tr>" + "".join(f"<td>{fmt(r)}</td>" for _, fmt in cols) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main():
    mid, d30, d7, d1 = H["mid"], H["d30"], H["d7"], H["d1"]
    liq = R["liquidity_mid"]
    early = R["early_resolution"]
    fav = [r for r in R["favourites_mid"] if r["n"] >= 30]
    worst = min(fav, key=lambda r: r["yes_rate"] - r["mean_price"])
    best = max(fav, key=lambda r: r["yes_rate"] - r["mean_price"])
    e = R.get("d1_by_early", {})

    horizon_rows = [("Halfway through life", mid), ("30 days before", d30), ("7 days before", d7),
                    ("1 day before", d1)]
    horizon_table = table(horizon_rows, [
        ("Price taken", lambda r: r[0]),
        ("Markets", lambda r: f"{r[1]['n']:,}"),
        ("Brier", lambda r: f"{r[1]['brier']:.3f}"),
        ("Base-rate Brier", lambda r: f"{r[1]['brier_base_rate']:.3f}"),
        ("Skill vs base rate", lambda r: pct(r[1]["skill_vs_base_rate"], 0)),
        ("Log loss", lambda r: f"{r[1]['log_loss']:.3f}"),
        ("Slope", lambda r: f"{r[1]['slope']:.2f}"),
    ])
    liq_table = table(list(liq.items()), [
        ("Traders", lambda r: f"{r[0]} ({r[1]['traders_range'][0]}–{r[1]['traders_range'][1]})"),
        ("Markets", lambda r: f"{r[1]['n']:,}"),
        ("Brier", lambda r: f"{r[1]['brier']:.3f}"),
        ("Skill vs base rate", lambda r: pct(r[1]["skill_vs_base_rate"], 0)),
        ("Slope (95% CI)", lambda r: f"{r[1]['slope']:.2f} ({r[1]['slope_ci'][0]:.2f}–{r[1]['slope_ci'][1]:.2f})"),
    ])
    fav_table = table(fav, [
        ("Favourite's price", lambda r: f"{pct(r['bin_low'], 0)}–{pct(r['bin_high'], 0)}"),
        ("Markets", lambda r: f"{r['n']:,}"),
        ("Average price", lambda r: pct(r["mean_price"])),
        ("Won", lambda r: pct(r["yes_rate"])),
        ("95% CI", lambda r: f"{pct(r['ci_low'], 0)}–{pct(r['ci_high'], 0)}"),
    ])

    liq_vals = list(liq.values())
    liq_better = liq_vals[-1]["brier"] < liq_vals[0]["brier"]
    early_note = ""
    if "True" in e and "False" in e:
        early_note = (f" One day out, markets that resolved early scored a Brier of {e['True']['brier']:.3f}, "
                      f"against {e['False']['brier']:.3f} for markets that ran to their scheduled close.")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Manifold calibration study</title>
<style>
@page {{ size: Letter; margin: 0.6in 0.7in; }}
body {{ font-family: "Times New Roman", Times, serif; font-size: 10.5pt; line-height: 1.3; color: #111; }}
h1 {{ font-size: 18pt; margin: 0 0 2pt; }}
.sub {{ color: #555; font-size: 9.5pt; margin-bottom: 10pt; }}
h2 {{ font-size: 11pt; text-transform: uppercase; border-bottom: 0.6pt solid #555; padding-bottom: 1.5pt; margin: 12pt 0 5pt; }}
p {{ margin: 0 0 6pt; text-align: justify; }}
.lead {{ font-size: 11.5pt; border-left: 3pt solid #1f2a44; padding: 4pt 8pt; background: #f4f5f8; }}
.figs {{ display: flex; gap: 10pt; align-items: flex-start; }}
.figs img {{ width: 50%; }}
img {{ max-width: 100%; }}
.full img {{ width: 78%; display: block; margin: 0 auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 4pt 0 8pt; }}
th, td {{ border-bottom: 0.5pt solid #ccc; padding: 2.5pt 4pt; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ border-bottom: 0.8pt solid #555; }}
ul {{ margin: 0 0 6pt; padding-left: 14pt; }} li {{ margin-bottom: 2pt; }}
.small {{ font-size: 9pt; color: #444; }}
.pb {{ break-before: page; }}
</style></head><body>

<h1>How well do prediction-market prices predict outcomes?</h1>
<div class="sub">A calibration study of {R['markets']:,} resolved Manifold markets (created {R['created_range'][0]} – {R['created_range'][1]}) · Raj Jha · code and data: github.com/0xr4j/manifold-calibration</div>

<p class="lead">Halfway through their lives, market prices beat always quoting the base rate by
<b>{pct(mid['skill_vs_base_rate'], 0)}</b> on Brier score ({mid['brier']:.3f} vs {mid['brier_base_rate']:.3f}).
The calibration slope was <b>{mid['slope']:.2f}</b> (95% CI {mid['slope_ci'][0]:.2f}–{mid['slope_ci'][1]:.2f}),
{slope_reading(mid)}; there was {intercept_reading(mid)}.</p>

<h2>Question</h2>
<p>An event contract's price is meant to be the probability of the event. If that holds, markets priced at 70% should resolve YES about 70% of the time,
at every price level and at every point in a market's life. This study checks that against {R['markets']:,} resolved YES/NO markets with at least 20 traders each:
the {R['markets_by_sample'].get('popular', 0):,} most popular resolved markets and {R['markets_by_sample'].get('smaller', 0):,} smaller recent ones.
For each market it reads the last traded price before four moments and compares it with the outcome.</p>

<div class="figs">{img('calibration_mid.png')}{img('favourites_mid.png')}</div>
<p class="small">Left: each dot is a price bin, with the share of its markets that resolved YES and a 95% Wilson interval; the histogram shows where prices sat.
Right: every market folded onto its favourite. A dot below zero means favourites at that price won less often than their price said.</p>

<h2>Favourites and long shots</h2>
<p>Among price bands with at least 30 markets, favourites priced {pct(worst['bin_low'], 0)}–{pct(worst['bin_high'], 0)} won
{pct(worst['yes_rate'])} of the time against an average price of {pct(worst['mean_price'])}, the largest shortfall.
Favourites priced {pct(best['bin_low'], 0)}–{pct(best['bin_high'], 0)} won {pct(best['yes_rate'])} against {pct(best['mean_price'])}.</p>
{fav_table}

<h2 class="pb">Accuracy against time left</h2>
{horizon_table}
<div class="full">{img('brier_by_horizon.png')}</div>
<p>Accuracy improves sharply as resolution nears, but part of the late improvement is look-ahead rather than skill.
{pct(early['yes'], 0)} of YES markets resolved more than a day before their scheduled close, against {pct(early['no'], 0)} of NO markets:
"will X happen by date" markets resolve as soon as X happens, so a price taken a day before resolution often already reflects the answer.{early_note}
This is why the headline uses the halfway price.</p>

<h2>Liquidity</h2>
{liq_table}
<p>Markets with the most traders were {'more' if liq_better else 'less'} accurate halfway through their lives than markets with the fewest
(Brier {liq_vals[-1]['brier']:.3f} vs {liq_vals[0]['brier']:.3f}). The most-traded third comes mostly from the popular sample, which spans more years,
so trader count and time period are partly confounded.</p>

<h2>What this means for pricing</h2>
<ul>
<li><b>The last trade is a usable prior, not an answer.</b> Halfway through, prices removed {pct(mid['skill_vs_base_rate'], 0)} of the base rate's Brier error.
A pricing system can start from the market price and look for systematic adjustments, rather than rebuilding every forecast from scratch.</li>
<li><b>Correct for the shape of the error, not just its size.</b> A calibration slope of {mid['slope']:.2f} gives a one-line re-mapping:
fair probability = sigmoid({mid['intercept']:.2f} + {mid['slope']:.2f} × logit(price)). It must be re-fitted on fresh, real-money data before it is trusted.</li>
<li><b>Measure skill where the answer is not yet known.</b> Late prices look excellent partly because early-resolving markets leak the outcome.
Backtests of event-contract strategies need the same care, or they will credit a model for information it could not have had.</li>
</ul>

<h2>Limits</h2>
<ul class="small">
<li>Manifold uses play money (mana), so incentives are weaker than on Kalshi or Polymarket, which were not reachable from the network this was run on. All {R['markets']:,} markets traded in mana.</li>
<li>The popular sample favours heavily traded markets and the smaller sample favours recent ones. Creators resolve their own markets.</li>
<li>Prices are last trades, not mid-quotes. Markets with no trade before a moment were left out of that horizon:
{mid['dropped_no_price']} halfway, {d30['dropped_no_price']} at 30 days, {d7['dropped_no_price']} at 7 days and {d1['dropped_no_price']} at 1 day.</li>
</ul>
<p class="small">Method: Brier score and log loss against a base-rate forecaster; Wilson intervals per price bin; calibration slope and intercept from a logistic
regression of the outcome on logit(price), with 1,000 bootstrap resamples of markets; Murphy decomposition (halfway: reliability {mid['reliability']:.4f},
resolution {mid['resolution']:.4f}, uncertainty {mid['uncertainty']:.4f}). Code: fetch.py, metrics.py, analyze.py, report.py, with tests on synthetic data.
I used an AI coding assistant (Claude) to help write the code; the design and conclusions are mine, and every number here is generated from the data.</p>
</body></html>"""
    out = ROOT / "report"
    out.mkdir(exist_ok=True)
    (out / "report.html").write_text(html, encoding="utf-8")
    print("wrote", out / "report.html")


if __name__ == "__main__":
    main()
