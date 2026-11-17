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
        return "consistent with 1: prices were neither too extreme nor too timid overall"
    if lo > 1:
        return ("above 1: outcomes were more extreme than prices suggested, so prices were too timid, "
                "with favourites underpriced and long shots overpriced")
    return ("below 1: prices were too extreme, so long shots won more often than their prices implied "
            "and heavy favourites less often")


def intercept_reading(r):
    lo, hi = r["intercept_ci"]
    if lo <= 0 <= hi:
        return "no significant tilt toward YES or NO"
    return "a tilt toward YES: YES resolved more often than priced" if lo > 0 else \
        "a tilt toward NO: YES resolved less often than priced"


def table(rows, cols):
    head = "".join(f"<th>{c}</th>" for c, _ in cols)
    body = "".join("<tr>" + "".join(f"<td>{fmt(r)}</td>" for _, fmt in cols) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main():
    o7, mid, d1 = H["o7"], H["mid"], H["d1"]
    liq = R["liquidity_headline"]
    early = R["early_resolution"]
    fav = [r for r in R["favourites_headline"] if r["n"] >= 30]
    worst = min(fav, key=lambda r: r["yes_rate"] - r["mean_price"])
    best = max(fav, key=lambda r: r["yes_rate"] - r["mean_price"])
    e = R["d1_by_early"]

    horizon_rows = [("7 days after opening", H["o7"]), ("Halfway through life", mid), ("30 days before end", H["d30"]),
                    ("7 days before end", H["d7"]), ("1 day before end", d1)]
    horizon_table = table(horizon_rows, [
        ("Price taken", lambda r: r[0]),
        ("Markets", lambda r: f"{r[1]['n']:,}"),
        ("Brier", lambda r: f"{r[1]['brier']:.3f}"),
        ("Base-rate Brier", lambda r: f"{r[1]['brier_base_rate']:.3f}"),
        ("Skill vs base rate", lambda r: pct(r[1]["skill_vs_base_rate"], 0)),
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
    liq_word = "more" if liq_vals[-1]["brier"] < liq_vals[0]["brier"] else "less"
    wf = R["walk_forward"]
    lo, hi = wf["avg_profit_ci"]
    if lo > 0:
        wf_reading = "The edge held out of sample, though it would need to survive fees, price impact and real money."
    elif hi < 0:
        wf_reading = "It lost money out of sample: the pattern did not persist."
    else:
        wf_reading = ("That is not distinguishable from zero, and about the size of a typical 1¢ fee. "
                      f"The slope learned before {wf['split']} was {wf['fit']['slope']:.2f}; on the later markets it was "
                      f"{wf['test_period_fit']['slope']:.2f}. The mispricing was real, but its size shifted between periods, "
                      "so a pricing model has to keep re-learning it rather than fix it once.")

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
.full img {{ width: 62%; display: block; margin: 0 auto; }}
tr {{ break-inside: avoid; }} table {{ break-inside: auto; }} h2 {{ break-after: avoid; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 4pt 0 8pt; }}
th, td {{ border-bottom: 0.5pt solid #ccc; padding: 2.5pt 4pt; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ border-bottom: 0.8pt solid #555; }}
ul {{ margin: 0 0 6pt; padding-left: 14pt; }} li {{ margin-bottom: 2pt; }}
.small {{ font-size: 9pt; color: #444; }}
.pb {{ break-before: page; }}
</style></head><body>

<h1>How well do prediction-market prices predict outcomes?</h1>
<div class="sub">A calibration study of {R['markets']:,} resolved Manifold markets (created {R['created_range'][0]} – {R['created_range'][1]}) · Raj Jha ·
code, data and interactive site: github.com/0xr4j/manifold-calibration</div>

<p class="lead">Seven days after opening, market prices beat always quoting the base rate by <b>{pct(o7['skill_vs_base_rate'], 0)}</b> on Brier score
({o7['brier']:.3f} vs {o7['brier_base_rate']:.3f}, {o7['n']:,} markets). The calibration slope was <b>{o7['slope']:.2f}</b>
(95% CI {o7['slope_ci'][0]:.2f}–{o7['slope_ci'][1]:.2f}), {slope_reading(o7)}; there was {intercept_reading(o7)}.</p>

<h2>Question</h2>
<p>An event contract's price is meant to be the probability of the event. If that holds, markets priced at 70% should resolve YES about 70% of the time.
This study checks that against {R['markets']:,} resolved YES/NO markets with at least 20 traders each: the {R['markets_by_sample'].get('popular', 0):,} most popular
resolved markets and {R['markets_by_sample'].get('smaller', 0):,} smaller recent ones. For each market it reads the last traded price before fixed moments and
compares it with the outcome. The headline moment is <b>7 days after the market opened</b>, because it is the only one a trader could choose in advance (see "Why the moment matters").</p>

<div class="figs">{img('calibration_o7.png')}{img('favourites_o7.png')}</div>
<p class="small">Left: each dot is a price bin, with the share of its markets that resolved YES and a 95% Wilson interval; the histogram shows where prices sat.
Right: every market folded onto its favourite. A dot below zero means favourites at that price won less often than their price said.</p>

<h2>Favourites and long shots</h2>
<p>Among price bands with at least 30 markets, favourites priced {pct(worst['bin_low'], 0)}–{pct(worst['bin_high'], 0)} won {pct(worst['yes_rate'])} of the time
against an average price of {pct(worst['mean_price'])}, the largest shortfall. Favourites priced {pct(best['bin_low'], 0)}–{pct(best['bin_high'], 0)} won
{pct(best['yes_rate'])} against {pct(best['mean_price'])}.</p>
{fav_table}

<h2>Why the moment matters</h2>
{horizon_table}
<div class="full">{img('brier_by_horizon.png')}</div>
<p>Prices look far more accurate close to resolution, but every moment except the first is counted back from the resolution time, which no trader knows in advance.
That timing also depends on the outcome: <b>{pct(early['yes'], 0)}</b> of YES markets were resolved while trading was still open, against <b>{pct(early['no'], 0)}</b>
of NO markets, because "will X happen by a date" resolves YES as soon as X happens and NO only at the deadline. A day before resolution, markets that ran to their
deadline scored a Brier of {e['False']['brier']:.3f} (with a day left, the answer was usually settled; only {pct(R['d1_ran_to_close_after_close'], 0)} of these prices were
read after trading had closed), while markets resolved while still open scored {e['True']['brier']:.3f}, often because the event itself was a surprise.
Late accuracy therefore says as much about how a question resolves as about forecasting skill, and a backtest keyed to it would trade on information it could not have had.</p>
<p class="small">Manifold moves a market's close time to the moment it resolves when an open market is resolved, so a close within a minute of resolution marks a
market resolved while trading was still open. This includes early resolutions and some resolved on their deadline day before trading closed.</p>

<h2>Liquidity</h2>
{liq_table}
<p>Seven days after opening, markets with the most traders were {liq_word} accurate than markets with the fewest
(Brier {liq_vals[-1]['brier']:.3f} vs {liq_vals[0]['brier']:.3f}; slope {liq_vals[-1]['slope']:.2f} vs {liq_vals[0]['slope']:.2f}). More traders did not mean
better prices here. The most-traded third comes mostly from the popular sample, which spans more years and attracts long-running, contested questions,
so trader count, question type and time period are confounded; this is a description, not evidence that liquidity hurts accuracy.</p>

<h2>What this means for pricing</h2>
<ul>
<li><b>The market price is a usable prior, not an answer.</b> A week in, prices removed {pct(o7['skill_vs_base_rate'], 0)} of the base rate's Brier error.
A pricing system can start from the market and look for systematic adjustments.</li>
<li><b>A real mispricing is not yet a tradable edge.</b> A calibration slope of {o7['slope']:.2f} suggests a one-line re-mapping,
fair probability = sigmoid(a + b × logit(price)). Fitted only on the {wf['train_markets']:,} markets resolved before {wf['split']} and applied to the
{wf['test_markets']:,} prices read after it (buying whichever side the correction favoured by at least {pct(wf['min_edge'], 0)}), it made
<b>{100 * wf['avg_profit']:+.1f}¢</b> per contract over {wf['trades']:,} trades (95% CI {100 * wf['avg_profit_ci'][0]:+.1f}¢ to {100 * wf['avg_profit_ci'][1]:+.1f}¢), before fees.
{wf_reading}</li>
<li><b>Only test on moments a trader could have chosen.</b> Horizons counted back from resolution leak the outcome through its timing. The interactive site's backtest
trades only at 7 days after opening and learns its correction only from markets already resolved.</li>
</ul>

<h2>Limits</h2>
<ul class="small">
<li>Manifold uses play money (mana), so incentives are weaker than on Kalshi or Polymarket, which were not reachable from the network this was run on. All {R['markets']:,} markets traded in mana.</li>
<li>Markets were selected using their final popularity and trader count, which are not known 7 days in. A market must still be open 7 days in to have a price then
({o7['dropped_no_price']:,} resolved sooner and are excluded at that moment). Creators resolve their own markets.</li>
<li>Prices are last trades, not quotes. A market is left out of a moment that came before it opened or before its first trade:
{mid['dropped_no_price']} halfway, {H['d30']['dropped_no_price']:,} at 30 days before the end (mostly markets that lived less than 30 days),
{H['d7']['dropped_no_price']} at 7 days and {d1['dropped_no_price']} at 1 day.</li>
</ul>
<p class="small">Method: Brier score and log loss against a base-rate forecaster; Wilson intervals per price bin; calibration slope and intercept from a logistic
regression of the outcome on logit(price), with 1,000 bootstrap resamples of markets; Murphy decomposition (7 days after opening: reliability {o7['reliability']:.4f},
resolution {o7['resolution']:.4f}, uncertainty {o7['uncertainty']:.4f}). Code: fetch.py, metrics.py, analyze.py, report.py, with tests on synthetic data.
Every number here is generated from the data.</p>
</body></html>"""
    out = ROOT / "report"
    out.mkdir(exist_ok=True)
    (out / "report.html").write_text(html, encoding="utf-8")
    print("wrote", out / "report.html")
    update_readme(o7, early, fav, wf, liq_vals)


def update_readme(o7, early, fav, wf, liq_vals):
    """Rewrite the Results section of README.md (between the RESULTS markers) from results.json."""
    worst = min(fav, key=lambda r: r["yes_rate"] - r["mean_price"])
    lo, hi = wf["avg_profit_ci"]
    block = f"""<!-- RESULTS: generated by report.py from results.json; do not edit by hand -->
## Results

Using the price **7 days after each market opened** ({o7['n']:,} markets), the only moment a trader could choose in advance:

- **Prices carry real information.** They beat always quoting the base rate by **{pct(o7['skill_vs_base_rate'], 0)}** on Brier score ({o7['brier']:.3f} vs {o7['brier_base_rate']:.3f}).
- **But they were too extreme.** The calibration slope was **{o7['slope']:.2f}** (95% CI {o7['slope_ci'][0]:.2f}–{o7['slope_ci'][1]:.2f}); 1.00 is perfect. Favourites priced {pct(worst['bin_low'], 0)}–{pct(worst['bin_high'], 0)} won {pct(worst['yes_rate'])} of the time against an average price of {pct(worst['mean_price'])}.
- **Timing leaks the answer.** {pct(early['yes'], 0)} of YES markets were resolved while trading was still open, against {pct(early['no'], 0)} of NO markets, so any horizon counted back from resolution quietly uses the outcome.
- **The mispricing was not a reliable edge.** A correction learned on markets resolved before {wf['split']} earned {100 * wf['avg_profit']:+.1f}¢ per contract on {wf['trades']:,} later trades (95% CI {100 * lo:+.1f}¢ to {100 * hi:+.1f}¢), before fees. The slope moved from {wf['fit']['slope']:.2f} to {wf['test_period_fit']['slope']:.2f} between the two periods.
- **More traders did not mean better prices.** The most-traded third had a slope of {liq_vals[-1]['slope']:.2f} against {liq_vals[0]['slope']:.2f} for the least-traded, though trader count is confounded with question type and time period.

![Calibration 7 days after opening](figures/calibration_o7.png)
<!-- /RESULTS -->"""
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    start = text.index("<!-- RESULTS")
    end = text.index("<!-- /RESULTS -->") + len("<!-- /RESULTS -->") if "<!-- /RESULTS -->" in text \
        else start + len("<!-- RESULTS -->")
    path.write_text(text[:start] + block + text[end:], encoding="utf-8")
    print("updated README.md results")


if __name__ == "__main__":
    main()
