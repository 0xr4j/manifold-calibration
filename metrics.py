"""Scoring and calibration statistics for probability forecasts.

p is an array of forecast probabilities for YES, y is an array of outcomes (1 = YES, 0 = NO).
Everything here is plain numpy so each formula is easy to check by hand.
"""

import numpy as np

# Finer bins at the tails, where favourite-longshot effects show up.
BINS = np.array([0, .02, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95, .98, 1.0])
EPS = 0.005  # clip prices before taking logs; Manifold prices sit between 1% and 99%


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def log_loss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z ** 2 / n
    centre = (phat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
    return (centre - half, centre + half)


def calibration_table(p, y, bins=BINS):
    """For each price bin: count, mean price, observed YES rate and its 95% interval."""
    idx = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
    rows = []
    for b in range(len(bins) - 1):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        k = int(y[mask].sum())
        lo, hi = wilson(k, n)
        rows.append({
            "bin_low": float(bins[b]), "bin_high": float(bins[b + 1]), "n": n,
            "mean_price": float(p[mask].mean()), "yes_rate": k / n, "ci_low": lo, "ci_high": hi,
        })
    return rows


def murphy_decomposition(p, y, bins=BINS):
    """Brier ~= reliability - resolution + uncertainty (Murphy, 1973), using price bins.

    Binning leaves a small within-bin term, so the three parts only approximately add up to
    the Brier score.
    """
    idx = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
    base = y.mean()
    rel = res = 0.0
    for b in np.unique(idx):
        mask = idx == b
        w = mask.mean()
        rel += w * (p[mask].mean() - y[mask].mean()) ** 2
        res += w * (y[mask].mean() - base) ** 2
    return {"reliability": float(rel), "resolution": float(res), "uncertainty": float(base * (1 - base))}


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_fit(p, y, iters=50):
    """Logistic regression of the outcome on logit(price): y ~ intercept + slope * logit(p).

    A perfectly calibrated market has intercept 0 and slope 1. A slope above 1 means prices are
    too timid (outcomes are more extreme than prices suggest); below 1 means prices are too extreme.
    Fitted with Newton's method.
    """
    X = np.column_stack([np.ones_like(p), logit(p)])
    beta = np.array([0.0, 1.0])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-X @ beta))
        w = mu * (1 - mu)
        grad = X.T @ (y - mu)
        # A tiny ridge term keeps the fit stable when a small resample is perfectly separable.
        hess = X.T @ (X * w[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            break
    return {"intercept": float(beta[0]), "slope": float(beta[1])}


def bootstrap_fit(p, y, n_boot=1000, seed=0):
    """95% intervals for the calibration intercept and slope, resampling markets."""
    rng = np.random.default_rng(seed)
    n = len(p)
    fits = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        fits.append(calibration_fit(p[i], y[i]))
    out = {}
    for key in ("intercept", "slope"):
        vals = np.array([f[key] for f in fits])
        out[key + "_ci"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    return out


def favourite_table(p, y, bins=(.5, .6, .7, .8, .9, .95, .98, 1.0)):
    """Fold every market onto its favourite: price of the likelier side, and whether it won."""
    fav_price = np.maximum(p, 1 - p)
    fav_won = np.where(p >= 0.5, y, 1 - y)
    return calibration_table(fav_price, fav_won, bins=np.array(bins))
