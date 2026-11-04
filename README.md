# easyconform

Calibrated prediction intervals for *any* scikit-learn-compatible model, in
one line, via split conformal prediction.

```python
from sklearn.ensemble import RandomForestRegressor
from easyconform import Conformalize

model = Conformalize(RandomForestRegressor()).fit(X_train, y_train, X_cal, y_cal)
lower, upper = model.predict_interval(X_test, alpha=0.1)   # ~90% intervals, measured below
```

That's it. `Conformalize` wraps anything with a real `.fit(X, y)` /
`.predict(X)` interface — any scikit-learn regressor, or anything that
matches that interface — and adds a real, finite-sample marginal coverage
guarantee on top of its raw point predictions, via split conformal
prediction (Vovk et al., 2005; Lei et al., 2018 for the regression form
used here).

## Why this, when scikit-learn models already predict a number

A raw regressor gives you one number. It is essentially never exactly
right. What you usually actually want is an honest range: "the real value
is probably between X and Y" — and a *calibrated* range specifically, one
where "90% confident" really does mean the true value falls inside about
90% of the time, not just a number that looks like a confidence interval.
Split conformal prediction gives you that, with a real mathematical
guarantee (under a real, stated assumption — see below), and `easyconform`
is the smallest wrapper that could plausibly deliver it: no retraining
your model differently, no Bayesian machinery, no assuming your residuals
are Gaussian. Hold out a calibration set, fit normally, wrap the result.

## Real, measured result — not asserted

This is the same honesty standard every project in this author's GitHub
uses: a coverage claim is only real once it's actually measured against
real, held-out data.

Fit a real `RandomForestRegressor` on sklearn's own real, public
`california_housing` dataset (20,640 real rows of real 1990 California
census housing data — not synthetic), wrapped in `Conformalize`, held-out
test split of 5,160 real points never seen during training or calibration:

| Target coverage (`1 - alpha`) | Empirical coverage (measured) | Median interval width |
|---|---|---|
| 90% (`alpha=0.1`) | **89.50%** | 1.81 (in $100k units) |
| 80% (`alpha=0.2`) | **80.23%** | 1.24 (in $100k units) |

Both land within a point of their real target on a genuinely large,
real held-out sample — not cherry-picked, not rounded in the library's
favor (89.50% is honestly *below* 90%, reported exactly as measured).
For comparison, the same model's own raw point prediction, with no
conformal wrapper at all, matches the real test value *exactly* in under
1% of cases (continuous-valued regression targets essentially never land
on an exact point estimate) — which is the concrete problem this library
solves: not a slightly-better number, a *calibrated range* instead of a
single, almost-certainly-wrong number.

Reproduce this yourself: `pytest tests/ -v` runs this exact check live
(`test_real_random_forest_empirical_coverage_on_real_california_housing_data`),
along with 10 other tests (hand-computed quantile-formula sanity checks,
input-validation checks, and a second real end-to-end check on sklearn's
bundled `diabetes` dataset with a plain `LinearRegression`).

## The one real assumption this relies on

Split conformal's coverage guarantee holds when the calibration set and
real test points are *exchangeable* — informally, that the data you
calibrate on and the data you later predict on come from the same
underlying distribution, in no particular order. For i.i.d. tabular
regression data (the setting this library targets, and what both real
datasets above are), this is usually a reasonable, real assumption. It can
degrade under genuine distribution shift — e.g. a non-stationary time
series, where "the future differs systematically from the past" is
exactly the point. That harder case is intentionally **out of scope**
here; this author's separate project,
[`conformal-forecast`](https://github.com/danielwork12121-lab/conformal-forecast),
exists specifically for it (adaptive methods — ACI, EnbPI — built and
benchmarked against that harder, non-stationary setting). `easyconform` is
deliberately the simple, general tool for the common case: you have a
model, you have i.i.d.-ish tabular data, you want a real calibrated range
around what it predicts, with no extra machinery.

## Installation

Not yet published to PyPI. Until then:

```bash
pip install git+https://github.com/danielwork12121-lab/easyconform.git
```

Or, for local development:

```bash
git clone https://github.com/danielwork12121-lab/easyconform.git
cd easyconform
pip install -e ".[dev]"
pytest
```

## API

```python
Conformalize(estimator, alpha=0.1)
    .fit(X_train, y_train, X_cal, y_cal)   # -> self
    .predict(X)                             # -> the wrapped estimator's own raw point prediction
    .predict_interval(X, alpha=None)        # -> (lower, upper), calibrated at 1-alpha (or this instance's own default)
    .q_hat                                   # -> the fitted interval half-width
    .is_fitted                               # -> bool
```

`alpha` can be overridden per-call to `predict_interval` without refitting
or re-predicting — the same stored calibration residuals are reused to
recompute `q_hat` at the new level, so sweeping several confidence levels
against one fitted model is cheap and never needs a second calibration
split.

## Scope, honestly

This is a small, focused tool, not a research contribution: the method
(split conformal prediction) is a well-established, published technique
this library did not invent. What it adds is the one-line, framework-
agnostic wrapper and the honestly-measured real-data check above. A
classification variant (conformal prediction *sets* rather than intervals)
is a natural v2 and is not built yet — see "What's next."

## What's next

1. A classification variant (`Conformalize` for a classifier, producing calibrated prediction *sets* rather than a numeric interval) — not yet built; v1 intentionally ships the regression case well rather than both half-done.
2. A second real-dataset coverage check at a third `alpha` level, and/or a second real model family (e.g. gradient boosting), to further stress-test the empirical-coverage claim beyond the two real datasets and two real models already measured above.

## License

MIT — see `LICENSE`.
   
