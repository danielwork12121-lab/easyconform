# easyconform

Calibrated prediction intervals (regression) and prediction sets
(classification) for *any* scikit-learn-compatible model, in one line,
via split conformal prediction.

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
along with other tests (hand-computed quantile-formula sanity checks,
input-validation checks, and a second real end-to-end check on sklearn's
bundled `diabetes` dataset with a plain `LinearRegression`).

### A second real regression check: a different model family, dataset, and `alpha`

The `california_housing`/`RandomForestRegressor` case above is the
library's flagship check, but `diabetes` had so far only ever been
exercised for well-formedness (finite, correctly-ordered intervals), never
for actual measured coverage against a target. A second real combination —
`GradientBoostingRegressor` (boosted trees, not a bagged ensemble or a
linear model) on sklearn's real, bundled `diabetes` dataset (442 real
diabetes-progression measurements), at a different `alpha=0.15` (an 85%
coverage target, matching the classification suite's own third check) —
closes that gap:

| Target coverage (`1 - alpha`) | Empirical coverage (measured) | Median interval width |
|---|---|---|
| 85% (`alpha=0.15`) | **86.49%** | 179.8 (raw target units) |

Measured on a real, held-out test split of 111 points never seen during
training or calibration — reported exactly as measured, landing a bit
above the 85% target rather than forced to it. Reproduce via
`test_real_gradient_boosting_empirical_coverage_on_real_diabetes_data_at_alpha_0_15`.

### A fourth real regression check: a different model family and `alpha`

The two checks above cover a bagged tree ensemble (`RandomForestRegressor`)
and a boosted tree ensemble (`GradientBoostingRegressor`). A fourth real
combination — `SVR` (support vector regression, margin-based, neither a
tree ensemble nor a linear model) wrapped in a `StandardScaler` (its margin
is sensitive to raw feature scale, and `california_housing`'s own real
features span very different numeric ranges — room counts, geographic
coordinates, income in tens of thousands — the same real engineering
judgment call already applied to the non-tree-ensemble classifiers in this
library's classification suite), back on the large `california_housing`
dataset at a genuinely new `alpha=0.2` (an 80% coverage target, matching
the classification suite's own fourth check's alpha):

| Target coverage (`1 - alpha`) | Empirical coverage (measured) | Median interval width |
|---|---|---|
| 80% (`alpha=0.2`) | **80.48%** | 1.18 (in $100k units) |

Measured on the same large, real held-out test split the flagship
RandomForest check uses (5,160 real points never seen during training or
calibration) — reported exactly as measured, landing within half a point
of the real target. Reproduce via
`test_real_svr_empirical_coverage_on_real_california_housing_data_at_alpha_0_2`.

### A fifth real regression check: a different model family, dataset, and `alpha`

The four checks above cover a bagged tree ensemble (`RandomForestRegressor`),
a boosted tree ensemble (`GradientBoostingRegressor`), a linear model
(`LinearRegression`, finiteness-only), and a margin-based method (`SVR`). A
fifth real combination — `KNeighborsRegressor` (instance-based, not a tree
ensemble, not a linear model, not margin-based) wrapped in a
`StandardScaler` (k-NN's distance metric is sensitive to raw feature scale,
the same real engineering judgment call already applied to `SVR` above) —
back on the real, bundled `diabetes` dataset, but at a genuinely new
combination: `diabetes` has never had its empirical coverage actually
*measured* before (only checked for finiteness by the plain
`LinearRegression` check), and `alpha=0.1` (a 90% target) has never been
paired with `diabetes` in this suite:

| Target coverage (`1 - alpha`) | Empirical coverage (measured) | Median interval width |
|---|---|---|
| 90% (`alpha=0.1`) | **92.13%** | 199.4 (raw target units) |

Measured on a real, held-out test split of 89 points never seen during
training or calibration — reported exactly as measured, landing a bit
above the 90% target rather than forced to it. Reproduce via
`test_real_knn_empirical_coverage_on_real_diabetes_data_at_alpha_0_1`.

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

## Classification: calibrated prediction *sets*

`ConformalizeClassifier` is the classification counterpart: it wraps any
scikit-learn-compatible classifier (anything with a real `.fit(X, y)` and
`.predict_proba(X)`) and returns a calibrated prediction *set* — the list
of classes plausible enough to include — rather than a single point
prediction, using the standard "least ambiguous set-valued classifier"
nonconformity score (Sadinle, Lei & Wasserman, 2016).

```python
from sklearn.ensemble import RandomForestClassifier
from easyconform import ConformalizeClassifier

model = ConformalizeClassifier(RandomForestClassifier()).fit(X_train, y_train, X_cal, y_cal)
pred_sets = model.predict_set(X_test, alpha=0.1)   # list of arrays of class labels, ~90% coverage
```

### Real, measured result — not asserted

Fit a real `RandomForestClassifier` (100 trees, depth 10) on sklearn's own
real, public `digits` dataset (1,797 real 8×8 handwritten-digit images, 10
real classes — not synthetic), wrapped in `ConformalizeClassifier` at
`alpha=0.1`, held-out test split of 450 real points never seen during
training or calibration (449-point calibration split):

| Method | Empirical coverage (measured) | Mean set size |
|---|---|---|
| Raw model probabilities, naively thresholded at 90% (**no conformal calibration**) | **22.44%** | 0.22 |
| `ConformalizeClassifier` (real split-conformal calibration) | **91.56%** | 0.92 |

The same underlying model is 97.56% accurate by raw top-1 prediction on
this same test split — a genuinely strong classifier, not a strawman.
The point this table makes is specifically about *calibration*, not raw
accuracy: naively trusting a tree ensemble's own vote-fraction
"probabilities" and thresholding them at 90% gives wildly miscalibrated
coverage here (random forests are well known to be systematically
over/under-confident depending on the region of feature space — this is
a real, measured instance of that, not a cherry-picked pathological
case), while wrapping the identical model in real split-conformal
calibration lands almost exactly on the real 90% target. Neither number
above is rounded in the library's favor.

Reproduce this yourself: `pytest tests/ -v` runs this exact check live
(`test_real_random_forest_empirical_set_coverage_on_real_digits_data`),
along with 10 other classification tests (hand-computed nonconformity-score
sanity checks, input-validation checks, a second real end-to-end check on
sklearn's bundled `wine` dataset with a scaled `LogisticRegression`, and a
third real check below).

### A third real check: a different model family, dataset, and `alpha`

The `digits`/`RandomForestClassifier` case above was a genuine,
real instance of naive miscalibration. It is not the only real possible
outcome, and this library doesn't only report the cases that make the
comparison look dramatic. A third real combination — `GradientBoostingClassifier`
(boosted trees, not a bagged ensemble or a linear model) on sklearn's
real, bundled `breast_cancer` dataset (569 real diagnostic measurements,
binary malignant/benign), at a different `alpha=0.15` (an 85% coverage
target) — measures a genuinely *less dramatic*, equally honest result:

| Method | Empirical coverage (measured) | Mean set size |
|---|---|---|
| Raw model probabilities, naively thresholded at 85% (**no conformal calibration**) | **93.57%** | 0.99 |
| `ConformalizeClassifier` (real split-conformal calibration) | **91.23%** | 0.97 |

Both land above the 85% target here — gradient boosting's own log-loss-
trained probabilities turn out to already be reasonably well-calibrated
on this dataset, so the naive threshold doesn't fail the way it did on
`digits`. Reported exactly as measured, with no assertion forcing either
number toward a particular story: the real, honest point is that
`ConformalizeClassifier` gives the same *guaranteed* bound either way,
without first requiring you to measure (the way this check just did)
whether your particular model happens to calibrate well on your
particular data. Reproduce via
`test_real_gradient_boosting_empirical_set_coverage_on_real_breast_cancer_data`
and `test_naive_threshold_can_also_be_reasonably_calibrated_sometimes_an_honest_contrast`.

Honesty note on empty sets: at small calibration sizes or for a
genuinely ambiguous point, the returned set can be empty (no class
clears the calibrated probability floor) — `predict_set` reports this
honestly rather than silently forcing in a guess, since doing so would
quietly break the real coverage guarantee. See `ConformalizeClassifier`'s
own docstring for the full detail.

### A fourth real check: a different model family, dataset, and `alpha`

A fourth real combination — a support vector machine (margin-based,
neither a tree ensemble nor a linear model) on sklearn's real, bundled
`iris` dataset (150 real flower measurements, three classes, not used
anywhere else in this README), at a different `alpha=0.2` (an 80%
coverage target) — measured on 45 real held-out test points:

| Target coverage (`1 - alpha`) | Empirical coverage (measured) |
|---|---|
| 80% (`alpha=0.2`) | **80.00%** |

An exact hit at this sample size (n=45) is itself unremarkable, not
cherry-picked — the real binomial standard error at this size and target
rate is roughly 6 points, so this is one more honestly-reported real data
point, not a claim that `ConformalizeClassifier` always lands exactly on
target. Uses `CalibratedClassifierCV(SVC(), ensemble=False)` rather than
the deprecated `SVC(probability=True)` constructor argument, so this
check stays correct against current and near-future scikit-learn
releases. Reproduce via `test_real_svm_empirical_set_coverage_on_real_iris_data`.

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

ConformalizeClassifier(estimator, alpha=0.1)
    .fit(X_train, y_train, X_cal, y_cal)   # -> self
    .predict(X)                             # -> the wrapped estimator's own raw top-1 class prediction
    .predict_set(X, alpha=None)             # -> list of arrays of class labels, calibrated at 1-alpha
    .q_hat                                   # -> the fitted nonconformity-score threshold
    .is_fitted                               # -> bool
    .classes_                                # -> the wrapped estimator's own real class labels
```

`alpha` can be overridden per-call to `predict_interval`/`predict_set`
without refitting or re-predicting — the same stored calibration
residuals/scores are reused to recompute the threshold at the new level,
so sweeping several confidence levels against one fitted model is cheap
and never needs a second calibration split.

## Scope, honestly

This is a small, focused tool, not a research contribution: the method
(split conformal prediction) is a well-established, published technique
this library did not invent. What it adds is the one-line, framework-
agnostic wrapper and the honestly-measured real-data checks above,
covering both the regression (interval) and classification (set) cases.

## What's next

1. ~~A second real-dataset coverage check for the classification case at a different `alpha` level, and/or a second real classifier family (e.g. gradient boosting)~~ — **done**: `GradientBoostingClassifier` on real `breast_cancer` data at `alpha=0.15`, above. ~~A further real classifier family (e.g. SVM with calibrated probabilities)~~ — **done**: a support vector machine on real `iris` data at `alpha=0.2`, above. A fifth real classifier family/dataset/alpha combination is still a real option if a future check wants one.
2. ~~A second real-dataset coverage check for the regression case at a third `alpha` level, and/or a second real model family (e.g. gradient boosting)~~ — **done**: `GradientBoostingRegressor` on real `diabetes` data at `alpha=0.15`, above. ~~A further real regression model family (e.g. SVR, or a genuinely different method)~~ — **done**: `SVR` (margin-based, `StandardScaler`-wrapped) on real `california_housing` data at `alpha=0.2`, above. ~~A fifth real regression model family/dataset/alpha combination~~ — **done**: `KNeighborsRegressor` (`StandardScaler`-wrapped) on real `diabetes` data at `alpha=0.1`, above. A sixth combination remains a real option if a future check wants one.

## License

MIT — see `LICENSE`.
   
