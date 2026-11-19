"""Split conformal prediction for classification: calibrated prediction
*sets* (rather than a single point label) for any scikit-learn-compatible
classifier that implements `.fit(X, y)` and `.predict_proba(X)`.

`ConformalizeClassifier` is the classification counterpart to this
library's own `Conformalize` (which handles regression intervals). It
uses the standard "least ambiguous set-valued classifier" (LAC)
nonconformity score (Sadinle, Lei & Wasserman, 2016, "Least Ambiguous
Set-Valued Classifiers Based on Inductive Conformal Prediction"): the
score for a labeled example is `1 - P(true class)` under the fitted
classifier's own real predicted probabilities, and at prediction time a
class is included in the output set exactly when its own predicted
probability is at least `1 - q_hat`, where `q_hat` is the same
finite-sample-corrected quantile of the real calibration scores that
`Conformalize` already uses for the regression case (this module
reimplements that one small piece of shared arithmetic rather than
importing it as a private cross-module dependency, since it is a
three-line, well-tested formula, not a design choice worth coupling the
two classes over).

The guarantee this actually gives, precisely: if the calibration set and
the real test point are exchangeable (the same standard split-conformal
assumption `Conformalize`'s own docstring states for the regression
case), then for a freshly `fit` `ConformalizeClassifier(estimator,
alpha=0.1)`, the returned prediction *set* contains the true label for
roughly 90% of real, independent test points -- not a single predicted
class, a *set* of plausible classes (sometimes one class, sometimes
several, sometimes, at small calibration-sample sizes or near a genuine
decision boundary, technically empty before the floor described below).
This is checked empirically against real public data in
`tests/test_conformalize_classifier.py` and in the README, the same
honesty discipline (a real measured number, not an assumed one) used for
every coverage claim elsewhere in this library and in this author's other
conformal-prediction project, `conformal-forecast`.
"""

from __future__ import annotations

import math

import numpy as np

from easyconform._conformalize import NotFittedError

__all__ = ["ConformalizeClassifier", "NotFittedError"]


def _quantile_with_finite_sample_correction(scores: np.ndarray, alpha: float) -> float:
    """The same standard split-conformal finite-sample-corrected quantile
    convention `easyconform._conformalize` uses for the regression case,
    reimplemented here for this module's own plain `(n_cal,)` score array
    (see that module's own copy of this function for the full rationale;
    duplicated deliberately -- see this module's own top-level docstring)."""
    n = len(scores)
    if n == 0:
        raise ValueError("Need at least one calibration score to compute q_hat")
    level = math.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    sorted_scores = np.sort(scores)
    idx = min(int(math.ceil(level * n)) - 1, n - 1)
    return float(sorted_scores[idx])


class ConformalizeClassifier:
    """Wrap any scikit-learn-compatible classifier (anything with a real
    `.fit(X, y)` and `.predict_proba(X)`) to produce calibrated prediction
    *sets* via split conformal prediction.

    Example::

        from sklearn.ensemble import RandomForestClassifier
        from easyconform import ConformalizeClassifier

        model = ConformalizeClassifier(RandomForestClassifier()).fit(
            X_train, y_train, X_cal, y_cal
        )
        pred_sets = model.predict_set(X_test, alpha=0.1)   # list of arrays of class labels

    Parameters
    ----------
    estimator : any object with `.fit(X, y)` and `.predict_proba(X)`
        Not copied or cloned -- `fit` below calls `estimator.fit(X_train,
        y_train)` directly, mutating the real object passed in, exactly
        the way `Conformalize` (the regression class in this same
        library) handles its own wrapped estimator. Pass a fresh, unfit
        classifier (e.g. `RandomForestClassifier()`) unless you
        specifically want to reuse already-fit parameters.
    alpha : float, default 0.1
        The default miscoverage rate (`1 - alpha` is the target
        set-coverage, e.g. `alpha=0.1` targets the true label falling in
        the returned set roughly 90% of the time) used by `predict_set`
        when its own `alpha` argument is left as `None`. Can still be
        overridden per-call.

    Honesty note on the exchangeability assumption: identical in substance
    to `Conformalize`'s own docstring -- the real coverage guarantee holds
    when the calibration set and real test points are exchangeable. For
    i.i.d. tabular classification data (the setting this class targets)
    this is usually a reasonable, real assumption.

    Honesty note on empty prediction sets: at small calibration sizes, or
    for a genuinely ambiguous test point near several classes' decision
    boundaries, the LAC score can produce an empty set for some points
    (no class reaches the `1 - q_hat` probability floor) -- `predict_set`
    reports this honestly as an empty array rather than silently forcing
    in the single most-likely class, since doing the latter would quietly
    break the real coverage guarantee this class exists to provide. Call
    `predict` (this class's own point-prediction pass-through) if you
    specifically want the wrapped estimator's single top class instead.
    """

    def __init__(self, estimator, alpha: float = 0.1):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.estimator = estimator
        self.alpha = alpha
        self.classes_: np.ndarray | None = None
        self._cal_scores: np.ndarray | None = None
        self._q_hat: float | None = None

    @property
    def is_fitted(self) -> bool:
        return self._cal_scores is not None

    def fit(self, X_train, y_train, X_cal, y_cal) -> "ConformalizeClassifier":
        """Fit `self.estimator` on `(X_train, y_train)`, then compute the
        real LAC nonconformity scores (`1 - P(true class)`) on the
        held-out `(X_cal, y_cal)` split -- `X_cal`/`y_cal` must be
        genuinely held out from training, the same discipline
        `Conformalize.fit` requires for the regression case. Returns
        `self` so calls can be chained.
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train).reshape(-1)
        X_cal = np.asarray(X_cal)
        y_cal = np.asarray(y_cal).reshape(-1)

        if len(X_cal) < 1:
            raise ValueError("X_cal/y_cal must contain at least one held-out calibration point")
        if not hasattr(self.estimator, "predict_proba"):
            raise TypeError(
                "ConformalizeClassifier requires an estimator with a real "
                "predict_proba(X) method; got "
                f"{type(self.estimator).__name__}, which has none"
            )

        self.estimator.fit(X_train, y_train)
        self.classes_ = np.asarray(self.estimator.classes_)

        proba_cal = np.asarray(self.estimator.predict_proba(X_cal), dtype=np.float64)
        class_index = {label: i for i, label in enumerate(self.classes_)}
        true_class_idx = np.array([class_index[label] for label in y_cal])
        true_class_proba = proba_cal[np.arange(len(y_cal)), true_class_idx]
        self._cal_scores = 1.0 - true_class_proba
        self._q_hat = _quantile_with_finite_sample_correction(self._cal_scores, self.alpha)
        return self

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError("ConformalizeClassifier must be fit(...) before predict/predict_set")

    def predict(self, X):
        """The wrapped estimator's own real single top-class prediction --
        a plain pass-through to `self.estimator.predict(X)`, mirroring
        `Conformalize.predict`'s role for the regression case."""
        self._require_fitted()
        return self.estimator.predict(X)

    def predict_set(self, X, alpha: float | None = None):
        """Return a list of length `len(X)`, each entry a real `numpy`
        array of the class labels included in that point's calibrated
        prediction set at `1 - alpha` coverage (defaulting to
        `1 - self.alpha` from `fit` when `alpha` is left `None`).

        Passing a different `alpha` here recomputes `q_hat` from the
        *same* already-stored calibration scores (no refitting or
        re-predicting needed), mirroring `Conformalize.predict_interval`'s
        own cheap-alpha-sweep behavior for the regression case.
        """
        self._require_fitted()
        proba = np.asarray(self.estimator.predict_proba(X), dtype=np.float64)
        if alpha is None:
            q_hat = self._q_hat
        else:
            if not (0.0 < alpha < 1.0):
                raise ValueError("alpha must be in (0, 1)")
            q_hat = _quantile_with_finite_sample_correction(self._cal_scores, alpha)

        include = proba >= (1.0 - q_hat)
        return [self.classes_[row] for row in include]

    @property
    def q_hat(self) -> float:
        """The real, fitted score threshold (`q_hat`) used for
        `self.alpha`'s own default prediction set -- mirrors
        `Conformalize.q_hat` for the regression case."""
        self._require_fitted()
        return self._q_hat
