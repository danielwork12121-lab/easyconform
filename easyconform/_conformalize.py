"""Split conformal prediction for any scikit-learn-compatible estimator.

`Conformalize` wraps an estimator that already implements `.fit(X, y)` and
`.predict(X)` (any real scikit-learn regressor, or anything that matches
that interface) and adds a real, finite-sample marginal coverage guarantee
on top of it, via split conformal prediction (Vovk et al., 2005; Lei et
al., 2018 for the regression form used here) -- the same general method
Daniel's own `conformal-forecast` project (https://github.com/danielwork12121-lab/conformal-forecast)
implements for time-series point forecasters specifically. This module is
an independent, from-scratch implementation for the general
`.fit`/`.predict` regressor case (a genuinely different interface --
arbitrary feature matrices, not time-series lookback windows -- so it is
not literally shared code), using the identical published method and the
identical finite-sample quantile correction convention, for anyone who
just wants calibrated intervals around a model they already have, with no
time-series machinery required.

The guarantee this actually gives, precisely: if the calibration set
(`X_cal`, `y_cal`) and the test point are exchangeable (the standard, real
assumption behind split conformal prediction -- see the class docstring
below), then for a freshly `fit` `Conformalize(estimator, alpha=0.1)`,
roughly 90% of real, independent test points will have their true `y`
fall inside the returned interval. This is checked empirically against
real public data and real held-out coverage in `tests/test_conformalize.py`
and in the README -- not just asserted.
"""

from __future__ import annotations

import math

import numpy as np


class NotFittedError(RuntimeError):
    """Raised when `predict`/`predict_interval` is called before `fit`."""


def _quantile_with_finite_sample_correction(residuals: np.ndarray, alpha: float) -> float:
    """The standard split-conformal finite-sample-corrected quantile of
    `residuals`: the `ceil((n + 1) * (1 - alpha)) / n` empirical quantile,
    clipped to 1.0 (i.e. "take the max residual") when the calibration set
    is small enough that the raw formula would ask for a quantile level
    above 1 -- the same convention `conformal-forecast`'s own
    `SplitConformalForecaster._quantile_with_finite_sample_correction`
    uses, reimplemented here independently for this project's own, plain
    `(n_cal,)` residual array (no time-series windowing involved).
    """
    n = len(residuals)
    if n == 0:
        raise ValueError("Need at least one calibration residual to compute q_hat")
    level = math.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    sorted_residuals = np.sort(residuals)
    idx = min(int(math.ceil(level * n)) - 1, n - 1)
    return float(sorted_residuals[idx])


class Conformalize:
    """Wrap any scikit-learn-compatible regressor to produce calibrated
    prediction intervals via split conformal prediction.

    Example::

        from sklearn.ensemble import RandomForestRegressor
        from easyconform import Conformalize

        model = Conformalize(RandomForestRegressor()).fit(X_train, y_train, X_cal, y_cal)
        lower, upper = model.predict_interval(X_test, alpha=0.1)   # ~90% intervals

    Parameters
    ----------
    estimator : any object with `.fit(X, y)` and `.predict(X)`
        Not copied or cloned -- `fit` below calls `estimator.fit(X_train,
        y_train)` directly, mutating the real object passed in, exactly
        the way a plain scikit-learn estimator's own `.fit` would. Pass a
        fresh, unfit estimator (e.g. `RandomForestRegressor()`, not one
        you've already called `.fit` on with different data) unless you
        specifically want to reuse already-fit parameters.
    alpha : float, default 0.1
        The default miscoverage rate (`1 - alpha` is the target coverage,
        e.g. `alpha=0.1` targets ~90% coverage) used by `predict_interval`
        when its own `alpha` argument is left as `None`. Can still be
        overridden per-call.

    Honesty note on the exchangeability assumption, stated plainly (the
    same discipline `conformal-forecast`'s own README uses): split
    conformal's real coverage guarantee holds under the assumption that the
    calibration set and the real test points are exchangeable with each
    other. For i.i.d. data (the classic regression setting this class
    targets) this is usually a reasonable, real assumption. For data with
    real distribution shift between calibration and test time (e.g. a
    genuinely non-stationary time series), the guarantee can and does
    degrade -- which is exactly why `conformal-forecast` exists as a
    *separate*, time-series-specific project with its own adaptive
    (ACI/EnbPI) methods for that harder case, rather than this class
    claiming to solve it too.
    """

    def __init__(self, estimator, alpha: float = 0.1):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.estimator = estimator
        self.alpha = alpha
        self._cal_residuals: np.ndarray | None = None
        self._q_hat: float | None = None

    @property
    def is_fitted(self) -> bool:
        return self._cal_residuals is not None

    def fit(self, X_train, y_train, X_cal, y_cal) -> "Conformalize":
        """Fit `self.estimator` on `(X_train, y_train)`, then compute the
        real calibration residuals on the held-out `(X_cal, y_cal)` split
        -- `X_cal`/`y_cal` must be genuinely held out from training (never
        seen by `self.estimator.fit`), the same discipline any split
        conformal method requires to be honest. Returns `self` so calls can
        be chained, e.g. `Conformalize(model).fit(...).predict_interval(...)`.
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train).reshape(-1)
        X_cal = np.asarray(X_cal)
        y_cal = np.asarray(y_cal).reshape(-1)

        if len(X_cal) < 1:
            raise ValueError("X_cal/y_cal must contain at least one held-out calibration point")

        self.estimator.fit(X_train, y_train)
        cal_pred = np.asarray(self.estimator.predict(X_cal), dtype=np.float64).reshape(-1)
        self._cal_residuals = np.abs(cal_pred - y_cal)
        self._q_hat = _quantile_with_finite_sample_correction(self._cal_residuals, self.alpha)
        return self

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError("Conformalize must be fit(...) before predict/predict_interval")

    def predict(self, X):
        """The wrapped estimator's own real point prediction -- a plain
        pass-through to `self.estimator.predict(X)`, for convenience (so
        `Conformalize` can be used as a drop-in point-predictor too, not
        only for intervals)."""
        self._require_fitted()
        return self.estimator.predict(X)

    def predict_interval(self, X, alpha: float | None = None):
        """Return `(lower, upper)`, each a real `numpy` array the same
        length as `X`, forming a calibrated prediction interval at
        `1 - alpha` coverage (defaulting to `1 - self.alpha` from `fit`,
        i.e. this instance's own default, when `alpha` is left `None`).

        Passing a different `alpha` here recomputes `q_hat` from the
        *same* already-stored calibration residuals (no refitting or
        re-predicting needed) -- so sweeping several `alpha` values against
        one fitted `Conformalize` instance is cheap, and always honest: it
        still only ever uses the one real, held-out calibration sample from
        `fit`, never a different one.
        """
        self._require_fitted()
        point = np.asarray(self.estimator.predict(X), dtype=np.float64).reshape(-1)
        if alpha is None:
            q_hat = self._q_hat
        else:
            if not (0.0 < alpha < 1.0):
                raise ValueError("alpha must be in (0, 1)")
            q_hat = _quantile_with_finite_sample_correction(self._cal_residuals, alpha)
        lower = point - q_hat
        upper = point + q_hat
        return lower, upper

    @property
    def q_hat(self) -> float:
        """The real, fitted half-width (`q_hat`) used for `self.alpha`'s
        own default interval -- exposed for inspection/debugging, mirroring
        `conformal-forecast`'s own `ConformalPrediction.q_hat` field."""
        self._require_fitted()
        return self._q_hat
