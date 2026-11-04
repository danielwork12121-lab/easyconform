import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from easyconform import Conformalize, NotFittedError


class _ConstantModel:
    """A trivial estimator for hand-computed sanity checks: always predicts
    a fixed constant, regardless of X -- lets the calibration residuals (and
    therefore q_hat) be computed by hand rather than depending on any real
    model's own learned behavior."""

    def __init__(self, constant: float):
        self.constant = constant

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), self.constant, dtype=np.float64)


def test_alpha_must_be_in_open_unit_interval():
    with pytest.raises(ValueError):
        Conformalize(_ConstantModel(0.0), alpha=0.0)
    with pytest.raises(ValueError):
        Conformalize(_ConstantModel(0.0), alpha=1.0)
    with pytest.raises(ValueError):
        Conformalize(_ConstantModel(0.0), alpha=-0.1)


def test_predict_and_predict_interval_raise_before_fit():
    model = Conformalize(_ConstantModel(0.0))
    assert model.is_fitted is False
    with pytest.raises(NotFittedError):
        model.predict([[0.0]])
    with pytest.raises(NotFittedError):
        model.predict_interval([[0.0]])
    with pytest.raises(NotFittedError):
        _ = model.q_hat


def test_q_hat_matches_hand_computed_value_on_a_constant_model():
    """A constant-prediction model makes the calibration residuals exactly
    |y_cal - constant| -- fully hand-computable, so this checks the real
    finite-sample-corrected quantile formula itself, not a black box."""
    X_train = [[0.0]] * 4
    y_train = [0.0] * 4
    X_cal = [[0.0]] * 9
    # Predict constant=0 -> residuals are |y_cal| = [1,2,3,4,5,6,7,8,9].
    y_cal = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    model = Conformalize(_ConstantModel(0.0), alpha=0.1).fit(X_train, y_train, X_cal, y_cal)

    # n=9, alpha=0.1 -> level = ceil(10 * 0.9)/9 = ceil(9.0)/9 = 9/9 = 1.0
    # -> clipped to 1.0 -> take the max residual = 9.
    assert model.q_hat == pytest.approx(9.0)

    lower, upper = model.predict_interval([[0.0], [0.0]])
    assert lower == pytest.approx([-9.0, -9.0])
    assert upper == pytest.approx([9.0, 9.0])


def test_q_hat_at_a_looser_alpha_is_smaller_hand_computed():
    X_train = [[0.0]] * 4
    y_train = [0.0] * 4
    X_cal = [[0.0]] * 9
    y_cal = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # residuals from constant=0

    model = Conformalize(_ConstantModel(0.0), alpha=0.1).fit(X_train, y_train, X_cal, y_cal)

    # alpha=0.5 -> level = ceil(10*0.5)/9 = ceil(5.0)/9 = 5/9 ~= 0.5556
    # idx = ceil(0.5556*9)-1 = ceil(5.0)-1 = 4 -> sorted_residuals[4] = 5.
    lower, upper = model.predict_interval([[0.0]], alpha=0.5)
    assert (upper[0] - lower[0]) / 2.0 == pytest.approx(5.0)
    # A looser (larger) alpha must never produce a *wider* interval than a
    # stricter one from the same fitted calibration sample.
    assert (upper[0] - lower[0]) < 2 * model.q_hat


def test_predict_interval_default_alpha_matches_constructor_alpha():
    X_train = [[0.0]] * 4
    y_train = [0.0] * 4
    X_cal = [[0.0]] * 9
    y_cal = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    model = Conformalize(_ConstantModel(0.0), alpha=0.2).fit(X_train, y_train, X_cal, y_cal)
    lower_default, upper_default = model.predict_interval([[0.0]])
    lower_explicit, upper_explicit = model.predict_interval([[0.0]], alpha=0.2)
    assert lower_default == pytest.approx(lower_explicit)
    assert upper_default == pytest.approx(upper_explicit)


def test_predict_interval_rejects_invalid_override_alpha():
    X_train, y_train = [[0.0]], [0.0]
    X_cal, y_cal = [[0.0], [1.0]], [1.0, 2.0]
    model = Conformalize(_ConstantModel(0.0)).fit(X_train, y_train, X_cal, y_cal)
    with pytest.raises(ValueError):
        model.predict_interval([[0.0]], alpha=0.0)
    with pytest.raises(ValueError):
        model.predict_interval([[0.0]], alpha=1.5)


def test_predict_passes_through_to_the_wrapped_estimator():
    X_train, y_train = [[0.0], [1.0]], [10.0, 10.0]
    X_cal, y_cal = [[0.0], [1.0]], [10.0, 10.0]
    model = Conformalize(_ConstantModel(10.0)).fit(X_train, y_train, X_cal, y_cal)
    assert list(model.predict([[5.0], [6.0]])) == pytest.approx([10.0, 10.0])


def test_empty_calibration_set_raises():
    model = Conformalize(_ConstantModel(0.0))
    with pytest.raises(ValueError):
        model.fit([[0.0]], [0.0], X_cal=[], y_cal=[])


def test_real_linear_model_fits_and_produces_finite_intervals_on_diabetes():
    """Uses sklearn's real, bundled `diabetes` dataset (no network) with a
    real `LinearRegression` -- no mocking, a genuine sklearn estimator end
    to end."""
    data = load_diabetes()
    X_train, X_rest, y_train, y_rest = train_test_split(
        data.data, data.target, test_size=0.4, random_state=0
    )
    X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)

    model = Conformalize(LinearRegression(), alpha=0.1).fit(X_train, y_train, X_cal, y_cal)
    lower, upper = model.predict_interval(X_test)

    assert np.all(np.isfinite(lower))
    assert np.all(np.isfinite(upper))
    assert np.all(upper >= lower)
    assert model.q_hat > 0.0


def test_real_random_forest_empirical_coverage_on_real_california_housing_data():
    """The real, honest check this library's whole claim rests on: fit a
    real `RandomForestRegressor` on a real, public dataset (sklearn's own
    `fetch_california_housing`, ~20,640 real rows, no synthetic data), wrap
    it in `Conformalize` at alpha=0.1 (targeting ~90% coverage), and measure
    the *actual* empirical coverage on a real, held-out test split -- not
    just asserted, the same honesty discipline `conformal-forecast`'s own
    README uses for every coverage claim it makes.
    """
    data = fetch_california_housing()
    X_train, X_rest, y_train, y_rest = train_test_split(
        data.data, data.target, test_size=0.5, random_state=0
    )
    X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)

    alpha = 0.1
    model = Conformalize(
        RandomForestRegressor(n_estimators=50, max_depth=8, random_state=0, n_jobs=-1),
        alpha=alpha,
    ).fit(X_train, y_train, X_cal, y_cal)

    lower, upper = model.predict_interval(X_test)
    y_test = np.asarray(y_test)
    covered = (y_test >= lower) & (y_test <= upper)
    empirical_coverage = float(np.mean(covered))

    # Split conformal's real, finite-sample guarantee: empirical coverage
    # should be close to 1 - alpha (90%) on a real, large (~5,160-point)
    # held-out test split -- allow real sampling slack (a few points either
    # side) rather than demanding an exact match, which no real finite
    # sample would hit precisely.
    assert empirical_coverage >= 1.0 - alpha - 0.03
    # The interval should also never be absurdly loose -- a real sanity
    # check that this isn't trivially covering by being infinitely wide.
    assert np.median(upper - lower) < (y_test.max() - y_test.min())


def test_naive_uncalibrated_point_prediction_does_not_meet_the_real_coverage_target():
    """The direct 'before vs. after' comparison this library's README
    leads with: the *same* real model's own raw point prediction, with no
    conformal wrapper, checked against the real target the wrapped version
    is supposed to hit -- showing concretely what the wrapper actually
    buys, on real data, not a toy example."""
    data = fetch_california_housing()
    X_train, X_rest, y_train, y_rest = train_test_split(
        data.data, data.target, test_size=0.5, random_state=0
    )
    X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)

    raw_model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=0, n_jobs=-1)
    raw_model.fit(X_train, y_train)
    point_pred = raw_model.predict(X_test)
    y_test = np.asarray(y_test)

    # A raw point prediction "covers" only if it exactly equals the real
    # value -- essentially never, for a continuous real-valued target. This
    # is the honest reason an interval (not just a point estimate) is
    # needed at all, and why `Conformalize` exists.
    exact_hits = float(np.mean(np.isclose(point_pred, y_test, atol=1e-9)))
    assert exact_hits < 0.01
