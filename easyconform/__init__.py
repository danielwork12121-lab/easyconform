"""easyconform: calibrated prediction intervals for any scikit-learn-compatible
regressor, in one line, via split conformal prediction.

See `Conformalize`'s own docstring for the full interface and the honest
exchangeability-assumption caveat.
"""

from easyconform._conformalize import Conformalize, NotFittedError

__all__ = ["Conformalize", "NotFittedError"]
__version__ = "0.1.0"
