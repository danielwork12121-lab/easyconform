"""easyconform: calibrated prediction intervals (regression) and prediction
sets (classification) for any scikit-learn-compatible model, in one line,
via split conformal prediction.

See `Conformalize`'s own docstring for the regression interface and
`ConformalizeClassifier`'s own docstring for the classification interface
-- both state the same honest exchangeability-assumption caveat.
"""

from easyconform._conformalize import Conformalize, NotFittedError
from easyconform._conformalize_classifier import ConformalizeClassifier

__all__ = ["Conformalize", "ConformalizeClassifier", "NotFittedError"]
__version__ = "0.2.9"
