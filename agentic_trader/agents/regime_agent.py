"""RegimeAgent — classifies market regime from mid-frequency features.

Current implementation: rule-based classifier.  The class is designed so that
`classify()` can be replaced by an ML model (gradient boosted trees, HMM, etc.)
without changing any caller code — just swap the implementation of `_predict`.

Feature vector (from TradingEnv.get_regime_features()):
    [rolling_vol, mean_return, abs_inventory_fraction, spread_norm, step_frac]
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime taxonomy
# ---------------------------------------------------------------------------


class MarketRegime(str, Enum):
    """Discrete market regime labels."""

    CALM = "CALM"
    HIGH_VOL = "HIGH_VOL"
    CRISIS = "CRISIS"


# ---------------------------------------------------------------------------
# RegimeAgent
# ---------------------------------------------------------------------------


class RegimeAgent:
    """Classifies the current market regime from a feature vector.

    Args:
        vol_high_threshold: rolling_vol (normalised) above this → HIGH_VOL.
        vol_crisis_threshold: rolling_vol above this → CRISIS.
        smoothing: exponential smoothing factor for the volatility signal (0 = off).

    Feature indices (matching TradingEnv.get_regime_features):
        0 → rolling_vol  (std of recent log-returns)
        1 → mean_return
        2 → abs_inventory_fraction
        3 → spread_norm
        4 → step_frac
    """

    VOL_IDX = 0
    MEAN_RET_IDX = 1
    SPREAD_IDX = 3

    def __init__(
        self,
        vol_high_threshold: float = 0.010,
        vol_crisis_threshold: float = 0.018,
        smoothing: float = 0.4,
    ):
        self.vol_high_threshold = vol_high_threshold
        self.vol_crisis_threshold = vol_crisis_threshold
        self.smoothing = smoothing
        self._smoothed_vol: float = 0.0

    def classify(self, features: np.ndarray) -> MarketRegime:
        """Return the regime label for the current feature vector."""
        features = np.asarray(features, dtype=float).flatten()
        regime = self._predict(features)
        logger.debug(
            "RegimeAgent: vol=%.5f → %s",
            features[self.VOL_IDX] if len(features) > self.VOL_IDX else 0,
            regime.value,
        )
        return regime

    def _predict(self, features: np.ndarray) -> MarketRegime:
        """Rule-based classifier.  Replace with ML model as needed.

        Rules (in priority order):
            1. Smoothed vol > crisis_threshold            → CRISIS
            2. Smoothed vol > high_threshold             → HIGH_VOL
            3. Otherwise                                 → CALM
        """
        raw_vol = float(features[self.VOL_IDX]) if len(features) > self.VOL_IDX else 0.0
        self._smoothed_vol = (
            self.smoothing * raw_vol + (1.0 - self.smoothing) * self._smoothed_vol
        )

        if self._smoothed_vol >= self.vol_crisis_threshold:
            return MarketRegime.CRISIS
        if self._smoothed_vol >= self.vol_high_threshold:
            return MarketRegime.HIGH_VOL
        return MarketRegime.CALM

    def reset(self) -> None:
        """Reset internal smoothing state at the start of a new episode."""
        self._smoothed_vol = 0.0

    # ------------------------------------------------------------------
    # ML upgrade path
    # ------------------------------------------------------------------

    @classmethod
    def from_sklearn_model(cls, model: object) -> "MLRegimeAgent":
        """Wrap a trained sklearn-compatible classifier as a RegimeAgent."""
        return MLRegimeAgent(model)


class MLRegimeAgent(RegimeAgent):
    """RegimeAgent backed by an sklearn-compatible classifier.

    The model must implement `predict(X)` returning one of the MarketRegime
    string values, or integer labels 0/1/2 mapped to CALM/HIGH_VOL/CRISIS.
    """

    _LABEL_MAP = {0: MarketRegime.CALM, 1: MarketRegime.HIGH_VOL, 2: MarketRegime.CRISIS}

    def __init__(self, model: object):
        super().__init__()
        self._model = model

    def _predict(self, features: np.ndarray) -> MarketRegime:
        result = self._model.predict(features.reshape(1, -1))[0]  # type: ignore[attr-defined]
        if isinstance(result, (int, np.integer)):
            return self._LABEL_MAP.get(int(result), MarketRegime.CALM)
        return MarketRegime(str(result))
