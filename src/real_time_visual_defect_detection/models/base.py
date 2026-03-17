from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ModelOutput:
    score: float
    is_anomaly: bool


class BaseModel:
    def fit(
        self,
        train_paths: List[Path],
        fit_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Train the model on *train_paths*.

        Default implementation is a no-op for models that require no
        training (e.g. reference-free heuristics).  Override in
        subclasses that need a fitting step.
        """

    def predict(self, x: np.ndarray) -> ModelOutput:
        raise NotImplementedError

    def get_embedding(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Return a 1-D feature embedding for image *x*, or ``None``.

        Models that expose an internal representation (e.g. a bottleneck
        vector) should override this method and return a 1-D float32
        numpy array.  Models that do not support embeddings keep the
        default ``None`` return value; the pipeline will skip UMAP
        generation in that case.

        Parameters
        ----------
        x:
            Preprocessed image array as produced by the pipeline
            (BGR float32/64 in [0, 1]).

        Returns
        -------
        numpy.ndarray or None
            1-D float32 embedding vector, or ``None`` if unsupported.
        """
        return None

