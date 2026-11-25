"""BPI process profile lenses; all event-log semantics stay in this example."""

from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from reflex.core.models.features import EventWindowFeatures

FEATURES = ("event_count", "duration_hours", "activity_count", "resource_count",
            "offer_count", "rework_count")


def _matrix(rows, centers, scales):
    return np.asarray([[(row.feature_values[name] - centers[i]) / scales[i]
                        for i, name in enumerate(FEATURES)] for row in rows], dtype=float)


def _robust_stats(rows):
    centers = [median(row.feature_values[name] for row in rows) for name in FEATURES]
    scales = [1.4826 * median(abs(row.feature_values[name] - centers[i]) for row in rows)
              for i, name in enumerate(FEATURES)]
    return centers, [scale if scale > 0 else 1.0 for scale in scales]


class ProcessProfileCandidate:
    """Pooled robust profile or a robustly scaled KMeans peer profile."""

    lens_id = "bpi2017.application-process"
    feature_names = FEATURES
    metric_name = "median_normalized_reconstruction_error"
    provider = "examples.reflex-builder.bpi2017"
    provider_version = "1"

    def __init__(self, *, clusters: int):
        self.clusters = clusters
        self.version = "clustered-v1" if clusters else "pooled-v1"

    def fit(self, rows: Sequence[EventWindowFeatures]) -> dict[str, Any]:
        centers, scales = _robust_stats(rows)
        values = _matrix(rows, centers, scales)
        model: dict[str, Any] = {"centers": centers, "scales": scales, "cluster_centers": []}
        if self.clusters:
            cluster_count = min(self.clusters, max(2, len(rows) // 1000))
            fitted = KMeans(n_clusters=cluster_count, n_init=10, random_state=17).fit(values)
            model["cluster_centers"] = fitted.cluster_centers_.tolist()
        return model

    def _residual(self, model: Mapping[str, Any], row: EventWindowFeatures):
        values = np.asarray([(row.feature_values[name] - model["centers"][i]) / model["scales"][i]
                             for i, name in enumerate(FEATURES)], dtype=float)
        if not model["cluster_centers"]:
            return values, "pooled"
        centers = np.asarray(model["cluster_centers"], dtype=float)
        distances = np.mean(np.abs(centers - values), axis=1)
        nearest = int(np.argmin(distances))
        return values - centers[nearest], f"cluster-{nearest + 1}"

    def validation_loss(self, model: Mapping[str, Any], rows: Sequence[EventWindowFeatures]) -> float:
        return median(float(np.mean(np.abs(self._residual(model, row)[0]))) for row in rows)

    def score(self, model: Mapping[str, Any], rows: Sequence[EventWindowFeatures]) -> list[float]:
        return [float(np.mean(np.abs(self._residual(model, row)[0]))) for row in rows]

    def explain(self, model: Mapping[str, Any], row: EventWindowFeatures) -> Mapping[str, Any]:
        residual, cluster = self._residual(model, row)
        return {"cluster": cluster,
                "feature_residuals": {name: float(residual[i]) for i, name in enumerate(FEATURES)},
                "features": dict(row.feature_values)}

    def encode(self, model: Mapping[str, Any]) -> Mapping[str, Any]:
        return model

    def decode(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return dict(payload)


CANDIDATES = (ProcessProfileCandidate(clusters=0), ProcessProfileCandidate(clusters=4))
