"""Fit, persist, and apply the versioned statistical lenses.

The core split is chronological: fit preprocessing and models on training
rows, choose score cutoffs on calibration rows, then score later rows. This
module accepts already-separated rows and does not receive evaluator labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch import nn

from .features import FEATURE_NAMES, matrix
from .models.events import stable_id
from .models.features import EventWindowFeatures
from .models.lenses import LensKind, LensVersion, Observation

# Small tabular fits are quicker and more repeatable without a large pool of
# competing BLAS/PyTorch worker threads in this local research workflow.
torch.set_num_threads(1)


class Autoencoder(nn.Module):
    """Small feed-forward reconstruction model used as one statistical lens."""

    def __init__(self, input_dim: int, latent_dim: int = 4):
        super().__init__()
        hidden = max(4, input_dim // 2)
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, input_dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Reconstruct standardized feature rows from their compressed code."""
        return self.decoder(self.encoder(values))


@dataclass
class FittedLenses:
    """In-memory models plus the cutoff and metadata needed to score later rows.

    The scaler is fitted on training rows. ``thresholds`` are fitted from the
    calibration interval, and ``training_cutoff`` records the temporal edge
    that separates model fitting from later data.
    """
    artifact_dir: Path
    scaler: StandardScaler
    autoencoder: Autoencoder
    isolation_forest: IsolationForest
    thresholds: dict[str, float]
    model_version: str
    training_cutoff: datetime
    config: dict[str, Any]


def _write_json(path: Path, payload: Any) -> None:
    """Write stable, readable JSON metadata beside a model artifact."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=lambda v: v.isoformat() if isinstance(v, datetime) else str(v)) + "\n")


def rows_signature(rows: list[EventWindowFeatures]) -> str:
    """Hash the row identities, times, feature values, and evidence references."""
    payload = [{"subject": row.subject, "start": row.window_start.isoformat(), "end": row.window_end.isoformat(),
                "features": dict(row.feature_values), "events": row.supporting_event_ids}
               for row in rows]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fit_lenses(
    training_rows: list[EventWindowFeatures], calibration_rows: list[EventWindowFeatures],
    artifact_dir: str | Path, *, seed: int = 11, epochs: int = 35, contamination: float = 0.01,
    training_cutoff: datetime | None = None,
) -> FittedLenses:
    """Fit the autoencoder and Isolation Forest on training rows only.

    A ``StandardScaler`` is learned from ``training_rows`` and reused for the
    calibration rows. The ``contamination`` quantile of each calibration score
    distribution becomes that lens's frozen cutoff. Tied values can result in
    more calibration alerts than the nominal contamination rate, and later
    drift can change the alert rate further.

    Args:
        training_rows: Feature windows available for model fitting.
        calibration_rows: Later, separate windows used only to set cutoffs.
        artifact_dir: Directory for model files and the JSON manifest.
        seed: Random seed used by PyTorch and Isolation Forest.
        epochs: Number of full-batch autoencoder optimization steps.
        contamination: Upper-tail calibration quantile, between zero and one-half.
        training_cutoff: Explicit UTC boundary to record in model provenance.

    Returns:
        A ``FittedLenses`` bundle ready to score later feature windows.

    This function does not read hidden labels or decide whether a flagged row
    is a real-world anomaly.
    """
    if len(training_rows) < 12 or not calibration_rows:
        raise ValueError("at least 12 training windows and one calibration window are required")
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5")
    target = Path(artifact_dir)
    target.mkdir(parents=True, exist_ok=True)
    x_train = matrix(training_rows)
    x_calibration = matrix(calibration_rows)
    # Fit every preprocessing statistic on training only. Fitting this scaler
    # on calibration or holdout rows would leak their distribution into scores.
    scaler = StandardScaler().fit(x_train)
    z_train = scaler.transform(x_train).astype(np.float32)
    z_calibration = scaler.transform(x_calibration).astype(np.float32)
    torch.manual_seed(seed)
    model = Autoencoder(input_dim=z_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    x_tensor = torch.from_numpy(z_train)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        residual = model(x_tensor) - x_tensor
        loss = torch.mean(residual * residual)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_ae = ((model(x_tensor) - x_tensor) ** 2).mean(dim=1).numpy()
        cal_tensor = torch.from_numpy(z_calibration)
        calibration_ae = ((model(cal_tensor) - cal_tensor) ** 2).mean(dim=1).numpy()
    forest = IsolationForest(n_estimators=160, contamination=contamination, random_state=seed, n_jobs=1)
    forest.fit(z_train)
    train_if = -forest.decision_function(z_train)
    calibration_if = -forest.decision_function(z_calibration)
    # The calibration slice is separate from model fitting, but empirical
    # quantiles are not a workload guarantee when scores are tied or drift.
    thresholds = {
        "autoencoder": float(np.quantile(calibration_ae, 1 - contamination)),
        "isolation_forest": float(np.quantile(calibration_if, 1 - contamination)),
    }
    cutoff = training_cutoff or max(row.window_end for row in training_rows)
    config = {"seed": seed, "epochs": epochs, "contamination": contamination, "latent_dim": 4,
              "feature_names": list(FEATURE_NAMES), "training_rows": len(training_rows),
              "calibration_rows": len(calibration_rows), "training_cutoff": cutoff.isoformat(),
              "calibration_thresholds": thresholds, "torch_version": str(torch.__version__),
              "training_signature": rows_signature(training_rows),
              "calibration_signature": rows_signature(calibration_rows)}
    torch.save({"state_dict": model.state_dict(), "input_dim": z_train.shape[1], "config": config}, target / "autoencoder.pt")
    joblib.dump({"forest": forest, "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}, target / "isolation_forest.joblib")
    np.savez(target / "preprocessing.npz", mean=scaler.mean_, scale=scaler.scale_, feature_names=np.asarray(FEATURE_NAMES))
    digest = hashlib.sha256((target / "autoencoder.pt").read_bytes() + (target / "isolation_forest.joblib").read_bytes()).hexdigest()[:16]
    _write_json(target / "manifest.json", {**config, "model_version": digest})
    return FittedLenses(target, scaler, model, forest, thresholds, digest, cutoff, config)


def load_lenses(artifact_dir: str | Path) -> FittedLenses:
    """Load a fitted bundle after checking its recorded model-file digest.

    The saved feature names and training row count restore the scaler shape.
    The artifact directory must be trusted because the Isolation Forest is
    loaded from a joblib serialization.
    """
    target = Path(artifact_dir)
    manifest = json.loads((target / "manifest.json").read_text())
    digest = hashlib.sha256((target / "autoencoder.pt").read_bytes() + (target / "isolation_forest.joblib").read_bytes()).hexdigest()[:16]
    if digest != manifest.get("model_version"):
        raise ValueError("model artifact content does not match its recorded version")
    preprocessing = np.load(target / "preprocessing.npz", allow_pickle=False)
    scaler = StandardScaler()
    scaler.mean_ = preprocessing["mean"]
    scaler.scale_ = preprocessing["scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(preprocessing["feature_names"])
    scaler.n_samples_seen_ = int(manifest["training_rows"])
    autoencoder_payload = torch.load(target / "autoencoder.pt", map_location="cpu", weights_only=True)
    model = Autoencoder(autoencoder_payload["input_dim"], autoencoder_payload["config"]["latent_dim"])
    model.load_state_dict(autoencoder_payload["state_dict"])
    model.eval()
    forest_payload = joblib.load(target / "isolation_forest.joblib")
    cutoff = datetime.fromisoformat(manifest["training_cutoff"])
    return FittedLenses(target, scaler, model, forest_payload["forest"], manifest["calibration_thresholds"], manifest["model_version"], cutoff, manifest)


def score_rows(rows: list[EventWindowFeatures], lenses: FittedLenses, *,
               state_snapshot: dict[str, Any] | Callable[[EventWindowFeatures], dict[str, Any]] | None = None) -> list[Observation]:
    """Score windows and return one evidence-linked observation per lens.

    Larger scores are more unusual for both detectors. ``is_anomaly`` is the
    simple comparison ``score >= calibrated threshold``; it is not a verified
    label. A callable ``state_snapshot`` is evaluated for each output
    observation so each lens result can receive state as of its window end.
    """
    if not rows:
        return []
    z = lenses.scaler.transform(matrix(rows)).astype(np.float32)
    with torch.no_grad():
        tensor = torch.from_numpy(z)
        residuals = (lenses.autoencoder(tensor) - tensor).numpy()
        ae_scores = np.mean(residuals * residuals, axis=1)
        latent = lenses.autoencoder.encoder(tensor).numpy()
    if_scores = -lenses.isolation_forest.decision_function(z)
    ae_lens = LensVersion(lens_id="stat.autoencoder", version=lenses.model_version, kind=LensKind.AUTOENCODER,
                          description="Per-window reconstruction residual score", feature_schema=tuple(FEATURE_NAMES),
                          training_cutoff=lenses.training_cutoff, artifact_path=str(lenses.artifact_dir), config=lenses.config)
    if_lens = LensVersion(lens_id="stat.isolation_forest", version=lenses.model_version, kind=LensKind.ISOLATION_FOREST,
                          description="Per-window Isolation Forest anomaly score", feature_schema=tuple(FEATURE_NAMES),
                          training_cutoff=lenses.training_cutoff, artifact_path=str(lenses.artifact_dir), config=lenses.config)
    observations = []
    for index, row in enumerate(rows):
        for lens, kind, score, threshold in (
            (ae_lens, "autoencoder_reconstruction_score", float(ae_scores[index]), lenses.thresholds["autoencoder"]),
            (if_lens, "isolation_forest_score", float(if_scores[index]), lenses.thresholds["isolation_forest"]),
        ):
            observations.append(Observation(
                observation_id=stable_id(lens.lens_id, lens.version, row.subject, row.window_start.isoformat(), row.window_end.isoformat()),
                observation_type=kind, lens_id=lens.lens_id, lens_version=lens.version,
                subject=row.subject, window_start=row.window_start, window_end=row.window_end,
                value={"score": score, "threshold": threshold, "is_anomaly": score >= threshold,
                       "feature_residuals": {name: float(residuals[index, offset]) for offset, name in enumerate(FEATURE_NAMES)} if lens.kind == LensKind.AUTOENCODER else None,
                       "features": row.feature_values},
                supporting_event_ids=row.supporting_event_ids,
                state_snapshot=(state_snapshot(row) if callable(state_snapshot) else state_snapshot) or {}, model_version=lenses.model_version,
                quality={"calibrated": True, "training_cutoff": lenses.training_cutoff.isoformat(), "cohort": row.cohort},
                provenance={"artifact_dir": str(lenses.artifact_dir), "threshold_source": "held_out_calibration", "latent_embedding": latent[index].tolist() if lens.kind == LensKind.ISOLATION_FOREST else None},
            ))
    return observations


def cluster_observations(observations: list[Observation], *, min_samples: int = 2) -> list[dict[str, Any]]:
    """Group flagged windows by standardized score and feature similarity.

    DBSCAN receives the score and feature vector for each flagged observation.
    This helper does not use labels or evidence text, and it does not currently
    encode time overlap despite older descriptions of the clustering. Treat
    clusters as a review aid, not an independent finding or causal account.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler as Scaler
    candidates = [obs for obs in observations if isinstance(obs.value, dict) and obs.value.get("is_anomaly")]
    if not candidates:
        return []
    rows = []
    for obs in candidates:
        features = obs.value.get("features") or {}
        # Detection scores and window features keep semantically related records close.
        rows.append([float(obs.value.get("score", 0)), *[float(features.get(name, 0)) for name in FEATURE_NAMES]])
    vectors = Scaler().fit_transform(np.asarray(rows))
    labels = DBSCAN(eps=2.2, min_samples=min_samples).fit_predict(vectors)
    result = []
    for label in sorted(set(labels)):
        members = [obs for obs, assigned in zip(candidates, labels, strict=True) if assigned == label]
        if label == -1:
            for member in members:
                result.append({"cluster_id": f"singleton:{member.observation_id}", "observation_ids": [member.observation_id], "subjects": [member.subject], "noise": True})
        else:
            result.append({"cluster_id": f"cluster:{label}", "observation_ids": [m.observation_id for m in members], "subjects": sorted({m.subject for m in members}), "noise": False})
    return result

