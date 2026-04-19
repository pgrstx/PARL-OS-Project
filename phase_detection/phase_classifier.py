"""
PhaseClassifier — clusters feature vectors into 5 distinct phases using k-means.

Plain English:
  Imagine plotting each window's 6 features as a dot in 6D space. Windows from
  the same type of workload (e.g., sequential scan) will cluster together.
  K-means finds 5 natural clusters and labels each window with its cluster ID.

  We use ONLINE (incremental) k-means because we're processing a live stream —
  we can't re-cluster everything from scratch every window. MiniBatchKMeans
  updates its cluster centers incrementally as new data arrives.

  During warm-up (first 10 windows), we just return phase 0 until we have
  enough data to make meaningful cluster assignments.
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from .feature_extractor import FeatureExtractor


class PhaseClassifier:
    def __init__(self, n_phases: int = 5, window_size: int = 1000,
                 warmup_windows: int = 10, random_state: int = 42):
        self._n_phases = n_phases
        self._window_size = window_size
        self._warmup_windows = warmup_windows
        self._feature_extractor = FeatureExtractor(window_size)

        self._kmeans = MiniBatchKMeans(
            n_clusters=n_phases,
            random_state=random_state,
            n_init=3,
            batch_size=max(10, warmup_windows),
        )

        self._windows_seen = 0
        self._warmup_buffer: list[np.ndarray] = []
        self._fitted = False

        # Current phase state
        self._current_phase_id: int = 0
        self._current_embedding: np.ndarray = np.zeros(6, dtype=np.float32)

        # Access counter within current window
        self._access_count = 0

    def update(self, page_id: int, is_hit: bool):
        """Feed one page access to the classifier."""
        self._feature_extractor.push(page_id, is_hit)
        self._access_count += 1

        # Update phase every full window
        if self._access_count >= self._window_size:
            self._access_count = 0
            self._update_phase()

    def _update_phase(self):
        features = self._feature_extractor.extract()

        if not self._fitted:
            self._warmup_buffer.append(features)
            if len(self._warmup_buffer) >= self._warmup_windows:
                X = np.array(self._warmup_buffer)
                self._kmeans.partial_fit(X)
                self._fitted = True
                self._windows_seen = len(self._warmup_buffer)
        else:
            # Online update
            self._kmeans.partial_fit(features.reshape(1, -1))
            self._windows_seen += 1

        # Predict current phase
        if self._fitted:
            phase_id = int(self._kmeans.predict(features.reshape(1, -1))[0])
            self._current_phase_id = phase_id
            self._current_embedding = self._kmeans.cluster_centers_[phase_id].astype(np.float32)
        else:
            self._current_phase_id = 0
            self._current_embedding = features

    @property
    def phase_id(self) -> int:
        return self._current_phase_id

    @property
    def phase_embedding(self) -> np.ndarray:
        """8-dimensional embedding of current phase for RL state input."""
        emb = self._current_embedding
        # Pad or truncate to exactly 8 dimensions
        if len(emb) < 8:
            emb = np.pad(emb, (0, 8 - len(emb)))
        elif len(emb) > 8:
            emb = emb[:8]
        return emb.astype(np.float32)

    @property
    def is_warmed_up(self) -> bool:
        return self._fitted

    def reset(self):
        self._feature_extractor.reset()
        self._warmup_buffer.clear()
        self._fitted = False
        self._windows_seen = 0
        self._access_count = 0
        self._current_phase_id = 0
        self._current_embedding = np.zeros(6, dtype=np.float32)
        self._kmeans = MiniBatchKMeans(
            n_clusters=self._n_phases,
            random_state=42,
            n_init=3,
            batch_size=max(10, self._warmup_windows),
        )
