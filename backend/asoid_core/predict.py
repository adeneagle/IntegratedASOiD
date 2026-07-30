"""RandomForest prediction helpers.

Vendored from `asoid/utils/predict.py`, keeping only the *noscale* paths —
A-SOID trains on unscaled features (see the scaling note in `features.py`), so
the scaler-taking variants were dead weight. `stqdm` is replaced by an optional
progress callback.

Two prediction modes:

**Binned** (`predict_binned`) — one prediction per feature row, i.e. per
`bin_width` native frames. Cheap; resolution is the bin.

**Frame-shifted** (`frameshift_predict_proba`) — runs the binning at every
phase offset and interleaves the results, recovering a prediction for *every*
native frame at `bin_width`× the cost. This is what upstream uses for
prediction, and it is what the desktop UI wants: the active-learning panel lets
the user jump to an individual frame, so blocky per-bin uncertainty would send
them to the start of a bin rather than the actual moment of confusion.
"""

from __future__ import annotations

import math

import numpy as np

from .features import asoid_extract_numba_3d, bsoid_extract_numba
from numba.typed import List


def _typed(arr):
    typed = List()
    typed.append(np.ascontiguousarray(arr, dtype=np.float64))
    return typed


def predict_binned(features, clf):
    """Per-feature-row class predictions."""
    return clf.predict(np.nan_to_num(features))


def predict_proba_binned(features, clf):
    """Per-feature-row class probabilities."""
    return clf.predict_proba(np.nan_to_num(features))


def normalized_entropy(proba: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row, scaled to [0, 1].

    Matches the uncertainty metric the labeling UI's heatmap already expects
    from the TSM backend, so `renderer.js` needs no changes: 0 = fully
    confident, 1 = uniform over classes.
    """
    proba = np.asarray(proba, dtype=float)
    n_classes = proba.shape[1]
    if n_classes <= 1:
        return np.zeros(proba.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(proba > 0, np.log(proba), 0.0)
    entropy = -(proba * logs).sum(axis=1)
    return np.clip(entropy / math.log(n_classes), 0.0, 1.0)


def frameshift_predict_proba(
    pose_array,
    clf,
    frames2integ,
    is_3d: bool = False,
    progress_cb=None,
):
    """Per-native-frame class predictions and probabilities.

    Ported from `frameshift_predict_proba`, dropping the scaler and the outer
    multi-file loop (callers handle one recording at a time so progress can be
    reported per file).

    Args:
        pose_array: (n_frames, n_bodyparts * dims) filtered pose.
        clf: fitted RandomForestClassifier.
        frames2integ: bin-scaling constant; see `indexing`.
        is_3d: use the 3D feature kernels.
        progress_cb: called as `progress_cb(done, total)` per phase shift.

    Returns:
        (labels (n_frames,), proba (n_frames, n_classes)) — both indexed by
        native video frame.
    """
    extract = asoid_extract_numba_3d if is_3d else bsoid_extract_numba
    n_frames = pose_array.shape[0]

    feats_new = [np.asarray(f) for f in extract(_typed(pose_array), frames2integ)]
    n_shifts = len(feats_new)

    labels = []
    proba = []
    for i, feats in enumerate(feats_new):
        p = predict_proba_binned(feats, clf)
        labels.append(clf.classes_[p.argmax(axis=1)])
        proba.append(p)
        if progress_cb is not None:
            progress_cb(i + 1, n_shifts)

    # Right-align each shifted sequence, then carry the previous shift's head
    # into the gap. This is upstream's padding scheme, preserved verbatim:
    # shift s produces one fewer usable bin at the front, so its first s
    # entries are borrowed from shift s-1.
    max_len = max(len(l) for l in labels)
    labels_pad = -1 * np.ones([n_shifts, max_len])
    proba_pad = -1 * np.ones([n_shifts, max_len, proba[0].shape[1]])
    for n, l in enumerate(labels):
        rev = l[::-1]
        labels_pad[n][0:len(rev)] = rev
        labels_pad[n] = labels_pad[n][::-1]
        if n > 0:
            labels_pad[n][0:n] = labels_pad[n - 1][0:n]
    for n, p in enumerate(proba):
        rev = p[::-1, :]
        proba_pad[n][0:len(rev), :] = rev
        proba_pad[n] = proba_pad[n][::-1, :]
        if n > 0:
            proba_pad[n][0:n, :] = proba_pad[n - 1][0:n, :]

    # Interleave the shifts back into native frame order.
    n_interleave = math.floor(frames2integ / 10)
    flat_labels = np.array([labels_pad[l] for l in range(n_interleave)]).flatten("F")
    stacked = np.array([proba_pad[l] for l in range(n_interleave)])
    flat_proba = stacked.reshape(-1, stacked.shape[2])

    # Feature extraction consumes frames from the front, so the prediction
    # sequence is shorter than the video; edge-pad the head to realign index 0.
    pad_n = n_frames - len(flat_labels)
    if pad_n > 0:
        flat_labels = np.pad(flat_labels, (pad_n, 0), "edge")
        flat_proba = np.pad(flat_proba, [(pad_n, 0), (0, 0)], "edge")
    elif pad_n < 0:
        flat_labels = flat_labels[:n_frames]
        flat_proba = flat_proba[:n_frames]

    return flat_labels.astype(int), flat_proba
