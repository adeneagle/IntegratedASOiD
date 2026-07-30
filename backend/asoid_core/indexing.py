"""Feature-row ↔ native-video-frame mapping.

This is the single most load-bearing piece of the port. The rest of the desktop
app (labels, the uncertainty heatmap, misclassification ticks, "jump to lowest
confidence") indexes everything by **native video frame number**. A-SOID's
feature extractor instead works in *bins*: it collapses a run of consecutive
frames into one feature row. Get the conversion wrong and nothing errors — the
heatmap just silently points at the wrong part of the video.

Where the numbers come from
---------------------------
`frames2integ = round(framerate * min_duration / 0.1)`
    from `asoid/apps/B_extract_features.py:60`. With the defaults
    (30 fps, min_duration 0.1 s) this is 30.

`bin_width = int(frames2integ / 10)`
    from `fast_feature_binning`, which takes its bin width as
    `int(framerate / 10)` but is *called* with `frames2integ` in the
    `framerate` position (see `bsoid_extract_numba`). So the effective bin is
    `frames2integ // 10` native frames — 3 frames, i.e. 100 ms at 30 fps.

`targets[(num2skip - 1)::num2skip]`
    from `Extract.downsample_labels`, confirming the same stride is used to
    align labels with feature rows, and that a bin is represented by its
    *last* frame.
"""

from __future__ import annotations

import numpy as np


def frames2integ_from(framerate: float, min_duration: float = 0.1) -> int:
    """Bin-scaling constant A-SOID threads through feature extraction.

    Mirrors `B_extract_features.py:60` exactly, including its use of banker's
    rounding via the builtin `round`.
    """
    return int(round(framerate * (min_duration / 0.1)))


def bin_width_from(frames2integ: int) -> int:
    """Native frames per feature row (`num2skip` in upstream `downsample_labels`)."""
    return max(1, int(frames2integ) // 10)


def row_to_frames(row_index: int, bin_width: int) -> range:
    """Native frame indices covered by feature row `row_index`.

    Bins are contiguous and non-overlapping, so row *i* owns
    ``[i * bin_width, (i + 1) * bin_width)``.
    """
    start = row_index * bin_width
    return range(start, start + bin_width)


def frame_to_row(frame_index: int, bin_width: int) -> int:
    """Inverse of :func:`row_to_frames` — which feature row covers this frame."""
    return int(frame_index) // bin_width


def expand_rows_to_frames(
    values: np.ndarray, bin_width: int, n_frames: int | None = None
) -> np.ndarray:
    """Repeat one value per feature row out to one value per native frame.

    Args:
        values: shape (n_rows,) or (n_rows, k) — per-bin predictions/probabilities.
        bin_width: native frames per row.
        n_frames: truncate/pad to exactly this many frames. Feature extraction
            drops the remainder frames that don't fill a whole bin, so the
            expansion is short by up to `bin_width - 1` frames; the tail is
            edge-padded to keep the array aligned with the video.

    Returns:
        Array whose first axis is native frame index.
    """
    values = np.asarray(values)
    expanded = np.repeat(values, bin_width, axis=0)
    if n_frames is None:
        return expanded
    if expanded.shape[0] >= n_frames:
        return expanded[:n_frames]
    pad = n_frames - expanded.shape[0]
    pad_spec = [(0, pad)] + [(0, 0)] * (expanded.ndim - 1)
    return np.pad(expanded, pad_spec, mode="edge")


def downsample_targets(targets: np.ndarray, bin_width: int, n_rows: int) -> np.ndarray:
    """Collapse per-frame labels to one label per feature row.

    Upstream (`Extract.downsample_labels`) takes the *mode* of each bin, so a
    bin straddling a bout boundary gets the label that dominates it rather than
    whichever frame happens to land on the sampling stride.
    """
    targets = np.asarray(targets)
    usable = min(n_rows, targets.shape[0] // bin_width)
    if usable == 0:
        return np.zeros(0, dtype=int)
    trimmed = targets[: usable * bin_width].reshape(usable, bin_width)
    # Bin-count mode along axis 1. Labels are small non-negative ints, so a
    # bincount per row is both faster and dependency-free versus scipy.stats.
    n_classes = int(trimmed.max()) + 1 if trimmed.size else 1
    counts = np.zeros((usable, n_classes), dtype=np.int64)
    for c in range(n_classes):
        counts[:, c] = (trimmed == c).sum(axis=1)
    return counts.argmax(axis=1).astype(int)
