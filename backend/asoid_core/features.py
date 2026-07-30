"""Spatio-temporal feature extraction from pose.

Vendored from `asoid/utils/extract_features_2D.py` and `extract_features_3D.py`.
Streamlit/stqdm imports and the `feature_extraction*` UI wrappers are dropped;
the numba kernels are byte-for-byte equivalent to upstream.

For every pair of selected body parts, each 100 ms bin yields:
  * the mean **length** of the inter-bodypart vector,
  * the summed frame-to-frame **angular change** of that vector,
  * the summed per-bodypart **displacement**.

So the feature width is `nchoose2(n_bodyparts) * 2 + n_bodyparts`. That makes
the *set and order* of selected body parts part of the model's identity — a
model cannot be applied to a pose file filtered to a different selection. See
`active_learning.fingerprint_bodyparts`.

NOTE ON SCALING: upstream fits a `StandardScaler` in `feature_extraction()` and
then throws it away — `Extract.save_features_targets` persists the *unscaled*
features, `RF_Classify` trains on those, and inference uses the explicitly named
`bsoid_predict_numba_noscale`. The scaler was vestigial. It is omitted here
rather than reintroduced: a RandomForest splits on thresholds and is invariant
to feature scaling, so this is both faithful to upstream and one less artifact
to keep in sync with a checkpoint.
"""

from __future__ import annotations

import numpy as np
from numba import jit
from numba.typed import List


# ── numba helpers ─────────────────────────────────────────────────────────────

@jit(nopython=True)
def fast_standardize(data):
    a_ = (data - np.mean(data)) / np.std(data)
    return a_


def fast_nchoose2(n, k):
    """Index pairs for all n-choose-k combinations (used only with k=2).

    Upstream places `return a` *inside* the `for j` loop. With k=2 the loop body
    executes exactly once, so hoisting the return out is behaviourally
    identical — it just no longer reads as an accident.
    """
    a = np.ones((k, n - k + 1), dtype=int)
    a[0] = np.arange(n - k + 1)
    for j in range(1, k):
        reps = (n - k + j) - a[j - 1]
        a = np.repeat(a, reps, axis=1)
        ind = np.add.accumulate(reps)
        a[j, ind[:-1]] = 1 - reps[1:]
        a[j, 0] = j
        a[j] = np.add.accumulate(a[j])
    return a


@jit(nopython=True)
def fast_running_mean(x, N):
    out = np.zeros_like(x, dtype=np.float64)
    dim_len = x.shape[0]
    for i in range(dim_len):
        if N % 2 == 0:
            a, b = i - (N - 1) // 2, i + (N - 1) // 2 + 2
        else:
            a, b = i - (N - 1) // 2, i + (N - 1) // 2 + 1
        a = max(0, a)
        b = min(dim_len, b)
        out[i] = np.mean(x[a:b])
    return out


@jit(nopython=True)
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@jit(nopython=True)
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)


@jit(nopython=True)
def np_std(array, axis):
    return np_apply_along_axis(np.std, axis, array)


@jit(nopython=True)
def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


@jit(nopython=True)
def angle_between(vector1, vector2):
    """Returns the signed angle in radians between given vectors."""
    v1_u = unit_vector(vector1)
    v2_u = unit_vector(vector2)
    minor = np.linalg.det(
        np.stack((v1_u[-2:], v2_u[-2:]))
    )
    if minor == 0:
        sign = 1
    else:
        sign = -np.sign(minor)
    dot_p = np.dot(v1_u, v2_u)
    dot_p = min(max(dot_p, -1.0), 1.0)
    return sign * np.arccos(dot_p)


@jit(nopython=True)
def fast_smooth(data, n):
    data_boxcar_avg = np.zeros((data.shape[0], data.shape[1]))
    for body_part in range(data.shape[1]):
        data_boxcar_avg[:, body_part] = fast_running_mean(data[:, body_part], n)
    return data_boxcar_avg


# ── 2D ────────────────────────────────────────────────────────────────────────

@jit(nopython=True)
def fast_displacment(data, reduce=False):
    data_length = data.shape[0]
    if reduce:
        displacement_array = np.zeros((data_length, int(data.shape[1] / 10)), dtype=np.float64)
    else:
        displacement_array = np.zeros((data_length, int(data.shape[1] / 2)), dtype=np.float64)
    for r in range(data_length):
        if r < data_length - 1:
            if reduce:
                count = 0
                for c in range(int(data.shape[1] / 2 - 2), data.shape[1], int(data.shape[1] / 2)):
                    displacement_array[r, count] = np.linalg.norm(data[r + 1, c:c + 2] - data[r, c:c + 2])
                    count += 1
            else:
                for c in range(0, data.shape[1], 2):
                    displacement_array[r, int(c / 2)] = np.linalg.norm(data[r + 1, c:c + 2] - data[r, c:c + 2])
    return displacement_array


@jit(nopython=True)
def fast_length_angle(data, index):
    data_length = data.shape[0]
    length_2d_array = np.zeros((data_length, index.shape[1], 2), dtype=np.float64)
    for r in range(data_length):
        for i in range(index.shape[1]):
            ref = index[0, i]
            target = index[1, i]
            length_2d_array[r, i, :] = data[r, ref:ref + 2] - data[r, target:target + 2]
    length_array = np.zeros((data_length, length_2d_array.shape[1]), dtype=np.float64)
    angle_array = np.zeros((data_length, length_2d_array.shape[1]), dtype=np.float64)
    for k in range(length_2d_array.shape[1]):
        for kk in range(data_length):
            length_array[kk, k] = np.linalg.norm(length_2d_array[kk, k, :])
            if kk < data_length - 1:
                try:
                    angle_array[kk, k] = np.rad2deg(
                        angle_between(length_2d_array[kk, k, :], length_2d_array[kk + 1, k, :]))
                except Exception:
                    pass
    return length_array, angle_array


@jit(nopython=True)
def fast_feature_extraction(data, framerate, index, smooth):
    window = int(np.round(0.05 / (1 / framerate)) * 2 - 1)
    features = List()
    for n in range(len(data)):
        displacement_raw = fast_displacment(data[n])
        length_raw, angle_raw = fast_length_angle(data[n], index)
        if smooth:
            displacement_run_mean = fast_smooth(displacement_raw, window)
            length_run_mean = fast_smooth(length_raw, window)
            angle_run_mean = fast_smooth(angle_raw, window)
            features.append(np.hstack((length_run_mean[1:, :], angle_run_mean[:-1, :], displacement_run_mean[:-1, :])))
        else:
            features.append(np.hstack((length_raw[:, :], angle_raw[:, :], displacement_raw[:, :])))
    return features


@jit(nopython=True)
def fast_feature_binning(features, framerate, index):
    """Collapse per-frame features into bins of `int(framerate / 10)` frames.

    Callers pass `frames2integ` in the `framerate` slot, so the effective bin is
    `frames2integ // 10` native frames. Lengths/angles are averaged over the
    bin; displacements are summed (distance travelled). Returns one entry per
    phase shift `s` — index 0 is the non-shifted binning used for training;
    the full list drives frame-shifted prediction.
    """
    binned_features_list = List()
    for n in range(len(features)):
        bin_width = int(framerate / 10)
        for s in range(bin_width):
            binned_features = np.zeros((int(features[n].shape[0] / bin_width), features[n].shape[1]), dtype=np.float64)
            for b in range(bin_width + s, features[n].shape[0], bin_width):
                binned_features[int(b / bin_width) - 1, 0:index.shape[1]] = np_mean(features[n][(b - bin_width):b,
                                                                                    0:index.shape[1]], 0)
                binned_features[int(b / bin_width) - 1, index.shape[1]:] = np.sum(features[n][(b - bin_width):b,
                                                                                  index.shape[1]:], axis=0)
            binned_features_list.append(binned_features)
    return binned_features_list


def bsoid_extract_numba(data, fps):
    """2D extraction. `data` is a numba typed List of (n_frames, n_bodyparts*2) arrays."""
    smooth = False
    index = fast_nchoose2(int(data[0].shape[1] / 2), 2)
    features = fast_feature_extraction(data, fps, index * 2, smooth)
    binned_features = fast_feature_binning(features, fps, index * 2)
    return binned_features


# ── 3D ────────────────────────────────────────────────────────────────────────

@jit(nopython=True)
def fast_displacment_3d(data, reduce=False):
    data_length = data.shape[0]
    if reduce:
        displacement_array = np.zeros((data_length, int(data.shape[1] / 10)), dtype=np.float64)
    else:
        displacement_array = np.zeros((data_length, int(data.shape[1] / 3)), dtype=np.float64)
    for r in range(data_length):
        if r < data_length - 1:
            if reduce:
                count = 0
                for c in range(int(data.shape[1] / 2 - 2), data.shape[1], int(data.shape[1] / 2)):
                    displacement_array[r, count] = np.linalg.norm(data[r + 1, c:c + 2] - data[r, c:c + 2])
                    count += 1
            else:
                for c in range(0, data.shape[1], 3):
                    displacement_array[r, int(c / 3)] = np.linalg.norm(data[r + 1, c:c + 3] - data[r, c:c + 3])
    return displacement_array


@jit(nopython=True)
def fast_length_angle_3d(data, index):
    data_length = data.shape[0]
    length_2d_array = np.zeros((data_length, index.shape[1], 3), dtype=np.float64)
    for r in range(data_length):
        for i in range(index.shape[1]):
            ref = index[0, i]
            target = index[1, i]
            length_2d_array[r, i, :] = data[r, ref:ref + 3] - data[r, target:target + 3]
    length_array = np.zeros((data_length, length_2d_array.shape[1]), dtype=np.float64)
    angle_array = np.zeros((data_length, length_2d_array.shape[1]), dtype=np.float64)
    for k in range(length_2d_array.shape[1]):
        for kk in range(data_length):
            length_array[kk, k] = np.linalg.norm(length_2d_array[kk, k, :])
            if kk < data_length - 1:
                try:
                    angle_array[kk, k] = np.rad2deg(
                        angle_between(length_2d_array[kk, k, :], length_2d_array[kk + 1, k, :]))
                except Exception:
                    pass
    return length_array, angle_array


@jit(nopython=True)
def fast_feature_extraction_3d(data, framerate, index, smooth):
    # UPSTREAM FIX: was `np.int(...)`, removed in numpy >= 1.24 — the 3D path
    # raised AttributeError on import-time-modern numpy and cannot have been
    # exercised recently upstream.
    window = int(np.round(0.05 / (1 / framerate)) * 2 - 1)
    features = List()
    for n in range(len(data)):
        displacement_raw = fast_displacment_3d(data[n])
        length_raw, angle_raw = fast_length_angle_3d(data[n], index)
        if smooth:
            displacement_run_mean = fast_smooth(displacement_raw, window)
            length_run_mean = fast_smooth(length_raw, window)
            angle_run_mean = fast_smooth(angle_raw, window)
            features.append(np.hstack((length_run_mean[1:, :], angle_run_mean[:-1, :], displacement_run_mean[:-1, :])))
        else:
            features.append(np.hstack((length_raw[1:, :], angle_raw[:-1, :], displacement_raw[:-1, :])))
    return features


def asoid_extract_numba_3d(data, fps):
    """3D extraction. `data` is a typed List of (n_frames, n_bodyparts*3) arrays."""
    smooth = False
    index = fast_nchoose2(int(data[0].shape[1] / 3), 2)
    # UPSTREAM FIX: upstream passes `index * 2` here, but 3D pose stores three
    # coordinates per body part, so a body-part index must be scaled by 3 to
    # become a column offset. With *2 the kernels read overlapping, misaligned
    # slices and the resulting features are meaningless. Combined with the
    # `np.int` crash above, the upstream 3D path could not have produced a
    # working model; there is no numerical compatibility to preserve here.
    features = fast_feature_extraction_3d(data, fps, index * 3, smooth)
    binned_features = fast_feature_binning(features, fps, index * 3)
    return binned_features


# ── public API ────────────────────────────────────────────────────────────────

def extract_features(pose_arrays, frames2integ, is_3d=False, all_shifts=False):
    """Extract binned features for one or more pose recordings.

    Args:
        pose_arrays: sequence of float64 arrays, each (n_frames, n_bodyparts * dims),
            already likelihood-filtered and restricted to the selected body parts.
        frames2integ: `round(framerate * min_duration / 0.1)`; see `indexing`.
        is_3d: use the 3D kernels.
        all_shifts: return every phase shift (for frame-shifted prediction)
            instead of only the non-shifted binning (for training).

    Returns:
        If `all_shifts` is False: one (n_rows, n_features) array, the recordings
        stacked vertically. Otherwise a list (per recording) of lists of
        `bin_width` shifted arrays.
    """
    extract = asoid_extract_numba_3d if is_3d else bsoid_extract_numba

    per_recording = []
    for arr in pose_arrays:
        typed = List()
        typed.append(np.ascontiguousarray(arr, dtype=np.float64))
        binned = extract(typed, frames2integ)
        per_recording.append([np.asarray(b) for b in binned])

    if all_shifts:
        return per_recording
    # Index 0 is the non-shifted binning — the one whose row i maps to native
    # frames [i * bin_width, (i+1) * bin_width). See `indexing.row_to_frames`.
    return np.vstack([r[0] for r in per_recording])


def feature_width(n_bodyparts: int, is_3d: bool = False) -> int:
    """Number of feature columns produced for a given body-part count.

    Used to reject a pose selection that doesn't match a trained model.
    """
    n_pairs = int(fast_nchoose2(n_bodyparts, 2).shape[1])
    return n_pairs * 2 + n_bodyparts
