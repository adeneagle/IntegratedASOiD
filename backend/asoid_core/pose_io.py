"""Pose-file loading, body-part selection and likelihood filtering.

Vendored from `asoid/utils/import_data.py` (loaders) and
`asoid/utils/preprocessing.py` (`adp_filt`), with Streamlit removed.

All loaders return a DataFrame with a MultiIndex column schema
``(Animal, bodypart, coords)`` where `coords` is x/y[/z]/likelihood, so
downstream selection is uniform across DeepLabCut, SLEAP and OpenMonkeyStudio.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

SUPPORTED_ORIGINS = ("deeplabcut", "sleap", "openmonkeystudio")


# ── Schema helpers ────────────────────────────────────────────────────────────

def get_animals(df: pd.DataFrame, lvl: int = 1) -> list[str]:
    """Animal list from a df in the internal MultiIndex format. Excludes Label."""
    animals = list(df.columns.get_level_values(lvl).unique())
    if "Label" in animals:
        animals.remove("Label")
    return animals


def get_bodyparts(df: pd.DataFrame, lvl: int = 1) -> list[str]:
    """Body-part list from a df in the internal MultiIndex format."""
    bodyparts = list(df.columns.get_level_values(lvl).unique())
    # Drop empties left behind by MultiIndex levels that don't apply
    while "" in bodyparts:
        bodyparts.remove("")
    return bodyparts


def level_indices(origin: str, multi_animal: bool) -> tuple[int | None, int]:
    """Which MultiIndex levels hold the animal and the body part.

    DeepLabCut keeps a `scorer` level at 0, so multi-animal DLC has animals at
    level 1 and body parts at 2. SLEAP files converted by `load_sleap_data` have
    no scorer level, so animals sit at 0 and body parts at 1.
    """
    if not multi_animal:
        return None, 1
    if origin.lower() == "deeplabcut":
        return 1, 2
    return 0, 1


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_dlc_data(path, multi_animal: bool = False) -> pd.DataFrame:
    """Load a DeepLabCut csv export."""
    if not multi_animal:
        return pd.read_csv(path, header=[0, 1, 2], sep=",", index_col=0)
    return pd.read_csv(path, header=[0, 1, 2, 3], sep=",", index_col=0)


def load_sleap_data(path, multi_animal: bool = False) -> pd.DataFrame:
    """Load a SLEAP analysis .h5 and normalize it to the internal schema.

    SLEAP nodes have no intrinsic order, so they are sorted alphabetically for
    determinism. SLEAP also exports no per-point score — it drops lost tracks to
    NaN instead — so a synthetic likelihood column is added (1.0 normally, 0.0
    where the point was NaN) to keep `adp_filt` applicable.
    """
    with h5py.File(path, "r") as f:
        tracks_matrix = f["tracks"][:].T
        animals = []
        temp_tracks = None
        for an in np.arange(tracks_matrix.shape[3]):
            animals.append("Animal{}".format(an))
            if temp_tracks is None:
                temp_tracks = tracks_matrix[:, :, :, an]
            else:
                temp_tracks = np.concatenate([temp_tracks, tracks_matrix[:, :, :, an]], axis=1)
        bodyparts = f["node_names"][:].astype("U13")

    likelihood_matrix = np.ones((temp_tracks.shape[0], temp_tracks.shape[1], 1))
    sleap_matrix = np.concatenate((temp_tracks, likelihood_matrix), axis=2)
    sleap_array = np.reshape(
        sleap_matrix, (sleap_matrix.shape[0], sleap_matrix.shape[1] * sleap_matrix.shape[2])
    )
    column_names = pd.MultiIndex.from_product(
        [animals, bodyparts, ("x", "y", "likelihood")],
        names=["Animal", "bodypart", "coords"],
    )
    df = pd.DataFrame(sleap_array, columns=column_names)

    # Mark lost tracking (NaN) as zero likelihood so adp_filt carries the last
    # good point forward instead of propagating NaN into the features.
    for ani in animals:
        for bp in bodyparts:
            any_nan = np.any(pd.isna(df[[(ani, bp, "x"), (ani, bp, "y")]]).values, axis=1)
            nan_indexes = np.argwhere(any_nan).flatten()
            df.loc[df.index[nan_indexes], (ani, bp, "likelihood")] = 0

    clean_df = df.interpolate(method="linear", limit_direction="forward")
    clean_df = clean_df.bfill()
    return clean_df.reindex(columns=sorted(bodyparts), level=1)


def load_opm_data(path) -> pd.DataFrame:
    """Load OpenMonkeyStudio 3D pose from a .mat or .txt file."""
    path = Path(path)
    if path.name.endswith(".mat"):
        mat = loadmat(str(path))
        monkey_pose = pd.DataFrame(mat["coords"], columns=["idx", "x", "y", "z"])
    elif path.name.endswith(".txt"):
        monkey_pose = pd.read_csv(path, sep=" ", header=None, names=["idx", "x", "y", "z"])
    else:
        raise ValueError(f"OpenMonkeyPose file {path.name} is not a .mat or .txt file.")

    n_frames, n_bodyparts = np.unique(monkey_pose["idx"], return_counts=True)
    if not np.all(n_bodyparts == n_bodyparts[0]):
        raise ValueError("OpenMonkeyStudio file has a ragged number of body parts per frame.")
    bodyparts_opm = (
        "Nose, Head, Neck, RShoulder, RHand, LShoulder, LHand, Hip, "
        "RKnee, RFoot, LKnee, LFoot, Tail"
    ).split(", ")
    n_bodyparts = n_bodyparts[0]
    if n_bodyparts != len(bodyparts_opm):
        raise ValueError(
            f"Expected {len(bodyparts_opm)} OpenMonkeyStudio body parts, found {n_bodyparts}."
        )
    n_frames = len(n_frames)

    monkey_pose["likelihood"] = 1
    monkey_pose = monkey_pose.drop(columns="idx", errors="ignore")
    monkey_pose = monkey_pose.values.reshape((n_frames, n_bodyparts * 4))

    return pd.DataFrame(
        monkey_pose,
        columns=pd.MultiIndex.from_product(
            [["OpenMonkeyStudio"], bodyparts_opm, ["x", "y", "z", "likelihood"]],
            names=["Animal", "bodypart", "coords"],
        ),
    )


def load_pose(path, origin: str, multi_animal: bool = False) -> pd.DataFrame:
    """Load a pose-estimation file based on its origin software."""
    origin = origin.lower()
    if origin == "deeplabcut":
        return load_dlc_data(path, multi_animal)
    if origin == "sleap":
        return load_sleap_data(path, multi_animal)
    if origin == "openmonkeystudio":
        if multi_animal:
            raise ValueError("OpenMonkeyStudio does not support multi-animal pose estimation.")
        return load_opm_data(path)
    raise ValueError(f"Pose estimation file origin {origin!r} is not supported.")


# ── Inspection (drives the body-part picker in the UI) ────────────────────────

def inspect_pose(path, origin: str, multi_animal: bool = False) -> dict:
    """Summarize a pose file without committing to a selection."""
    df = load_pose(path, origin, multi_animal)
    animal_lvl, bp_level = level_indices(origin, multi_animal)
    coords = list(df.columns.get_level_values(-1).unique())
    return {
        "animals": get_animals(df, lvl=animal_lvl) if animal_lvl is not None else [],
        "bodyparts": get_bodyparts(df, lvl=bp_level),
        "n_frames": int(df.shape[0]),
        "is_3d": "z" in coords,
        "has_likelihood": "likelihood" in coords,
    }


def select_columns(
    df: pd.DataFrame,
    bodyparts: list[str],
    origin: str,
    animals: list[str] | None = None,
    multi_animal: bool = False,
) -> tuple[list[int], list[int], bool]:
    """Resolve body-part names to column indices.

    Mirrors `load_preprocess.Preprocess.get_config_params` + `compile_data`:
    gather every column belonging to a selected body part (and, for multi-animal
    projects, a selected animal), sort them, then split off the likelihood
    columns.

    The result is sorted by *file* column order, not by the order the caller
    listed body parts — so the feature layout depends only on (file schema,
    selected set), which is what makes a checkpoint's body-part fingerprint
    reproducible.

    Returns:
        (coordinate column indices, likelihood column indices, is_3d)
    """
    animal_lvl, bp_level = level_indices(origin, multi_animal)

    animal_idx = None
    if multi_animal and animals:
        parts = [
            np.argwhere(df.columns.get_level_values(animal_lvl) == an).flatten()
            for an in animals
        ]
        animal_idx = set(np.concatenate(parts).tolist()) if parts else set()

    bp_index_list = []
    for bp in bodyparts:
        bp_index = np.argwhere(df.columns.get_level_values(bp_level) == bp).flatten()
        if animal_idx is not None:
            bp_index = np.array([i for i in bp_index if i in animal_idx], dtype=int)
        if bp_index.size == 0:
            raise ValueError(f"Body part {bp!r} not found in this pose file.")
        bp_index_list.append(bp_index)

    selected = np.sort(np.concatenate(bp_index_list)).tolist()

    coords_level = df.columns.get_level_values(-1)
    idx_llh = [i for i in selected if coords_level[i] == "likelihood"]
    idx_selected = [i for i in selected if i not in idx_llh]
    is_3d = any(coords_level[i] == "z" for i in idx_selected)
    return idx_selected, idx_llh, is_3d


# ── Likelihood filtering ──────────────────────────────────────────────────────

def adp_filt(pose: pd.DataFrame, idx_selected, idx_llh, llh_value: float):
    """Hold the last confident position wherever likelihood drops below `llh_value`.

    Verbatim from `asoid/utils/preprocessing.py::adp_filt`. 2D only — 3D pose
    carries no meaningful per-point score (see `filter_pose`).

    Returns:
        (filtered (n_frames, n_bodyparts*2) array, per-bodypart rejected fraction)
    """
    datax = np.array(pose.iloc[:, idx_selected[::2]])
    datay = np.array(pose.iloc[:, idx_selected[1::2]])
    data_lh = np.array(pose.iloc[:, idx_llh])
    currdf_filt = np.zeros((datax.shape[0], (datax.shape[1]) * 2))
    perc_rect = [0] * data_lh.shape[1]
    for x in range(data_lh.shape[1]):
        llh = llh_value
        data_lh_float = data_lh[:, x].astype(float)
        perc_rect[x] = np.sum(data_lh_float < llh) / data_lh.shape[0]
        currdf_filt[0, (2 * x):(2 * x + 2)] = np.hstack([datax[0, x], datay[0, x]])
        for i in range(1, data_lh.shape[0]):
            if data_lh_float[i] < llh:
                currdf_filt[i, (2 * x):(2 * x + 2)] = currdf_filt[i - 1, (2 * x):(2 * x + 2)]
            else:
                currdf_filt[i, (2 * x):(2 * x + 2)] = np.hstack([datax[i, x], datay[i, x]])
    return currdf_filt.astype(float), perc_rect


def filter_pose(df: pd.DataFrame, idx_selected, idx_llh, llh_value: float, is_3d: bool):
    """Likelihood-filter (2D) or plain-select (3D) the chosen coordinate columns."""
    if is_3d:
        # 3D sources carry a constant dummy likelihood, so filtering is a no-op;
        # upstream skips it entirely.
        return np.asarray(df.iloc[:, idx_selected].values, dtype=float), []
    return adp_filt(df, idx_selected, idx_llh, llh_value)


def load_and_filter(
    path,
    origin: str,
    bodyparts: list[str],
    llh_value: float = 0.6,
    animals: list[str] | None = None,
    multi_animal: bool = False,
):
    """One-shot: load a pose file and return the array feature extraction wants.

    Returns:
        (array (n_frames, n_bodyparts * dims), is_3d, per-bodypart rejected fraction)
    """
    df = load_pose(path, origin, multi_animal)
    idx_selected, idx_llh, is_3d = select_columns(
        df, bodyparts, origin, animals=animals, multi_animal=multi_animal
    )
    if not is_3d and not idx_llh:
        warnings.warn(
            "No likelihood columns found for the selected body parts; "
            "skipping adaptive filtering.",
            RuntimeWarning,
        )
        return np.asarray(df.iloc[:, idx_selected].values, dtype=float), is_3d, []
    filtered, perc_rect = filter_pose(df, idx_selected, idx_llh, llh_value, is_3d)
    return filtered, is_3d, perc_rect


def keypoints_for_overlay(
    path,
    origin: str,
    bodyparts: list[str],
    animals: list[str] | None = None,
    multi_animal: bool = False,
):
    """Raw (unfiltered) x/y/likelihood per body part, for drawing on the video.

    Deliberately *not* likelihood-filtered: the point of the overlay is to show
    the user where tracking actually failed, which the hold-last-position filter
    would paper over.
    """
    df = load_pose(path, origin, multi_animal)
    animal_lvl, bp_level = level_indices(origin, multi_animal)
    coords_level = df.columns.get_level_values(-1)

    series = []
    names = []
    for bp in bodyparts:
        cols = np.argwhere(df.columns.get_level_values(bp_level) == bp).flatten()
        if multi_animal and animals:
            keep = set()
            for an in animals:
                keep.update(np.argwhere(df.columns.get_level_values(animal_lvl) == an).flatten().tolist())
            cols = np.array([c for c in cols if c in keep], dtype=int)
        if cols.size == 0:
            continue
        x_col = next((c for c in cols if coords_level[c] == "x"), None)
        y_col = next((c for c in cols if coords_level[c] == "y"), None)
        l_col = next((c for c in cols if coords_level[c] == "likelihood"), None)
        if x_col is None or y_col is None:
            continue
        likelihood = (
            df.iloc[:, l_col].to_numpy(dtype=float)
            if l_col is not None
            else np.ones(df.shape[0])
        )
        series.append(
            np.column_stack(
                [
                    df.iloc[:, x_col].to_numpy(dtype=float),
                    df.iloc[:, y_col].to_numpy(dtype=float),
                    likelihood,
                ]
            )
        )
        names.append(bp)

    if not series:
        return names, np.zeros((0, 0, 3))
    # (n_frames, n_bodyparts, 3)
    return names, np.stack(series, axis=1)
