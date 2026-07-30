"""RandomForest training with A-SOID's active-learning loop.

Vendored from `asoid/utils/auto_active_learning.py::RF_Classify`, with all
Streamlit and Plotly removed. The arithmetic — the 80/20 split, the per-class
initial subsample, the `predict_proba.max(1) < conf_threshold` selection rule,
the RandomForest hyperparameters — is preserved exactly, so the F1 trajectory
this produces is comparable with the reference Streamlit app.

What the loop does
------------------
1. Fit a **ceiling** model on *all* labeled training data. Its per-class F1 is
   the "with full data" line the user is trying to reach.
2. Fit a **seed** model on `init_ratio` of each class (iteration 0).
3. Repeatedly: find training samples the current model is least sure about
   (max class probability below `conf_threshold`), draw up to
   `max_samples_iter` of them at random, add them to the training set, refit,
   and score on the held-out split.

The point is that step 3 reaches the ceiling using a small fraction of the
labels — and, in the desktop app, tells the user *which frames* are worth
their attention.

Upstream's large commented-out per-split blocks are dropped rather than carried
over; the single-split behaviour they were superseded by is what actually runs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


RF_KWARGS = dict(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    criterion="gini",
    class_weight="balanced_subsample",
)


def fingerprint_bodyparts(bodyparts: list[str], is_3d: bool) -> str:
    """Stable identity for a body-part selection.

    Feature width is derived from `nchoose2(n_bodyparts)`, and column order
    follows the pose file's schema restricted to the selection — so a model is
    only valid for the exact selection it was trained on. Applying it to a
    different one silently produces garbage (or, if the width happens to match,
    a plausible-looking wrong answer). Upstream hit exactly this: A-SOID commit
    1710d15, "Fixed error in prediction if keypoints where deselected".

    This fingerprint is stored on the checkpoint and re-checked at inference.
    """
    payload = "|".join(bodyparts) + f"|3d={bool(is_3d)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class IterationResult:
    """One row of the training log the UI streams."""

    iteration: int
    macro_f1: float
    f1_per_class: list[float]
    n_train: int
    n_lowconf_remaining: int
    n_samples_added: int


@dataclass
class ActiveLearningResult:
    model: RandomForestClassifier = None
    class_names: list[str] = field(default_factory=list)
    # Ceiling: a model fit on every labeled sample.
    full_macro_f1: float = 0.0
    full_f1_per_class: list[float] = field(default_factory=list)
    # Seed model (iteration 0) and each refinement iteration.
    iterations: list[IterationResult] = field(default_factory=list)


class RFActiveLearner:
    """Subsample → find low-confidence → relabel → refit.

    Args:
        features: (n_samples, n_features) unscaled feature matrix.
        targets: (n_samples,) integer class codes; the *last* code is "other".
        class_names: names indexed by class code, "other" last.
        init_ratio: fraction of each class used to seed iteration 0.
        max_iter: maximum refinement iterations.
        max_samples_iter: samples added per iteration.
        conf_threshold: a sample is "low confidence" below this max probability.
        exclude_other: drop the "other" class from training entirely.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        class_names: list[str],
        init_ratio: float = 0.01,
        max_iter: int = 100,
        max_samples_iter: int = 20,
        conf_threshold: float = 0.5,
        exclude_other: bool = False,
    ):
        self.features = np.asarray(features)
        self.targets = np.asarray(targets)
        self.class_names = list(class_names)
        self.init_ratio = init_ratio
        self.max_iter = max_iter
        self.max_samples_iter = max_samples_iter
        self.conf_threshold = conf_threshold
        self.exclude_other = exclude_other
        # "other" is by construction the last class code.
        self.label_code_other = max(np.arange(len(class_names))) if class_names else 0

        self.features_train = None
        self.features_heldout = None
        self.targets_train = None
        self.targets_heldout = None

    # ── data ──────────────────────────────────────────────────────────────

    def split_data(self):
        n = min(self.features.shape[0], self.targets.shape[0])
        X, y = self.features[:n], self.targets[:n]
        (
            self.features_train,
            self.features_heldout,
            self.targets_train,
            self.targets_heldout,
        ) = train_test_split(X, y, test_size=0.20, random_state=42)

    def _scored_classes(self):
        """Class codes present in the data, minus "other" if excluded."""
        unique = np.unique(
            np.hstack([np.hstack(self.targets_train), np.hstack(self.targets_heldout)])
        )
        if self.exclude_other:
            unique = unique[unique != self.label_code_other]
        return unique

    def _score(self, model):
        """Per-class and macro F1 on the held-out split.

        "other" is always excluded from scoring, even when it is trained on:
        it is the *absence* of a label, so counting it would inflate the score
        with the majority of frames.
        """
        predict = model.predict(np.nan_to_num(self.features_heldout))
        mask = self.targets_heldout != self.label_code_other
        curr_targets = self.targets_heldout[mask]
        flt_predict = predict[mask]
        labels = np.unique(curr_targets)
        per_class = f1_score(curr_targets, flt_predict, average=None, labels=labels)
        macro = f1_score(curr_targets, flt_predict, average="macro", labels=labels)
        return np.asarray(per_class, dtype=float), float(macro)

    def _train_probabilities(self, model):
        """Model confidence on the *labeled* training pool (excluding "other")."""
        return model.predict_proba(
            np.nan_to_num(self.features_train[self.targets_train != self.label_code_other])
        )

    # ── the loop ──────────────────────────────────────────────────────────

    def run(self, progress_cb=None) -> ActiveLearningResult:
        """Run ceiling fit + seed fit + refinement iterations.

        Args:
            progress_cb: called with a dict per emitted event, suitable for
                writing straight to the metrics JSONL the SSE endpoint tails.
        """

        def emit(payload):
            if progress_cb is not None:
                progress_cb(payload)

        self.split_data()
        result = ActiveLearningResult(class_names=self.class_names)

        unique_classes = self._scored_classes()
        if unique_classes.size == 0:
            raise ValueError(
                "No labeled samples to train on. Label some bouts first, or "
                "uncheck 'exclude other'."
            )
        if unique_classes.size < 2:
            raise ValueError(
                f"Only one class ({self.class_names[int(unique_classes[0])]}) has labeled "
                "samples. A classifier needs at least two."
            )

        emit({
            "event": "start",
            "n_samples": int(self.features.shape[0]),
            "n_features": int(self.features.shape[1]),
            "n_train": int(self.features_train.shape[0]),
            "n_heldout": int(self.features_heldout.shape[0]),
            "classes": self.class_names,
            "class_counts": {
                self.class_names[int(c)]: int(np.sum(self.targets == c))
                for c in np.unique(self.targets)
            },
            "exclude_other": self.exclude_other,
        })

        # 1. Ceiling model — every labeled sample.
        X_all = np.vstack([self.features_train[self.targets_train == c] for c in unique_classes])
        Y_all = np.hstack([self.targets_train[self.targets_train == c] for c in unique_classes])
        all_model = RandomForestClassifier(**RF_KWARGS)
        all_model.fit(X_all, Y_all)
        full_per_class, full_macro = self._score(all_model)
        result.full_f1_per_class = full_per_class.tolist()
        result.full_macro_f1 = full_macro
        emit({
            "event": "ceiling",
            "macro_f1": full_macro,
            "f1_per_class": full_per_class.tolist(),
            "n_train": int(X_all.shape[0]),
        })

        # 2. Seed model — init_ratio of each class.
        samples2train = [int(np.sum(self.targets_train == c) * self.init_ratio) for c in unique_classes]
        X_seed, Y_seed = [], []
        for n_samples, c in zip(samples2train, unique_classes):
            if n_samples > 0:
                X_seed.append(self.features_train[self.targets_train == c][:n_samples])
                Y_seed.append(self.targets_train[self.targets_train == c][:n_samples])
        if not X_seed:
            raise ValueError(
                f"Initial ratio {self.init_ratio} selects zero samples from every class. "
                "Raise it, or label more data."
            )
        X_train = np.vstack(X_seed)
        Y_train = np.hstack(Y_seed)
        if np.unique(Y_train).size < 2:
            raise ValueError(
                f"Initial ratio {self.init_ratio} seeds only one class. Raise it, or "
                "balance the labeled data."
            )

        model = RandomForestClassifier(**RF_KWARGS)
        model.fit(X_train, Y_train)
        per_class, macro = self._score(model)
        predict_prob = self._train_probabilities(model)
        n_lowconf = int(np.sum(predict_prob.max(1) < self.conf_threshold))

        result.iterations.append(IterationResult(
            iteration=0,
            macro_f1=macro,
            f1_per_class=per_class.tolist(),
            n_train=int(X_train.shape[0]),
            n_lowconf_remaining=n_lowconf,
            n_samples_added=int(X_train.shape[0]),
        ))
        emit({
            "event": "iteration",
            "iteration": 0,
            "macro_f1": macro,
            "f1_per_class": per_class.tolist(),
            "n_train": int(X_train.shape[0]),
            "n_lowconf_remaining": n_lowconf,
            "n_samples_added": int(X_train.shape[0]),
            "ceiling_macro_f1": full_macro,
        })
        result.model = model

        # 3. Refinement iterations.
        pool_X = self.features_train[self.targets_train != self.label_code_other]
        pool_Y = self.targets_train[self.targets_train != self.label_code_other]
        rng = np.random.RandomState(42)

        for it in range(self.max_iter):
            idx_lowconf = np.where(predict_prob.max(1) < self.conf_threshold)[0]
            if idx_lowconf.shape[0] == 0:
                emit({
                    "event": "converged",
                    "iteration": it,
                    "message": "No confusing samples remain — the model did the best it could.",
                })
                break

            n_draw = min(self.max_samples_iter, idx_lowconf.shape[0])
            idx_sampled = rng.choice(np.arange(idx_lowconf.shape[0]), n_draw, replace=False)
            new_X = pool_X[idx_lowconf][idx_sampled, :]
            new_Y = pool_Y[idx_lowconf][idx_sampled]

            X_train = np.vstack((X_train, new_X))
            Y_train = np.hstack((Y_train, new_Y))

            model = RandomForestClassifier(**RF_KWARGS)
            model.fit(X_train, Y_train)
            per_class, macro = self._score(model)
            predict_prob = self._train_probabilities(model)
            n_lowconf = int(np.sum(predict_prob.max(1) < self.conf_threshold))

            result.model = model
            result.iterations.append(IterationResult(
                iteration=it + 1,
                macro_f1=macro,
                f1_per_class=per_class.tolist(),
                n_train=int(X_train.shape[0]),
                n_lowconf_remaining=n_lowconf,
                n_samples_added=int(n_draw),
            ))
            emit({
                "event": "iteration",
                "iteration": it + 1,
                "macro_f1": macro,
                "f1_per_class": per_class.tolist(),
                "n_train": int(X_train.shape[0]),
                "n_lowconf_remaining": n_lowconf,
                "n_samples_added": int(n_draw),
                "ceiling_macro_f1": full_macro,
            })

        emit({"event": "done", "iterations": len(result.iterations)})
        return result
