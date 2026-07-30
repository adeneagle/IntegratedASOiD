"""A-SOID numeric core — pose → features → RandomForest → active learning.

Vendored from the upstream A-SOID Streamlit application (YttriLab/A-SOID,
`asoid/utils/`) with every Streamlit, stqdm and Plotly dependency removed so the
code is importable inside the FastAPI backend and unit-testable without a UI.

The arithmetic is intentionally kept identical to upstream so that models and
metrics are comparable between this app and the reference Streamlit
implementation. Where a genuine upstream bug had to be corrected to run at all,
the change is called out in an `UPSTREAM FIX` comment.

Modules:
    features        spatio-temporal feature extraction (numba)
    pose_io         DeepLabCut / SLEAP / OpenMonkeyStudio loaders + likelihood filter
    predict         RandomForest prediction helpers (incl. frame-shifted)
    active_learning the subsample → low-confidence → retrain loop
    indexing        feature-row ↔ native-video-frame mapping
"""
