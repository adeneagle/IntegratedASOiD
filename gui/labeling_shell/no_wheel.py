"""Widgets that don't steal the mouse wheel.

Qt's combo boxes and spin boxes change value on wheel events even when they
don't have focus. Inside a scrolling container that is a data-loss bug rather
than a nicety: scrolling the bouts table past a behavior combo silently
reassigns whichever bout happened to be under the cursor, and scrolling the
Train tab past a spin box silently edits a hyperparameter.

These subclasses ignore the wheel unless the widget has keyboard focus, so the
event propagates to the scroll area instead. Click into a control first and the
wheel works normally on it.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class _NoWheelMixin:
    def wheelEvent(self, event):  # noqa: N802 (Qt naming)
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoWheelComboBox(QtWidgets.QComboBox):
    """A combo box that never changes its current item on the wheel.

    Qt's default QComboBox.wheelEvent cycles the selected item on any hover
    scroll, popup open or not. That's fine for a lone dropdown, but inside a
    scrolling table it silently reassigns whatever row happens to be under the
    cursor. The fix is to always ignore the wheel on the closed control — the
    event then propagates to the scroll area instead. This does not affect
    scrolling within the popup list itself once it's open: that list is a
    separate top-level widget with its own wheel handling, untouched here.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):  # noqa: N802 (Qt naming)
        event.ignore()


class NoWheelSpinBox(_NoWheelMixin, QtWidgets.QSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)


class NoWheelDoubleSpinBox(_NoWheelMixin, QtWidgets.QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
