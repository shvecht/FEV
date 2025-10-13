from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class CollapsibleSection(QtWidgets.QFrame):
    """A helper widget that animates expansion/collapse of its content."""

    toggled = QtCore.Signal(bool)

    def __init__(
        self,
        title: str,
        content: QtWidgets.QWidget,
        *,
        expanded: bool = True,
        animation_duration_ms: int = 150,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)

        self._content = content
        self._content.setParent(self)
        self._content.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        self._content.setMaximumHeight(0)
        self._content.setMinimumHeight(0)

        self._toggle = QtWidgets.QToolButton(self)
        self._toggle.setText(title)
        self._toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(
            QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow
        )
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setCursor(QtCore.Qt.PointingHandCursor)
        self._toggle.setAutoRaise(True)

        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addWidget(self._toggle)

        content_layout = QtWidgets.QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self._content)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 10)
        layout.setSpacing(10)
        layout.addLayout(header_layout)
        layout.addLayout(content_layout)

        self._animation = QtCore.QPropertyAnimation(self._content, b"maximumHeight", self)
        self._animation.setDuration(animation_duration_ms)
        self._animation.setEasingCurve(QtCore.QEasingCurve.InOutCubic)
        self._animation.finished.connect(self._handle_animation_finished)

        self._toggle.clicked.connect(self._on_toggled)

        self._set_expanded(expanded, animate=False)

    def is_expanded(self) -> bool:
        return self._toggle.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        if self.is_expanded() == expanded:
            return
        self._toggle.setChecked(expanded)
        self._on_toggled(expanded)

    def content_widget(self) -> QtWidgets.QWidget:
        return self._content

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(
            QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow
        )
        self._set_expanded(checked, animate=True)
        self.toggled.emit(checked)

    def _set_expanded(self, expanded: bool, *, animate: bool) -> None:
        target_height = self._content.sizeHint().height() if expanded else 0
        if expanded and target_height <= 0:
            target_height = self._content.minimumSizeHint().height()
        if not animate:
            self._animation.stop()
            if expanded:
                self._content.setMaximumHeight(QtWidgets.QWIDGETSIZE_MAX)
            else:
                self._content.setMaximumHeight(0)
            self._content.setVisible(expanded)
            return

        self._content.setVisible(True)
        self._animation.stop()
        start_value = self._content.maximumHeight()
        if start_value in (0, QtWidgets.QWIDGETSIZE_MAX):
            start_value = self._content.height()
        self._content.setMaximumHeight(start_value)
        self._animation.setStartValue(start_value)
        self._animation.setEndValue(target_height)
        self._animation.start()

    def _handle_animation_finished(self) -> None:
        if not self.is_expanded():
            self._content.setVisible(False)
            self._content.setMaximumHeight(0)
        else:
            self._content.setMaximumHeight(QtWidgets.QWIDGETSIZE_MAX)

