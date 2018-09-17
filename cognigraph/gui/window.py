from typing import List

from PyQt5 import QtCore, QtGui, QtWidgets

from ..pipeline import Pipeline
from .controls import Controls

from .screen_recorder import ScreenRecorder


class GUIWindow(QtGui.QMainWindow):
    def __init__(self, pipeline=Pipeline()):
        super().__init__()
        self._pipeline = pipeline  # type: Pipeline
        self._controls = Controls(pipeline=self._pipeline)
        self._controls_widget = self._controls.widget

        # Start button
        self.run_button = QtGui.QPushButton("Start")
        self.run_button.clicked.connect(self._toggle_run_button)

        # Record gif button and recorder
        self._gif_recorder = ScreenRecorder()
        self.gif_button = QtGui.QPushButton("Record gif")
        self.gif_button.clicked.connect(self._toggle_gif_button)

        # Resize screen
        self.resize(QtCore.QSize(
            QtGui.QDesktopWidget().availableGeometry().width() * 0.9,
            QtGui.QDesktopWidget().availableGeometry().height() * 0.9))

    def init_ui(self):
        self._controls.initialize()

        central_widget = QtWidgets.QSplitter()
        self.setCentralWidget(central_widget)

        # Build the controls portion of the window
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.addWidget(self._controls_widget)

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(self.run_button)
        buttons_layout.addWidget(self.gif_button)

        controls_layout.addLayout(buttons_layout)

        self._controls_widget.setMinimumWidth(400)

        # Add control portion to the main widget
        controls_layout_wrapper = QtGui.QWidget()
        controls_layout_wrapper.setLayout(controls_layout)
        self.centralWidget().addWidget(controls_layout_wrapper)

    def initialize(self):
        self._pipeline.initialize_all_nodes()
        for node_widget in self._node_widgets:
            node_widget.setMinimumWidth(600)

            # insert widget at before-the-end pos (just before controls widget)
            self.centralWidget().insertWidget(self.centralWidget().count() - 1,
                                              node_widget)
            self.centralWidget().insertWidget(self.centralWidget().count()-1, node_widget)

    def moveEvent(self, event):
        self._reset_gif_sector()
        return super(GUIWindow, self).moveEvent(event)

    def _reset_gif_sector(self):
        widgetRect = self.centralWidget().widget(0).geometry()
        widgetRect.moveTopLeft(self.centralWidget().mapToGlobal(widgetRect.topLeft()))
        self._gif_recorder.sector = (widgetRect.left(), widgetRect.top(), widgetRect.right(), widgetRect.bottom())

    def _toggle_run_button(self):
        if self.run_button.text() == "Pause":
            self.run_button.setText("Start")
        else:
            self.run_button.setText("Pause")

    def _toggle_gif_button(self):
        if self.gif_button.text() == "Stop recording":
            self.gif_button.setText("Record gif")

            self._gif_recorder.stop()
            save_path = QtGui.QFileDialog.getSaveFileName(
                caption="Save the recording", filter="Gif image (*.gif)")[0]
            self._gif_recorder.save(save_path)
        else:
            self._reset_gif_sector()
            self.gif_button.setText("Stop recording")
            self._gif_recorder.start()

    @property
    def _node_widgets(self) -> List[QtGui.QWidget]:
        node_widgets = list()
        for node in self._pipeline.all_nodes:
            try:
                node_widgets.append(node.widget)
            except AttributeError:
                pass
        return node_widgets
