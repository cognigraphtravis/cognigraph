from typing import List

from pyqtgraph import QtCore, QtGui

from ..pipeline import Pipeline
from .controls import Controls


class GUIWindow(QtGui.QMainWindow):
    def __init__(self, pipeline=Pipeline()):
        super().__init__()
        self._pipeline = pipeline  # type: Pipeline
        self._controls = Controls(pipeline=self._pipeline)
        self._controls_widget = self._controls.widget
        
        #Start button and timer
        self.timer = QtCore.QTimer()
        self._control_button = QtGui.QPushButton("Start")
        self._control_button.clicked.connect(self._toggle_timer)
        
        self.main_layout = None  # type: QtGui.QBoxLayout
        
    def init_ui(self):
        self._controls.initialize()

        central_widget = QtGui.QWidget()
        self.setCentralWidget(central_widget)
        
        # Build the controls portion of the window
        controls_layout = QtGui.QVBoxLayout()
        controls_layout.addWidget(self._controls_widget)
        controls_layout.addWidget(self._control_button)        
        
        # Add control portion to the main layout
        main_layout = QtGui.QHBoxLayout()
        self._controls_widget.setMinimumWidth(400)
        main_layout.addLayout(controls_layout)

        self.centralWidget().setLayout(main_layout)
        self.main_layout = main_layout

    def initialize(self):
        self._pipeline.initialize_all_nodes()
        for node_widget in self._node_widgets:
            node_widget.setMinimumWidth(400)
            
            # insert widget at before-the-end pos (just before controls widget)
            self.main_layout.insertWidget(self.main_layout.count()-1, node_widget)

    def _toggle_timer(self):
        if self.timer.isActive():
            self.timer.stop()
            self._control_button.setText("Start")
        else:
            self.timer.start()
            self._control_button.setText("Pause")            
            
    @property
    def _node_widgets(self) -> List[QtGui.QWidget]:
        node_widgets = list()
        for node in self._pipeline.all_nodes:
            try:
                node_widgets.append(node.widget)
            except AttributeError:
                pass
        return node_widgets
